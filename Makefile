# Everything a reviewer needs, in the order they would run it.
#
#   make setup      create the venv and install
#   make data       generate bl_full_data.csv (skipped if a real extract is present)
#   make ingest     CSV -> Delta table, with the data-quality gate
#   make train      evaluation mode: time split, per-day metrics, nothing registered
#   make train-prod production mode: fit on all data, register a version, move champion
#   make serve      run the API locally
#   make loadtest   open-loop latency sweep against a running API
#   make test       the fast suite
#   make test-all   everything, including the equivalence and gender-table checks
#
#   make all        data -> ingest -> production (registers) -> evaluation
#   make up         the whole stack in Docker (MLflow, scheduler, API)

SHELL := /bin/bash
PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
export PYTHONPATH := $(CURDIR)/src
# Exported so `make rollback VERSION=n MLFLOW_TRACKING_URI=http://localhost:5000` reaches
# the recipe's environment: registry.py resolves the registry from this variable, and
# without it a host rollback silently targets the local file store under mlruns/ instead
# of the registry the Docker deployment reads. Unset here, it stays the local file store.
export MLFLOW_TRACKING_URI
COMPOSE := docker compose -f docker/docker-compose.yml

# The offline-safe backend. Override for the real thing:
#   make train-prod BACKEND=tabpfn_local
#   make train-prod BACKEND=tabpfn_client   (needs TABPFN_TOKEN)
BACKEND ?= catboost_fallback
PORT ?= 8080

.PHONY: help setup setup-tabpfn data ingest train train-prod serve loadtest test test-all \
        lint clean all up down mlflow schedule rollback versions data-history

# Only the header block above, not every later comment that happens to show a command.
help:
	@awk '/^#/ {sub(/^#[ ]?/, ""); print; next} {exit}' Makefile

# `uv venv` deliberately creates an environment with no pip in it, so the install has
# to go through `uv pip`. Falling through to the stdlib venv keeps the target working
# on a machine that has neither uv nor a system pip in the new environment.
setup:
	@if command -v uv >/dev/null 2>&1; then \
	  uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[train,serve,dev]"; \
	else \
	  python3.12 -m venv .venv && $(PIP) install -e ".[train,serve,dev]"; \
	fi
	@echo "TabPFN is optional and large (torch). Install it with:"
	@echo "  make setup-tabpfn"

setup-tabpfn:
	@if command -v uv >/dev/null 2>&1; then \
	  uv pip install --python .venv/bin/python -e ".[tabpfn]"; \
	else \
	  $(PIP) install -e ".[tabpfn]"; \
	fi

data:
	$(PYTHON) -m bl_ranking.data.generate

ingest: data
	$(PYTHON) -m bl_ranking.data.ingest

train: ingest
	BL_MODEL__PAYOUT__BACKEND=$(BACKEND) $(PYTHON) -m bl_ranking.training.job --mode train_test

train-prod: ingest
	BL_MODEL__PAYOUT__BACKEND=$(BACKEND) $(PYTHON) -m bl_ranking.training.job --mode production

# Production first so a model is registered before the slower evaluation runs - the
# same ordering the Databricks job uses.
all: train-prod train

# BACKEND is passed through only when it was asked for on the command line. Serving
# otherwise reads the backend out of the bundle it loaded, which is what lets a rollback
# carry the backend its version was trained with; setting the variable unconditionally
# made that unreachable. `make serve BACKEND=surrogate` still overrides it.
serve:
	BL_SERVING__PORT=$(PORT) 	$(if $(filter command line,$(origin BACKEND)),BL_MODEL__PAYOUT__BACKEND=$(BACKEND),) 	./scripts/serve.sh

# Sweep rather than a single rate: the point at which achieved rps falls behind the
# target is the capacity number, and a single point cannot show it.
# GENERATORS, not workers: one asyncio process cannot schedule much past ~200 arrivals per
# second while sharing four cores with the server, and when it falls behind it charges its
# own lateness to the server - which is how this harness once reported 54 s p50 for an
# endpoint answering in 8 ms. Three generators offer the whole sweep with a send lag in the
# low milliseconds. Override with `make loadtest GENERATORS=1` to see the difference.
GENERATORS ?= 3

# 30 s rather than 20: the published numbers used to move by 2x with the duration, because
# a saturated generator's backlog never reaches steady state and every percentile grew with
# however long the run was left going. Long enough matters at the top of the sweep in the
# other direction too - a 20 s run read 400 rps as comfortable and three 30 s runs could not
# offer that rate at all. Even at 30 s, run the sweep more than once: 300 rps came back at
# p99 48, 54 and 750 ms on three identical runs, so a single sweep can only mislead. See
# docs/load-test.md, which publishes medians of three with the spread.
loadtest:
	$(PYTHON) loadtest/run_load.py --url http://127.0.0.1:$(PORT)/rank \
	  --rps 25,50,100,200,300,350,400 --duration 30 --warmup 5 \
	  --processes $(GENERATORS) \
	  --out loadtest/results/latency.json

schedule:
	$(PYTHON) -m bl_ranking.ops.schedule --show

# Rollback is a registry operation, not a redeploy: move the alias, restart the API.
#   make rollback VERSION=3
# Targets the local file store by default. To roll back the Docker deployment, name its
# registry so the alias moves where the API reads it:
#   make rollback VERSION=3 MLFLOW_TRACKING_URI=http://localhost:5000
rollback:
	@test -n "$(VERSION)" || (echo "usage: make rollback VERSION=<n>" && exit 1)
	$(PYTHON) -m bl_ranking.ops.registry rollback --version $(VERSION)

versions:
	$(PYTHON) -m bl_ranking.ops.registry list

data-history:
	$(PYTHON) -m bl_ranking.data.delta_cli history

test:
	$(PYTHON) -m pytest tests -m "not slow" -q

test-all:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m ruff check src tests loadtest

mlflow:
	$(COMPOSE) up mlflow -d
	@echo "MLflow UI: http://localhost:5000"

up:
	$(COMPOSE) up -d --build mlflow api scheduler

down:
	$(COMPOSE) down

clean:
	rm -rf runs mlruns data/delta loadtest/results .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
