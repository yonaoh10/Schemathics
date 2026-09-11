# Two images from one file: the weekly training job and the serving API.
#
# They are deliberately different. Training needs names-dataset (2.4 GB resident, 18 s
# to construct) to build the gender lookup table; serving reads that table out of the
# model bundle and never imports the library. Splitting the images is what turns a
# 20-second, multi-gigabyte cold start into a 2-second one.
#
#   docker build -f docker/Dockerfile --target training -t bl-ranking-train .
#   docker build -f docker/Dockerfile --target serving  -t bl-ranking-api   .

# Python 3.12 is a hard floor: the research scripts use PEP 701 f-strings and do not
# parse on 3.11.
FROM python:3.12-slim AS base

# LC_ALL is load-bearing, not cosmetic. session_day_of_week comes from
# Series.dt.day_name(), which follows LC_TIME: an image with a different locale emits
# 'Dienstag' - a categorical level the model has never seen - with no error anywhere.
# Pinned in BOTH images so training and serving agree.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src


# --------------------------------------------------------------------------------- #
FROM base AS training

RUN pip install --no-cache-dir ".[train]"

COPY conf ./conf
COPY scripts ./scripts

# Default to the evaluation mode; compose and the scheduler override it.
CMD ["python", "-m", "bl_ranking.training.job", "--mode", "train_test"]


# --------------------------------------------------------------------------------- #
FROM base AS serving

# mlflow-skinny is enough to resolve `models:/name@champion` and download the bundle;
# the full mlflow package pulls a web server and a database stack the API never uses.
RUN pip install --no-cache-dir ".[serve]"

COPY conf ./conf
COPY scripts ./scripts
RUN chmod +x scripts/*.sh

# One thread per worker: the scoring path is GIL-bound, and letting each of N workers
# start one BLAS thread per core collapses p99 through CPU oversubscription.
#
# TABPFN_DISABLE_TELEMETRY: the library reports usage to a third party by default.
# TABPFN_CLIENT_CI_MODE: the hosted client otherwise wraps every call in a thread and
#   busy-polls every 200 ms while drawing a spinner, quantising latency onto that grid.
# TABPFN_MODEL_CACHE_DIR: where baked-in weights live, so a container needs no network.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    BL_SERVING__PORT=8080 \
    TABPFN_DISABLE_TELEMETRY=1 \
    TABPFN_CLIENT_CI_MODE=true \
    TABPFN_MODEL_CACHE_DIR=/opt/tabpfn

EXPOSE 8080

# Readiness, not liveness: /readyz only returns 200 after a real scoring call has
# succeeded inside this worker, so traffic never reaches a cold process.
HEALTHCHECK --interval=10s --timeout=3s --start-period=90s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8080/readyz || exit 1

CMD ["./scripts/serve.sh"]
