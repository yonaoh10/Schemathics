#!/usr/bin/env bash
# Start the ranking API locally.
#
# Parallelism is by process, not thread: the scoring path is GIL-bound, so extra
# threads only add queueing. Each worker pins the numeric libraries to one thread
# (BL_SERVING__THREADS_PER_WORKER) so N workers do not oversubscribe the CPU.
#
#   ./scripts/serve.sh                 # host, port and workers from conf/config.yaml
#   BL_SERVING__WORKERS=1 ./scripts/serve.sh
#   BL_MODEL__PAYOUT__BACKEND=catboost_fallback ./scripts/serve.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
# A container (or a machine where the package is installed globally) has no
# .venv. Fall back rather than exec a path that does not exist.
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3 || command -v python)"; fi
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# Host, port and worker count come from the same place the app reads them: conf/config.yaml
# under BL_* overrides. The header above promised "workers from conf/config.yaml" while this
# script hard-coded 3 unless BL_SERVING__WORKERS was set, so editing the file the repo calls
# the single source of truth changed nothing. One Settings.load() costs ~0.2 s at start-up
# and makes the promise true; it also validates the configuration before uvicorn forks.
read -r HOST PORT WORKERS < <("$PYTHON" -c '
from bl_ranking.config import Settings
s = Settings.load().serving
print(s.host, s.port, s.workers)
')
# Set before numpy/torch initialise their thread pools.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# TabPFN phones home by default, and the hosted client's progress spinner busy-polls on
# a 200 ms grid. Neither belongs on a latency-sensitive path.
export TABPFN_DISABLE_TELEMETRY="${TABPFN_DISABLE_TELEMETRY:-1}"
export TABPFN_CLIENT_CI_MODE="${TABPFN_CLIENT_CI_MODE:-true}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$REPO_ROOT/.tabpfn_models}"

# Prometheus metrics are per-process objects, so with more than one worker a scrape would
# report only the worker that answered it - a third of the traffic at the default of 3.
# This variable makes prometheus_client keep the counters in shared mmap'd files instead,
# and /metrics merge them. It must be exported before the library is imported, which is
# why it is set here and not in the app. Left unset for a single worker, where the plain
# registry is already correct - unless the operator set it, in which case the app honours
# it whatever the worker count, so the clean-up below has to as well.
if (( WORKERS > 1 )) || [[ -n "${PROMETHEUS_MULTIPROC_DIR:-}" ]]; then
  export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-$REPO_ROOT/.metrics}"
  mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
  # The previous run's files are removed first: they are named by pid, and a recycled pid
  # would otherwise add a dead run's totals to this one. Only the files prometheus_client
  # writes are touched - never the directory. An earlier version did `rm -rf` on the path,
  # which for an operator who pointed the variable at a directory holding anything else
  # deleted that too, on every start, with no log line.
  find "$PROMETHEUS_MULTIPROC_DIR" -maxdepth 1 -type f -name '*.db' -delete
fi

exec "$PYTHON" -m uvicorn bl_ranking.serving.app:app \
  --host "$HOST" --port "$PORT" --workers "$WORKERS" \
  --loop uvloop --http httptools \
  --no-access-log --log-level warning \
  --timeout-keep-alive 15
