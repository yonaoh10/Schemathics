#!/usr/bin/env bash
# Start the ranking API locally.
#
# Parallelism is by process, not thread: the scoring path is GIL-bound, so extra
# threads only add queueing. Each worker pins the numeric libraries to one thread
# (BL_SERVING__THREADS_PER_WORKER) so N workers do not oversubscribe the CPU.
#
#   ./scripts/serve.sh                 # workers from conf/config.yaml
#   BL_SERVING__WORKERS=1 ./scripts/serve.sh
#   BL_MODEL__PAYOUT__BACKEND=catboost_fallback ./scripts/serve.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
# A container (or a machine where the package is installed globally) has no
# .venv. Fall back rather than exec a path that does not exist.
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3 || command -v python)"; fi
HOST="${BL_SERVING__HOST:-0.0.0.0}"
PORT="${BL_SERVING__PORT:-8080}"
WORKERS="${BL_SERVING__WORKERS:-3}"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
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
# why it is set here and not in the app. The directory is wiped first: the files are named
# by pid, and a recycled pid from the previous run would otherwise add its totals to this
# one. Left unset for a single worker, where the plain registry is already correct.
if (( WORKERS > 1 )); then
  export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-$REPO_ROOT/.metrics}"
  rm -rf "$PROMETHEUS_MULTIPROC_DIR"
  mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
fi

exec "$PYTHON" -m uvicorn bl_ranking.serving.app:app \
  --host "$HOST" --port "$PORT" --workers "$WORKERS" \
  --loop uvloop --http httptools \
  --no-access-log --log-level warning \
  --timeout-keep-alive 15
