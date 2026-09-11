"""FastAPI application for the brand-ranking endpoint.

Shape of the service
--------------------
    POST /rank        the ranking plus which model produced it
    POST /rank/bare   the research dictionary alone, for drop-in compatibility
    GET  /healthz     liveness - the process is up
    GET  /readyz      readiness - the models are loaded AND a real scoring call passed
    GET  /model       the serving version, backend and brand count
    GET  /metrics     Prometheus exposition

Design choices that matter for tail latency
-------------------------------------------
* The route handlers are `def`, not `async def`. The work is CPU-bound pandas and
  CatBoost; doing it inside a coroutine blocks the event loop and every other in-flight
  request behind it. As a sync handler it runs in the threadpool and the loop keeps
  accepting connections.
* The models are loaded in the lifespan hook, before the app serves anything, and
  `/readyz` stays 503 until a synthetic scoring call has actually succeeded. A load
  balancer that routes on readiness therefore never sends a user to a cold worker.
* Parallelism comes from worker *processes*, not threads: the scoring path is
  GIL-bound, so threads add queueing without adding throughput. Each worker pins
  CatBoost to `serving.threads_per_worker` so N workers do not oversubscribe the CPU
  and destroy p99.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from bl_ranking.config import Settings
from bl_ranking.serving.fast_features import MissingRegisterDate
from bl_ranking.serving.model_source import resolve_bundle
from bl_ranking.serving.ranker import BrandRanker, InsufficientSurveyData
from bl_ranking.serving.schemas import RankRequest, RankResponse

log = logging.getLogger("bl_ranking.serving")

# Buckets chosen around the observed distribution: sub-millisecond resolution where the
# service actually lives, and enough headroom to make a stall visible.
LATENCY_BUCKETS = (0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.025,
                   0.05, 0.1, 0.25, 0.5, 1.0, 2.5)

REQUESTS = Counter("bl_rank_requests_total", "Ranking requests", ["outcome"])
LATENCY = Histogram("bl_rank_latency_seconds", "End-to-end handler latency",
                    buckets=LATENCY_BUCKETS)
BRANDS = Gauge("bl_rank_brands", "Brands in the current model's universe")
READY = Gauge("bl_rank_ready", "1 once the worker has served a successful warm-up call")


class ServiceState:
    """Process-local singletons. One instance per uvicorn worker."""

    def __init__(self) -> None:
        self.ranker: BrandRanker | None = None
        self.settings: Settings = Settings.load()
        self.bundle_dir: Path | None = None
        self.error: str | None = None
        self.meta_template: dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return self.ranker is not None


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model before the first request, not during it."""
    settings = state.settings
    _pin_threads(settings.serving.threads_per_worker)
    started = time.perf_counter()
    try:
        bundle_dir = resolve_bundle(settings)
        state.bundle_dir = bundle_dir
        # BrandRanker.load() ends with a real scoring call; if it raises, this worker
        # must not report ready.
        state.ranker = BrandRanker.load(bundle_dir, settings)
        described = state.ranker.describe()
        state.meta_template = {
            "model_version": str(described["model_version"]),
            "payout_backend": str(described["payout_backend"]),
            "payout_exact": bool(described["payout_exact"]),
        }
        BRANDS.set(state.ranker.warm.n_brands)
        READY.set(1)
        log.info("model ready in %.1fs from %s (%s)", time.perf_counter() - started,
                 bundle_dir, state.ranker.describe())
    except Exception as exc:  # noqa: BLE001 - surfaced through /readyz, not swallowed
        state.error = f"{type(exc).__name__}: {exc}"
        READY.set(0)
        log.exception("model load failed; the worker will report not-ready")
    yield


app = FastAPI(
    title="Business Loans brand ranking",
    version="1.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


@app.post("/rank", response_model=None, responses={200: {"model": RankResponse}})
def rank(payload: RankRequest) -> ORJSONResponse:
    """Rank every brand for one post-funnel user.

    `response_model=None` is deliberate. FastAPI would otherwise re-validate the
    response against the Pydantic model on every call, which costs ~1 ms for a
    15-brand ranking and proves nothing: the dictionary is built by code we control,
    from floats we just computed. The schema is still published for the OpenAPI docs
    via `responses=`.
    """
    ranker = _require_ranker()
    started = time.perf_counter()
    try:
        ranking = ranker.rank(payload.to_user_data())
        outcome = "ok"
    except InsufficientSurveyData:
        # The user answered too little for the research pipeline to score them. This is
        # a property of the request, not a server fault, and the funnel should fall back
        # to its static ordering.
        REQUESTS.labels("insufficient_data").inc()
        raise HTTPException(status_code=422, detail={
            "error": "insufficient_survey_answers",
            "message": "at least 5 of the 8 survey answers are required",
            "ranking": {},
        }) from None
    except MissingRegisterDate:
        # The research code raises here too: a user who never submitted the survey
        # cannot become a lead. A bad request, not a server fault.
        REQUESTS.labels("no_register_date").inc()
        raise HTTPException(status_code=422, detail={
            "error": "register_date_absent",
            "message": "user cannot be a lead - register_date is absent",
            "ranking": {},
        }) from None
    except Exception as exc:  # noqa: BLE001
        REQUESTS.labels("error").inc()
        log.exception("ranking failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - started
    LATENCY.observe(elapsed)
    REQUESTS.labels(outcome).inc()

    # The model identity does not change between requests, so it is built once at
    # load time rather than reassembled on every call.
    meta = dict(state.meta_template)
    meta["n_brands"] = len(ranking)
    meta["latency_ms"] = round(elapsed * 1000, 3)
    return ORJSONResponse({"ranking": ranking, "meta": meta})


@app.post("/rank/bare", response_model=None)
def rank_bare(payload: RankRequest) -> ORJSONResponse:
    """The research function's return value verbatim, with no envelope.

    Provided so the endpoint is a drop-in for code that already consumes
    BLPayoutModelsPredict.predict_().
    """
    ranker = _require_ranker()
    started = time.perf_counter()
    try:
        ranking = ranker.rank(payload.to_user_data())
    except (InsufficientSurveyData, MissingRegisterDate):
        REQUESTS.labels("insufficient_data").inc()
        return ORJSONResponse({}, status_code=422)
    LATENCY.observe(time.perf_counter() - started)
    REQUESTS.labels("ok").inc()
    return ORJSONResponse(ranking)


@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    """Liveness only. A worker that is up but has no model is still alive."""
    return "ok"


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    if not state.ready:
        raise HTTPException(status_code=503, detail={
            "ready": False,
            "error": state.error or "model is still loading",
        })
    return {"ready": True, "bundle": str(state.bundle_dir)}


@app.get("/model")
def model_info() -> dict[str, Any]:
    ranker = _require_ranker()
    info = dict(ranker.describe())
    info["bundle"] = str(state.bundle_dir)
    info["manifest"] = ranker.warm.manifest.as_tags()
    return info


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)


class TimingMiddleware:
    """Adds X-Process-Time-Ms, as raw ASGI rather than a Starlette HTTP middleware.

    `@app.middleware("http")` wraps every request in BaseHTTPMiddleware, which spawns a
    task and an anyio stream per call. Measured here that costs roughly 0.5 ms per
    request - more than the model scoring itself. Intercepting the response-start
    message directly costs microseconds and gives the same header, which the load
    generator uses to separate handler time from queueing.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                elapsed = f"{(time.perf_counter() - started) * 1000:.3f}".encode()
                message.setdefault("headers", []).append((b"x-process-time-ms", elapsed))
            await send(message)

        await self.app(scope, receive, send_wrapper)


app.add_middleware(TimingMiddleware)


def _require_ranker() -> BrandRanker:
    if state.ranker is None:
        raise HTTPException(status_code=503, detail=state.error or "model is still loading")
    return state.ranker


def _pin_threads(threads: int) -> None:
    """Stop the numeric libraries from each grabbing every core.

    With several worker processes on a small box, the default (one thread per core per
    library per process) oversubscribes badly: throughput barely moves and p99 doubles.
    These have to be set before numpy/torch initialise their pools, which is why it
    happens in the lifespan hook rather than at import.
    """
    value = str(max(1, threads))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, value)
    try:
        import torch
        torch.set_num_threads(max(1, threads))
    except ImportError:
        pass
