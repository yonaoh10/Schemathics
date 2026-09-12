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

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse, PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

from bl_ranking.config import Settings
from bl_ranking.serving import model_source
from bl_ranking.serving.fast_features import MissingRegisterDate
from bl_ranking.serving.model_source import resolve_bundle
from bl_ranking.serving.ranker import BrandRanker, InsufficientSurveyData
from bl_ranking.serving.schemas import RankRequest, RankResponse

log = logging.getLogger("bl_ranking.serving")

# Buckets chosen around the observed distribution: sub-millisecond resolution where the
# service actually lives, and enough headroom to make a stall visible.
LATENCY_BUCKETS = (0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.025,
                   0.05, 0.1, 0.25, 0.5, 1.0, 2.5)

# Bounds on the 422 body. The request has 22 fields, so reporting more errors than that
# says nothing more; 600 characters is several times the longest message here.
MAX_REPORTED_ERRORS = 22
MAX_MESSAGE_CHARS = 600


def _clipped(message: str) -> str:
    if len(message) <= MAX_MESSAGE_CHARS:
        return message
    return f"{message[:MAX_MESSAGE_CHARS]}... ({len(message)} characters)"


def _multiprocess_dir() -> str | None:
    """The directory prometheus_client shares counters through, or None for one worker.

    Every metric below is a module-level object, so it lives in the worker that imported
    it. The service runs `serving.workers` of those (3 by default), and each answers its
    own share of the traffic - so a scrape used to report whichever worker the load
    balancer happened to route it to and undercount everything by roughly the other two
    thirds. bl_rank_requests_total read a third of the requests, and p99 was one worker's
    p99, which is exactly the number an operator must not have to guess at.

    prometheus_client's answer is a shared directory of mmap'd files, switched on by this
    variable *before the library is imported*; scripts/serve.sh and the compose file set
    it, and it is created here rather than required so that neither has to. With it unset
    - a single worker, or a test - the default registry is used unchanged.
    """
    # The lower-case spelling is prometheus_client's own deprecated one, still honoured by
    # the library; an operator following an older guide has it set, and silently ignoring
    # it here would mean single-worker metrics with nothing to say why.
    path = os.environ.get("PROMETHEUS_MULTIPROC_DIR") or os.environ.get(
        "prometheus_multiproc_dir")  # noqa: SIM112
    if not path:
        return None
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


MULTIPROC_DIR = _multiprocess_dir()

REQUESTS = Counter("bl_rank_requests_total", "Ranking requests", ["outcome"])
# The share of traffic whose survey answer matched no band. docs/design.md promised this
# before it existed: a copy change that renames a funnel answer sends every user to the
# -99 sentinel silently, and it would otherwise surface as revenue rather than as an error.
BAND_SENTINELS = Counter(
    "bl_rank_band_sentinel_total",
    "Requests whose band mapping fell through to the -99 sentinel",
    ["feature"],
)
# Observed by the middleware, so it covers the whole request. Observed in the handler it
# covered the scoring call only: body parsing and Pydantic validation happen before the
# handler is entered, so a flood of invalid bodies that took the caller seconds was
# recorded in the 7.5 ms bucket - the metric said the endpoint was healthy while it was
# not answering.
LATENCY = Histogram("bl_rank_latency_seconds", "End-to-end request latency, all routes",
                    buckets=LATENCY_BUCKETS)
# What the handler itself spent, kept separately. The gap between the two is parsing,
# validation and the ASGI stack, which is the first thing to look at when the endpoint is
# slow and the model is not.
SCORING = Histogram("bl_rank_scoring_seconds", "Time inside the ranking call",
                    buckets=LATENCY_BUCKETS)
# Counters and histograms add up across workers; a gauge has to be told how. Both of
# these describe a state every worker shares rather than a quantity it contributes, so
# they aggregate over the workers that are alive: the brand count is the same everywhere
# (max), and readiness is only true when it is true of all of them (min) - one worker
# that failed to load must not be hidden by two that did.
BRANDS = Gauge("bl_rank_brands", "Brands in the current model's universe",
               multiprocess_mode="livemax")
READY = Gauge("bl_rank_ready", "1 once the worker has served a successful warm-up call",
              multiprocess_mode="livemin")


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
    started = time.perf_counter()
    try:
        # Inside the try: anything that raises before the model is loaded has to leave
        # the worker reporting not-ready, rather than killing the process outside the
        # one place that knows how to say so.
        _pin_threads(settings.serving.threads_per_worker)
        bundle_dir = resolve_bundle(settings)
        state.bundle_dir = bundle_dir
        # BrandRanker.load() ends with a real scoring call; if it raises, this worker
        # must not report ready.
        state.ranker = BrandRanker.load(bundle_dir, settings)
        state.ranker.on_band_sentinels = lambda feature: (
            BAND_SENTINELS.labels(feature).inc()
        )
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


@app.exception_handler(RequestValidationError)
async def invalid_request(_request, exc: RequestValidationError) -> Response:
    """Render a rejected payload ourselves, because the default renderer can crash.

    FastAPI's built-in handler echoes each error's `input` value back. Two kinds of
    input make that fail, and both turned a 422 into a 500 - the wrong status, no log
    line the caller can act on, and nothing counted, so a funnel sending malformed
    traffic was invisible on the dashboard:

      * a non-finite float (`NaN`, `Infinity`, which json.loads accepts) - not
        representable in JSON, so the encoder raises;
      * a lone UTF-16 surrogate from a `\ud800` escape - not encodable as UTF-8, so
        the encoder raises while reporting the string that caused it.

    So only `loc`, `msg` and `type` are returned, serialised with ensure_ascii, which
    escapes a surrogate back into the form it arrived in. The caller's own value is
    never echoed - it cannot be rendered, and repeating unvalidated input in an error
    body is not something to do on a public endpoint anyway.

    Each message is also truncated. A validator that quotes the value it refused is what
    makes a 422 useful, but a quoted value is caller-controlled input in a response this
    handler builds on the event loop: a 32 MB string in one field produced a 32 MB message
    and a 33 MB body, two seconds during which that worker answered nobody. BodyLimit caps
    the input and schemas._shown caps the quoting, and this caps the result of both - so no
    message added later can reopen it by forgetting either.
    """
    REQUESTS.labels("invalid_request").inc()
    detail = [
        {
            "loc": [str(part) for part in error.get("loc", ())],
            "msg": _clipped(str(error.get("msg", ""))),
            "type": str(error.get("type", "")),
        }
        for error in exc.errors()[:MAX_REPORTED_ERRORS]
    ]
    return Response(
        content=json.dumps({"detail": detail}, ensure_ascii=True),
        status_code=422,
        media_type="application/json",
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
    SCORING.observe(elapsed)
    REQUESTS.labels(outcome).inc()

    # The model identity does not change between requests, so it is built once at
    # load time rather than reassembled on every call.
    meta = dict(state.meta_template)
    # `brands_ranked`, not `n_brands`: GET /model reports n_brands as the size of the brand
    # universe, and this is the number that came back for *this* user. The two were the same
    # name for two different quantities, so a partial ranking read as a shrunken universe.
    meta["brands_ranked"] = len(ranking)
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
    except InsufficientSurveyData:
        REQUESTS.labels("insufficient_data").inc()
        return ORJSONResponse({}, status_code=422)
    except MissingRegisterDate:
        # Counted under its own label, as on /rank. Folding it into insufficient_data
        # here made the two endpoints label the same refusal differently, so an operator
        # reading no_register_date was seeing /rank traffic only and could not tell a
        # funnel that had stopped sending register_date from one asking too little.
        REQUESTS.labels("no_register_date").inc()
        return ORJSONResponse({}, status_code=422)
    except Exception as exc:  # noqa: BLE001
        # Without this the failure returned a bare 500 from the ASGI stack: the error
        # counter never moved and nothing was logged, so an endpoint failing on every
        # request looked identical to one nobody was calling.
        REQUESTS.labels("error").inc()
        log.exception("ranking failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    SCORING.observe(time.perf_counter() - started)
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
    # Which of the three resolution tiers actually answered. A worker that fell back to a
    # local run directory is healthy by every other measure, and the commonest reason is a
    # mistyped registered model or alias - which looks identical to a registry with nothing
    # promoted yet. Saying so here is what makes the two distinguishable without ssh.
    info["bundle_source"] = model_source.last_bundle_source
    info["manifest"] = ranker.warm.manifest.as_tags()
    return info


def _forget_dead_workers(path: str) -> None:
    """Delete the live-gauge files of workers that no longer exist.

    `livemin`/`livemax` mean "across the workers that are alive" only if somebody removes
    the files of the ones that are not, and prometheus_client leaves that to the
    application. Nothing else can do it here - uvicorn's supervisor gives the app no
    worker-exit hook - so it happens on scrape, which is the one moment the answer is
    needed. Counter and histogram files are deliberately left alone: a request a since
    replaced worker served still happened, and the totals have to keep counting it.
    """
    for name in os.listdir(path):
        if not name.startswith("gauge_live"):
            continue
        pid = name.rsplit("_", 1)[-1].removesuffix(".db")
        try:
            os.kill(int(pid), 0)
        except (ProcessLookupError, ValueError):
            multiprocess.mark_process_dead(pid, path)
        except PermissionError:
            continue                      # alive, just not ours to signal


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    """One scrape covers every worker, not just the one that answered it."""
    if MULTIPROC_DIR is None:
        return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)
    _forget_dead_workers(MULTIPROC_DIR)
    # A fresh registry per scrape, because the collector reads the files as they are now.
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=MULTIPROC_DIR)
    return PlainTextResponse(generate_latest(registry).decode(),
                             media_type=CONTENT_TYPE_LATEST)


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
                seconds = time.perf_counter() - started
                elapsed = f"{seconds * 1000:.3f}".encode()
                message.setdefault("headers", []).append((b"x-process-time-ms", elapsed))
                # Here rather than in the handler: everything before the handler - reading
                # the body, parsing the JSON, validating 22 fields - is latency the caller
                # paid, and a 422 costs it too. Measured in the handler, a flood of invalid
                # bodies left the histogram looking idle.
                if scope.get("path", "").startswith("/rank"):
                    LATENCY.observe(seconds)
            await send(message)

        await self.app(scope, receive, send_wrapper)


class BodyLimit:
    """Refuse a request body larger than `serving.max_body_bytes`, before reading it.

    A /rank payload is one funnel session, about 1 KB. Nothing bounded the body, so a
    caller could post 32 MB of nonsense and the worker would read it, parse it as JSON and
    build a rejection for it - 2 s measured, all of it on the event loop, so that worker
    answered nobody else for two seconds. With three workers and three such requests the
    endpoint is down, from one client, with no error anywhere. 413 costs microseconds.

    Raw ASGI for the same reason as TimingMiddleware: a Starlette HTTP middleware spawns a
    task and an anyio stream per request, which is more expensive than the check.
    """

    def __init__(self, app, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _declared_length(scope)
        if declared is not None:
            # The common case: every ordinary JSON client sends Content-Length, so the
            # decision costs one header lookup and the body is never read at all.
            if declared > self.limit:
                await self._too_large(send, declared)
                return
            await self.app(scope, receive, send)
            return
        # Chunked, so the size is only knowable by counting. Read it here, up to one byte
        # past the limit, and replay what we read to the app: buffering ~1 KB is nothing
        # next to letting an unbounded stream through.
        body, overflowed = await self._read_capped(receive)
        if overflowed:
            await self._too_large(send, None)
            return
        await self.app(scope, _replay(body), send)

    async def _read_capped(self, receive) -> tuple[list[dict], bool]:
        chunks: list[dict] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                chunks.append(message)
                return chunks, False          # a disconnect; let the app see it
            size += len(message.get("body", b""))
            chunks.append(message)
            if size > self.limit:
                return chunks, True
            if not message.get("more_body", False):
                return chunks, False

    async def _too_large(self, send, declared: int | None) -> None:
        REQUESTS.labels("too_large").inc()
        said = f"{declared} bytes" if declared is not None else "more"
        detail = (f"request body is too large: {said} against a limit of {self.limit}. "
                  f"One ranking payload is about 1 KB.")
        body = json.dumps({"detail": detail}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def _declared_length(scope) -> int | None:
    """The Content-Length header as an int, or None if absent or unreadable."""
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _replay(chunks: list[dict]):
    """A `receive` that hands back the messages BodyLimit already consumed."""
    pending = list(chunks)

    async def receive():
        if pending:
            return pending.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    return receive


# Order matters: TimingMiddleware is added second and so runs first, which means a 413
# is still timed and still carries X-Process-Time-Ms like every other response.
app.add_middleware(BodyLimit, limit=state.settings.serving.max_body_bytes)
app.add_middleware(TimingMiddleware)


def _require_ranker() -> BrandRanker:
    if state.ranker is None:
        # Counted, because this is the total-outage case: a worker whose model failed to
        # load answers every request 503 while emitting no request metrics at all, so on
        # the dashboard it is indistinguishable from a worker nobody is calling.
        REQUESTS.labels("not_ready").inc()
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
