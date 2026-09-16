"""The HTTP contract: request validation, the response shape, and the failure modes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bl_ranking.serving.ranker import WARMUP_USER
from bl_ranking.serving.schemas import RankRequest


@pytest.fixture(scope="module")
def client(ranker, bundle, settings):
    """A TestClient over the real app, with the already-built ranker injected.

    The lifespan hook is bypassed on purpose: it would resolve and load a second copy
    of the model, and the point here is the HTTP layer, not model loading.
    """
    from bl_ranking.serving import app as app_module

    app_module.state.ranker = ranker
    app_module.state.bundle_dir = bundle
    app_module.state.settings = settings
    described = ranker.describe()
    app_module.state.meta_template = {
        "model_version": str(described["model_version"]),
        "payout_backend": str(described["payout_backend"]),
        "payout_exact": bool(described["payout_exact"]),
    }
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_rank_returns_the_documented_shape(client, example_user):
    response = client.post("/rank", json=example_user)
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"ranking", "meta"}

    ranking = body["ranking"]
    assert ranking, "no brands returned for a well-formed user"
    for brand, entry in ranking.items():
        assert isinstance(brand, str)
        assert set(entry) == {"rank", "expected_payout"}
        assert entry["expected_payout"] > 0.01   # the research code's floor

    # Ranks are 1..n in the order the dictionary is serialised, descending by value.
    ranks = [entry["rank"] for entry in ranking.values()]
    assert ranks == [float(i + 1) for i in range(len(ranking))]
    payouts = [entry["expected_payout"] for entry in ranking.values()]
    assert payouts == sorted(payouts, reverse=True)


def test_meta_identifies_the_serving_version(client, example_user, ranker):
    meta = client.post("/rank", json=example_user).json()["meta"]
    assert meta["model_version"] == ranker.describe()["model_version"]
    assert meta["payout_backend"] == ranker.describe()["payout_backend"]
    assert isinstance(meta["payout_exact"], bool)
    assert meta["latency_ms"] > 0


def test_bare_endpoint_matches_the_enveloped_one(client, example_user):
    enveloped = client.post("/rank", json=example_user).json()["ranking"]
    bare = client.post("/rank/bare", json=example_user).json()
    assert bare == enveloped


def test_timing_header_is_present(client, example_user):
    response = client.post("/rank", json=example_user)
    assert "x-process-time-ms" in {k.lower() for k in response.headers}
    assert float(response.headers["x-process-time-ms"]) >= 0


def test_unparseable_phone_is_normalised_not_rejected(client, example_user):
    """The research pipeline does cellphone.astype(int); the schema guarantees it can."""
    for value in ["(305) 555-0142", "+1 786 991 4030", "n/a", "", None]:
        user = dict(example_user)
        user["cellphone"] = value
        assert client.post("/rank", json=user).status_code == 200, value


def test_phone_normalisation_matches_ingestion():
    """Training and serving must derive cellphone_prefix the same way."""
    import pandas as pd

    from bl_ranking.data.ingest import sanitise

    raw = ["(305) 555-0142", "+1 786 991 4030", "7869914030", "n/a"]
    frame = pd.DataFrame({
        "cellphone": raw, "payout": [0] * 4,
        "session_dt": ["2026-01-06 19:24:22"] * 4,
        "conversion_dt": [None] * 4, "register_date": [None] * 4,
        **{c: ["x"] * 4 for c in ["credit_score", "industry", "loan_amount",
                                  "loan_reason", "monthly_revenue", "time_in_business",
                                  "device_type", "business_type"]},
    })
    ingested, _ = sanitise(frame)
    served = [RankRequest.normalise_cellphone(v) for v in raw]
    assert list(ingested["cellphone"]) == served


def test_sparse_survey_gives_422_with_an_empty_ranking(client, example_user):
    user = dict(example_user)
    for column in ["credit_score", "industry", "loan_amount", "loan_reason",
                   "monthly_revenue"]:
        user[column] = None
    response = client.post("/rank", json=user)
    assert response.status_code == 422
    assert response.json()["detail"]["ranking"] == {}
    assert response.json()["detail"]["error"] == "insufficient_survey_answers"


@pytest.mark.parametrize("mangle", [
    lambda user: {k: v for k, v in user.items() if k != "register_date"},
    lambda user: {**user, "register_date": None},
])
def test_a_user_who_cannot_be_a_lead_is_told_why(client, example_user, mangle):
    """The documented 422 carries `register_date_absent`, and for a long time no request
    could produce it: the field was required, so every such payload came back as a
    generic validation failure instead - which tells a funnel nothing about why."""
    response = client.post("/rank", json=mangle(example_user))
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "register_date_absent"


def test_a_malformed_register_date_is_not_an_absent_one(client, example_user):
    """Absent is a data condition; unreadable is a formatting problem. Folding the second
    into the first would tell the caller they have a user who cannot be a lead when what
    they have is a broken date format."""
    response = client.post("/rank", json={**example_user, "register_date": "not a date"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)                      # a validation error, not ours
    assert "parseable timestamp" in detail[0]["msg"]


def test_unknown_fields_are_ignored(client, example_user):
    """A funnel adding a field must not break ranking."""
    user = dict(example_user)
    user["some_new_tracking_field"] = "whatever"
    assert client.post("/rank", json=user).status_code == 200


def test_unseen_categorical_values_do_not_fail_the_request(client, example_user):
    """A brand-new campaign or city appears constantly; it must not be an outage."""
    user = dict(example_user)
    user["campaign_id"] = 999999999999999
    user["auto_city"] = "Nowheresville"
    user["page"] = "top10us.com/app/brand-new-lander"
    assert client.post("/rank", json=user).status_code == 200


def test_health_readiness_and_model_endpoints(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").json()["ready"] is True

    info = client.get("/model").json()
    assert info["n_brands"] > 0
    assert info["feature_path"] in {"fast", "research"}
    assert "payout_backend" in info


def test_metrics_are_exposed(client, example_user):
    client.post("/rank", json=example_user)
    body = client.get("/metrics").text
    assert "bl_rank_requests_total" in body
    assert "bl_rank_latency_seconds" in body
    assert "bl_rank_ready" in body


def test_worker_that_cannot_load_a_model_reports_alive_but_not_ready(monkeypatch):
    """A worker whose model load failed must fail readiness, not liveness.

    Behind a load balancer that routes on readiness, that is the difference between a
    bad deploy taking one worker out of rotation and it serving 500s to real users.
    The failure is injected where it actually happens - resolving the bundle - so the
    lifespan hook's error handling is what is under test.
    """
    from bl_ranking.serving import app as app_module

    saved = (app_module.state.ranker, app_module.state.error)
    monkeypatch.setattr(app_module, "resolve_bundle",
                        lambda settings: (_ for _ in ()).throw(FileNotFoundError("no bundle")))
    app_module.state.ranker = None
    try:
        with TestClient(app_module.app, raise_server_exceptions=False) as cold:
            assert cold.get("/healthz").status_code == 200    # the process is alive
            readyz = cold.get("/readyz")
            assert readyz.status_code == 503                  # but must not be routed to
            assert "no bundle" in readyz.json()["detail"]["error"]
            assert cold.post("/rank", json={}).status_code in (422, 503)
    finally:
        app_module.state.ranker, app_module.state.error = saved


# A malformed request is a 422. A 500 on the funnel's critical path is an outage, and
# every case below returned one before these were written.
@pytest.mark.parametrize("field,value", [
    ("campaign_id", "not-a-number"),   # declared int|str, but a numeric model feature
    ("session_dt", ["2026-01-06 19:24:22"]),  # a list escaped the scalar check
    ("session_dt", {"when": "now"}),
    ("session_dt", "not a date"),
])
def test_malformed_fields_are_rejected_not_crashed(client, field, value):
    body = dict(WARMUP_USER) | {field: value}
    response = client.post("/rank", json=body)
    assert response.status_code != 500, response.text
    assert response.status_code in (200, 422), response.status_code


def test_an_omitted_cellphone_is_normalised_like_an_explicit_null(client):
    """`cellphone` is optional, and omitting it used to be a 500.

    A `before` validator does not run for an absent key, so the documented
    normalisation to 0 was skipped in exactly the case it was written for, and the
    research pipeline's `.astype(int)` took the request down.
    """
    omitted = dict(WARMUP_USER)
    omitted.pop("cellphone", None)
    explicit = dict(WARMUP_USER) | {"cellphone": None}

    a = client.post("/rank", json=omitted)
    b = client.post("/rank", json=explicit)
    assert a.status_code == 200, a.text
    assert b.status_code == 200, b.text
    assert a.json()["ranking"] == b.json()["ranking"]


def test_a_quoted_campaign_id_keeps_every_digit(client):
    """Real ids exceed 2^53, so a caller may quote one to keep it out of a float."""
    from bl_ranking.serving.schemas import RankRequest

    exact = 120227360861540306
    assert RankRequest(**(dict(WARMUP_USER) | {"campaign_id": str(exact)})).campaign_id == exact
    # Junk degrades to the same 0 level the ingestion gate assigns it, not a 500.
    assert RankRequest(**(dict(WARMUP_USER) | {"campaign_id": "junk"})).campaign_id == 0

@pytest.mark.parametrize("value", ["9999-12-31 23:59:59", "1600-01-01 00:00:00"])
def test_timestamps_outside_the_representable_range_are_rejected(client, value):
    """Both feature paths must agree, including on what they refuse.

    pandas holds timestamps as nanoseconds in an int64, so a year outside roughly
    1677..2262 has no representation. The research path died inside CatBoost on such a
    row while the vectorised path scored it and returned a confident ranking - a
    divergence in the one contract the two paths have.
    """
    response = client.post("/rank", json=dict(WARMUP_USER) | {"session_dt": value})
    assert response.status_code == 422, response.text


def test_an_oversized_phone_number_does_not_overflow(client):
    """cellphone.astype(int) is an int64 cast; a 20-digit number used to raise there."""
    from bl_ranking.serving.schemas import RankRequest

    assert RankRequest(**(dict(WARMUP_USER) | {"cellphone": 99999999999999999999})).cellphone == 0
    response = client.post("/rank", json=dict(WARMUP_USER) | {"cellphone": 99999999999999999999})
    assert response.status_code == 200, response.text

def test_an_unexpected_failure_on_the_bare_endpoint_is_counted(client, monkeypatch):
    """/rank/bare had no catch-all, so failures were invisible to monitoring.

    The bare 500 that the ASGI stack returned never touched the error counter and was
    never logged, which makes an endpoint failing on every request indistinguishable
    from one nobody is calling.
    """
    from bl_ranking.serving import app as app_module

    def boom(_user):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.state.ranker, "rank", boom)
    response = client.post("/rank/bare", json=dict(WARMUP_USER))

    assert response.status_code == 500
    assert 'bl_rank_requests_total{outcome="error"}' in client.get("/metrics").text

@pytest.mark.parametrize("value", ["1" + "0" * 400, "9" * 20, "junk"])
def test_a_campaign_id_too_large_for_int64_degrades_like_training(client, value):
    """The training column is int64, so a larger id is the 0 level there.

    Without the same bound at serving the Python int reached CatBoost, which raised on
    anything past float range - a 500 leaking a library message - and silently scored
    everything below it differently from how training saw it.
    """
    from bl_ranking.serving.schemas import RankRequest

    assert RankRequest(**(dict(WARMUP_USER) | {"campaign_id": value})).campaign_id == 0
    assert client.post("/rank", json=dict(WARMUP_USER) | {"campaign_id": value}).status_code == 200


def test_a_timestamp_pair_that_cannot_be_subtracted_is_refused(client):
    """pandas can hold 1677..2262 but a timedelta spans only ~292 years.

    `from_start_to_register` subtracts the two, so a pair the schema accepted
    individually could still overflow - and the research path raised while the
    vectorised path returned a ranking, breaking the equivalence contract on input
    neither implementation should have taken.
    """
    body = dict(WARMUP_USER) | {"session_dt": "1700-01-01 00:00:00"}
    assert client.post("/rank", json=body).status_code == 422

@pytest.mark.parametrize("value", [7448788, "7448788", "7448788.0", 7448788.0, "007448788", None])
def test_attribution_ids_normalise_the_same_way_on_both_sides(value):
    """sub1/sub2/sub3 are categorical, so the level is whatever string is produced.

    The ingestion gate normalises them so '1815195.0' and '1815195' are one id.
    Nothing did the same at the request boundary, so a caller quoting an id with a
    trailing '.0' - exactly what a JSON encoder emits for a float-typed column
    upstream - scored against a level training had never seen. The rule is imported
    from the gate rather than restated, because two copies are two chances to drift.
    """
    from bl_ranking.data.ingest import _as_identifier

    served = RankRequest(**(dict(WARMUP_USER) | {"sub1": value})).sub1
    assert served == _as_identifier(value)



@pytest.mark.parametrize(("label", "mangle"), [
    # json.loads accepts both of these; neither can be rendered back into JSON.
    ("not-a-number", lambda body: body.replace('"industry": "construction"',
                                               '"industry": NaN')),
    ("infinity", lambda body: body.replace('"industry": "construction"',
                                           '"industry": Infinity')),
    ("lone surrogate", lambda body: body.replace('"construction"',
                                                 r'"const\ud800ruction"')),
])
def test_an_unrenderable_payload_is_a_422_not_a_500(client, example_user, label, mangle):
    """FastAPI's own validation-error renderer echoes the offending input back, and
    both of these break the encoder while it tries to - so a malformed payload became a
    500 with no useful body, and nothing was counted. The surrogate case also crashed
    CatBoost itself with a SystemError no handler recognises."""
    import json

    body = mangle(json.dumps({**example_user, "industry": "construction"}))
    response = client.post("/rank", content=body.encode(),
                           headers={"content-type": "application/json"})

    assert response.status_code == 422, f"{label}: {response.text[:200]}"
    detail = response.json()["detail"]
    assert detail and all({"loc", "msg", "type"} == set(entry) for entry in detail)


def test_a_rejected_payload_is_counted(client, example_user):
    """An uncounted failure mode is an invisible one: a funnel sending malformed traffic
    has to show up on the dashboard, not only in a traceback."""
    from bl_ranking.serving.app import REQUESTS

    before = REQUESTS.labels("invalid_request")._value.get()
    client.post("/rank", json={**example_user, "session_dt": "not a timestamp"})
    assert REQUESTS.labels("invalid_request")._value.get() == before + 1


@pytest.mark.parametrize(("spelling", "expected"), [
    (13055550142, 3055550142),
    # A JSON caller has no integers. Both of these are the same payload to a browser,
    # and str() used to render the float as '13055550142.0' - twelve digits, so the
    # country-code strip did not fire and the prefix feature became '130' not '305'.
    (13055550142.0, 3055550142),
    ("13055550142", 3055550142),
    ("(305) 555-0142", 3055550142),
])
def test_one_phone_number_gives_one_prefix(example_user, spelling, expected):
    assert RankRequest.model_validate(
        {**example_user, "cellphone": spelling}).cellphone == expected


@pytest.mark.parametrize("spelling", [
    120227360861540306, "120227360861540306",
    # Going via `int(float(text))` rounds this to ...304 - the exact float64 demotion
    # the ingestion gate exists to undo, reintroduced at the request boundary.
    "120227360861540306.0",
])
def test_a_17_digit_campaign_id_stays_exact(example_user, spelling):
    assert RankRequest.model_validate(
        {**example_user, "campaign_id": spelling}).campaign_id == 120227360861540306


def test_both_endpoints_label_the_same_refusal_the_same_way(client, example_user):
    """/rank/bare folded a missing register_date into insufficient_data, so an operator
    reading no_register_date saw /rank traffic only - and could not tell a funnel that
    had stopped sending register_date from one asking too few survey questions."""
    from bl_ranking.serving.app import REQUESTS

    payload = {k: v for k, v in example_user.items() if k != "register_date"}
    before = REQUESTS.labels("no_register_date")._value.get()
    for path in ("/rank", "/rank/bare"):
        assert client.post(path, json=payload).status_code == 422
    assert REQUESTS.labels("no_register_date")._value.get() == before + 2


@pytest.mark.parametrize("field", ["session_dt", "register_date", "conversion_dt"])
@pytest.mark.parametrize("value", [20260115, 20260115.0, 1767225600])
def test_a_numeric_timestamp_is_refused_at_the_door(client, example_user, field, value):
    """The ingestion gate refuses a numeric date column; the endpoint used to accept one and
    serve a ranking built from 1970-01-01. pandas reads a number here as nanoseconds since
    the epoch, and the epoch is the floor of the accepted range - so session_day,
    session_day_of_week, session_hour and from_start_to_register were all confidently wrong
    behind a 200."""
    response = client.post("/rank", json={**example_user, field: value})
    assert response.status_code == 422
    assert "nanoseconds" in str(response.json()["detail"])


def test_the_model_endpoint_says_where_its_bundle_came_from(client):
    """A worker that fell back to a local run directory is healthy by every other measure,
    and the commonest cause is a mistyped registered model or alias - which looks identical
    to a registry with nothing promoted yet."""
    from bl_ranking.serving.model_source import (
        BUNDLE_SOURCE_LOCAL,
        BUNDLE_SOURCE_PINNED,
        BUNDLE_SOURCE_REGISTRY,
    )

    source = client.get("/model").json()["bundle_source"]
    assert source in {BUNDLE_SOURCE_LOCAL, BUNDLE_SOURCE_PINNED, BUNDLE_SOURCE_REGISTRY}


def test_a_renamed_funnel_answer_shows_up_as_a_metric(client, example_user):
    """docs/design.md promised this signal before it existed. A copy change that renames a
    survey answer sends every user to the -99 band sentinel silently: the feature still has
    a value, the request still succeeds with a 200, and the loss shows up as revenue rather
    than as an error."""
    from bl_ranking.serving.app import BAND_SENTINELS

    before = BAND_SENTINELS.labels("credit_score_num")._value.get()
    assert client.post("/rank", json=example_user).status_code == 200
    assert BAND_SENTINELS.labels("credit_score_num")._value.get() == before

    renamed = {**example_user, "credit_score": "Reasonably Good"}
    assert client.post("/rank", json=renamed).status_code == 200
    assert BAND_SENTINELS.labels("credit_score_num")._value.get() == before + 1


def test_the_two_brand_counts_have_two_names(client, example_user):
    """GET /model reports `n_brands` as the size of the brand universe; the response meta
    reports how many came back for this user. They were the same name for two different
    quantities, so a partial ranking read as a shrunken universe."""
    meta = client.post("/rank", json=example_user).json()["meta"]
    info = client.get("/model").json()

    assert "n_brands" not in meta
    assert meta["brands_ranked"] == len(
        client.post("/rank", json=example_user).json()["ranking"])
    assert info["n_brands"] >= meta["brands_ranked"]


def test_a_worker_with_no_model_is_visible_on_the_dashboard(client, example_user, monkeypatch):
    """A worker whose model failed to load answers every request 503 while emitting no
    request metrics at all, so it looked exactly like a worker nobody was calling.

    A well-formed request, because validation runs first and an empty body is a 422
    before readiness is ever consulted - which is also why /model, which this test used
    to count on, no longer counts: it is a probe, not a ranking request."""
    from bl_ranking.serving import app as app_module

    before = app_module.REQUESTS.labels("not_ready")._value.get()
    monkeypatch.setattr(app_module.state, "ranker", None)
    assert client.post("/rank", json=example_user).status_code == 503
    assert client.get("/model").status_code == 503
    assert app_module.REQUESTS.labels("not_ready")._value.get() == before + 1


@pytest.mark.parametrize("keyword", ["now", "today"])
def test_a_relative_keyword_is_not_a_timestamp(client, example_user, keyword):
    """pandas resolves both against the server's clock, so a funnel sending either had its
    own timestamp silently replaced - and session_day, session_day_of_week and session_hour
    became today's, behind a 200."""
    response = client.post("/rank", json={**example_user, "session_dt": keyword})
    assert response.status_code == 422
    assert "Relative keywords" in str(response.json()["detail"])


@pytest.mark.parametrize(("register", "accepted"), [
    ("2026-01-06 19:26:07", True),      # the survey after the session, as always
    ("2026-01-06 19:24:00", True),      # 22 s early: clock skew between two services
    ("2026-01-06 19:20:00", False),     # four minutes early
    ("2026-01-05 19:26:07", False),     # a day early
])
def test_a_time_inverted_funnel_is_refused(client, example_user, register, accepted):
    """from_start_to_register is register_date minus session_dt and every training row has
    it positive. An inverted pair produced -86,400 seconds behind a 200 - a value millions
    of standard deviations outside anything the model was fitted on, scored confidently."""
    payload = {**example_user, "session_dt": "2026-01-06 19:24:22", "register_date": register}
    response = client.post("/rank", json=payload)
    if accepted:
        assert response.status_code == 200
    else:
        assert response.status_code == 422
        assert "before session_dt" in str(response.json()["detail"])


@pytest.fixture
def other_worker(tmp_path_factory):
    """A directory holding the metrics of a worker in another process, now exited.

    Function-scoped, and each caller gets a fresh directory: reading /metrics deletes the
    live-gauge files of workers that have gone, so two tests sharing one directory would
    have the first sweep away what the second is about to look for.

    Production runs `serving.workers` uvicorn processes and every metric in app.py is a
    module-level object, so each worker counts only its own traffic. This fixture is the
    other workers: a real second process that records some requests and then dies, leaving
    its counters in the shared directory for the scrape to find.
    """
    import os
    import subprocess
    import sys

    directory = tmp_path_factory.mktemp("multiproc")
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ,
           "PROMETHEUS_MULTIPROC_DIR": str(directory),
           "PYTHONPATH": str(root / "src"),
           # The child only touches counters; loading a model would make it slow for
           # nothing, and the lifespan hook that loads one is never entered on import.
           "BL_SERVING__WORKERS": "3"}
    program = (
        "from bl_ranking.serving import app\n"
        "for _ in range(7): app.REQUESTS.labels('ok').inc()\n"
        # A worker whose model failed to load. The interesting case for readiness: it must
        # not be averaged away by the workers that did load, and must not outlive itself.
        "app.READY.set(0)\n"
    )
    subprocess.run([sys.executable, "-c", program], check=True, env=env, timeout=600)
    return directory


def test_a_scrape_covers_the_workers_that_did_not_answer_it(client, monkeypatch, other_worker):
    """One scrape must report the whole service, not the worker it happened to reach.

    With the documented 3 workers, a plain registry gave a scrape one worker's counters:
    bl_rank_requests_total read about a third of the traffic and the latency histogram was
    one worker's p99. Both are numbers an operator sizes capacity from, and being wrong low
    by two thirds is worse than not having them.
    """
    from bl_ranking.serving import app as app_module

    monkeypatch.setattr(app_module, "MULTIPROC_DIR", str(other_worker))
    body = client.get("/metrics").text
    # 7 requests that this process never saw, and would not have reported.
    assert 'bl_rank_requests_total{outcome="ok"} 7.0' in body


def test_a_dead_workers_readiness_does_not_outlive_it(client, monkeypatch, other_worker):
    """`livemin` means "across the workers that are alive", which needs the files of the
    others deleted - prometheus_client leaves that to the application, and uvicorn offers
    no worker-exit hook. Without the sweep, one worker that failed to load would pin
    bl_rank_ready to 0 for the lifetime of the service, including after it was replaced by
    a healthy one.
    """
    from prometheus_client import CollectorRegistry, generate_latest, multiprocess

    from bl_ranking.serving import app as app_module

    library = CollectorRegistry()
    multiprocess.MultiProcessCollector(library, path=str(other_worker))
    assert "bl_rank_ready 0.0" in generate_latest(library).decode()

    monkeypatch.setattr(app_module, "MULTIPROC_DIR", str(other_worker))
    body = client.get("/metrics").text
    assert "bl_rank_ready 0.0" not in body
    # The requests it served still count: only the gauge describes a state that died with it.
    assert 'bl_rank_requests_total{outcome="ok"} 7.0' in body


# --- Request size ---------------------------------------------------------------------
#
# Nothing bounded the request body. Measured on one worker, eight connections posting 1 MB
# bodies took valid /rank p50 from 7.8 ms to 404 ms and cut the requests it answered in
# eight seconds from 123 to 19; a single 32 MB body cost 2.0 s and came back as a 33.5 MB
# error, because the validator quoted the value it refused. The sender paid nothing for
# either. 64 KB is 60x an ordinary payload and refusing above it costs 0.05 ms.

def test_a_body_over_the_limit_is_refused_without_being_parsed(client):
    """413, not 422: the difference says whether the caller should fix their payload or
    stop sending 1 MB of it. And it must not be parsed first - parsing is the cost."""
    from bl_ranking.serving import app as app_module

    limit = app_module.state.settings.serving.max_body_bytes
    before = app_module.REQUESTS.labels("too_large")._value.get()
    # Valid JSON, and valid against the schema up to the size: the point is that the size
    # decides, before anything looks at the content.
    response = client.post("/rank", content=b'{"session_dt": "' + b"x" * (limit + 1) + b'"}',
                           headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert str(limit) in response.json()["detail"]
    assert app_module.REQUESTS.labels("too_large")._value.get() == before + 1


def test_a_body_the_client_does_not_measure_is_still_limited(client):
    """Chunked transfer sends no Content-Length, so the size is only knowable by counting.
    A limit that a caller can bypass by omitting a header is not a limit."""
    def chunks():
        for _ in range(40):
            yield b"x" * 4096

    response = client.post("/rank", content=chunks(),
                           headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_a_body_under_the_limit_is_unaffected(client, example_user):
    """The limit must be invisible to ordinary traffic, including on the counted path:
    reading the body to measure it and replaying it to the app has to be lossless."""
    import json as jsonlib

    encoded = jsonlib.dumps(example_user).encode()

    def one_chunk():
        yield encoded

    assert client.post("/rank", json=example_user).status_code == 200
    streamed = client.post("/rank", content=one_chunk(),
                           headers={"content-type": "application/json"})
    assert streamed.status_code == 200
    assert streamed.json()["ranking"]


def test_a_rejection_does_not_echo_the_value_it_rejected(client, example_user):
    """A 422 that quotes its input is an amplifier: 32 MB in, 33.5 MB out, built on the
    event loop. The message still names the value, which is what makes it actionable -
    just bounded, so the response size is a property of this service and not of the
    caller's payload."""
    long_value = "9" * 20_000                    # under the body limit, so it is validated
    response = client.post("/rank", json={**example_user, "session_dt": long_value})
    assert response.status_code == 422
    assert len(response.content) < 2_000
    message = response.json()["detail"][0]["msg"]
    assert "characters" in message               # says how much was withheld
    assert long_value not in message


def test_a_long_message_is_clipped_even_if_a_validator_forgets():
    """The bound is enforced where the body is written too, not only where the messages
    are. A validator added later cannot reopen this by quoting its input in full."""
    from bl_ranking.serving.app import MAX_MESSAGE_CHARS, _clipped

    assert _clipped("short") == "short"
    clipped = _clipped("x" * 5000)
    assert len(clipped) < MAX_MESSAGE_CHARS + 60
    assert "5000 characters" in clipped


# --- Per-field cost ---------------------------------------------------------------------
#
# The body limit bounds the bytes, not the work they buy. A 64 KB body whose session_dt
# was 64 KB of spaces in front of a valid date passed the limit, parsed cleanly in
# pd.to_datetime at ~17 us per byte, returned 200, and held the event loop for ~1 s; four
# such connections stopped a single worker answering /healthz. Every string field is now
# capped before any validator touches it, and the timestamps more tightly still.

@pytest.mark.parametrize("field", ["session_dt", "register_date", "conversion_dt"])
def test_a_padded_timestamp_is_refused_before_it_is_parsed(client, example_user, field):
    from bl_ranking.serving.schemas import MAX_TIMESTAMP_CHARS

    padded = " " * 2000 + "2026-01-06 19:26:07"
    response = client.post("/rank", json={**example_user, field: padded})
    assert response.status_code == 422
    message = str(response.json()["detail"])
    assert "characters long" in message
    # And a timestamp that is merely generous - an offset, fractional seconds - still fits.
    assert len("2026-01-06T19:26:07.123456+05:30") < MAX_TIMESTAMP_CHARS
    fine = client.post("/rank", json={**example_user, field: "2026-01-06 19:26:07"})
    assert fine.status_code == 200


def test_a_string_field_over_the_cap_is_refused_whatever_it_holds(client, example_user):
    from bl_ranking.serving.schemas import MAX_TEXT_CHARS

    response = client.post("/rank", json={**example_user, "page": "x" * (MAX_TEXT_CHARS + 1)})
    assert response.status_code == 422
    assert "characters long" in str(response.json()["detail"])
    assert client.post("/rank", json={**example_user, "page": "x" * MAX_TEXT_CHARS}).status_code == 200


def test_the_cap_costs_nothing_measurable_and_the_parse_is_never_reached(example_user):
    """Validated in-process so the timing is the validator's alone: the whole point is
    that refusing 64 KB of padding must not cost 64 KB of parsing."""
    import time

    from bl_ranking.serving.schemas import RankRequest

    RankRequest(**example_user)                                  # warm
    padded = {**example_user, "session_dt": " " * 60_000 + "2020-01-01"}
    started = time.perf_counter()
    with pytest.raises(ValueError):
        RankRequest(**padded)
    assert time.perf_counter() - started < 0.05                 # was ~0.8 s


def test_a_body_the_parser_cannot_read_is_counted(client):
    """FastAPI answers these 400 itself, before any code here runs, and the latency
    histogram timed them while no outcome counted them - so the two metrics disagreed by
    exactly the number of such bodies and the traffic was invisible."""
    from bl_ranking.serving import app as app_module

    before = app_module.REQUESTS.labels("bad_body")._value.get()
    nested = ("[" * 30_000 + "]" * 30_000).encode()
    response = client.post("/rank", content=nested, headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert app_module.REQUESTS.labels("bad_body")._value.get() == before + 1


@pytest.mark.parametrize(("header", "expected"), [
    (b"1024", 1024),
    (b"  1024  ", 1024),
    (b"-1", None),           # int() read this as "under the limit" and skipped the cap
    (b"1_0", None),          # int() accepts underscores; the wire format does not
    (b"+5", None),
    (b"abc", None),
])
def test_a_content_length_is_digits_or_it_is_not_a_length(header, expected):
    from bl_ranking.serving.app import _declared_length

    assert _declared_length({"headers": [(b"content-length", header)]}) == expected


def test_a_probe_of_model_is_not_a_refused_ranking_request(client, example_user, monkeypatch):
    """Five GET /model polls on a not-ready worker read as five refused users."""
    from bl_ranking.serving import app as app_module

    saved = app_module.state.ranker
    try:
        before = app_module.REQUESTS.labels("not_ready")._value.get()
        monkeypatch.setattr(app_module.state, "ranker", None)
        for _ in range(3):
            assert client.get("/model").status_code == 503
        assert app_module.REQUESTS.labels("not_ready")._value.get() == before
        assert client.post("/rank", json=example_user).status_code == 503
        assert app_module.REQUESTS.labels("not_ready")._value.get() == before + 1
    finally:
        app_module.state.ranker = saved


def test_the_published_schema_names_the_field_the_response_carries(client, example_user):
    """RankMeta is only read by OpenAPI - the handler is response_model=None - and it kept
    `n_brands` after the handler moved to `brands_ranked`, so a client generated from
    /openapi.json required a field no response ever carried."""
    meta = client.post("/rank", json=example_user).json()["meta"]
    schema = client.get("/openapi.json").json()["components"]["schemas"]["RankMeta"]
    assert set(schema["required"]) <= set(meta)
    assert "brands_ranked" in schema["properties"]
    assert "n_brands" not in schema["properties"]


# --- The metrics directory is shared state, and shared state gets junk in it ------------

def test_a_stray_entry_in_the_metrics_directory_does_not_break_the_scrape(client, monkeypatch,
                                                                          other_worker):
    """A directory named like a gauge file, a 0-byte file (which a starting worker has for
    a moment), and a file of junk each turned every scrape into a 500 until removed."""
    from bl_ranking.serving import app as app_module

    (other_worker / "gauge_livemin_x.db").mkdir()
    (other_worker / "counter_999999.db").write_bytes(b"")
    (other_worker / "histogram_abc.db").write_bytes(b"junk")
    (other_worker / "not_a_metric.txt").write_text("hello")
    monkeypatch.setattr(app_module, "MULTIPROC_DIR", str(other_worker))
    response = client.get("/metrics")
    assert response.status_code == 200
    assert 'bl_rank_requests_total{outcome="ok"} 7.0' in response.text


def test_losing_the_race_to_delete_a_dead_workers_file_is_not_an_error(tmp_path, monkeypatch):
    """Two scrapes sweeping at once both find the same dead worker; only one can delete."""
    from prometheus_client import multiprocess

    from bl_ranking.serving import app as app_module

    (tmp_path / "gauge_livemin_300001.db").write_bytes(b"\x08\x00\x00\x00" + b"\x00" * 60)
    calls = []

    def racing(pid, path=None):
        calls.append(pid)
        raise FileNotFoundError("the other scrape got there first")

    monkeypatch.setattr(multiprocess, "mark_process_dead", racing)
    app_module._forget_dead_workers(str(tmp_path))       # must not raise
    assert calls == [300001]


def test_the_plain_registry_exposition_carries_the_values_not_only_the_names(client, example_user):
    """`# HELP` lines exist for untouched metrics, so asserting names proved nothing."""
    from bl_ranking.serving import app as app_module

    before = app_module.REQUESTS.labels("ok")._value.get()
    client.post("/rank", json=example_user)
    body = client.get("/metrics").text
    assert f'bl_rank_requests_total{{outcome="ok"}} {before + 1}' in body
    assert "bl_rank_latency_seconds_count" in body
    latency_count = float(next(line.split()[-1] for line in body.splitlines()
                               if line.startswith("bl_rank_latency_seconds_count")))
    assert latency_count >= before + 1
