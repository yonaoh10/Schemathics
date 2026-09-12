"""The HTTP contract: request validation, the response shape, and the failure modes."""

from __future__ import annotations

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
