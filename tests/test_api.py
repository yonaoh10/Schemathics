"""The HTTP contract: request validation, the response shape, and the failure modes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

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


def test_missing_register_date_is_a_422(client, example_user):
    user = dict(example_user)
    user.pop("register_date")
    assert client.post("/rank", json=user).status_code == 422


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
