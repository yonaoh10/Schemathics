"""Configuration, ingestion, scheduling, the bundle contract, and the payout backends."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import pandas as pd
import pytest

from bl_ranking.config import Settings
from bl_ranking.models import bundle as bundle_files

# --------------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------------- #

def test_env_overrides_are_typed(monkeypatch):
    """BL_SERVING__WORKERS=7 must arrive as an int, not the string '7'."""
    monkeypatch.setenv("BL_SERVING__WORKERS", "7")
    monkeypatch.setenv("BL_MODEL__PAYOUT__BACKEND", "surrogate")
    monkeypatch.setenv("BL_DATA__LOOKBACK_DAYS", "30")
    loaded = Settings.load()
    assert loaded.serving.workers == 7
    assert loaded.model.payout.backend == "surrogate"
    assert loaded.data.lookback_days == 30


def test_flat_view_covers_every_nested_key():
    flat = Settings.load().flat()
    for key in ("model.catboost.n_estimators", "model.payout.backend",
                "schedule.cron", "serving.feature_path", "data.days_for_test"):
        assert key in flat, key


def test_research_hyperparameters_match_the_research_code():
    """conf/config.yaml is logged to MLflow as the run's parameters; if it drifts from
    what bl_models_train.py actually uses, every comparison becomes a lie."""
    source = (Path(__file__).resolve().parents[1]
              / "src/bl_ranking/research/bl_models_train.py").read_text()
    catboost = Settings.load().model.catboost
    assert f"depth={catboost.depth}" in source
    assert f"n_estimators={catboost.n_estimators}" in source
    assert f"random_seed={catboost.random_seed}" in source
    assert f"eval_metric='{catboost.eval_metric}'" in source
    assert f"CONTEXT_SIZE = {Settings.load().model.payout.context_size}" in source


# --------------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------------- #

def test_sanitise_enforces_the_research_code_input_contract():
    from bl_ranking.data.ingest import sanitise

    frame = pd.DataFrame({
        "cellphone": ["(305) 555-0142", "+1 786 991 4030", "n/a", "", 7869914030],
        "credit_score": [None] * 5,                  # all-null -> float64 without help
        "industry": ["construction"] * 5,
        "loan_amount": ["$25,000 - $49,999"] * 5,
        "loan_reason": ["Payroll"] * 5,
        "monthly_revenue": ["$20,000 - $49,999"] * 5,
        "time_in_business": ["2+ years"] * 5,
        "device_type": ["mobile"] * 5,
        "business_type": ["LLC"] * 5,
        "payout": ["42.5", None, "not a number", 10, 0],
        "session_dt": ["2026-01-06 19:24:22"] * 4 + [None],
        "conversion_dt": ["2026-01-06 19:26:10"] * 5,
        "register_date": ["2026-01-06 19:26:07"] * 5,
    })
    clean, repairs = sanitise(frame)

    # 1. cellphone must survive .astype(int)
    assert clean["cellphone"].dtype == "int64"
    clean["cellphone"].astype(int).astype(str).str[:3]
    assert clean.loc[0, "cellphone"] == 3055550142
    # The US country code is stripped so '+1 786...' and '786...' share a prefix.
    assert clean.loc[1, "cellphone"] == 7869914030
    assert repairs["cellphone"] == 2                 # 'n/a' and ''

    # 2. survey columns must support the .str accessor
    clean["credit_score"].astype("object").str.lower()

    # 3. payout must be numeric
    assert pd.api.types.is_numeric_dtype(clean["payout"])
    assert repairs["payout"] == 1                    # 'not a number'

    # A row without a session timestamp cannot be placed on the split timeline.
    assert repairs["dropped_no_session_dt"] == 1
    assert len(clean) == 4


def test_null_sub_ids_do_not_create_train_serve_skew(tmp_path):
    """The highest-impact skew in the original code, and the reason ingest exists.

    `import_preprocess` does `fillna('Other')` then `.astype(str)` on sub1/sub2/sub3.
    One null anywhere in the column makes pandas read it as float64, so training sees
    '1815195.0'. A JSON request carries the same id as an integer, so serving sees
    '1815195'. Three of the fourteen categorical features then miss on every single
    request, forever, with nothing in any log to say so.

    Goes through a real CSV because that is where the damage happens - by the time a
    float has been rounded, no downstream code can undo it.
    """
    from bl_ranking.data.ingest import READ_AS_TEXT, sanitise

    csv = tmp_path / "extract.csv"
    csv.write_text(
        "sub1,sub2,sub3,campaign_id,cellphone,payout,session_dt,conversion_dt,register_date,"
        "credit_score,industry,loan_amount,loan_reason,monthly_revenue,time_in_business,"
        "device_type,business_type\n"
        "1815195,01121993 Ad set,1513124082,120227360861540306,7869914030,0,"
        "2026-01-06 19:24:22,,,x,x,x,x,x,x,x,x\n"
        ",,,,,0,2026-01-06 19:25:22,,,x,x,x,x,x,x,x,x\n"
    )

    # Precondition: pandas' own inference is what breaks both columns.
    naive = pd.read_csv(csv, low_memory=False)
    assert naive["sub1"].fillna("Other").astype(str).iloc[0] == "1815195.0"
    assert naive["campaign_id"].dtype == "float64"      # one null demotes the column
    assert int(naive["campaign_id"].iloc[0]) == 120227360861540304   # ...306 was lost

    clean, repairs = sanitise(pd.read_csv(csv, low_memory=False, dtype=READ_AS_TEXT))

    # sub ids now stringify the way a JSON request does, and nulls still mean 'Other'.
    stringified = clean["sub1"].fillna("Other").astype(str).tolist()
    assert stringified == ["1815195", "Other"]
    assert repairs["sub_ids"] > 0

    # campaign_id stays numeric, as the model expects, and stays exact.
    assert clean["campaign_id"].dtype == "int64"
    assert clean.loc[0, "campaign_id"] == 120227360861540306


def test_missing_required_column_fails_loudly():
    from bl_ranking.data.ingest import _require_columns

    with pytest.raises(ValueError, match="missing columns"):
        _require_columns(pd.DataFrame({"session_id": [1]}))


def test_delta_write_creates_a_new_version_each_time(settings, ingested):
    from bl_ranking.data.delta import read_snapshot, table_version, write_snapshot

    before = table_version(settings.paths.delta_table)
    snapshot = read_snapshot(settings.paths.delta_table)
    after = write_snapshot(snapshot.frame, settings.paths.delta_table)
    assert after == before + 1

    # Time travel back to the version the earlier run read.
    old = read_snapshot(settings.paths.delta_table, version=before)
    assert old.version == before
    assert len(old.frame) == len(snapshot.frame)


# --------------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------------- #

def test_quartz_expression_fires_sunday_0500():
    from datetime import datetime

    from apscheduler.triggers.cron import CronTrigger

    from bl_ranking.ops.schedule import parse_quartz

    fields = parse_quartz(Settings.load().schedule.cron)
    trigger = CronTrigger(timezone="UTC", **fields.as_apscheduler_kwargs())

    friday = datetime(2026, 9, 11, 14, 0, tzinfo=UTC)
    first = trigger.get_next_fire_time(None, friday)
    assert first.strftime("%A %H:%M") == "Sunday 05:00"

    # And weekly thereafter.
    second = trigger.get_next_fire_time(first, first)
    assert (second - first).days == 7


@pytest.mark.parametrize("expression", ["0 5 * * SUN", "0 0 5 ? * SUN * extra extra"])
def test_non_quartz_expressions_are_rejected(expression):
    """A 5-field unix cron silently means something else in Quartz; refuse it."""
    from bl_ranking.ops.schedule import parse_quartz

    with pytest.raises(ValueError, match="Quartz"):
        parse_quartz(expression)


def test_local_and_databricks_schedules_are_the_same_string():
    import yaml

    bundle_path = Path(__file__).resolve().parents[1] / "databricks" / "databricks.yml"
    spec = yaml.safe_load(bundle_path.read_text())
    job = spec["resources"]["jobs"]["bl_weekly_training"]
    assert job["schedule"]["quartz_cron_expression"] == Settings.load().schedule.cron
    assert job["schedule"]["timezone_id"] == Settings.load().schedule.timezone


# --------------------------------------------------------------------------------- #
# Bundle and registry
# --------------------------------------------------------------------------------- #

def test_bundle_is_complete_and_self_describing(bundle):
    assert bundle_files.verify(bundle) == []

    manifest = bundle_files.Manifest.read(bundle)
    assert manifest.mlflow_run_id
    assert manifest.delta_version >= 0
    assert manifest.rows_train > 0
    assert manifest.n_brands > 0
    assert len(manifest.train_columns) == 25
    assert manifest.research_code_sha

    # The manifest must describe the files that are actually there.
    clients = pd.read_csv(bundle / bundle_files.CLIENTS_FILE)
    assert int((clients["client_name"] != "other").sum()) == manifest.n_brands


# The digests of the two scripts exactly as they were handed over. Pinning them is the
# whole audit trail: without this the checksum is recorded faithfully every run and
# still never fails anything, because nothing compares it to a known-good value.
VENDORED_SHA256 = {
    "bl_models_train.py":
        "b1274c55fb6b8eff0aad333ca540aba53e991a6862edefd9eaa434e5b43a573c",
    "bl_exp_payout_predictor.py":
        "d9ac9696ede0404c0f5273903394f085e2cb1b538187593afd0f1cb0b332b911",
}
VENDORED_COMBINED_SHA = "e9384db687f76093"


def test_vendored_research_scripts_are_byte_for_byte_as_delivered():
    """The claim 'we did not change the given code', as a failing test rather than prose."""
    import hashlib

    import bl_ranking.research as pkg

    directory = Path(pkg.__file__).parent
    for name, expected in VENDORED_SHA256.items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert actual == expected, (
            f"{name} has been modified. The brief forbids changing the given functions "
            f"and logic; if the change is deliberate, update VENDORED_SHA256 in the "
            f"same commit so the edit is visible in review."
        )


def test_logged_digest_matches_the_vendored_scripts():
    """The value written to MLflow is the one the pinned files produce."""
    import bl_ranking.training.job as job

    assert job.research_code_sha() == VENDORED_COMBINED_SHA


def test_research_code_checksum_detects_an_edit(tmp_path, monkeypatch):
    """Proves the detector works, so a passing checksum test means something."""
    import bl_ranking.training.job as job

    original = job.research_code_sha()
    fake = tmp_path / "research"
    fake.mkdir()
    (fake / "bl_models_train.py").write_text("# edited\n")
    (fake / "bl_exp_payout_predictor.py").write_text("# edited\n")
    monkeypatch.setattr(job, "_research_dir", lambda: fake)
    assert job.research_code_sha() != original


def test_checksum_refuses_to_report_over_a_missing_file(tmp_path, monkeypatch):
    """A digest computed over an incomplete set would attest to nothing."""
    import pytest

    import bl_ranking.training.job as job

    half = tmp_path / "research"
    half.mkdir()
    (half / "bl_models_train.py").write_text("# only one of the two\n")
    monkeypatch.setattr(job, "_research_dir", lambda: half)
    with pytest.raises(FileNotFoundError):
        job.research_code_sha()


def test_champion_alias_points_at_the_trained_version(bundle, settings):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow.resolved_tracking_uri())
    client = MlflowClient()
    version = client.get_model_version_by_alias(
        settings.mlflow.registered_model, settings.mlflow.serving_alias)
    manifest = bundle_files.Manifest.read(bundle)
    assert version.run_id == manifest.mlflow_run_id


def test_researcher_log_is_attached_to_the_run(bundle, settings):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow.resolved_tracking_uri())
    client = MlflowClient()
    run_id = bundle_files.Manifest.read(bundle).mlflow_run_id
    artifacts = [a.path for a in client.list_artifacts(run_id, "researcher_log")]
    assert artifacts, "no researcher log logged"
    assert artifacts[0].endswith(".log")

    local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifacts[0])
    text = Path(local).read_text()
    # The log is the research code's own output, so its own phrases must be in it.
    assert "raw data rows" in text
    assert "Catboost classifier for lead finished" in text


def test_run_parameters_make_runs_comparable(bundle, settings):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow.resolved_tracking_uri())
    client = MlflowClient()
    run = client.get_run(bundle_files.Manifest.read(bundle).mlflow_run_id)

    for key in ("data.delta_version", "data.rows", "data.window_start",
                "cfg.model.catboost.n_estimators", "cfg.model.payout.backend",
                "lib.pandas", "lib.catboost"):
        assert key in run.data.params, key
    for key in ("mode", "git_sha", "research_code_sha", "payout_backend"):
        assert key in run.data.tags, key


# --------------------------------------------------------------------------------- #
# Payout backends
# --------------------------------------------------------------------------------- #

def test_catboost_fallback_fits_and_predicts(bundle):
    import joblib

    from bl_ranking.config import PayoutSettings
    from bl_ranking.models.payout import create_backend

    context = joblib.load(bundle / bundle_files.PAYOUT_CONTEXT_FILE)
    backend = create_backend(PayoutSettings(backend="catboost_fallback"))
    backend.fit(context["x"], context["y"])

    predictions = backend.predict(context["x"].head(20))
    assert len(predictions) == 20
    assert predictions.dtype.kind == "f"
    # A backend that is not TabPFN must say so, on every response.
    assert backend.describe()["payout_exact"] is False


def test_unknown_backend_is_rejected():
    from bl_ranking.config import PayoutSettings
    from bl_ranking.models.payout import create_backend

    with pytest.raises(ValueError, match="Unknown payout backend"):
        create_backend(PayoutSettings(), name="not_a_backend")


def test_serving_degrades_rather_than_failing_when_a_backend_cannot_load(bundle):
    """An endpoint on the funnel's critical path must keep answering."""
    import joblib

    from bl_ranking.config import PayoutSettings
    from bl_ranking.models.payout import PayoutContext, prepare_backend

    raw = joblib.load(bundle / bundle_files.PAYOUT_CONTEXT_FILE)
    context = PayoutContext(x=raw["x"], y=raw["y"], columns=list(raw["columns"]))

    seen: list[str] = []
    # tabpfn_client with no token cannot authenticate, which is the shape of a real
    # outage (bad credentials, unreachable API, missing weights).
    backend = prepare_backend(
        PayoutSettings(backend="tabpfn_client"), context, bundle,
        on_fallback=lambda name, exc: seen.append(name),
    )
    assert seen == ["tabpfn_client"]
    assert backend.name == "catboost_fallback"
    assert len(backend.predict(context.x.head(5))) == 5


def test_manifest_round_trips(tmp_path):
    manifest = bundle_files.Manifest(
        trained_at="2026-09-11T00:00:00Z", mlflow_run_id="abc", delta_version=3,
        rows_train=100, n_brands=15, payout_backend="tabpfn_local",
        train_columns=["a", "b"],
    )
    manifest.write(tmp_path)
    assert json.loads((tmp_path / bundle_files.MANIFEST_FILE).read_text())["delta_version"] == 3
    assert bundle_files.Manifest.read(tmp_path).payout_backend == "tabpfn_local"

def test_attribution_ids_survive_the_csv_staging_round_trip(tmp_path):
    """The train/serve skew this repo claims to have closed, as an executable check.

    `sanitise` normalises sub1/sub2/sub3 and leaves their nulls null, because the
    research code fills them itself. Staging writes the frame back to CSV and the
    research code re-reads it with a bare `pd.read_csv`, which infers float64 for a
    digits-plus-empties column - and `.astype(str)` then yields '7448788.0' in
    training against '7448788' from a request. Two of the fourteen categorical
    features would miss on every single call, with nothing in any log.

    This asserts the level a training run produces equals the level the serving
    feature path produces, for the same raw id.
    """
    import pandas as pd

    from bl_ranking.data import ingest
    from bl_ranking.serving import fast_features
    from bl_ranking.serving.ranker import WARMUP_USER

    raw_id = "7448788"
    frame = pd.DataFrame({
        # A null in the column is what triggers the demotion, so it must be present.
        "sub1": pd.array([raw_id, None], dtype="object"),
        "sub2": pd.array(["07448788 Ad set", None], dtype="object"),
        "sub3": pd.array(["1021632077", None], dtype="object"),
    })
    ingest.stage_for_research_code(frame, tmp_path / "input")
    staged = pd.read_csv(tmp_path / "input" / "bl_full_data.csv")

    # The research code's own two lines, verbatim in effect.
    training_levels = staged["sub1"].fillna("Other").astype(str).tolist()

    user = dict(WARMUP_USER) | {"sub1": int(raw_id)}
    serving_level = fast_features.build_feature_row(user, None)["sub1"]

    assert training_levels[0] == serving_level == raw_id, (
        f"train/serve skew is back: training produced {training_levels[0]!r}, "
        f"serving sends {serving_level!r}"
    )
    # The null still means what it always meant.
    assert training_levels[1] == "Other"

def test_a_bundle_with_mismatched_feature_order_is_refused(bundle, settings, tmp_path):
    """The mis-slotting hazard the atomic bundle exists to prevent, as a test.

    `prediction_expected_payout` reorders the frame by the payout model's column list
    before handing it to the classifier. If those two disagree the features are scored
    in the wrong slots - industry as page, city as device_type - with no error and no
    log line, just a quietly worse ranking for as long as that version is champion.

    Before the load-time check, a permuted column list loaded clean and changed the
    returned ordering.
    """
    import shutil

    import joblib
    import pytest

    from bl_ranking.models import bundle as bundle_files
    from bl_ranking.serving.ranker import BrandRanker

    damaged = tmp_path / "bundle"
    shutil.copytree(bundle, damaged)

    context = joblib.load(damaged / bundle_files.PAYOUT_CONTEXT_FILE)
    columns = list(context["columns"])
    columns[1], columns[2] = columns[2], columns[1]
    context["columns"] = columns
    joblib.dump(context, damaged / bundle_files.PAYOUT_CONTEXT_FILE)

    with pytest.raises(ValueError, match="inconsistent"):
        BrandRanker.load(damaged, settings)


def test_a_healthy_bundle_passes_the_consistency_check(bundle, settings):
    """The guard must not reject what training actually produces."""
    from bl_ranking.serving.ranker import BrandRanker

    assert BrandRanker.load(bundle, settings) is not None

def test_one_unrepresentable_id_does_not_corrupt_the_rest_of_its_column():
    """The gate must not round a real id because another row is malformed.

    `pd.to_numeric` chooses one dtype for the whole column, so a single value too large
    for int64 demoted every other value to float64 - and 120227360861540306, a real
    campaign id, came back as 120227360861540304 while the repair counters reported
    nothing. One bad row silently corrupted the column it shared, which is the exact
    failure this gate exists to prevent.
    """
    import pandas as pd

    from bl_ranking.data.ingest import sanitise

    exact = 120227360861540306
    frame = pd.DataFrame({
        "campaign_id": ["999999999999999999999999", str(exact), f"{exact}.0"],
        "cellphone": ["99999999999999999999", "7869914030", "(305) 555-0142"],
        "payout": [0] * 3,
        "session_dt": ["2026-01-06 19:24:22"] * 3,
        "conversion_dt": [None] * 3, "register_date": [None] * 3,
        **{c: ["x"] * 3 for c in ["credit_score", "industry", "loan_amount", "loan_reason",
                                  "monthly_revenue", "time_in_business", "device_type",
                                  "business_type"]},
    })
    cleaned, repairs = sanitise(frame)

    ids = cleaned["campaign_id"].tolist()
    assert ids[1] == exact, "a well-formed id was rounded by a malformed sibling"
    assert ids[2] == exact, "the '.0' form must resolve to the same id"
    assert ids[0] == 0, "an unrepresentable id becomes the 0 level"
    assert repairs["campaign_id"] == 1, "the repair must be counted, not silent"

    phones = cleaned["cellphone"].tolist()
    assert phones[0] == 0, "an oversized phone number used to wrap to INT64_MIN"
    assert phones[1] == 7869914030
    assert repairs["cellphone"] == 1

def test_a_bundle_with_no_brands_is_refused(bundle, settings, tmp_path):
    """An empty brand universe used to be a silent total outage.

    The worker loaded, warm-up passed, /readyz went green, and every request returned
    200 with an empty ranking - a service that looks healthy from the outside while
    answering nothing.
    """
    import shutil

    import pandas as pd
    import pytest

    from bl_ranking.models import bundle as bundle_files
    from bl_ranking.serving.ranker import BrandRanker

    empty = tmp_path / "bundle"
    shutil.copytree(bundle, empty)
    clients = empty / bundle_files.CLIENTS_FILE
    pd.read_csv(clients).head(0).to_csv(clients, index=False)

    with pytest.raises(ValueError, match="empty brand universe"):
        BrandRanker.load(empty, settings)


def test_a_bundle_without_a_manifest_is_refused(bundle, settings, tmp_path):
    """The bundle is the unit of versioning; the manifest is what identifies it."""
    import shutil

    import pytest

    from bl_ranking.models import bundle as bundle_files
    from bl_ranking.serving.ranker import BrandRanker

    incomplete = tmp_path / "bundle"
    shutil.copytree(bundle, incomplete)
    (incomplete / bundle_files.MANIFEST_FILE).unlink()

    with pytest.raises(FileNotFoundError, match=bundle_files.MANIFEST_FILE):
        BrandRanker.load(incomplete, settings)


def test_the_pyfunc_reports_errors_beside_the_ranking_not_inside_it(bundle, settings):
    """The MLflow path is a serving surface and owes the same contract as HTTP.

    A failing row used to come back as a ranking containing a brand named __error__
    whose value was a string, where every real entry is a {rank, expected_payout}
    object. Any consumer iterating the ranking either crashed or treated the message
    as a lender. It also skipped request validation entirely, so a row's features
    depended on which other rows shared its batch.
    """
    import json

    import pandas as pd

    from bl_ranking.serving.pyfunc import BrandRankerModel
    from bl_ranking.serving.ranker import WARMUP_USER, BrandRanker

    model = BrandRankerModel()
    model._ranker = BrandRanker.load(bundle, settings)

    good = dict(WARMUP_USER)
    malformed = dict(WARMUP_USER) | {"session_dt": "not a date"}
    frame = model.predict(None, pd.DataFrame([good, malformed]))

    assert "error" in frame.columns
    for payload in frame["ranking"]:
        assert "__error__" not in json.loads(payload)

    assert frame["error"][0] is None
    assert frame["error"][1] is not None

    # The same row scored alone must give the same answer as in a mixed batch.
    alone = model.predict(None, pd.DataFrame([good]))["ranking"][0]
    assert json.loads(alone) == json.loads(frame["ranking"][0])

def test_an_unreachable_registry_does_not_undo_a_rollback(bundle, settings, monkeypatch):
    """Falling back to the newest local bundle is the opposite of a rollback.

    `_latest_local_bundle` picks the most recent run on disk, which immediately after
    a rollback is the version the operator rolled back *from*. A worker restarting
    during a brief registry outage would therefore quietly restore the bad model and
    report itself healthy doing it.
    """
    import pytest

    from bl_ranking.serving import model_source

    def unreachable(uri, s):
        raise ConnectionError("registry down")

    monkeypatch.setattr(model_source, "_from_uri", unreachable)

    # A deployment with a tracking server: the registry decides, so refuse to guess.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    with pytest.raises(RuntimeError, match="rolled back from"):
        model_source.resolve_bundle(settings)

    # Offline development has no registry to consult, so the documented fallback
    # stands and resolves to the local bundle the fixture built.
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(settings.mlflow, "tracking_uri", None, raising=False)
    assert model_source.resolve_bundle(settings).exists()

