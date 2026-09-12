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
    what bl_models_train.py actually uses, every comparison becomes a lie.

    Checked through the same function the training job calls, so the test and the run
    cannot disagree about what counts as drift."""
    from bl_ranking.training.job import assert_config_mirrors_research

    assert_config_mirrors_research(Settings.load())
    source = (Path(__file__).resolve().parents[1]
              / "src/bl_ranking/research/bl_models_train.py").read_text()
    # context_size is deliberately not in that function: the trainer overrides
    # tabpfn_regression_payout and really does apply it, so it is a control rather than
    # a mirror. It still has to start life matching the research default.
    assert f"CONTEXT_SIZE = {Settings.load().model.payout.context_size}" in source


@pytest.mark.parametrize(("setting", "value"), [
    ("model.catboost.depth", 2),
    ("model.catboost.n_estimators", 50),
    ("model.catboost.random_seed", 7),
    ("model.catboost.eval_metric", "Logloss"),
    ("data.days_for_test", 14),
])
def test_a_hyperparameter_that_cannot_be_applied_cannot_be_claimed(setting, value):
    """These settings mirror values the research scripts hard-code, and the brief forbids
    editing those - so nothing reads them back into the model. `BL_MODEL__CATBOOST__DEPTH=2`
    used to produce a run whose params said depth 2 while the shipped classifier had depth
    8, and the registered version's tags said so too. A researcher sweeping depth would
    have compared two identical models and drawn a conclusion from the noise."""
    from bl_ranking.training.job import assert_config_mirrors_research

    settings = Settings.load()
    target = settings
    *path, leaf = setting.split(".")
    for part in path:
        target = getattr(target, part)
    setattr(target, leaf, value)

    with pytest.raises(ValueError, match=setting):
        assert_config_mirrors_research(settings)


# --------------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------------- #

def test_sanitise_enforces_the_research_code_input_contract():
    from bl_ranking.data.ingest import sanitise

    frame = pd.DataFrame({
        "cellphone": ["(305) 555-0142", "+1 786 991 4030", "n/a", "", 7869914030],
        # Numeric survey codes: an object column of ints still refuses `.str`, so the
        # gate has to null them rather than merely move the dtype.
        "credit_score": ["550-599", 7, None, "720+", 9],
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

    # 2. survey columns must support the .str accessor as they stand, with no help
    #    from the caller - the vendored code calls it on whatever pandas hands it.
    clean["credit_score"].str.lower()
    assert list(clean["credit_score"])[:2] == ["550-599", None]
    assert repairs["survey_nonstring"] == 2          # the 7 and the 9

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

@pytest.mark.parametrize("names,expected", [
    ([None], "blank brand name"),
    ([""], "blank brand name"),
    (["sba central", "sba central"], "more than once"),
    ([], "empty brand universe"),
])
def test_a_degenerate_brand_universe_is_refused(bundle, settings, tmp_path, names, expected):
    """Each of these used to load and serve something wrong rather than nothing.

    A null name comes back from pandas as NaN and reaches the funnel as a lender
    literally called 'nan'. Duplicates collapse in the ranking, so the response carries
    fewer brands than the bundle claims.
    """
    import shutil

    import pandas as pd

    from bl_ranking.models import bundle as bundle_files
    from bl_ranking.serving.ranker import BrandRanker

    damaged = tmp_path / "bundle"
    shutil.copytree(bundle, damaged)
    pd.DataFrame({bundle_files.CLIENT_NAME_COLUMN: names}).to_csv(
        damaged / bundle_files.CLIENTS_FILE, index=False)

    with pytest.raises(ValueError, match=expected):
        BrandRanker.load(damaged, settings)


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

def test_one_malformed_cell_does_not_fail_the_whole_pyfunc_batch(bundle, settings):
    """Per-row isolation has to survive the normalisation step too.

    `pd.isna` returns an array for a list cell, and branching on that raises "truth
    value of an array is ambiguous" - which escaped the per-row handler and failed
    every row in the batch over one bad cell.
    """
    import pandas as pd

    from bl_ranking.serving.pyfunc import BrandRankerModel
    from bl_ranking.serving.ranker import WARMUP_USER, BrandRanker

    model = BrandRankerModel()
    model._ranker = BrandRanker.load(bundle, settings)

    good = dict(WARMUP_USER)
    frame = model.predict(None, pd.DataFrame([good, dict(good) | {"fname": ["a", "b"]}, good]))

    assert len(frame) == 3
    assert frame["error"][0] is None and frame["error"][2] is None
    assert frame["error"][1] is not None

@pytest.mark.parametrize("uri,authoritative", [
    ("http://mlflow:5000", True),
    ("databricks", True),
    ("postgresql://user@host/mlflow", True),
    ("sqlite:////var/lib/mlflow.db", True),
    ("file:///tmp/mlruns", False),
    ("/tmp/mlruns", False),
])
def test_every_registry_backend_counts_as_authoritative(settings, monkeypatch, uri, authoritative):
    """Only a plain directory of files is the offline case the fallback is for.

    An earlier version listed the remote schemes it knew - http, https, databricks -
    which silently left the rollback-undoing fallback live for every postgresql://,
    mysql:// and sqlite:// registry, which are the ordinary production setups.
    """
    from bl_ranking.serving.model_source import _registry_is_authoritative

    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    assert _registry_is_authoritative(settings) is authoritative


@pytest.mark.parametrize(("uri", "authoritative"), [
    # Unity Catalog's own spelling, which carries no scheme separator and was read as a
    # relative directory - the fallback left live on the most administered setup there is.
    ("databricks-uc", True),
    ("databricks-uc://profile", True),
    # A single-slash typo is not a directory either, and reading it as one re-enabled the
    # fallback on a deployment whose operator plainly meant a server.
    ("http:/mlflow:5000", True),
    ("./runs", False),
])
def test_the_databricks_family_and_a_typo_are_not_directories(
        settings, monkeypatch, uri, authoritative):
    from bl_ranking.serving.model_source import _registry_is_authoritative

    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    assert _registry_is_authoritative(settings) is authoritative


def test_a_registry_somewhere_else_still_counts(settings, monkeypatch):
    """MLflow lets the registry live apart from the tracking store, and the registry is
    what holds the alias - so reading only the tracking URI left the rollback-undoing
    fallback live for exactly the setup where the alias is furthest away."""
    from bl_ranking.serving.model_source import _registry_is_authoritative

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "mlruns")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "https://registry.internal/")
    assert _registry_is_authoritative(settings) is True

    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "")
    assert _registry_is_authoritative(settings) is False

def _gate_frame(rows=3, **override):
    """A minimal well-formed extract, for tests that damage one column of it."""
    import pandas as pd

    survey = ["credit_score", "industry", "loan_amount", "loan_reason",
              "monthly_revenue", "time_in_business", "device_type", "business_type"]
    frame = pd.DataFrame({
        "cellphone": ["7869914030"] * rows,
        "campaign_id": ["1"] * rows,
        "payout": [0.0] * rows,
        "session_dt": ["2026-01-06 19:24:22"] * rows,
        "conversion_dt": [None] * rows,
        "register_date": [None] * rows,
        # additional_features takes .str.len() of both, so the gate has an invariant for
        # them and a frame without them is not a realistic extract.
        "fname": ["Rigoberto"] * rows,
        "lname": ["Rodriguez"] * rows,
        **{c: ["x"] * rows for c in survey},
    })
    for column, values in override.items():
        frame[column] = values
    return frame


def test_the_phone_rule_is_the_same_on_both_sides():
    """cellphone_prefix is a model feature, so the two sides must derive it identically.

    Ingestion stripped the first digit of ANY 11-digit number while serving only strips
    a leading US country code, so an 11-digit number starting with anything else got a
    different prefix in training than at request time.
    """
    from bl_ranking.data.ingest import sanitise
    from bl_ranking.serving.schemas import RankRequest

    numbers = ["27869914030", "17869914030", "7869914030"]
    ingested, _ = sanitise(_gate_frame(cellphone=numbers))
    served = [RankRequest.normalise_cellphone(n) for n in numbers]
    assert ingested["cellphone"].tolist() == served


def test_a_non_finite_payout_is_removed_and_counted():
    """inf survives to_numeric, then reaches the payout model and the `payout > 0` mask."""
    import numpy as np

    from bl_ranking.data.ingest import sanitise

    cleaned, repairs = sanitise(_gate_frame(payout=[float("inf"), float("-inf"), 12.5]))
    assert repairs["payout_not_finite"] == 2
    finite = cleaned["payout"].dropna()
    assert np.isfinite(finite).all()
    assert 12.5 in finite.tolist()


def test_mixed_timezone_offsets_neither_crash_nor_drop_rows():
    """A DST transition makes one export column carry two different offsets.

    Parsing without utc=True raises on the .dt accessor; parsing with it turns every
    naive value into NaT, which the row filter then drops. Both happened here in turn.
    """
    from bl_ranking.data.ingest import sanitise

    cleaned, repairs = sanitise(_gate_frame(session_dt=[
        "2026-01-06 19:24:22+00:00",   # UTC
        "2026-01-06 19:24:22-05:00",   # same wall clock, five hours later in UTC
        "2026-01-06 19:24:22",         # naive, assumed UTC on both sides
    ]))

    assert len(cleaned) == 3, "a valid row was dropped"
    assert repairs.get("dropped_no_session_dt", 0) == 0
    assert cleaned["session_dt"].tolist() == [
        "2026-01-06 19:24:22", "2026-01-07 00:24:22", "2026-01-06 19:24:22"]

def test_re_promoting_the_serving_version_does_not_destroy_the_way_back(settings, bundle):
    """champion_previous must never point at the version that is currently serving.

    MLflow returns `.version` as a string while a caller may hold an int, and
    "3" != 3 is always true - so the guard never fired. Re-running the documented
    rollback pointed champion_previous at the version being rolled back *to*,
    overwriting the only pointer back to the one it replaced.

    With a single registered version the correct outcome is that the alias is not
    written at all: nothing was replaced, so there is nothing to point back to.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from bl_ranking.ops.registry import set_alias

    mlflow.set_tracking_uri(settings.mlflow.resolved_tracking_uri())
    client = MlflowClient()
    name = settings.mlflow.registered_model
    alias = settings.mlflow.serving_alias
    serving = client.get_model_version_by_alias(name, alias).version

    # A careless operator repeating the command, once as a string and once as an int:
    # argparse hands over a string, MLflow's file store reports an int, and the two
    # never compared equal.
    set_alias(settings, str(serving))
    set_alias(settings, int(serving))

    # Compared as strings: the file store returns an int here while a SQL-backed store
    # returns a string, which is the type mismatch the fix is about in the first place.
    assert str(client.get_model_version_by_alias(name, alias).version) == str(serving)
    try:
        previous = client.get_model_version_by_alias(name, f"{alias}_previous").version
    except Exception:
        return          # never written, which is right: nothing was replaced
    assert str(previous) != str(serving), (
        "champion_previous points at the serving version, so there is no way back"
    )

def test_an_explicit_backend_override_beats_the_bundle(bundle, monkeypatch):
    """The README documents a one-variable switch, and it did nothing.

    A bundle records the backend it was built with so a rollback carries its own, which
    is deliberate - but it was also overruling an explicit override, so the variable
    changed the config, the manifest won, and GET /model reported the bundle's backend
    as though nothing had been asked for.
    """
    from bl_ranking.config import Settings
    from bl_ranking.serving.ranker import BrandRanker

    monkeypatch.setenv("BL_MODEL__PAYOUT__BACKEND", "catboost_fallback")
    asked = BrandRanker.load(bundle, Settings.load())
    assert asked.describe()["payout_backend"] == "catboost_fallback"

    # With nothing set, the bundle still decides - that half of the rule must hold too.
    monkeypatch.delenv("BL_MODEL__PAYOUT__BACKEND", raising=False)
    from bl_ranking.models import bundle as bundle_files
    manifest = bundle_files.Manifest.read(bundle)
    default = BrandRanker.load(bundle, Settings.load())
    assert default.describe()["payout_backend"] == manifest.payout_backend

def test_a_payout_model_with_the_wrong_column_order_is_refused(bundle, settings):
    """The classifier check covered half the ranking; this covers the other half.

    The CatBoost-based payout backends slice the frame by a column list they persisted
    at training time, so a bundle whose payout model and payout context disagree scores
    the dollar half in the wrong slots - while the classifier half stays correct, which
    makes the result look plausible rather than broken.

    The guard is exercised directly rather than by damaging a bundle on disk: only the
    surrogate backend persists column metadata, so a disk-level test skips entirely
    under the offline backend CI runs, which would leave the check unprotected.
    """
    import pytest

    from bl_ranking.serving.ranker import BrandRanker, _assert_payout_columns_match

    ranker = BrandRanker.load(bundle, settings)
    payout = ranker.warm.payout
    columns = list(ranker.warm.columns)

    # A backend with no fixed column order has nothing to check and must not raise.
    _assert_payout_columns_match(payout, columns, bundle)

    class Mismatched:
        name = "mismatched"

        def expected_columns(self):
            swapped = list(columns)
            swapped[1], swapped[2] = swapped[2], swapped[1]
            return swapped

    with pytest.raises(ValueError, match="mis-slot"):
        _assert_payout_columns_match(Mismatched(), columns, bundle)

    class Shorter:
        name = "shorter"

        def expected_columns(self):
            return columns[:-1]

    with pytest.raises(ValueError, match="mis-slot"):
        _assert_payout_columns_match(Shorter(), columns, bundle)

def test_an_entirely_null_column_does_not_break_ingestion(tmp_path, settings):
    """A quiet window has no conversions, and conversion_dt is then wholly null.

    Arrow infers the `Null` type for an empty column and Delta rejects it outright -
    "Invalid data type for Delta Lake: Null" - failing the whole ingestion with a
    message that names neither the column nor the file. It is not an exotic input: any
    window where nobody converted produces it, as does a freshly launched funnel.
    """
    import pandas as pd

    from bl_ranking.data.delta import read_snapshot
    from bl_ranking.data.ingest import ingest

    survey = ["credit_score", "industry", "loan_amount", "loan_reason",
              "monthly_revenue", "time_in_business", "device_type", "business_type"]
    rows = 6
    extract = pd.DataFrame({
        "session_id": [f"s{i}" for i in range(rows)],
        "cellphone": ["7869914030"] * rows,
        "campaign_id": ["120227360861540306"] * rows,
        "payout": [0.0] * rows,
        "session_dt": ["2026-01-06 19:24:22"] * rows,
        "conversion_dt": [None] * rows,          # nobody converted
        "register_date": ["2026-01-06 19:26:07"] * rows,
        "client_name": ["sba central"] * rows,
        "disposition": ["Rejected"] * rows,
        "disposition_source": ["x"] * rows,
        "page": ["p"] * rows,
        "auto_city": ["Miami"] * rows, "auto_state": ["Florida"] * rows,
        "auto_country": ["United States"] * rows,
        "sub1": ["1"] * rows, "sub2": ["2"] * rows, "sub3": ["3"] * rows,
        "fname": ["Michael"] * rows, "lname": ["Smith"] * rows,
        **{c: ["x"] * rows for c in survey},
    })

    raw = tmp_path / "raw"
    raw.mkdir()
    extract.to_csv(raw / "bl_full_data.csv", index=False)

    local = Settings.load()
    local.paths.raw_dir = str(raw)
    local.paths.delta_table = str(tmp_path / "delta")

    report = ingest(local)
    assert report.rows_out == rows
    assert len(read_snapshot(local.paths.delta_table).frame) == rows

@pytest.mark.parametrize("quartz,day_of_week", [
    ("0 0 5 ? * SUN *", "sun"),
    ("0 0 5 ? * 1 *", "sun"),      # Quartz 1 is Sunday; APScheduler 1 is Tuesday
    ("0 0 5 ? * 2 *", "mon"),
    ("0 0 5 ? * 7 *", "sat"),
    ("0 0 5 ? * 2-6 *", "mon-fri"),
    ("0 0 5 ? * MON,WED *", "mon,wed"),
    ("0 0 5 ? * * *", "*"),
])
def test_quartz_day_numbers_are_translated_not_passed_through(quartz, day_of_week):
    """Quartz numbers days 1=SUN..7=SAT; APScheduler numbers 0=MON..6=SUN.

    A number passed through unchanged moves the schedule two days with no error, so a
    weekly retrain written as '1' for Sunday would have run on Tuesday. Both systems
    read the same three-letter names, so numbers become names.
    """
    from bl_ranking.ops.schedule import parse_quartz

    assert parse_quartz(quartz).day_of_week == day_of_week


def test_an_impossible_quartz_day_is_rejected():
    from bl_ranking.ops.schedule import parse_quartz

    with pytest.raises(ValueError, match="outside 1-7"):
        parse_quartz("0 0 5 ? * 9 *")

@pytest.mark.parametrize("mutate", [
    lambda m: m.update({"a_field_from_a_newer_version": "x"}),
    lambda m: m.pop("git_sha", None),
])
def test_a_manifest_from_a_different_version_still_loads(bundle, settings, tmp_path, mutate):
    """A rolling deploy has old replicas reading bundles written by the new pipeline.

    Passing the JSON straight into the constructor made one added field a TypeError
    that bricked the worker, so shipping a new manifest key would have taken down the
    fleet it was meant to roll through. Missing keys are the same problem from the
    other side, when an old bundle is read after a rollback.
    """
    import json
    import shutil

    from bl_ranking.models import bundle as bundle_files
    from bl_ranking.serving.ranker import BrandRanker

    other = tmp_path / "bundle"
    shutil.copytree(bundle, other)
    path = other / bundle_files.MANIFEST_FILE
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))

    assert BrandRanker.load(other, settings) is not None

@pytest.mark.parametrize("asked,at_least,at_most", [(500, 400, 600), (5000, 4500, 5100)])
def test_the_distillation_sample_size_setting_can_lower_the_row_count(asked, at_least, at_most):
    """It was written as max(setting, len(context)), so it could only raise the count.

    Every value someone would pick to make a run faster sits at or below the context
    size, which is exactly the range the floor swallowed - so the knob appeared to do
    nothing at all in the only direction anyone turns it.
    """
    import numpy as np
    import pandas as pd

    from bl_ranking.config import PayoutSettings
    from bl_ranking.models.payout import SurrogateBackend

    context = pd.DataFrame({
        "client_name": [f"b{i % 15}" for i in range(1000)],
        "feature": np.arange(1000),
    })
    backend = SurrogateBackend(PayoutSettings(surrogate_sample_rows=asked))
    sample, users, brands = backend._distillation_sample(context)

    assert at_least <= len(sample) <= at_most
    assert brands == 15
    assert users >= 1, "every brand must still get at least one row"

def test_a_payout_sidecar_that_disagrees_with_its_model_is_refused():
    """Two files decide the payout half, and only their agreement makes it correct.

    The sidecar list is what the backend slices the frame by, so it decides what gets
    fed. The model's own recorded names are what it was fitted on, so they decide what
    should have been fed. Checking only one means the guard confirms a file against
    itself and passes while the model scores the wrong columns.

    Exercised directly rather than by damaging a bundle: only the surrogate keeps a
    sidecar, so a disk-level test skips under the offline backend CI runs - leaving the
    check unprotected in exactly the configuration that is tested, while the summary
    line still reads green.
    """
    from bl_ranking.models.payout import _fitted_feature_names

    class Model:
        def __init__(self, names):
            self.feature_names_ = names

    columns = ["a", "b", "c"]

    # Agreeing sources: the persisted list is what gets sliced, so it is returned.
    assert _fitted_feature_names(Model(list(columns)), list(columns)) == columns

    # Only one source available: use whichever exists.
    assert _fitted_feature_names(Model(None), list(columns)) == columns
    assert _fitted_feature_names(Model(list(columns)), []) == columns
    assert _fitted_feature_names(Model(None), []) is None

    # Disagreement is a bundle assembled from two runs, whichever way it differs.
    with pytest.raises(ValueError, match="inconsistent"):
        _fitted_feature_names(Model(["a", "c", "b"]), columns)
    with pytest.raises(ValueError, match="inconsistent"):
        _fitted_feature_names(Model(["a", "b"]), columns)


@pytest.mark.parametrize("timestamps,expected", [
    (["2026-01-06 19:24:22"] * 3, 0),
    (["2026-01-06 19:24:22", "06/01/2026 19:24", "2026-01-06 19:24:22"], 1),
])
def test_rows_needing_a_fallback_timestamp_parse_are_counted(timestamps, expected):
    """A row whose date format differs from its column is parsed by inference.

    '06/01/2026' becomes June 1st or January 6th depending on what pandas decides, and
    session_day and session_day_of_week are model features - so a misread date is a
    silently wrong feature rather than an error. Counted like every other repair, a
    funnel changing its date format shows up here before it shows up in the model.
    """
    from bl_ranking.data.ingest import sanitise

    cleaned, repairs = sanitise(_gate_frame(rows=3, session_dt=timestamps))
    assert repairs["timestamp_format_fallbacks"] == expected
    assert len(cleaned) == 3, "no row should be dropped for a format difference alone"



# --------------------------------------------------------------------------------- #
# The gate, second pass: what a real export does that a clean one does not
# --------------------------------------------------------------------------------- #

@pytest.mark.parametrize(("label", "values"), [
    # A DST transition in an export: the same column carries two different offsets.
    ("two offsets after a naive value",
     ["2026-01-15 10:00:00", "2026-03-08 01:30:00-05:00", "2026-03-08 03:30:00-04:00"]),
    # Two funnels writing the same column in two layouts.
    ("two layouts", ["2026-01-15 10:00:00", "15/01/2026 10:00", "2026-01-16 11:00:00"]),
])
def test_a_heterogeneous_timestamp_column_loses_no_row(label, values):
    """Neither pandas pass is correct alone, and each failed in its own direction.

    Without `utc=True` the retry returns an object column of mixed-offset datetimes and
    `.dt` raises on it - an AttributeError from inside the gate. Without
    `format="mixed"` the retry infers one layout for the subset and NaTs the rest, and
    the row filter then drops rows that parse perfectly on their own and that serving
    accepts.
    """
    from bl_ranking.data.ingest import _to_utc_naive

    parsed, fallbacks = _to_utc_naive(pd.Series(values, dtype="object"))
    assert int(parsed.isna().sum()) == 0, label
    assert fallbacks > 0, "a row that needed the second pass must be counted"


def test_a_numeric_survey_answer_is_nulled_not_stringified():
    """`.str.lower()` yields NaN for a non-string element and the research code then
    fills 'other', which is exactly what serving's mirror returns for one. Stringifying
    would put '12' in training against 'other' at serve time."""
    from bl_ranking.data.ingest import sanitise

    clean, repairs = sanitise(_gate_frame(rows=3, industry=["retail", 12, "retail"]))
    assert list(clean["industry"]) == ["retail", None, "retail"]
    assert repairs["survey_nonstring"] == 1
    clean["industry"].str.lower()          # the invariant itself, unassisted


def test_a_survey_column_with_no_text_at_all_is_refused():
    """It cannot be repaired: its nulls become empty cells in the staged CSV, pandas
    reads those back as float64, and `.str.lower()` raises on float64 - inside the
    vendored code, twenty minutes into the run."""
    from bl_ranking.data.ingest import sanitise

    with pytest.raises(ValueError, match="industry"):
        sanitise(_gate_frame(rows=3, industry=[None, None, None]))


def test_a_repeated_required_header_is_refused():
    """pandas renames the second copy to `payout.1`, so the missing-column check waves
    it through and the gate reads whichever copy came first. When that is the empty one
    the whole regression label becomes null with every repair counter reporting zero."""
    from bl_ranking.data.ingest import REQUIRED_COLUMNS, _require_columns

    frame = pd.DataFrame({c: ["x"] for c in REQUIRED_COLUMNS})
    frame["payout.1"] = ["42.5"]
    with pytest.raises(ValueError, match="repeats columns"):
        _require_columns(frame)


def test_an_id_with_a_leading_zero_survives_the_staged_csv(tmp_path):
    """The staged CSV is where the research code actually reads, and pandas re-infers
    dtypes there. A column of pure digits comes back as int64 whatever we write, so the
    ids themselves have to be canonical - and serving must use the same rule."""
    from bl_ranking.data.ingest import (
        RESEARCH_NULL_CATEGORY,
        _as_identifier,
        sanitise,
        stage_for_research_code,
    )

    clean, _ = sanitise(_gate_frame(rows=3, sub1=["007", "7448788", "12"],
                                    sub2=["a"] * 3, sub3=["b"] * 3))
    stage_dir, filename = stage_for_research_code(clean, tmp_path / "input")
    reread = pd.read_csv(stage_dir / filename)
    research_levels = list(reread["sub1"].fillna(RESEARCH_NULL_CATEGORY).astype(str))

    assert research_levels == ["7", "7448788", "12"]
    # The request boundary imports the same rule, so the two sides cannot drift.
    assert [_as_identifier(v) for v in ("007", 7448788, "12")] == research_levels


@pytest.mark.parametrize("ids", [
    # A fractional id is left exactly as it came - truncating '7.5' to '7' would silently
    # change it - and pandas then reads the column as float64, so its neighbour '12' comes
    # back as '12.0'. A level no request can ever match.
    ["7.5", "12"],
    # The literal text 'nan' in a CSV cell reads back as a null, which the research code's
    # own fillna turns into 'Other'.
    ["nan", "12"],
])
def test_an_id_the_round_trip_would_change_stops_the_run(tmp_path, ids):
    """Canonicalising covers the ids that occur; the check covers the ones that do not.
    Two rules are cheaper to maintain than one rule that has to be exhaustive."""
    from bl_ranking.data.ingest import sanitise, stage_for_research_code

    clean, _ = sanitise(_gate_frame(rows=2, sub1=ids))
    with pytest.raises(ValueError, match="round trip"):
        stage_for_research_code(clean, tmp_path / "input")


@pytest.mark.parametrize(("spelling", "canonical"), [
    ("007", "7"), ("7.0", "7"),
    # These two survived the gate untouched and then stopped the weekly retrain at the
    # round-trip check, because only a single trailing '.0' was handled. A SQL DECIMAL or
    # float column exports both.
    ("7448788.00", "7448788"), ("1e3", "1000"),
    ("1.2e17", "120000000000000000"),
    # Not a whole number, and not changed: truncating an id is worse than refusing it.
    ("7.5", "7.5"),
    # Numeric-looking but not numbers.
    ("nan", "nan"), ("Inf", "Inf"), ("abc007", "abc007"),
    # Past int64, so exact only if it is never parsed as a number.
    ("12345678901234567890", "12345678901234567890"),
])
def test_every_numeric_id_spelling_lands_on_one_level(spelling, canonical):
    from bl_ranking.data.ingest import _as_identifier

    assert _as_identifier(spelling) == canonical


def test_the_ingest_report_reaches_the_training_run(tmp_path):
    """Ingestion and training are separate jobs, so the repair counters have to travel
    with the Delta version or the run cannot report the quality of the rows it trained
    on. `as_params` had no caller at all before this.

    Carried in the commit rather than in a file beside the table: atomic with the version
    it describes, and present on object storage and Unity Catalog, where a local sibling
    directory would not exist at all.
    """
    from bl_ranking.data.delta import write_snapshot
    from bl_ranking.data.ingest import IngestReport, read_report

    table = tmp_path / "delta" / "bl_sessions"
    for index in range(2):
        report = IngestReport(source="/x/bl_full_data.csv", rows_in=10 + index,
                              rows_out=9, repairs={"cellphone": 2 + index})
        version = write_snapshot(pd.DataFrame({"a": [index]}), table,
                                 commit_metadata=report.as_commit_metadata())

    loaded = read_report(table, version)
    assert loaded is not None
    assert loaded.as_params()["ingest.repaired.cellphone"] == 3
    assert loaded.as_params()["ingest.rows_in"] == 11
    assert loaded.as_params()["ingest.delta_version"] == version
    # An older version keeps its own counters, not the newest ones.
    assert read_report(table, 0).as_params()["ingest.repaired.cellphone"] == 2
    # A version written without the gate carries none, which is an ordinary answer.
    plain = write_snapshot(pd.DataFrame({"a": [9]}), table)
    assert read_report(table, plain) is None


def test_a_brand_universe_of_only_the_sentinel_is_refused(tmp_path):
    """'other' is the research fill for a missing client_name, not a lender. n_brands
    and _brands both drop it, so counting it in the guard let a bundle with nothing to
    rank report /readyz green and answer every request 200 with an empty ranking."""
    from bl_ranking.serving.ranker import _assert_usable_brand_universe

    with pytest.raises(ValueError, match="empty brand universe"):
        _assert_usable_brand_universe(
            pd.DataFrame({"client_name": ["other"]}), tmp_path)
    # A real universe alongside the sentinel is fine.
    _assert_usable_brand_universe(
        pd.DataFrame({"client_name": ["acme", "other"]}), tmp_path)


def test_the_smallest_distillation_sample_still_fits():
    """`fit` holds out max(1, 20% of users) whole users, so one user per brand leaves
    nothing to fit on and CatBoost raises on an empty label vector - twenty minutes
    into the weekly run, for a value validate() accepted."""
    import numpy as np

    from bl_ranking.models.payout import create_backend

    cfg = Settings.load().model.payout
    cfg.backend, cfg.teacher, cfg.surrogate_sample_rows = "surrogate", "catboost_fallback", 1
    rng = np.random.RandomState(0)
    rows, brands = 40, [f"b{i}" for i in range(15)]
    x = pd.DataFrame({
        "client_name": rng.choice(brands, rows),
        "campaign_id": rng.randint(1, 5, rows).astype("int64"),
        "page": rng.choice(["p1", "p2"], rows),
        "from_start_to_register": rng.rand(rows) * 100,
    })
    fitted = create_backend(cfg, "surrogate").fit(x, pd.Series(rng.rand(rows) * 50))
    assert fitted.fidelity["surrogate_holdout_users"] >= 1


@pytest.mark.parametrize("start", range(1, 8))
@pytest.mark.parametrize("end", range(1, 8))
def test_every_quartz_day_range_fires_on_the_days_quartz_means(start, end):
    """A Quartz range runs in Quartz's week, which starts on Sunday; APScheduler's ends
    there. So the edges cannot be translated one at a time - '6-2' became 'fri-mon',
    which APScheduler refuses, and because Sunday is 1 in Quartz, *every* range starting
    on Sunday wrapped: '1-5', the ordinary weekday range, took the local runner down on
    start-up while Databricks accepted the same expression.

    Checked against what the trigger actually fires on, not against a rendered string.
    """
    from datetime import datetime, timedelta

    from apscheduler.triggers.cron import CronTrigger

    from bl_ranking.ops.schedule import parse_quartz

    # Quartz: 1=SUN..7=SAT, and a range wraps the week when start > end.
    quartz_days = ([start] if start == end
                   else list(range(start, end + 1)) if start < end
                   else list(range(start, 8)) + list(range(1, end + 1)))
    # Python's weekday(): 0=MON..6=SUN. Quartz 1 (SUN) is 6.
    expected = {(day + 5) % 7 for day in quartz_days}

    trigger = CronTrigger(timezone=UTC,
                          **parse_quartz(f"0 0 5 ? * {start}-{end} *").as_apscheduler_kwargs())
    fired, moment = set(), datetime(2026, 1, 1, tzinfo=UTC)
    for _ in range(8):
        # `None` as the previous fire time returns the next fire at or after `moment`,
        # so step past each hit or the loop sits on the same day forever.
        moment = trigger.get_next_fire_time(None, moment)
        fired.add(moment.weekday())
        moment = moment + timedelta(seconds=1)
    assert fired == expected


def test_an_unpromoted_registry_is_not_an_unreachable_one():
    """The refusal to fall back locally exists to protect an operator's rollback. An
    alias that was never set records no decision, so refusing there only made the
    documented local stack unstartable: the API would not serve the bundle in runs/ and
    so could never become ready for the first `make train-prod` to reach it.

    Classified from the failure MLflow already raised rather than by asking again: an
    unreachable registry retries with backoff, so a second question would double how
    long a worker takes to report a refusal it is going to report anyway.
    """
    from mlflow.exceptions import MlflowException

    # error_code is the protobuf enum, not the name the property reads back.
    from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE

    from bl_ranking.serving.model_source import _alias_is_unset

    answered = MlflowException("Registered model alias champion not found.",
                               error_code=INVALID_PARAMETER_VALUE)
    unreachable = MlflowException("API request failed", error_code=INTERNAL_ERROR)

    assert _alias_is_unset(answered) is True
    assert _alias_is_unset(unreachable) is False
    # Anything unrecognised falls through to the refusal: a 503 an operator can explain
    # costs less than silently serving the version they rolled back from.
    assert _alias_is_unset(RuntimeError("something else")) is False


def test_a_rollback_run_twice_keeps_the_way_back(tmp_path):
    """`champion_previous` is the documented way to undo a rollback. The guard that
    stops it pointing at the champion itself compared the caller's spelling of the
    version, not the version the registry resolved - so `make rollback VERSION=02`
    destroyed the only pointer back."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from bl_ranking.ops import registry

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    # MLflow's active experiment is process-global, so without this the runs below are
    # created against whichever experiment id an earlier test left behind.
    mlflow.set_experiment("rollback_guard")
    settings = Settings.load()
    settings.mlflow.tracking_uri = uri
    settings.mlflow.registered_model = "bl_rank_test"
    settings.mlflow.serving_alias = "champion"

    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_registered_model("bl_rank_test")
    for index in range(2):
        with mlflow.start_run() as run:
            pass
        client.create_model_version("bl_rank_test", source=f"file://{tmp_path}/m{index}",
                                    run_id=run.info.run_id)

    def alias(name):
        try:
            # str(): MLflow returns .version as an int from some stores and a string
            # from others, which is the very confusion the guard in set_alias exists for.
            return str(client.get_model_version_by_alias("bl_rank_test", name).version)
        except Exception:      # noqa: BLE001 - "not set yet" is a normal state
            return None

    registry.set_alias(settings, "1")
    registry.set_alias(settings, "2")
    assert (alias("champion"), alias("champion_previous")) == ("2", "1")

    registry.set_alias(settings, "1")                  # rollback
    assert (alias("champion"), alias("champion_previous")) == ("1", "2")
    registry.set_alias(settings, "01")                 # the same rollback, padded
    assert (alias("champion"), alias("champion_previous")) == ("1", "2")


@pytest.mark.parametrize("body", ["[1, 2, 3]", "null", '{"payout_backend"'])
def test_a_manifest_that_is_not_an_object_names_the_file(tmp_path, body):
    """Tolerating a manifest from a newer pipeline is not the same as tolerating a
    broken one. A truncated or half-written file took the worker down with "'NoneType'
    object is not iterable", naming neither the bundle nor the file."""
    (tmp_path / bundle_files.MANIFEST_FILE).write_text(body)
    with pytest.raises(ValueError, match=bundle_files.MANIFEST_FILE):
        bundle_files.Manifest.read(tmp_path)


def test_the_registered_signature_accepts_what_the_endpoint_accepts():
    """The signature was inferred from one example row, which made every field required
    and the four id-ish ones `long`. Both are narrower than the HTTP contract, and each
    broke a whole batch rather than a row: one null cellphone - an optional field -
    demotes its column to float64 and enforcement refuses the cast, and a 17-digit
    campaign_id cannot cross a columnar boundary as a number at all without becoming
    ...304, which is the same float64 demotion the ingestion gate reads these columns as
    text to avoid."""
    import pandas as pd
    from mlflow.models.utils import _enforce_schema

    from bl_ranking.serving.pyfunc import build_signature, request_example

    schema = build_signature().inputs
    example = request_example()

    _enforce_schema(example, schema)                                  # the example itself
    _enforce_schema(pd.concat([example, example.assign(cellphone=None)],
                              ignore_index=True), schema)             # one null optional
    _enforce_schema(example.drop(columns=["fname", "lname", "conversion_dt"]), schema)

    enforced = _enforce_schema(
        example.assign(campaign_id="120227360861540306"), schema)
    assert enforced["campaign_id"][0] == "120227360861540306"          # every digit


def test_a_batch_neighbour_cannot_change_a_row(bundle, settings):
    """The point of validating each record on its own. A null anywhere in a column makes
    pandas type the whole column around it, so a row's features used to depend on which
    other rows shared its batch - with no error anywhere."""
    import pandas as pd
    from mlflow.models.utils import _enforce_schema

    from bl_ranking.serving.pyfunc import (
        BrandRankerModel,
        build_signature,
        request_example,
    )
    from bl_ranking.serving.ranker import BrandRanker

    model = BrandRankerModel()
    model._ranker = BrandRanker.load(bundle, settings)
    schema = build_signature().inputs
    example = request_example().assign(campaign_id="120227360861540306")

    alone = model.predict(None, _enforce_schema(example, schema))["ranking"][0]
    neighboured = pd.concat(
        [example.assign(cellphone=None, sub1=None), example], ignore_index=True)
    in_batch = model.predict(None, _enforce_schema(neighboured, schema))["ranking"][1]

    assert alone == in_batch


def test_a_numeric_timestamp_column_is_refused():
    """pandas reads a number in a date column as nanoseconds since 1970, so a funnel
    switching to a YYYYMMDD integer date turns every row into 1970-01-01 - one value for
    the whole column, no row dropped, no repair counted. session_day and
    session_day_of_week are model features, so that is a silently constant feature rather
    than an error. 20260115 as a date and 20260115 as a nanosecond count cannot be told
    apart from inside the gate, so it says so instead of choosing."""
    import numpy as np

    from bl_ranking.data.ingest import sanitise

    with pytest.raises(ValueError, match="session_dt arrived as a numeric column"):
        sanitise(_gate_frame(rows=3, session_dt=[20260115, 20260116, 20260117]))

    # An all-empty column arrives as float64 too, and is not the same thing: no values.
    clean, _ = sanitise(_gate_frame(
        rows=3, register_date=pd.Series([np.nan] * 3, dtype="float64")))
    assert clean["register_date"].isna().all()


def test_a_blank_attribution_id_is_an_absent_one(tmp_path):
    """An empty tracking parameter reaches the CSV as an empty cell, which pandas reads as
    null and the research code's `fillna('Other')` turns into 'Other' - and serving's
    mirror does the same with None. Returning '' put an empty level in training against
    'Other' at serve time, and once the staged round trip was checked it stopped the
    weekly run outright over a blank sub parameter, which is ordinary attribution data."""
    from bl_ranking.data.ingest import (
        RESEARCH_NULL_CATEGORY,
        _as_identifier,
        sanitise,
        stage_for_research_code,
    )
    from bl_ranking.serving.schemas import RankRequest

    assert _as_identifier("") is None
    assert _as_identifier("   ") is None

    clean, _ = sanitise(_gate_frame(rows=3, sub1=["", "7448788", "  "]))
    stage_dir, filename = stage_for_research_code(clean, tmp_path / "input")
    reread = pd.read_csv(stage_dir / filename)
    assert list(reread["sub1"].fillna(RESEARCH_NULL_CATEGORY).astype(str)) == [
        "Other", "7448788", "Other"]

    # And the request boundary agrees, because it imports the same function.
    example = {"session_id": "s", "page": "p", "campaign_id": 1,
               "session_dt": "2026-01-06 19:24:22", "register_date": "2026-01-06 19:26:07",
               "credit_score": "550-599", "industry": "construction",
               "loan_amount": "a1", "loan_reason": "b1", "monthly_revenue": "c1",
               "time_in_business": "2+ years", "device_type": "mobile",
               "business_type": "llc"}
    assert RankRequest.model_validate({**example, "sub1": ""}).sub1 is None


@pytest.mark.parametrize("expression", ["0 0 5 L * ? *", "0 0 5 15W * ? *", "0 0 5 LW * ? *"])
def test_a_quartz_calendar_token_is_refused_here_not_by_apscheduler(expression):
    """Databricks accepts L, W and #; APScheduler cannot express them. Passed through it
    answered `Unrecognized expression "15W" for field "day"` when a worker started,
    naming neither the cron nor the setting - and the real cost is that the two
    schedulers would then disagree about when the weekly retrain runs."""
    from bl_ranking.ops.schedule import parse_quartz

    with pytest.raises(ValueError, match="calendar token"):
        parse_quartz(expression)


@pytest.mark.parametrize("uri", ["C:/mlruns", "C:\\mlruns", "d:/data/delta"])
def test_a_windows_drive_letter_is_a_directory_not_a_registry(settings, monkeypatch, uri):
    """A one-letter scheme is a drive, and reading it as a remote store told a developer
    on Windows that the registry was authoritative - refusing to serve their own bundle."""
    from bl_ranking.serving.model_source import _registry_is_authoritative

    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    assert _registry_is_authoritative(settings) is False


def test_the_run_reports_the_rows_it_actually_fitted(tmp_path):
    """The research code calls split_by_time in BOTH modes, so the fit never sees the last
    7 days of the window - production included. The manifest reported len(snapshot) as
    rows_train, which overstated it by about a tenth on the real extract, and the docs
    called production mode "fit on everything". The split is observed, not changed."""
    import numpy as np

    from bl_ranking.training.trainer import ProductionTrainer

    frame = pd.DataFrame({
        "session_dt": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(30), "D"),
        "payout": np.arange(30, dtype=float),
    })
    trainer = ProductionTrainer.__new__(ProductionTrainer)   # no research I/O needed
    trainer.split_counts = {}
    train, test = ProductionTrainer.split_by_time(trainer, frame.copy(), 7)

    assert trainer.split_counts == {
        "rows_preprocessed": 30, "rows_fitted": 23, "rows_held_out": 7,
        "days_for_test": 7,
    }
    assert len(train) == 23 and len(test) == 7
    # And the counts are the split, not the snapshot: the two differ by the held-out tail.
    assert trainer.split_counts["rows_fitted"] < trainer.split_counts["rows_preprocessed"]


def test_the_research_logger_takeover_is_actually_undone(tmp_path, monkeypatch):
    """preserve_root_logging has to be entered before the trainer is *constructed*:
    setup_bl_logger runs in BLPayoutModelsFit.__init__. Entered after, it saved the
    already-cleared handler list and restored that - so the scheduler went permanently
    mute after its first retrain, including the line naming the version it registered."""
    import logging

    from bl_ranking.training.trainer import preserve_root_logging

    root = logging.getLogger()
    saved = list(root.handlers)
    marker = logging.NullHandler()
    root.addHandler(marker)
    try:
        with preserve_root_logging():
            # What setup_bl_logger does, inside the guard as the job now arranges.
            root.handlers.clear()
            root.addHandler(logging.NullHandler())
        assert marker in root.handlers, "the process's own handler must come back"
    finally:
        root.handlers = saved


def test_a_payout_column_with_no_numbers_is_refused():
    """It cannot be repaired into a usable one and it fails a long way from the gate: every
    repair counter reports zero, _typed_for_delta types the all-null column as *string* so
    the table's schema silently changes, and the run dies half an hour later inside
    CatBoost with "Labels variable is empty" - naming neither the column nor the extract."""
    from bl_ranking.data.ingest import sanitise

    with pytest.raises(ValueError, match="payout arrived with no numeric values"):
        sanitise(_gate_frame(rows=3, payout=[None, None, None]))

    # All-zero is a real answer, not a missing column: nobody was paid that day.
    clean, _ = sanitise(_gate_frame(rows=3, payout=[0.0, 0.0, 0.0]))
    assert pd.api.types.is_numeric_dtype(clean["payout"])


def test_run_directories_are_stamped_in_utc_and_never_reused(tmp_path):
    """The name is also an ordering - serving's offline fallback picks the newest
    runs/*/bundle by sorting these. A DST fall-back repeats an hour, so two retrains could
    land on the same second-resolution local name, and exist_ok=True wrote the second run's
    artifacts into the first's directory."""
    from bl_ranking.training.job import _new_run_dir

    settings = Settings.load()
    settings.paths.run_root = str(tmp_path)

    first = _new_run_dir(settings, "production")
    second = _new_run_dir(settings, "production")

    assert first != second
    assert first.name.split("_")[2].endswith("Z"), first.name
    # Sorting the names has to order them by time, which is what the fallback relied on -
    # and is exactly what a '-1' collision suffix broke, because '-' precedes '_'.
    assert sorted([first.name, second.name]) == [first.name, second.name]


def test_a_delta_version_that_does_not_exist_says_which_ones_do(tmp_path):
    """delta-rs answers a missing version with a generic error, and a negative one by
    saying the table was not found - for a table that plainly is."""
    from bl_ranking.data.delta import read_snapshot, write_snapshot

    table = tmp_path / "delta" / "t"
    write_snapshot(pd.DataFrame({"session_dt": ["2026-01-01 10:00:00"]}), table)
    with pytest.raises(ValueError, match="has no version 7; versions 0..0"):
        read_snapshot(table, version=7)


def test_the_local_fallback_picks_the_newest_bundle_by_its_manifest(tmp_path):
    """Ordered by each bundle's own trained_at, not by directory name. The names are UTC
    now and sort correctly, but only for bundles this version produced - an earlier one
    stamped them in local time, so a container whose TZ moved backwards made the newest
    bundle stop being the last name alphabetically."""
    from bl_ranking.serving.model_source import _latest_local_bundle

    settings = Settings.load()
    settings.paths.run_root = str(tmp_path)
    # Deliberately named so that the *older* run sorts last, as a backwards TZ change did.
    for name, trained_at in (("20260301_010000_production", "2026-03-01T09:00:00Z"),
                             ("20260301_020000_production", "2026-03-01T08:00:00Z")):
        bundle = tmp_path / name / "bundle"
        bundle.mkdir(parents=True)
        bundle_files.Manifest(trained_at=trained_at).write(bundle)

    chosen = _latest_local_bundle(settings)
    assert chosen.parent.name == "20260301_010000_production"


@pytest.mark.parametrize("header", ["Payout", "payout "])
def test_a_header_that_only_looks_different_is_refused(header):
    """pandas keeps 'payout' and 'Payout' as two distinct columns, so neither the
    missing-column check nor the mangled-duplicate check sees anything wrong - and the gate
    then reads whichever copy is spelled exactly right, which in a warehouse view exporting
    both a snake_case and a display-cased column is as likely as not the empty one.
    Trailing whitespace in a header is something CSV exporters do routinely."""
    from bl_ranking.data.ingest import REQUIRED_COLUMNS, _require_columns

    frame = pd.DataFrame({c: ["x"] for c in REQUIRED_COLUMNS})
    frame[header] = ["42.5"]
    with pytest.raises(ValueError, match="case or whitespace"):
        _require_columns(frame)


@pytest.mark.parametrize(("names", "fragment"), [
    # Both differ from the sentinel only in spelling, and nothing downstream treats them
    # as the sentinel - so each would be offered to the funnel as a lender by that name.
    (["acme", "Other"], "case or whitespace"),
    (["acme", "other "], "case or whitespace"),
    (["other"], "empty brand universe"),
])
def test_a_brand_that_is_the_sentinel_in_all_but_spelling_is_refused(tmp_path, names, fragment):
    from bl_ranking.serving.ranker import _assert_usable_brand_universe

    with pytest.raises(ValueError, match=fragment):
        _assert_usable_brand_universe(pd.DataFrame({"client_name": names}), tmp_path)

    # And a real universe is still fine, sentinel included.
    _assert_usable_brand_universe(
        pd.DataFrame({"client_name": ["acme", "beta", "other"]}), tmp_path)


def test_a_name_column_the_research_code_cannot_read_is_refused():
    """additional_features calls `.str.len()` on fname and lname (bl_models_train.py lines
    209-210), and the gate had no invariant for either - so a numeric or fully redacted name
    column passed with every counter at zero and killed the run inside the vendored code."""
    from bl_ranking.data.ingest import sanitise

    clean, repairs = sanitise(_gate_frame(rows=3, fname=["John", 7, "Ann"],
                                          lname=["Smith"] * 3))
    assert repairs["name_nonstring"] == 1
    clean["fname"].str.len()                       # the invariant itself, unassisted

    with pytest.raises(ValueError, match="fname"):
        sanitise(_gate_frame(rows=3, fname=[None] * 3, lname=["Smith"] * 3))


def test_a_text_invariant_is_checked_on_the_rows_that_survive():
    """Evaluated before the no-session_dt drop, a column whose only text sat on rows the
    gate was about to discard satisfied it - and the research code died on .str.lower()
    anyway."""
    from bl_ranking.data.ingest import sanitise

    with pytest.raises(ValueError, match="industry"):
        sanitise(_gate_frame(
            rows=4,
            industry=[None, "retail", "retail", "retail"],
            session_dt=["2026-01-01 10:00:00", "nope", "nope", "nope"]))


@pytest.mark.parametrize(("dates", "ambiguous"), [
    (["2026-01-06 19:24:22", "2026-01-07 10:00:00"], 0),      # ISO: never ambiguous
    (["06/01/2026 08:00:00", "07/01/2026 08:00:00"], 2),      # could be either way round
    (["13/01/2026", "25/01/2026"], 0),                        # 13 cannot be a month
    (["06/06/2026"], 0),                                      # reads the same both ways
    (["2026/01/06"], 0),                                      # four-digit year first
])
def test_an_ambiguous_date_layout_is_counted(dates, ambiguous):
    """timestamp_format_fallbacks counts a row whose layout differs from its column's,
    which is the case pandas notices. It reports 0 for the documented failure it was named
    for - a whole column written DD/MM/YYYY, which inference reads as MM/DD with nothing
    looking unusual. This counts the rows where the reading is a coin flip, so the
    ambiguity is visible in the run's parameters instead of resolved silently."""
    from bl_ranking.data.ingest import _count_ambiguous_dates

    assert _count_ambiguous_dates(pd.Series(dates, dtype="object")) == ambiguous


def test_the_gate_reports_both_timestamp_counters():
    from bl_ranking.data.ingest import sanitise

    _, repairs = sanitise(_gate_frame(
        rows=2, session_dt=["06/01/2026 08:00:00", "07/01/2026 08:00:00"]))
    assert repairs["timestamp_ambiguous_layout"] == 2
    assert repairs["timestamp_format_fallbacks"] == 0      # the column is self-consistent


# --------------------------------------------------------------------------------- #
# The Databricks bundle: every value it duplicates from the repository
# --------------------------------------------------------------------------------- #

def _bundle() -> dict:
    import yaml

    path = Path(__file__).resolve().parents[1] / "databricks.yml"
    assert path.exists(), "databricks.yml must be at the repository root: a bundle's sync " \
                          "root is the directory holding it, and conf/ and dist/ have to " \
                          "be inside it"
    return yaml.safe_load(path.read_text())


def _shipped_config() -> dict:
    """conf/config.yaml as written, not as the environment overrides it.

    What the bundle duplicates is the *file*. `Settings.load()` here would merge this test
    session's own BL_ variables (conftest pins the payout backend for CI), so comparing
    against it would compare the bundle to the test harness.
    """
    import yaml

    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "conf" / "config.yaml").read_text())


def test_the_bundle_and_the_config_schedule_at_the_same_moment():
    """A bundle cannot read conf/config.yaml, so the Quartz string is copied into it by
    hand. The README claimed the two were one definition read twice; they are two, and the
    only way to keep that claim true is to fail here when either moves."""
    job = _bundle()["resources"]["jobs"]["bl_weekly_training"]
    schedule = _shipped_config()["schedule"]
    assert job["schedule"]["quartz_cron_expression"] == schedule["cron"]
    assert job["schedule"]["timezone_id"] == schedule["timezone"]


def test_the_bundle_installs_the_extras_the_tasks_import():
    """A `whl:` library spec installs the wheel's base dependencies and has no way to ask
    for an extra, so every task of the weekly job died at `import mlflow`. The extras are
    listed as explicit pinned pypi libraries instead, which only works while the two lists
    agree."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"]
    expected = set(extras["train"]) | set(extras["tabpfn"])

    for task in _bundle()["resources"]["jobs"]["bl_weekly_training"]["tasks"]:
        libraries = task["libraries"]
        assert {"whl": "./dist/*.whl"} in libraries, task["task_key"]
        installed = {entry["pypi"]["package"] for entry in libraries if "pypi" in entry}
        assert installed == expected, task["task_key"]


def test_the_bundle_points_delta_at_a_path_not_a_table_name():
    """delta-rs takes a filesystem path. A three-part Unity Catalog name produced a local
    directory of that name on the driver's ephemeral disk - 81k rows ingested, the real
    table untouched - so data/delta.py now refuses one, and the bundle must not set one."""
    from bl_ranking.data.delta import _reject_unity_catalog_name

    env = (_bundle()["resources"]["jobs"]["bl_weekly_training"]["job_clusters"][0]
           ["new_cluster"]["spark_env_vars"])
    _reject_unity_catalog_name(env["BL_PATHS__DELTA_TABLE"])       # raises if it is a name
    assert env["BL_PATHS__DELTA_TABLE"].startswith("/Volumes/")
    # And the ingest task needs somewhere to read from.
    assert env["BL_PATHS__RAW_DIR"].startswith("/Volumes/")
    # Both URIs, not just the registry: without the tracking URI every run logged to the
    # driver's ephemeral disk and vanished with the cluster.
    assert env["MLFLOW_TRACKING_URI"] == "databricks"
    assert env["MLFLOW_REGISTRY_URI"] == "databricks-uc"


def test_the_bundle_default_backend_matches_the_config():
    """It said tabpfn_local, which contradicted conf/config.yaml and the README's own
    latency argument - and needed a torch extra the library spec could not install."""
    assert (_bundle()["variables"]["payout_backend"]["default"]
            == _shipped_config()["model"]["payout"]["backend"])


def test_the_bundle_declares_a_real_single_node_cluster():
    """num_workers: 0 on its own is what the Databricks CLI's validator calls a
    misconfigured single-node cluster."""
    cluster = (_bundle()["resources"]["jobs"]["bl_weekly_training"]["job_clusters"][0]
               ["new_cluster"])
    assert cluster["num_workers"] == 0
    assert cluster["spark_conf"]["spark.databricks.cluster.profile"] == "singleNode"
    assert cluster["custom_tags"]["ResourceClass"] == "SingleNode"


def test_the_bundle_can_build_its_own_artifact():
    """databricks.yml declares `python -m build --wheel` as how the wheel is produced, and
    nothing installed the tool that runs it - so a first `databricks bundle deploy` failed
    on a missing artifact with no hint about what to install."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    dev = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"]["dev"]
    command = _bundle()["artifacts"]["bl_ranking_wheel"]["build"]

    assert command == "python -m build --wheel"
    assert any(pin.startswith("build==") for pin in dev), dev
    # And the artifact path has to be inside the sync root, which is this file's directory.
    assert _bundle()["artifacts"]["bl_ranking_wheel"]["path"] == "."


def test_no_shipped_launcher_pins_the_serving_backend():
    """Serving reads the backend from the bundle it loaded unless an operator asks for
    another, which is what lets a rollback carry the backend its version was trained with.
    Every shipped way of starting the API used to pin it, so that tier was unreachable in
    practice - and the fix only holds while the API's variable stays separate from the one
    .env.example fills in for training."""
    import yaml

    root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((root / "docker" / "docker-compose.yml").read_text())
    api = compose["services"]["api"]["environment"]["BL_MODEL__PAYOUT__BACKEND"]

    # No default after ':-', so an operator who sets nothing sends an empty value, which
    # config.env_override_keys treats as unset.
    assert api.endswith(":-}"), api
    assert "BL_PAYOUT_BACKEND" not in api, (
        "the API must not share the training services' variable: .env.example gives that "
        "one a value, which would pin the serving backend again")

    makefile = (root / "Makefile").read_text()
    serve_target = makefile.split("\nserve:", 1)[1].split("\n\n", 1)[0]
    assert "origin BACKEND" in serve_target, (
        "make serve must pass BACKEND through only when it was asked for on the command "
        f"line, got: {serve_target!r}")
