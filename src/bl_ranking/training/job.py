"""The training job: one MLflow run per execution, in either of the research code's modes.

    train_test=True   time-based split, per-day evaluation, nothing registered.
                      This is the mode a researcher runs to decide whether a change is
                      an improvement, so the run carries the researcher log verbatim
                      *and* the same numbers as MLflow metrics.

    train_test=False  no evaluation, produce artifacts, register a new model version
                      and move the `champion` alias onto it. This is the mode the
                      weekly Sunday 05:00 schedule runs.

Not "fit on everything", which is what this said for a long time and is not what the
research code does: `bl_preprocessing` calls `split_by_time(bl_data, days_for_test=7)`
unconditionally and fits on `bl_train`, so the most recent 7 days of the window are held
back in *both* modes. Production mode differs from train_test only in that it does not
evaluate on them. Changing that would mean editing the given code, so instead the run
reports it: `split.rows_fitted`, `split.rows_held_out` and the manifest's `rows_train`
are what the models actually saw, where `data.rows` is the snapshot that was read.

Both modes log the same technical parameters, so any two runs are directly comparable
in the MLflow UI regardless of mode: the input data version, the row counts, the model
hyper-parameters, the payout backend, the git SHA, and a checksum of the research code
itself so it is provable that the given scripts were not edited.

Which is also why `model.catboost.*` and `data.days_for_test` are refused as overrides -
see assert_config_mirrors_research. They mirror values the research scripts hard-code, so
setting one changes what a run claims and not what it does, and two runs that differ only
in a logged parameter are the worst possible thing to put in a comparison view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from bl_ranking.config import REPO_ROOT, Settings, resolve
from bl_ranking.data.delta import read_snapshot
from bl_ranking.data.ingest import read_report, stage_for_research_code
from bl_ranking.models import bundle as bundle_files
from bl_ranking.models.gender_lut import ARTIFACT_NAME as GENDER_ARTIFACT
from bl_ranking.models.gender_lut import GenderLookup
from bl_ranking.ops import registry
from bl_ranking.serving.pyfunc import (
    BUNDLE_ARTIFACT_KEY,
    BrandRankerModel,
    build_signature,
    request_example,
)
from bl_ranking.training.trainer import ProductionTrainer, preserve_root_logging

RESEARCH_DIR = REPO_ROOT / "src" / "bl_ranking" / "research"


@dataclass
class TrainingResult:
    run_id: str
    mode: str
    metrics: dict[str, float]
    bundle_dir: Path | None = None
    model_version: str | None = None
    duration_s: float = 0.0


def run(train_test: bool, settings: Settings | None = None,
        backend: str | None = None, register: bool = True,
        delta_version: int | None = None) -> TrainingResult:
    settings = settings or Settings.load()
    assert_config_mirrors_research(settings)
    started = time.perf_counter()
    mode = "train_test" if train_test else "production"

    mlflow.set_tracking_uri(settings.mlflow.resolved_tracking_uri())
    mlflow.set_experiment(settings.mlflow.experiment)

    snapshot = read_snapshot(settings.paths.delta_table, version=delta_version,
                             lookback_days=settings.data.lookback_days)
    run_dir = _new_run_dir(settings, mode)

    with mlflow.start_run(run_name=f"{mode}-{run_dir.name}") as active:
        mlflow.set_tags(_tags(mode, settings, backend))
        mlflow.log_params(_params(settings, snapshot, backend, run_dir))

        # The research code reads a CSV at `input_path + input_file`. Staging the Delta
        # snapshot back to CSV leaves that call untouched and reproduces the same dtype
        # inference the research runs had.
        stage_dir, stage_file = stage_for_research_code(snapshot.frame, run_dir / "input")

        # The context manager wraps the *construction* as well as the fit, and that
        # ordering is the whole point. setup_bl_logger runs inside
        # BLPayoutModelsFit.__init__ and clears the root logger's handlers, so entering
        # afterwards saved the already-cleared list and restored that - preserving
        # nothing. A one-shot `make train-prod` exits anyway, but the scheduler is
        # long-lived: after its first retrain close_log() removed the one handler that
        # was left and every later line went nowhere, including the one naming the model
        # version it had just registered.
        with preserve_root_logging():
            trainer = ProductionTrainer(
                input_path=str(stage_dir) + "/",
                input_file=stage_file,
                output_predictors_path=str(run_dir) + "/",
                train_test=train_test,
                payout_cfg=settings.model.payout,
                backend_name=backend,
                on_event=lambda event, payload: mlflow.set_tags(
                    {f"{event}.{k}": str(v) for k, v in payload.items()}
                ),
            )
            try:
                trainer.fit_()
            finally:
                trainer.close_log()
                # In the finally, because a failed run is when this artifact is worth
                # most: it is the only record of how far the fit got, and it was being
                # left on a worker's disk that does not outlive the job.
                _log_researcher_log(trainer, run_dir)

        if trainer.metrics:
            mlflow.log_metrics(trainer.metrics)
        # What the research time split actually did, in both modes. `data.rows` above is
        # the snapshot; these are the rows the models saw, and the gap between them is
        # the held-out tail the research code removes whether or not it evaluates on it.
        for name, value in trainer.split_counts.items():
            mlflow.log_metric(f"split.{name}", value)
        mlflow.log_metric("train.duration_s", time.perf_counter() - started)

        result = TrainingResult(run_id=active.info.run_id, mode=mode,
                                metrics=dict(trainer.metrics))

        if not train_test:
            bundle_dir = _assemble_bundle(run_dir, trainer, snapshot, settings,
                                          active.info.run_id, backend)
            result.bundle_dir = bundle_dir
            if register:
                result.model_version = _register(bundle_dir, settings)

        result.duration_s = time.perf_counter() - started
        mlflow.log_metric("job.duration_s", result.duration_s)
        return result


# --------------------------------------------------------------------------------- #
# MLflow bookkeeping
# --------------------------------------------------------------------------------- #

def _tags(mode: str, settings: Settings, backend: str | None) -> dict[str, str]:
    return {
        "mode": mode,
        "payout_backend": backend or settings.model.payout.backend,
        "git_sha": bundle_files.current_git_sha(REPO_ROOT),
        "research_code_sha": research_code_sha(),
        "python": platform.python_version(),
        "host": platform.node(),
    }


def _params(settings: Settings, snapshot, backend: str | None,
            run_dir: Path) -> dict[str, Any]:
    """Everything needed to reproduce the run, flattened for the comparison view."""
    frame = snapshot.frame
    session_dt = pd.to_datetime(frame["session_dt"], errors="coerce")
    params: dict[str, Any] = {
        "data.delta_table": snapshot.table_uri,
        "data.delta_version": snapshot.version,
        "data.rows": len(frame),
        "data.sessions": frame["session_id"].nunique(),
        "data.window_start": str(session_dt.min()),
        "data.window_end": str(session_dt.max()),
        "data.lookback_days": settings.data.lookback_days,
        "run.dir": run_dir.name,
        "lib.pandas": pd.__version__,
        "lib.mlflow": mlflow.__version__,
        "lib.python": sys.version.split()[0],
    }
    params.update({f"cfg.{k}": v for k, v in settings.flat().items()})

    # The gate's repair counters, carried across from the ingest job that produced this
    # exact Delta version. This is what makes "a sudden jump in repairs is visible"
    # true: without it the counters exist only in the ingest process's stdout, and a
    # training run cannot say anything about the quality of the rows it trained on.
    # Absent for a snapshot written before this was added, or by something other than
    # the gate - in which case the run simply carries no ingest.* params.
    ingest_report = read_report(snapshot.table_uri, snapshot.version)
    if ingest_report is not None:
        params.update(ingest_report.as_params())
    if backend:
        params["cfg.model.payout.backend"] = backend
    try:
        import catboost
        params["lib.catboost"] = catboost.__version__
    except ImportError:  # pragma: no cover
        pass
    # MLflow rejects params longer than 500 characters.
    return {k: _clip(v) for k, v in params.items()}


def _clip(value: Any, limit: int = 480) -> Any:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _research_dir() -> Path:
    """Where the vendored scripts actually live, source tree or installed wheel.

    RESEARCH_DIR is derived from the repository root, which only exists when running
    from a checkout. The Databricks tasks run from a wheel in site-packages, so resolve
    through the imported package instead and fall back to the repo layout.
    """
    try:
        import bl_ranking.research as pkg
        return Path(pkg.__file__).parent
    except Exception:  # noqa: BLE001 - the repo layout is the fallback, not an error
        return RESEARCH_DIR


# Settings that exist only to *mirror* a value the research code hard-codes, so the run
# can log what it used. The research scripts are fixed, so nothing reads these back into
# the model - which makes an override a lie in the MLflow comparison view rather than a
# change to the model. Each entry: the dotted setting, and the literal to look for in the
# research source.
MIRRORED_SETTINGS: tuple[tuple[str, str], ...] = (
    ("model.catboost.random_seed", "random_seed={value}"),
    ("model.catboost.depth", "depth={value}"),
    ("model.catboost.n_estimators", "n_estimators={value}"),
    ("model.catboost.eval_metric", "eval_metric='{value}'"),
    ("model.catboost.task_type", "task_type='{value}'"),
    ("data.days_for_test", "days_for_test={value}"),
)


def assert_config_mirrors_research(settings: Settings) -> None:
    """Refuse a run whose config claims a hyper-parameter the research code will not use.

    `model.catboost.*` and `data.days_for_test` are not controls. The research scripts
    hard-code those values and the brief forbids editing them, so these settings exist so
    that a run can *report* what it used. They are logged as run params and copied into
    the bundle manifest.

    Which makes an override the worst kind of wrong: `BL_MODEL__CATBOOST__DEPTH=2` used to
    produce a run whose params said depth 2 while the shipped classifier had depth 8 -
    and the registered version's tags said so too. A researcher sweeping depth would have
    compared two identical models and drawn a conclusion from the noise between them.

    So the value is checked against the research source itself, before anything is logged.
    `model.payout.context_size` is deliberately absent: the trainer overrides
    `tabpfn_regression_payout` and really does apply it.
    """
    directory = _research_dir()
    try:
        source = "\n".join(
            (directory / name).read_text()
            for name in ("bl_models_train.py", "bl_exp_payout_predictor.py")
        )
    except OSError as exc:
        # Same reasoning as research_code_sha: this check is only worth anything if it read
        # the real scripts, so a missing one is refused rather than skipped.
        raise FileNotFoundError(
            f"cannot read the vendored research scripts under {directory}: {exc}. The "
            f"mirrored hyper-parameters are checked against them, so refusing rather than "
            f"reporting values nothing verified."
        ) from exc
    flat = settings.flat()
    wrong = []
    for key, template in MIRRORED_SETTINGS:
        value = flat.get(key)
        if value is None:
            continue
        if template.format(value=value) not in source:
            wrong.append((key, value, template))
    if wrong:
        raise ValueError(
            "These settings only mirror values the research code hard-codes, so setting "
            "them changes what a run *claims* and not what it does:\n"
            + "\n".join(
                f"  {key} = {value!r} - not found in the research source as "
                f"{template.format(value=value)!r}"
                for key, value, template in wrong
            )
            + "\nRemove the override. To change the value for real, the research scripts "
            "would have to change, which the brief forbids."
        )


def research_code_sha() -> str:
    """Checksum of the vendored research scripts.

    The brief says not to change the given functions and logic. This makes that
    auditable: the digest is recorded on every run, so an edit to either file would
    show up as a different value on the next one.

    A missing file raises rather than being skipped. The earlier version hashed only
    what it found, so a deployment without the research directory produced a
    respectable-looking digest that attested to nothing at all - the one failure mode
    an audit trail must not have.
    """
    directory = _research_dir()
    digest = hashlib.sha256()
    for name in sorted(("bl_models_train.py", "bl_exp_payout_predictor.py")):
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(
                f"vendored research script missing: {path}. The checksum is the audit "
                f"trail for 'the given code was not changed'; refusing to report one "
                f"computed over an incomplete set."
            )
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _log_researcher_log(trainer: ProductionTrainer, run_dir: Path) -> None:
    """Attach the researcher-defined log as an artifact for accuracy assessment."""
    path = trainer.researcher_log
    if path is None or not Path(path).exists():
        # close_log() detaches the handler, so re-find the file it wrote.
        candidates = sorted(run_dir.glob("log_file_bl_train_*.log"))
        if not candidates:
            return
        path = candidates[-1]
    mlflow.log_artifact(str(path), artifact_path="researcher_log")


# --------------------------------------------------------------------------------- #
# Bundle assembly and registration
# --------------------------------------------------------------------------------- #

def _assemble_bundle(run_dir: Path, trainer: ProductionTrainer, snapshot,
                     settings: Settings, run_id: str,
                     backend: str | None) -> Path:
    """Collect the research artifacts plus ours into one directory, then log it."""
    bundle_dir = run_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    for name in (bundle_files.CATBOOST_FILE, bundle_files.PAYOUT_CONTEXT_FILE,
                 bundle_files.CLIENTS_FILE):
        source = run_dir / name
        if not source.exists():
            raise FileNotFoundError(
                f"the training run did not produce {name}; expected it in {run_dir}"
            )
        shutil.copy2(source, bundle_dir / name)

    # Removes names-dataset (18s, 2.4GB) from the serving image entirely.
    if settings.model.build_gender_lookup:
        GenderLookup.build(bundle_dir / GENDER_ARTIFACT)

    extra: list[str] = []
    if trainer.payout_backend is not None:
        extra = trainer.payout_backend.save_extra(bundle_dir)
        fidelity = getattr(trainer.payout_backend, "fidelity", {})
        if fidelity:
            mlflow.log_metrics(fidelity)

    payout_context = joblib.load(bundle_dir / bundle_files.PAYOUT_CONTEXT_FILE)
    clients = pd.read_csv(bundle_dir / bundle_files.CLIENTS_FILE)

    manifest = bundle_files.Manifest(
        trained_at=bundle_files.utc_now(),
        mlflow_run_id=run_id,
        delta_version=snapshot.version,
        delta_table=snapshot.table_uri,
        # What was fitted, not what was read. The research code splits the last
        # days_for_test days off in both modes (bl_models_train.py line 235), so the
        # snapshot size overstated this by about a tenth on the real extract - in the
        # shipped manifest, in the registered version's tags, and in `make versions`.
        rows_train=int(trainer.split_counts.get("rows_fitted", len(snapshot.frame))),
        rows_payout_context=len(payout_context["x"]),
        n_brands=int((clients["client_name"] != "other").sum()),
        payout_backend=(trainer.payout_backend.name if trainer.payout_backend
                        else (backend or settings.model.payout.backend)),
        payout_exact=(trainer.payout_backend.exact if trainer.payout_backend else True),
        git_sha=bundle_files.current_git_sha(REPO_ROOT),
        research_code_sha=research_code_sha(),
        train_columns=list(payout_context["columns"]),
        extra_files=extra,
        config=settings.flat(),
    )
    manifest.write(bundle_dir)
    mlflow.set_tags(manifest.as_tags())

    # Raw files too, so they can be pulled individually without unpacking the model.
    mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
    return bundle_dir


def _register(bundle_dir: Path, settings: Settings) -> str:
    """Log the pyfunc, register it, and move the serving alias onto the new version."""
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=BrandRankerModel(),
        artifacts={BUNDLE_ARTIFACT_KEY: str(bundle_dir)},
        # The research scripts and our package travel with the model so the artifact is
        # self-contained wherever it is deployed.
        code_paths=[str(REPO_ROOT / "src" / "bl_ranking")],
        signature=build_signature(),
        input_example=request_example(),
        registered_model_name=settings.mlflow.registered_model,
        pip_requirements=_serving_requirements(),
    )

    client = MlflowClient()
    version = _version_for(client, settings.mlflow.registered_model, info.run_id)
    # Promote through ops/registry.set_alias, not by writing the alias here.
    #
    # They are the same operation - "point champion at this version" - and writing the
    # alias directly skipped the half of it that records `champion_previous`. So the
    # documented way back existed only after a manual rollback had already been run,
    # and pointed at whatever the operator had just rejected. A weekly retrain is
    # exactly when an operator needs one command to undo it.
    registry.set_alias(settings, version)
    manifest = bundle_files.Manifest.read(bundle_dir)
    for key, value in manifest.as_tags().items():
        client.set_model_version_tag(settings.mlflow.registered_model, version, key, value)
    return version


def _version_for(client: MlflowClient, name: str, run_id: str) -> str:
    versions = client.search_model_versions(f"name='{name}'")
    for candidate in versions:
        if candidate.run_id == run_id:
            return candidate.version
    raise RuntimeError(f"no registered version of {name} found for run {run_id}")


def _serving_requirements() -> list[str]:
    """Pinned to what the serving container installs. Deliberately excludes
    names-dataset: the gender table in the bundle replaces it."""
    return [
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
        "catboost==1.2.7",
        "joblib==1.4.2",
        "pyarrow==18.1.0",
        "pyyaml==6.0.2",
    ]


def _new_run_dir(settings: Settings, mode: str) -> Path:
    """A fresh directory per run, stamped in UTC and never reused.

    UTC because the name is also an ordering: serving's offline fallback picks the newest
    `runs/*/bundle` by sorting these names (serving/model_source._latest_local_bundle).
    Local time breaks both halves of that. A DST fall-back repeats an hour, so two
    retrains an hour apart could land on the same second-resolution name - and
    `exist_ok=True` meant the second one wrote its artifacts over the first, into a
    directory whose manifest belonged to the other run. Moving the container's TZ
    backwards has the same effect on the ordering: the newest bundle stops being the last
    name alphabetically.

    Microsecond resolution rather than a counter suffix, because a suffix breaks the very
    ordering it was added to protect: '..._191130Z-1_production' sorts *before*
    '..._191130Z_production', since '-' precedes '_'. Re-stamping until the name is free
    keeps every name the same shape and in time order.
    """
    root = resolve(settings.paths.run_root)
    while True:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%fZ")
        path = root / f"{stamp}_{mode}"
        if not path.exists():
            path.mkdir(parents=True)
            return path


# --------------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BL training pipeline")
    parser.add_argument("--mode", choices=("train_test", "production"),
                        default="train_test",
                        help="train_test evaluates and registers nothing; "
                             "production skips evaluation and registers a version")
    parser.add_argument("--backend", default=None,
                        help="override model.payout.backend for this run")
    parser.add_argument("--no-register", action="store_true",
                        help="production mode: build the bundle but skip the registry")
    # The Delta table keeps every version and the module docstring in data/delta.py sells
    # time travel as the reason - but nothing could reach it: this was the only training
    # read and it always took the latest. "Retrain exactly what version N produced" is the
    # first thing anyone asks after a bad retrain.
    parser.add_argument("--delta-version", type=int, default=None,
                        help="train on a past Delta version instead of the latest")
    args = parser.parse_args()

    result = run(train_test=args.mode == "train_test",
                 backend=args.backend, register=not args.no_register,
                 delta_version=args.delta_version)

    print(f"mode           {result.mode}")
    print(f"mlflow run     {result.run_id}")
    print(f"duration       {result.duration_s:.1f}s")
    if result.bundle_dir:
        print(f"bundle         {result.bundle_dir}")
    if result.model_version:
        print(f"model version  {result.model_version}")
    if result.metrics:
        print("metrics")
        for key in sorted(result.metrics):
            if key.startswith("clf.day"):
                continue
            print(f"  {key:28s} {result.metrics[key]:.4f}")
    print(json.dumps({"run_id": result.run_id, "version": result.model_version}))


if __name__ == "__main__":
    main()
