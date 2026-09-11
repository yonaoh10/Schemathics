"""The training job: one MLflow run per execution, in either of the research code's modes.

    train_test=True   time-based split, per-day evaluation, nothing registered.
                      This is the mode a researcher runs to decide whether a change is
                      an improvement, so the run carries the researcher log verbatim
                      *and* the same numbers as MLflow metrics.

    train_test=False  fit on everything, produce artifacts, register a new model
                      version and move the `champion` alias onto it. This is the mode
                      the weekly Sunday 05:00 schedule runs.

Both modes log the same technical parameters, so any two runs are directly comparable
in the MLflow UI regardless of mode: the input data version, the row counts, the model
hyper-parameters, the payout backend, the git SHA, and a checksum of the research code
itself so it is provable that the given scripts were not edited.
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
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from bl_ranking.config import REPO_ROOT, Settings, resolve
from bl_ranking.data.delta import read_snapshot
from bl_ranking.data.ingest import stage_for_research_code
from bl_ranking.models import bundle as bundle_files
from bl_ranking.models.gender_lut import ARTIFACT_NAME as GENDER_ARTIFACT
from bl_ranking.models.gender_lut import GenderLookup
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
        backend: str | None = None, register: bool = True) -> TrainingResult:
    settings = settings or Settings.load()
    started = time.perf_counter()
    mode = "train_test" if train_test else "production"

    mlflow.set_tracking_uri(settings.mlflow.resolved_tracking_uri())
    mlflow.set_experiment(settings.mlflow.experiment)

    snapshot = read_snapshot(settings.paths.delta_table,
                             lookback_days=settings.data.lookback_days)
    run_dir = _new_run_dir(settings, mode)

    with mlflow.start_run(run_name=f"{mode}-{run_dir.name}") as active:
        mlflow.set_tags(_tags(mode, settings, backend))
        mlflow.log_params(_params(settings, snapshot, backend, run_dir))

        # The research code reads a CSV at `input_path + input_file`. Staging the Delta
        # snapshot back to CSV leaves that call untouched and reproduces the same dtype
        # inference the research runs had.
        stage_dir, stage_file = stage_for_research_code(snapshot.frame, run_dir / "input")

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

        # setup_bl_logger clears the root logger's handlers; restore them afterwards so
        # MLflow and the rest of the process keep logging.
        with preserve_root_logging():
            trainer.fit_()
        trainer.close_log()

        _log_researcher_log(trainer, run_dir)
        if trainer.metrics:
            mlflow.log_metrics(trainer.metrics)
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


def research_code_sha() -> str:
    """Checksum of the vendored research scripts.

    The brief says not to change the given functions and logic. This makes that
    auditable: the digest is recorded on every run, so an edit to either file would
    show up as a different value on the next one.
    """
    digest = hashlib.sha256()
    for name in sorted(("bl_models_train.py", "bl_exp_payout_predictor.py")):
        path = RESEARCH_DIR / name
        if path.exists():
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
        rows_train=len(snapshot.frame),
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
    client.set_registered_model_alias(
        settings.mlflow.registered_model, settings.mlflow.serving_alias, version,
    )
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
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = resolve(settings.paths.run_root) / f"{stamp}_{mode}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BL training pipeline")
    parser.add_argument("--mode", choices=("train_test", "production"),
                        default="train_test",
                        help="train_test evaluates and registers nothing; "
                             "production trains on all data and registers a version")
    parser.add_argument("--backend", default=None,
                        help="override model.payout.backend for this run")
    parser.add_argument("--no-register", action="store_true",
                        help="production mode: build the bundle but skip the registry")
    args = parser.parse_args()

    result = run(train_test=args.mode == "train_test",
                 backend=args.backend, register=not args.no_register)

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
