"""Deciding which model version a serving process loads.

Resolution order, first match wins:

  1. `serving.model_uri` (or BL_SERVING__MODEL_URI) - an explicit pin. Accepts
     `models:/<name>@<alias>`, `models:/<name>/<version>`, or a local directory.
  2. `models:/<registered_model>@<serving_alias>` - the normal production path.
  3. The newest local `runs/*/bundle` - so the repo runs offline, straight after
     `make train`, with no tracking server.

Rollback is deliberately an operation on the registry, not on the containers:

    mlflow: set alias `champion` -> version 7
    restart (or let the next deploy roll)

Because a model version is one atomic bundle (see models/bundle.py), moving the alias
moves the classifier, the payout context, the brand universe and the gender table
together. There is no way to end up with a mismatched pair.

`GET /model` reports the resolved version, so what a live worker is serving is always
answerable without guessing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from bl_ranking.config import Settings, resolve

log = logging.getLogger("bl_ranking.serving")

BUNDLE_ARTIFACT_PATH = "bundle"
_MODELS_URI = re.compile(r"^models:/(?P<name>[^/@]+)(?:@(?P<alias>.+)|/(?P<version>\d+))$")


def resolve_bundle(settings: Settings) -> Path:
    """Return a local directory containing the model bundle."""
    explicit = settings.serving.model_uri
    if explicit:
        return _from_uri(explicit, settings)

    default_uri = f"models:/{settings.mlflow.registered_model}@{settings.mlflow.serving_alias}"
    try:
        return _from_uri(default_uri, settings)
    except Exception as exc:  # noqa: BLE001 - registry unavailable is a normal dev case
        log.warning("could not resolve %s (%s); falling back to the newest local run",
                    default_uri, exc)
        return _latest_local_bundle(settings)


def _from_uri(uri: str, settings: Settings) -> Path:
    local = Path(uri)
    if local.exists():
        return local

    match = _MODELS_URI.match(uri)
    if not match:
        raise ValueError(
            f"model_uri {uri!r} is neither an existing directory nor a models:/ URI"
        )

    mlflow.set_tracking_uri(settings.mlflow.resolved_tracking_uri())
    client = MlflowClient()
    name = match.group("name")
    if match.group("alias"):
        version = client.get_model_version_by_alias(name, match.group("alias"))
    else:
        version = client.get_model_version(name, match.group("version"))

    log.info("resolved %s -> version %s (run %s)", uri, version.version, version.run_id)
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path=BUNDLE_ARTIFACT_PATH,
    )
    return Path(downloaded)


def _latest_local_bundle(settings: Settings) -> Path:
    run_root = resolve(settings.paths.run_root)
    candidates = sorted(run_root.glob("*/bundle"))
    if not candidates:
        raise FileNotFoundError(
            f"no model bundle found under {run_root}. Run `make train-prod` first, "
            f"or point serving.model_uri at a registered version."
        )
    chosen = candidates[-1]
    log.info("serving the local bundle at %s", chosen)
    return chosen
