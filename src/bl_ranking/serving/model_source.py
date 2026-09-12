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
    except Exception as exc:  # noqa: BLE001 - classified below
        if _registry_is_authoritative(settings):
            # The local fallback picks the NEWEST bundle on disk, which after a rollback
            # is precisely the version the operator rolled back from. Falling back here
            # would silently undo their decision and report the worker healthy while
            # doing it. Where a tracking server is configured, the registry is the only
            # thing that knows which version should be serving: if it cannot be reached,
            # this worker has no business guessing. Failing keeps /readyz at 503 and the
            # load balancer routes around it until the registry is back.
            raise RuntimeError(
                f"could not resolve {default_uri} ({exc}). A tracking server is "
                f"configured, so the registry decides which version serves; refusing to "
                f"fall back to a local bundle, which after a rollback would be the "
                f"version that was rolled back from."
            ) from exc
        log.warning("could not resolve %s (%s); no tracking server is configured, so "
                    "falling back to the newest local run", default_uri, exc)
        return _latest_local_bundle(settings)


def _registry_is_authoritative(settings: Settings) -> bool:
    """True unless this process is pointed at a plain directory of files.

    The distinction is the whole point. A file store under the repository is the
    offline development case the local fallback exists for, and pointing at one is not
    a statement about which version should be serving. Anything else - a tracking
    server, or any of the database backends MLflow supports - is administered, and its
    alias is an operator's decision, so a worker that cannot read it has nothing to
    fall back *to* that would not contradict that decision.

    Written as "not a file store" rather than as a list of remote schemes, because the
    list is the part that goes stale: an earlier version named http, https and
    databricks, which silently left the rollback-undoing fallback live for every
    postgresql://, mysql:// and sqlite:// registry - the ordinary production setups.
    """
    uri = str(settings.mlflow.resolved_tracking_uri()).strip()
    if not uri:
        return False
    # MLflow's Databricks URIs are the one form with no scheme separator: the literal
    # "databricks", or "databricks://<profile>".
    if uri.lower() == "databricks" or uri.lower().startswith("databricks:"):
        return True
    scheme, separator, _ = uri.partition("://")
    if not separator:
        return False          # a bare path is a local directory
    return scheme.lower() != "file"


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
