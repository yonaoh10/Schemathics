"""Deciding which model version a serving process loads.

Resolution order, first match wins:

  1. `serving.model_uri` (or BL_SERVING__MODEL_URI) - an explicit pin. Accepts
     `models:/<name>@<alias>`, `models:/<name>/<version>`, or a local directory.
  2. `models:/<registered_model>@<serving_alias>` - the normal production path.
  3. The newest local `runs/*/bundle` - so the repo runs offline, straight after
     `make train-prod`, with no tracking server. `train-prod`, not `train`: the
     evaluation mode registers nothing and writes no bundle, so there would be nothing
     for this tier to find.

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

import contextlib
import logging
import os
import re
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from bl_ranking.config import Settings, resolve
from bl_ranking.models import bundle as bundle_files

log = logging.getLogger("bl_ranking.serving")

BUNDLE_ARTIFACT_PATH = "bundle"
# A URI scheme as RFC 3986 spells it. A POSIX path cannot match at position 0.
# Two characters minimum, because a one-letter "scheme" is a Windows drive: C:/mlruns is
# a local directory, and reading it as a remote store told a developer on Windows that the
# registry was authoritative and refused to serve their own local bundle.
_URI_SCHEME = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]+):")
_MODELS_URI = re.compile(r"^models:/(?P<name>[^/@]+)(?:@(?P<alias>.+)|/(?P<version>\d+))$")


# How the bundle a worker is serving was chosen. Reported by GET /model, because "this
# worker is serving an unregistered local directory" is not something an operator should
# have to read a start-up log to discover: the likeliest cause is a mistyped
# BL_MLFLOW__REGISTERED_MODEL or BL_MLFLOW__SERVING_ALIAS, and the symptom is a healthy
# endpoint serving whatever happens to be newest on disk.
BUNDLE_SOURCE_PINNED = "pinned_uri"
BUNDLE_SOURCE_REGISTRY = "registry_alias"
BUNDLE_SOURCE_LOCAL = "local_run_directory"

# Set by resolve_bundle, read by the app when it builds /model.
last_bundle_source: str = BUNDLE_SOURCE_REGISTRY

# Startup must fail fast when the registry is unreachable. With MLflow's defaults
# (MLFLOW_HTTP_REQUEST_MAX_RETRIES=5, backoff factor 2) an unreachable registry took ~62 s
# to raise, and because resolve_bundle runs in the lifespan hook *before* uvicorn accepts
# connections, /healthz, /readyz and /model were all unreachable for 68 s - long enough for
# a liveness probe to kill the worker and for compose's `restart: unless-stopped` to loop
# it. Bounded here so the worker comes up within a few seconds and answers 503 with the
# reason while the registry is down, which is what the load balancer routes around.
_STARTUP_HTTP_RETRIES = "2"
_STARTUP_HTTP_TIMEOUT = "5"


@contextlib.contextmanager
def _bounded_registry_http():
    """Cap MLflow's HTTP retries/timeout for the start-up resolution only.

    An operator's own values win (setdefault), and anything this sets that was previously
    unset is removed again afterwards, so this changes nothing for the training job, which
    imports none of this and wants MLflow's normal retry budget for artifact uploads.
    """
    wanted = {
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": _STARTUP_HTTP_RETRIES,
        "MLFLOW_HTTP_REQUEST_TIMEOUT": _STARTUP_HTTP_TIMEOUT,
    }
    added = [k for k in wanted if not os.environ.get(k)]
    for key in added:
        os.environ[key] = wanted[key]
    try:
        yield
    finally:
        for key in added:
            os.environ.pop(key, None)


def resolve_bundle(settings: Settings) -> Path:
    """Return a local directory containing the model bundle."""
    global last_bundle_source
    explicit = settings.serving.model_uri
    if explicit:
        last_bundle_source = BUNDLE_SOURCE_PINNED
        with _bounded_registry_http():
            return _from_uri(explicit, settings)

    default_uri = f"models:/{settings.mlflow.registered_model}@{settings.mlflow.serving_alias}"
    try:
        with _bounded_registry_http():
            resolved = _from_uri(default_uri, settings)
    except Exception as exc:  # noqa: BLE001 - classified below
        authoritative = _registry_is_authoritative(settings)
        if authoritative and not _alias_is_unset(exc):
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
        # A missing MODEL (not a missing alias) against an administered registry that holds
        # other models is a mistyped BL_MLFLOW__REGISTERED_MODEL, not an empty registry -
        # and falling back there serves the newest local bundle (after a rollback, the one
        # rolled back from) under a green /readyz. The genuinely empty registry is the
        # documented `docker compose up` bootstrap and must still fall back so the first
        # train-prod can be served. The registry just answered, so the one question that
        # tells these apart - is it empty? - is affordable here.
        if authoritative and _is_missing_model(exc) and not _registry_is_empty(settings):
            raise RuntimeError(
                f"could not resolve {default_uri} ({exc}). The registry holds other models "
                f"but not {settings.mlflow.registered_model!r}, so this is a mistyped "
                f"registered model, not an empty registry; refusing to fall back to a local "
                f"bundle, which would serve whatever is newest on disk under a healthy "
                f"/readyz. Check BL_MLFLOW__REGISTERED_MODEL."
            ) from exc
        log.warning("could not resolve %s (%s); falling back to the newest local run. "
                    "GET /model reports bundle_source=%s so this is visible from outside "
                    "the worker - a mistyped registered model or alias looks exactly like "
                    "a registry with nothing promoted yet.",
                    default_uri, exc, BUNDLE_SOURCE_LOCAL)
        last_bundle_source = BUNDLE_SOURCE_LOCAL
        return _latest_local_bundle(settings)
    else:
        last_bundle_source = BUNDLE_SOURCE_REGISTRY
        return resolved


def _alias_is_unset(exc: BaseException) -> bool:
    """True when the registry answered, and what it said is that the alias is not set.

    "Unreachable" and "nothing promoted yet" are different situations and were treated
    the same, which made the documented local stack unstartable: bring up
    `docker compose up`, and the API refuses to serve the bundle sitting in runs/
    because `models:/bl_rank@champion` does not resolve - so it never becomes ready, and
    the first `make train-prod` cannot be reached through it.

    The difference matters because the refusal exists to protect an operator's decision.
    An alias that was never set records no decision, so serving a local bundle
    contradicts nothing. An unreachable registry may be hiding a rollback, and there the
    refusal stands.

    Read off the failure that already happened, rather than by asking the registry a
    second question: an unreachable one retries with backoff, so a probe would double how
    long a worker takes to report the refusal it is going to report anyway.

    Only the two codes a registry uses to say "not there" count - MLflow answers a
    missing alias with INVALID_PARAMETER_VALUE, an unreachable one with INTERNAL_ERROR.
    Anything unrecognised, including a code a future version renames, falls through to
    the refusal, which is the safe direction: the cost of refusing when the alias was
    merely unset is a 503 an operator can explain, and the cost of falling back when a
    rollback is in force is silently serving the version they rejected.
    """
    from mlflow.exceptions import MlflowException

    if not isinstance(exc, MlflowException):
        return False
    return getattr(exc, "error_code", "") in {
        "RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE",
    }


def _is_missing_model(exc: BaseException) -> bool:
    """True when the registry said the registered MODEL does not exist.

    RESOURCE_DOES_NOT_EXIST is the code for a missing model; a missing *alias* on a model
    that exists is INVALID_PARAMETER_VALUE (see _alias_is_unset). Only the former can be a
    typo in the model name, and only then is the is-the-registry-empty question worth asking.
    """
    from mlflow.exceptions import MlflowException

    if not isinstance(exc, MlflowException):
        return False
    return getattr(exc, "error_code", "") == "RESOURCE_DOES_NOT_EXIST"


def _registry_is_empty(settings: Settings) -> bool:
    """True when the registry holds no registered models at all - a bootstrap, not a typo.

    A mistyped model name leaves the correctly-named model in the registry, so the search
    returns something and this is False; a fresh registry returns nothing. Asked only after
    the registry has already answered RESOURCE_DOES_NOT_EXIST, so it is one cheap extra call
    (6 ms on the file store) and not a probe against an unreachable server. If the search
    itself fails, treat the registry as not-empty: refusing is the safe direction, because
    the alternative is serving a local bundle that may undo a rollback.
    """
    try:
        with _bounded_registry_http():
            mlflow.set_tracking_uri(settings.mlflow.resolved_tracking_uri())
            return not MlflowClient().search_registered_models(max_results=1)
    except Exception:  # noqa: BLE001 - unreachable now means refuse, the safe direction
        return False


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

    Both URIs are consulted, and either one being administered is enough. MLflow lets
    the registry live somewhere other than the tracking store (MLFLOW_REGISTRY_URI), and
    the registry is the thing that holds the alias - so reading only the tracking URI
    left the fallback live for exactly the setup where the alias is furthest away.
    """
    return any(_is_administered(uri) for uri in _registry_uris(settings))


def _registry_uris(settings: Settings) -> tuple[str, ...]:
    """Every URI that could be holding the registry, in no particular order."""
    return (
        str(settings.mlflow.resolved_tracking_uri() or "").strip(),
        os.environ.get("MLFLOW_REGISTRY_URI", "").strip(),
    )


def _is_administered(uri: str) -> bool:
    """True unless this URI names a plain directory of files."""
    if not uri:
        return False
    lowered = uri.lower()
    # The Databricks family, which is the only one that may carry no scheme separator:
    # "databricks", "databricks://<profile>", and Unity Catalog's "databricks-uc" forms.
    # Matched by prefix, because "databricks-uc" on its own was read as a relative path.
    if lowered.startswith("databricks"):
        return True
    # `scheme:`, not `scheme://`. A single-slash typo - `http:/mlflow:5000` - is not a
    # directory either, and treating it as one silently re-enabled the fallback on a
    # deployment whose operator plainly meant a server.
    match = _URI_SCHEME.match(uri)
    if match is None:
        return False          # a bare path is a local directory
    return match.group("scheme").lower() != "file"


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
    """The most recently trained local bundle.

    Ordered by each bundle's own `trained_at`, not by directory name. The names are UTC
    timestamps and do sort correctly, but only for bundles this version produced: an
    earlier one stamped them in local time, so a container whose TZ moved backwards made
    the newest bundle stop being the last name alphabetically. The manifest records when
    the run happened and is the thing that actually answers the question. A bundle with no
    readable manifest falls back to its name, which keeps the ordering total.
    """
    run_root = resolve(settings.paths.run_root)
    candidates = sorted(run_root.glob("*/bundle"))
    if not candidates:
        raise FileNotFoundError(
            f"no model bundle found under {run_root}. Run `make train-prod` first, "
            f"or point serving.model_uri at a registered version."
        )

    def trained_at(bundle: Path) -> tuple[str, str]:
        try:
            manifest = bundle_files.Manifest.read(bundle)
        except Exception:  # noqa: BLE001 - an unreadable manifest orders by name alone
            return ("", bundle.parent.name)
        return (manifest.trained_at or "", bundle.parent.name)

    chosen = max(candidates, key=trained_at)
    log.info("serving the local bundle at %s", chosen)
    return chosen
