"""Identify and roll back the serving version.

The whole point of registering one atomic bundle per training run is that "which model
is live?" and "put last week's model back" are both one command, and neither requires
rebuilding an image or copying files around.

    python -m bl_ranking.ops.registry list
    python -m bl_ranking.ops.registry current
    python -m bl_ranking.ops.registry rollback --version 3
    python -m bl_ranking.ops.registry promote --version 5

Rollback moves the `champion` alias. Because a model version contains the CatBoost
model, the payout context, the brand universe and the gender table together, the alias
move takes all four back at once - there is no way to end up with a classifier from one
run and a payout context from another.

Serving processes resolve the alias at start-up, so a rollback takes effect on the next
restart (or the next rolling deploy). That is deliberate: a model swapping underneath a
warm process mid-request is a harder thing to reason about than a restart.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from bl_ranking.config import Settings


def _client(settings: Settings) -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow.resolved_tracking_uri())
    return MlflowClient()


def list_versions(settings: Settings, limit: int = 20) -> list[dict[str, Any]]:
    client = _client(settings)
    name = settings.mlflow.registered_model
    alias = settings.mlflow.serving_alias

    try:
        champion = client.get_model_version_by_alias(name, alias).version
    except Exception:  # noqa: BLE001 - no alias set yet is a normal early state
        champion = None

    rows = []
    versions = sorted(client.search_model_versions(f"name='{name}'"),
                      key=lambda v: int(v.version), reverse=True)
    for version in versions[:limit]:
        rows.append({
            "version": version.version,
            "serving": version.version == champion,
            "run_id": version.run_id,
            "created": version.creation_timestamp,
            "delta_version": version.tags.get("delta_version", "?"),
            "payout_backend": version.tags.get("payout_backend", "?"),
            "payout_exact": version.tags.get("payout_exact", "?"),
            "git_sha": version.tags.get("git_sha", "?"),
            "rows_train": version.tags.get("rows_train", "?"),
        })
    return rows


def current(settings: Settings) -> dict[str, Any] | None:
    client = _client(settings)
    try:
        version = client.get_model_version_by_alias(
            settings.mlflow.registered_model, settings.mlflow.serving_alias)
    except Exception:  # noqa: BLE001
        return None
    return {
        "version": version.version,
        "run_id": version.run_id,
        "uri": f"models:/{settings.mlflow.registered_model}@{settings.mlflow.serving_alias}",
        "tags": dict(version.tags),
    }


def set_alias(settings: Settings, version: str, alias: str | None = None) -> dict[str, Any]:
    """Point the serving alias at a version. This is both promotion and rollback."""
    client = _client(settings)
    name = settings.mlflow.registered_model
    alias = alias or settings.mlflow.serving_alias

    # Fail before moving anything if the target does not exist.
    target = client.get_model_version(name, version)

    previous = None
    with contextlib.suppress(Exception):
        # No alias set yet is a normal state on the first promotion.
        previous = client.get_model_version_by_alias(name, alias).version

    # Keep a breadcrumb to the version being replaced, so an accidental rollback is
    # itself reversible without reading the run history.
    #
    # Compared against the *resolved* version, as strings.
    #
    # Strings because MLflow returns `.version` as one while the caller may hold an int,
    # and "3" != 3 is always true. Resolved because the caller's spelling is not the
    # version: `make rollback VERSION=02` and a version with stray whitespace both reach
    # the registry, which resolves them, while the guard was still comparing the text
    # typed on the command line. Either way the guard did not fire, and re-running the
    # same rollback pointed champion_previous at the version being rolled back *to* -
    # destroying the only pointer back to the one being replaced, which is the entire
    # reason this breadcrumb exists.
    resolved = str(target.version)
    if previous is not None and str(previous) != resolved:
        client.set_registered_model_alias(name, f"{alias}_previous", previous)
    client.set_registered_model_alias(name, alias, resolved)

    return {"model": name, "alias": alias, "from": previous, "to": target.version,
            "run_id": target.run_id}


def rollback_advice(tracking_uri: str) -> str:
    """What to do after moving the alias, given the registry it was moved in.

    The alias only takes effect where the serving processes read it. A host command with
    no MLFLOW_TRACKING_URI set resolves the local file store under mlruns/, while the Docker
    deployment reads http://mlflow:5000 - so a file-store rollback is a no-op for the
    deployment, and telling the operator to "restart api" then hides that. The advice is
    keyed off whether the registry that changed is the administered one the deployment uses.
    """
    from bl_ranking.serving.model_source import _is_administered

    if _is_administered(tracking_uri):
        return ("Restart the serving processes to pick it up:\n"
                "  docker compose -f docker/docker-compose.yml restart api")
    return ("This is a LOCAL file-store registry, not the one the Docker deployment reads\n"
            "(that is MLFLOW_TRACKING_URI=http://mlflow:5000). To roll back the deployment,\n"
            "point this command at that registry and restart the API:\n"
            "  make rollback VERSION=<n> MLFLOW_TRACKING_URI=http://localhost:5000\n"
            "  docker compose -f docker/docker-compose.yml restart api")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="registered versions, newest first")
    sub.add_parser("current", help="which version the serving alias points at")
    for name in ("rollback", "promote"):
        cmd = sub.add_parser(name, help=f"{name} the serving alias to a version")
        cmd.add_argument("--version", required=True)
        cmd.add_argument("--alias", default=None)

    args = parser.parse_args()
    settings = Settings.load()

    if args.command == "list":
        rows = list_versions(settings)
        if not rows:
            print(f"no versions registered for {settings.mlflow.registered_model}")
            return
        print(f"{'':2} {'ver':>4}  {'backend':<18} {'exact':<6} {'data':>5} "
              f"{'fit rows':>9}  {'git':<12} run")
        for row in rows:
            marker = "->" if row["serving"] else "  "
            print(f"{marker} {row['version']:>4}  {row['payout_backend']:<18} "
                  f"{row['payout_exact']:<6} {row['delta_version']:>5} "
                  f"{row['rows_train']:>9}  {row['git_sha']:<12} {row['run_id'][:12]}")
        print(f"\n-> = serving now (alias '{settings.mlflow.serving_alias}')")
        # "fit rows", not "rows": this is manifest.rows_train, the rows the models actually
        # fitted on (after the survey-sufficiency drop and, in production, the split). It is
        # not the snapshot size read from Delta - GET /model's data.rows is that - and the
        # two differ by 57% on the real extract. Labelling the column as the smaller
        # quantity is what stops a reader concluding the champion saw less data than a
        # version whose column happened to record the snapshot instead.
        print("fit rows = rows fitted (manifest.rows_train), not the Delta snapshot size")

    elif args.command == "current":
        info = current(settings)
        print(json.dumps(info, indent=2) if info else
              f"alias '{settings.mlflow.serving_alias}' is not set")

    else:
        # Which registry this actually talks to, resolved the same way serving resolves it,
        # reported so a rollback that landed in the local file store instead of the
        # deployment's registry is visible rather than silent.
        tracking = settings.mlflow.resolved_tracking_uri()
        result = set_alias(settings, args.version, args.alias)
        result["tracking_uri"] = tracking
        print(json.dumps(result, indent=2))
        print(f"\nalias moved in the registry at {tracking}")
        print(rollback_advice(tracking))


if __name__ == "__main__":
    main()
