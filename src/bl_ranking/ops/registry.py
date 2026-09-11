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
    if previous and previous != version:
        client.set_registered_model_alias(name, f"{alias}_previous", previous)
    client.set_registered_model_alias(name, alias, version)

    return {"model": name, "alias": alias, "from": previous, "to": target.version,
            "run_id": target.run_id}


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
              f"{'rows':>9}  {'git':<12} run")
        for row in rows:
            marker = "->" if row["serving"] else "  "
            print(f"{marker} {row['version']:>4}  {row['payout_backend']:<18} "
                  f"{row['payout_exact']:<6} {row['delta_version']:>5} "
                  f"{row['rows_train']:>9}  {row['git_sha']:<12} {row['run_id'][:12]}")
        print(f"\n-> = serving now (alias '{settings.mlflow.serving_alias}')")

    elif args.command == "current":
        info = current(settings)
        print(json.dumps(info, indent=2) if info else
              f"alias '{settings.mlflow.serving_alias}' is not set")

    else:
        result = set_alias(settings, args.version, args.alias)
        print(json.dumps(result, indent=2))
        print("\nRestart the serving processes to pick it up:")
        print("  docker compose -f docker/docker-compose.yml restart api")


if __name__ == "__main__":
    main()
