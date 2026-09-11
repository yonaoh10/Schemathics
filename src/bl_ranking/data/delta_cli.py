"""Inspect the Delta session table: versions, row counts, and what each run read.

    python -m bl_ranking.data.delta_cli history
    python -m bl_ranking.data.delta_cli show --version 3

Every training run logs the table version it trained on, so this is the other half of
reproducing a past model: `make versions` says which data version a model version used,
and this says what was in it.
"""

from __future__ import annotations

import argparse

import pandas as pd

from bl_ranking.config import Settings, resolve
from bl_ranking.data.delta import history, read_snapshot, table_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("history", help="recent commits to the table")
    show = sub.add_parser("show", help="summarise one version")
    show.add_argument("--version", type=int, default=None)

    args = parser.parse_args()
    settings = Settings.load()
    table = settings.paths.delta_table

    if args.command == "history":
        current = table_version(table)
        if current is None:
            print(f"no Delta table at {resolve(table)} - run `make ingest` first")
            return
        print(f"table   {resolve(table)}")
        print(f"latest  version {current}\n")
        print(f"{'ver':>4}  {'timestamp':<19}  {'operation':<12} {'rows':>10}  engine")
        for commit in history(table, limit=15):
            metrics = commit.get("operationMetrics") or {}
            # delta-rs and Spark name these differently; accept either.
            rows = next((metrics[k] for k in
                         ("num_added_rows", "numOutputRows", "num_output_rows")
                         if k in metrics), "?")
            stamp = pd.to_datetime(commit.get("timestamp"), unit="ms").strftime("%Y-%m-%d %H:%M:%S")
            print(f"{commit.get('version', '?'):>4}  {stamp:<19}  "
                  f"{commit.get('operation', '?'):<12} {rows:>10}  "
                  f"{commit.get('clientVersion', commit.get('engineInfo', '?'))}")
        return

    snapshot = read_snapshot(table, version=args.version)
    session_dt = pd.to_datetime(snapshot.frame["session_dt"], errors="coerce")
    print(f"version     {snapshot.version}")
    print(f"rows        {snapshot.rows:,}")
    print(f"sessions    {snapshot.frame['session_id'].nunique():,}")
    print(f"window      {session_dt.min()} .. {session_dt.max()}")
    print(f"brands      {snapshot.frame['client_name'].nunique()}")
    print(f"accepted    {(snapshot.frame['disposition'] == 'Lead').sum():,}")
    print(f"payout > 0  {(snapshot.frame['payout'].fillna(0) > 0).sum():,}")


if __name__ == "__main__":
    main()
