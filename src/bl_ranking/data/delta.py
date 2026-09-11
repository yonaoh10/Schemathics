"""Delta Lake storage for the session table.

Databricks reads training data from a Delta table; locally we use delta-rs
(the `deltalake` package), which writes the same on-disk format without needing Spark.
The parts that matter for this system are identical either way:

  * every write produces a new, immutable table version;
  * a training run records the version it read, so the exact input is recoverable;
  * time travel lets a rollback re-train on precisely the data a past run saw.

Swapping in Databricks means changing the URI to `catalog.schema.table` and replacing
these two functions with `spark.read.table` / `DataFrame.write.saveAsTable`. Nothing
upstream or downstream of them changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from deltalake import DeltaTable, write_deltalake

from bl_ranking.config import resolve


@dataclass(frozen=True)
class Snapshot:
    """A table read, tagged with the version it came from."""

    frame: pd.DataFrame
    version: int
    table_uri: str

    @property
    def rows(self) -> int:
        return len(self.frame)


def write_snapshot(frame: pd.DataFrame, table_uri: str | Path,
                   mode: str = "overwrite") -> int:
    """Write the frame and return the new table version.

    `overwrite` is right for this dataset: the upstream extract is a full dump of the
    two-month window, not an increment. Delta keeps the previous versions regardless,
    which is what gives us rollback.
    """
    path = resolve(table_uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_deltalake(str(path), frame, mode=mode, schema_mode="overwrite")
    return DeltaTable(str(path)).version()


def read_snapshot(table_uri: str | Path, version: int | None = None,
                  lookback_days: int | None = None) -> Snapshot:
    """Read the table (optionally at a past version, optionally windowed by date)."""
    path = resolve(table_uri)
    if not path.exists():
        raise FileNotFoundError(
            f"Delta table {path} does not exist. Run `make ingest` first."
        )
    table = DeltaTable(str(path), version=version)
    frame = table.to_pandas()

    if lookback_days:
        session_dt = pd.to_datetime(frame["session_dt"], errors="coerce")
        cutoff = session_dt.max() - pd.Timedelta(days=lookback_days)
        frame = frame[session_dt >= cutoff].reset_index(drop=True)

    return Snapshot(frame=frame, version=table.version(), table_uri=str(path))


def table_version(table_uri: str | Path) -> int | None:
    path = resolve(table_uri)
    if not path.exists():
        return None
    return DeltaTable(str(path)).version()


def history(table_uri: str | Path, limit: int = 10) -> list[dict]:
    """Recent commits, for `make data-history` and for debugging a bad training run."""
    path = resolve(table_uri)
    if not path.exists():
        return []
    return DeltaTable(str(path)).history(limit)
