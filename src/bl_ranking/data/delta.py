"""Delta Lake storage for the session table.

Databricks reads training data from a Delta table; locally we use delta-rs
(the `deltalake` package), which writes the same on-disk format without needing Spark.
The parts that matter for this system are identical either way:

  * every write produces a new, immutable table version;
  * a training run records the version it read, so the exact input is recoverable;
  * time travel lets a rollback re-train on precisely the data a past run saw -
    `python -m bl_ranking.training.job --mode production --delta-version N`, where N is
    the `data.delta_version` the run being reproduced logged. The flag exists because
    this claim was here long before anything could act on it: the training job took the
    latest version and offered no way to ask for another.

Swapping in Databricks means changing the URI to `catalog.schema.table` and replacing
these two functions with `spark.read.table` / `DataFrame.write.saveAsTable`. Nothing
upstream or downstream of them changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from deltalake import CommitProperties, DeltaTable, write_deltalake

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
                   mode: str = "overwrite",
                   commit_metadata: dict[str, str] | None = None) -> int:
    """Write the frame and return the new table version.

    `overwrite` is right for this dataset: the upstream extract is a full dump of the
    two-month window, not an increment. Delta keeps the previous versions regardless,
    which is what gives us rollback.

    `commit_metadata` rides along in the commit itself, which is how the gate's repair
    counters reach the training run that later reads this version (see
    data/ingest.read_report). Stored here rather than in a file beside the table because
    it is then atomic with the version it describes, and because it works on object
    storage and Unity Catalog - where a local sibling directory would not exist at all.
    """
    path = resolve(table_uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = ({"commit_properties": CommitProperties(custom_metadata=commit_metadata)}
             if commit_metadata else {})
    write_deltalake(str(path), _typed_for_delta(frame), mode=mode,
                    schema_mode="overwrite", **extra)
    return DeltaTable(str(path)).version()


def commit_metadata(table_uri: str | Path, version: int, key: str) -> str | None:
    """One custom metadata value from the commit that produced `version`.

    None for anything that means "not there": no such table, no such version, no such
    key. The only caller logs it as run parameters, and a missing counter must never be
    the reason a finished training run fails.
    """
    try:
        history = DeltaTable(str(resolve(table_uri))).history()
    except Exception:  # noqa: BLE001 - an unreadable table simply has no metadata to give
        return None
    for entry in history:
        if entry.get("version") == version:
            value = entry.get(key)
            return str(value) if value is not None else None
    return None


def _typed_for_delta(frame: pd.DataFrame) -> pd.DataFrame:
    """Give entirely-null columns a concrete type before the write.

    Arrow infers the `Null` type for a column with nothing in it, and Delta rejects
    that outright: "Invalid data type for Delta Lake: Null". It is not an exotic case -
    conversion_dt is empty in any window where nobody converted, which is an ordinary
    quiet period or a freshly launched funnel, and the whole ingestion then fails with
    a message that names neither the column nor the file.

    Typed as string because every column that can be wholly null here is either a
    timestamp already normalised to an ISO string or a categorical, and the research
    code reads the staged CSV back with its own inference regardless.
    """
    empty = [c for c in frame.columns if frame[c].isna().all()]
    if not empty:
        return frame
    typed = frame.copy()
    for column in empty:
        typed[column] = typed[column].astype("string")
    return typed


def read_snapshot(table_uri: str | Path, version: int | None = None,
                  lookback_days: int | None = None) -> Snapshot:
    """Read the table (optionally at a past version, optionally windowed by date)."""
    path = resolve(table_uri)
    if not path.exists():
        raise FileNotFoundError(
            f"Delta table {path} does not exist. Run `make ingest` first."
        )
    try:
        table = DeltaTable(str(path), version=version)
    except Exception as exc:  # noqa: BLE001 - delta-rs raises several types here
        # delta-rs answers a version that does not exist with a generic error, and a
        # negative one by saying the table was not found - for a table that plainly is,
        # since the path check above just passed. Say which version was asked for and
        # which ones there are.
        latest = DeltaTable(str(path)).version()
        raise ValueError(
            f"Delta table {path} has no version {version}; versions 0..{latest} exist."
        ) from exc
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
