"""Precomputed lookup table for the gender feature.

Both research scripts do this at module import (line 5):

    from names_dataset import NameDataset
    nd = NameDataset()

Measured on this machine, that single line costs **18.6 s and 2.4 GB of resident
memory**. In a script it is a one-off annoyance. In a serving container it is the
dominant cold-start cost, it multiplies by the number of worker processes, and it sets
the memory floor for the whole deployment.

`detect_gender_with_confidence` is a pure function of one first name, and the dataset's
first-name universe is finite and enumerable (727,556 entries). So the whole function
can be materialised once, offline, into a table:

    714,212 entries -> 4.2 MB parquet -> 290 MB resident, 3.9 s to load
    per-lookup cost drops from 77 us to 0.09 us

Names absent from the table return ('unknown', 0.0) - which is exactly what the
research implementation returns for a name the dataset does not know. The table is
therefore **exact**, not an approximation, and tests/test_gender_lut.py asserts that
against the live NameDataset whenever it is installed.

The table is built by the training job and travels inside the model version, so the
serving image never needs names-dataset at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ARTIFACT_NAME = "gender_lookup.parquet"

# What the research code returns when the dataset has no usable entry for a name.
UNKNOWN: tuple[str, float] = ("unknown", 0.0)


class GenderLookup:
    """In-memory name -> (gender, confidence) map with the research code's semantics."""

    def __init__(self, table: dict[str, tuple[str, float]]) -> None:
        self._table = table

    def __len__(self) -> int:
        return len(self._table)

    def lookup(self, fname: str) -> tuple[str, float]:
        """Key is the capitalised name, matching `str(x).strip().capitalize()` upstream."""
        return self._table.get(fname, UNKNOWN)

    @classmethod
    def from_mapping(cls, mapping: dict[str, tuple[str, float]]) -> GenderLookup:
        """Build directly from a dict. Used by tests and by any alternative source."""
        return cls(dict(mapping))

    @classmethod
    def load(cls, path: str | Path) -> GenderLookup:
        table = pq.read_table(path)
        names = table.column("name").to_pylist()
        genders = table.column("gender").to_pylist()
        confidences = table.column("confidence").to_pylist()
        return cls(dict(zip(names, zip(genders, confidences, strict=False), strict=False)))

    @classmethod
    def build(cls, path: str | Path | None = None, dataset=None) -> GenderLookup:
        """Enumerate names-dataset and materialise the function. Training-time only.

        Reuses an already-loaded NameDataset when there is one. The training job imports
        the research module, which builds one at module scope, so constructing a second
        would add another 2.4 GB to a process that already peaks around 7.5 GB.
        """
        dataset = dataset or _loaded_dataset() or _new_dataset()
        names: list[str] = []
        genders: list[str] = []
        confidences: list[float] = []

        for name in dataset.first_names:
            gender, confidence = _detect(dataset, name)
            if (gender, confidence) == UNKNOWN:
                continue  # identical to the miss path; storing it would only add size
            names.append(name)
            genders.append(gender)
            confidences.append(confidence)

        if path is not None:
            table = pa.table({
                "name": pa.array(names),
                # Two distinct values across 714k rows; dictionary encoding makes the
                # file 4 MB instead of 30.
                "gender": pa.array(genders).dictionary_encode(),
                # float64, not float32: the research code returns round(m/total, 3),
                # and float32 turns 0.992 into 0.9919999837875366. The feature is not
                # one of the 25 the models use, but a table that claims to be exact
                # has to be exact.
                "confidence": pa.array(confidences, type=pa.float64()),
            })
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, str(path), compression="zstd")

        return cls(dict(zip(names, zip(genders, confidences, strict=False), strict=False)))


def _loaded_dataset():
    """The NameDataset the research modules build at import, if either is loaded.

    Both scripts do `nd = NameDataset()` at module scope. Finding it here is not a
    hack around their design - it is the only way to avoid paying that 2.4 GB twice in
    a process that has already imported one of them.
    """
    for module_name in ("bl_ranking.research.bl_models_train",
                        "bl_ranking.research.bl_exp_payout_predictor"):
        module = sys.modules.get(module_name)
        existing = getattr(module, "nd", None) if module else None
        if existing is not None:
            return existing
    return None


def _new_dataset():
    from names_dataset import NameDataset

    return NameDataset()


def _detect(dataset, fname: str) -> tuple[str, float]:
    """Line-for-line copy of BLPayoutModelsFit.detect_gender_with_confidence.

    Duplicated on purpose: this is the definition the table must reproduce, so it has
    to be visible next to the builder. tests/test_gender_lut.py pins the two together.
    """
    result = dataset.search(fname)
    if result is None:
        return UNKNOWN
    first = result.get("first_name")
    if first is None:
        return UNKNOWN
    gender_data = first.get("gender")
    if gender_data is None:
        return UNKNOWN
    male = gender_data.get("Male", 0)
    female = gender_data.get("Female", 0)
    total = male + female
    if total == 0:
        return UNKNOWN
    if male >= female:
        return "male", round(male / total, 3)
    return "female", round(female / total, 3)
