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

import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("bl_ranking.models")

ARTIFACT_NAME = "gender_lookup.parquet"

# Stamped into the parquet's own metadata so a stale table announces itself. Bump this
# whenever the KEY changes: a table keyed the old way looks perfectly healthy from the
# outside - right row count, right columns - and answers 'unknown' for a fifth of all
# names. Only a rebuild fixes it, and only this marker makes the need visible.
KEY_SCHEME = b"capitalised-lookup-key-v1"
KEY_SCHEME_FIELD = b"bl_key_scheme"

# What the research code returns when the dataset has no usable entry for a name.
UNKNOWN: tuple[str, float] = ("unknown", 0.0)


class GenderLookup:
    """In-memory name -> (gender, confidence) map with the research code's semantics."""

    def __init__(self, table: dict[str, tuple[str, float]],
                 key_scheme_current: bool = True) -> None:
        self._table = table
        # False only for a table loaded from a file built before the key fix. Reported
        # by GET /model, so a degraded feature is visible from outside the worker and
        # not only in its start-up log.
        self.key_scheme_current = key_scheme_current

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
    def live(cls) -> GenderLookup:
        """A lookup backed by names-dataset itself, for a bundle built without a table.

        Same interface, same answers, and the same 2.4 GB the table exists to avoid -
        which is why it is only reached when `model.build_gender_lookup` was off. The
        alternative, and what used to happen, is that the vectorised path answered
        'unknown' for every name while the research path did the real lookup, so the
        two implementations silently stopped agreeing on a model feature.
        """
        dataset = _loaded_dataset() or _new_dataset()
        return _LiveGenderLookup(dataset)

    @classmethod
    def load(cls, path: str | Path) -> GenderLookup:
        """Read the table and check it is one, before a worker starts serving from it.

        A table nothing validates is a silent feature outage waiting to happen: an
        empty or truncated parquet answers 'unknown' for every name, the four bundle
        guards say nothing (they check the two models and the brand list), /readyz goes
        green, and the ranking simply changes. Every check here is on the file's own
        shape, so it costs a fraction of the read it follows.
        """
        table = pq.read_table(path)
        missing = [c for c in ("name", "gender", "confidence")
                   if c not in table.column_names]
        if missing:
            raise ValueError(
                f"{path} is not a gender lookup table: missing column(s) "
                f"{', '.join(missing)}; columns present are {table.column_names}."
            )
        if table.num_rows == 0:
            raise ValueError(
                f"{path} is an empty gender lookup table. Every first name would "
                f"resolve to 'unknown' while the research path resolves them, so the "
                f"two feature implementations would disagree on every request."
            )
        if table.column("name").null_count:
            raise ValueError(
                f"{path} has {table.column('name').null_count} null name key(s); the "
                f"lookup would never match them."
            )
        current = _warn_if_stale(table, path)
        names = table.column("name").to_pylist()
        genders = table.column("gender").to_pylist()
        confidences = table.column("confidence").to_pylist()
        return cls(
            dict(zip(names, zip(genders, confidences, strict=False), strict=False)),
            key_scheme_current=current,
        )

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

        # Key by what the serving path looks up, not by the dataset's own spelling.
        #
        # The research code does `str(x).strip().capitalize()` before the lookup, and
        # `capitalize()` lowercases everything after the first letter. names-dataset
        # normalises internally so `search("Anne-marie")` still finds "Anne-Marie", but
        # a dict keyed on the raw spelling does not: 102,602 of its 727,556 names -
        # every hyphenated and multi-word first name among them - differ from their own
        # capitalisation and silently missed. Building on the capitalised key and
        # asking `_detect` with that exact key makes the table exact by construction,
        # because it is the same question serving asks.
        seen: set[str] = set()
        for name in dataset.first_names:
            key = str(name).strip().capitalize()
            if key in seen:
                continue
            seen.add(key)
            gender, confidence = _detect(dataset, key)
            if (gender, confidence) == UNKNOWN:
                continue  # identical to the miss path; storing it would only add size
            names.append(key)
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
            table = table.replace_schema_metadata({KEY_SCHEME_FIELD: KEY_SCHEME})
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


class _LiveGenderLookup(GenderLookup):
    """GenderLookup backed by a names-dataset instance rather than a materialised table."""

    def __init__(self, dataset) -> None:  # noqa: D107 - see GenderLookup.live
        super().__init__({}, key_scheme_current=True)
        self._dataset = dataset

    def __len__(self) -> int:
        return 0

    def lookup(self, fname: str) -> tuple[str, float]:
        return _detect(self._dataset, fname)


def _warn_if_stale(table, path) -> bool:
    """Say so when a table was built before the lookup key was fixed.

    A table keyed on the dataset's own spelling is indistinguishable from a correct one
    by inspection: same row count, same columns, same file size. It simply answers
    'unknown' for every name whose capitalisation differs from its own - 141,897 of
    727,556, every hyphenated and multi-word first name among them - while the research
    path answers correctly, so the two feature implementations disagree silently.

    Warned rather than refused: an old bundle still serves, and refusing to load one
    would turn a degraded feature into an outage. The next training run rebuilds it.
    """
    metadata = table.schema.metadata or {}
    if metadata.get(KEY_SCHEME_FIELD) == KEY_SCHEME:
        return True
    log.warning(
        "%s predates the lookup-key fix (no %s marker). Roughly a fifth of first names "
        "will resolve to 'unknown' on the vectorised path while the research path "
        "resolves them correctly. Retrain to rebuild the table.",
        path, KEY_SCHEME_FIELD.decode(),
    )
    return False
