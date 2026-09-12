"""CSV -> Delta ingestion with the data-quality gate the research code depends on.

The research scripts are treated as fixed. They make three assumptions about their
input that a raw warehouse dump does not guarantee:

  1. `cellphone` casts cleanly to int   (bl_models_train.py line 100)
  2. the survey columns support `.str`  (line 110)
  3. `payout` is numeric                (line 74)

Rather than patch the model code, ingestion enforces those invariants once, at the
boundary, and reports how many rows it had to repair. That keeps the contract explicit
and auditable: the counters are logged to MLflow with every training run, so a sudden
jump in repairs is visible instead of silently changing a feature.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bl_ranking.config import Settings, resolve
from bl_ranking.data.delta import write_snapshot

# Columns bl_models_train.py selects in import_preprocess. Missing any of them is a
# hard failure: the upstream extract changed and the pipeline must not guess.
REQUIRED_COLUMNS: list[str] = [
    "session_id", "session_dt", "conversion_dt", "register_date",
    "campaign_id", "page", "auto_city", "auto_country", "auto_state", "device_type",
    "sub1", "sub2", "sub3", "business_type", "credit_score", "industry",
    "loan_amount", "loan_reason", "monthly_revenue", "time_in_business",
    "fname", "lname", "cellphone", "client_name", "payout", "disposition",
    "disposition_source",
]

SURVEY_COLUMNS: list[str] = [
    "credit_score", "industry", "loan_amount", "loan_reason",
    "monthly_revenue", "time_in_business", "device_type", "business_type",
]

# Columns that must be read as text, because pandas' own dtype inference destroys them.
#
#   campaign_id  real ids are ~1.2e17, past float64's 53-bit mantissa. One null in the
#                column makes the whole column float64 and 120227360861540306 becomes
#                ...304. Reading as text and converting afterwards keeps it exact.
#   sub1/2/3     one null makes the column float64, so the research code's
#                `fillna('Other').astype(str)` yields '1815195.0' in training while a
#                JSON request yields '1815195' - a permanent categorical miss.
#   cellphone    leading zeros and formatting survive; sanitise() extracts the digits.
#
# This has to happen at read time. Once pandas has rounded a float there is nothing
# downstream that can recover the original integer.
# The attribution ids the research code fills and stringifies together, and the value
# it fills them with (bl_models_train.py: `fillna('Other')` then `.astype(str)`).
SUB_ID_COLUMNS = ("sub1", "sub2", "sub3")
RESEARCH_NULL_CATEGORY = "Other"

# How pandas renames a repeated CSV header: the second `payout` becomes `payout.1`.
_MANGLED_DUPLICATE = re.compile(r"(?P<base>.+)\.\d+")

READ_AS_TEXT: dict[str, str] = {
    "campaign_id": "string",
    "sub1": "string",
    "sub2": "string",
    "sub3": "string",
    "cellphone": "string",
}


@dataclass
class IngestReport:
    """Row counts and repair counters, logged as MLflow params on every run."""

    source: str = ""
    rows_in: int = 0
    rows_out: int = 0
    delta_version: int = -1
    repairs: dict[str, int] = field(default_factory=dict)

    def as_params(self) -> dict[str, object]:
        """Flattened for MLflow. The repair counts describe the extract as read, so
        they are counted against rows_in - a row repaired and then dropped for having
        no session timestamp still says something about the extract's quality."""
        params = {
            "ingest.source": Path(self.source).name,
            "ingest.rows_in": self.rows_in,
            "ingest.rows_out": self.rows_out,
            "ingest.delta_version": self.delta_version,
        }
        params.update({f"ingest.repaired.{k}": v for k, v in self.repairs.items()})
        return params

    def save(self, table_uri: str | Path) -> Path | None:
        """Persist the report beside the table, keyed by the version it produced.

        Ingestion and training are separate jobs - the weekly schedule runs the gate,
        then the trainer reads a Delta *version*, not the CSV. Without this the repair
        counters die with the ingest process and the training run cannot report the
        quality of the data it just trained on. Written outside the table directory so
        delta-rs never sees a file it did not put there.
        """
        directory = _report_dir(table_uri)
        if directory is None:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"v{self.delta_version}.json"
        payload = {
            "source": self.source,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "delta_version": self.delta_version,
            "repairs": self.repairs,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path


def _report_dir(table_uri: str | Path) -> Path | None:
    """Sibling of the Delta table. None for a remote table, where a local sibling
    directory would be meaningless (on Databricks the counters come from the job run)."""
    text = str(table_uri)
    if "://" in text and not text.startswith("file://"):
        return None
    return resolve(table_uri).parent / "ingest_reports"


def read_report(table_uri: str | Path, version: int) -> IngestReport | None:
    """The report for one Delta version, or None if it was not written by this gate."""
    directory = _report_dir(table_uri)
    if directory is None:
        return None
    path = directory / f"v{version}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return IngestReport(
        source=str(payload.get("source", "")),
        rows_in=int(payload.get("rows_in", 0)),
        rows_out=int(payload.get("rows_out", 0)),
        delta_version=int(payload.get("delta_version", -1)),
        repairs={str(k): int(v) for k, v in (payload.get("repairs") or {}).items()},
    )


def ingest(settings: Settings | None = None, source: Path | None = None) -> IngestReport:
    """Read the raw CSV, enforce the input contract, write a new Delta version."""
    settings = settings or Settings.load()
    csv_path = resolve(source or settings.paths.raw_csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Drop the real bl_full_data.csv there, "
            f"or run `make data` to generate a synthetic one."
        )

    # low_memory=False matches the research code's own read; READ_AS_TEXT overrides
    # inference only for the columns where inference is actively harmful.
    #
    # pandas' own errors here name neither the file nor what it choked on - an empty
    # file raises "No columns to parse from file", a ragged one raises "Error
    # tokenizing data" with a row number and nothing else. This function goes out of
    # its way to name the path when the file is missing; it should do the same when the
    # file is there and unreadable.
    try:
        frame = pd.read_csv(csv_path, low_memory=False, dtype=READ_AS_TEXT)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{csv_path} has no header row: {exc}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"{csv_path} is not readable as CSV: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{csv_path} is not UTF-8 text: {exc}") from exc
    report = IngestReport(source=str(csv_path), rows_in=len(frame))
    if report.rows_in == 0:
        # Checked before the gate so the message is about the file, not about whichever
        # invariant an empty column happens to trip first.
        raise ValueError(f"{csv_path} has a header but no rows.")

    _require_columns(frame)
    frame, repairs = sanitise(frame)
    report.repairs = repairs
    report.rows_out = len(frame)

    # An empty frame reaches delta-rs as a schema with no data and comes back as a
    # "Generic error ... no data" that names neither the CSV nor the reason, while the
    # counters that do explain it are still sitting in this function. Say it here, and
    # leave the previous table version in place as the last known good input.
    if report.rows_out == 0:
        raise ValueError(
            f"{csv_path} left no usable rows after the gate "
            f"({report.rows_in} read). Repairs: {report.repairs}. "
            f"The Delta table was not written."
        )

    report.delta_version = write_snapshot(frame, settings.paths.delta_table)
    report.save(settings.paths.delta_table)
    return report


def sanitise(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Enforce the three input invariants. Returns the frame and the repair counts."""
    frame = frame.copy()
    repairs: dict[str, int] = {}

    # 1. cellphone -> int64. The research code takes the first three digits as a
    #    categorical feature, so anything unparseable becomes 0 (prefix '0') rather
    #    than crashing the run or, worse, silently dropping the row.
    raw_phone = frame["cellphone"]
    digits = raw_phone.astype(str).str.replace(r"\D", "", regex=True)
    # Strip a leading US country code so '+1 786...' and '786...' give the same prefix.
    # The "1" matters: without it this stripped the first digit of *any* 11-digit
    # number while serving/schemas.py only strips a leading 1, so the same phone
    # produced a different cellphone_prefix in training than at request time.
    is_us_country_code = (digits.str.len() == 11) & digits.str.startswith("1")
    digits = digits.mask(is_us_country_code, digits.str[1:])
    parsed_phone = _exact_int64_column(digits)
    # Counts unparseable AND out-of-int64 numbers. The previous to_numeric route let a
    # 20-digit value wrap to INT64_MIN while reporting zero repairs.
    repairs["cellphone"] = int(parsed_phone.isna().sum())
    frame["cellphone"] = parsed_phone.where(parsed_phone.notna(), 0).astype("int64")

    # 2. Survey columns must expose the .str accessor where the research code uses it,
    #    which is after a CSV round trip (see stage_for_research_code), not here.
    #
    #    pandas grants `.str` by the values a column holds, not by its dtype: an object
    #    column of integers still refuses it. So `astype("object")`, which is what this
    #    repair used to do, moved the dtype and left the invariant broken - and the
    #    counter reported a repair that had not happened.
    #
    #    A non-string answer is nulled rather than stringified. That is not a loss:
    #    research's own `.str.lower()` yields NaN for a non-string element and its
    #    `fillna("other")` then turns it into 'other', which is exactly what serving's
    #    mirror (_lower_or_other) returns for one. Stringifying would put '12' in
    #    training against 'other' at serve time - a new skew in place of a crash.
    nonstring = 0
    for col in SURVEY_COLUMNS:
        values = frame[col].astype("object")
        is_text = values.map(lambda value: isinstance(value, str))
        nonstring += int((values.notna() & ~is_text).sum())
        frame[col] = values.where(is_text, other=None)
    repairs["survey_nonstring"] = nonstring

    # A column left with no text at all cannot be repaired into one. Its nulls become
    # empty cells in the staged CSV, pandas reads those back as float64, and
    # `.str.lower()` raises on float64 - twenty minutes into the run, from inside the
    # vendored code. Filling a sentinel instead is not value-preserving either: the
    # research code drops rows with fewer than five survey answers *before* it lowers
    # them, so a filled null would keep rows the researcher's pipeline discards and
    # quietly change the training set. An entire survey question arriving empty is an
    # upstream outage, not a row-level repair, so it is refused here - before a bad
    # snapshot reaches Delta, while the previous version is still the one training reads.
    empty = [col for col in SURVEY_COLUMNS if not frame[col].notna().any()]
    if empty:
        raise ValueError(
            "Survey columns arrived with no text answers at all: "
            + ", ".join(empty)
            + ". The research code calls .str.lower() on these, which pandas refuses "
            "on a column it reads back as numeric. Check the upstream extract."
        )

    # 3. payout must be numeric for fillna(0) and the payout > 0 mask.
    payout = pd.to_numeric(frame["payout"], errors="coerce")
    repairs["payout"] = int((payout.isna() & frame["payout"].notna()).sum())
    # inf and -inf survive to_numeric, then reach the payout model and the `payout > 0`
    # mask the research code builds its TabPFN context from. A single one poisons the
    # regressor's target range while every repair counter reports zero.
    not_finite = payout.notna() & ~np.isfinite(payout)
    repairs["payout_not_finite"] = int(not_finite.sum())
    payout = payout.mask(not_finite)
    repairs["payout"] = int(repairs["payout"]) + int(not_finite.sum())
    frame["payout"] = payout

    # 4. Attribution ids must stringify identically in training and in serving.
    #    The research code does `fillna('Other')` then `.astype(str)` on sub1/sub2/sub3.
    #    If the CSV column holds even one null, pandas reads the whole column as
    #    float64, and the cast then produces '1815195.0'. A request carries the same id
    #    as a JSON integer, which becomes '1815195'. Same expression, two different
    #    category levels - so three of the fourteen categorical features miss on every
    #    single request, silently and permanently.
    #    Normalising to a clean string here makes both sides agree.
    #    Reading them as text (READ_AS_TEXT) is what makes the repair possible at all.
    sub_repairs = 0
    for col in SUB_ID_COLUMNS:
        if col not in frame.columns:
            continue
        before = frame[col].astype(str)
        frame[col] = frame[col].map(_as_identifier).astype(object)
        sub_repairs += int((before != frame[col].astype(str)).sum())
    repairs["sub_ids"] = sub_repairs

    # 5. campaign_id is used as a *numeric* feature, and the real ids exceed 2^53
    #    (120227360861540306). A single null in the column makes pandas read it as
    #    float64, at which point that id silently becomes 120227360861540304 in
    #    training while serving sends the exact integer. Casting to int64 keeps the
    #    feature numeric, as the model expects, and keeps the value exact.
    #    Converting from the text form preserves every digit; converting from a float
    #    that pandas already rounded cannot.
    if "campaign_id" in frame.columns:
        campaign = _exact_int64_column(frame["campaign_id"])
        repairs["campaign_id"] = int(campaign.isna().sum())
        frame["campaign_id"] = campaign.where(campaign.notna(), 0).astype("int64")

    # Timestamps are normalised to ISO strings so the Delta round trip reproduces
    # exactly what pd.read_csv would have handed the research code.
    timestamp_fallbacks = 0
    for col in ("session_dt", "conversion_dt", "register_date"):
        parsed, fallback_rows = _to_utc_naive(frame[col])
        # A row the first pass could not read and the second could is a row whose
        # format differs from the rest of its column, and pandas then infers the
        # layout: '06/01/2026' becomes June 1st or January 6th depending on what it
        # decides. session_day and session_day_of_week are model features, so a
        # misread date is a silently wrong feature rather than an error. Counted like
        # every other repair, so a funnel changing its date format shows up here
        # before it shows up in the model.
        timestamp_fallbacks += fallback_rows
        frame[col] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S").where(parsed.notna(), None)

    repairs["timestamp_format_fallbacks"] = timestamp_fallbacks

    # A row with no session timestamp cannot be placed on the train/test timeline.
    before = len(frame)
    frame = frame[frame["session_dt"].notna()].reset_index(drop=True)
    repairs["dropped_no_session_dt"] = before - len(frame)

    return frame, repairs



INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1



def _to_utc_naive(values: pd.Series) -> tuple[pd.Series, int]:
    """Parse timestamps to naive UTC, returning the series and the fallback-row count.

    Two passes, because the fast one is not always right.

    Pass 1 is `utc=True` with pandas' own format inference: it reads one layout for the
    whole column, which is what makes it fast, and it is what a clean export needs.
    A naive value is taken as UTC, which is the assumption serving/schemas.py states -
    so both sides read a timestamp without an offset the same way.

    Pass 2 re-reads only what pass 1 could not. `format="mixed"` parses element by
    element, which is the point: it is the only setting that survives the two things a
    real export does. A column spanning a DST change carries two different offsets, and
    inference picks a single one and NaTs the rest. A column whose rows come from two
    funnels carries two layouts, and inference picks one and NaTs the other. Element-wise
    parsing reads each row on its own terms - the same way serving reads the one
    timestamp in a request - so a row that is readable by itself is never lost.

    `utc=True` on the retry matters as much as the format: without it, pandas returns an
    object column of mixed-offset datetimes, and `.dt` below raises on that. With it,
    every row comes back tz-aware and comparable.

    Pass 2 stays a fallback rather than becoming the only pass because element-wise
    parsing is roughly two orders of magnitude slower, and because a row that needed it
    is a row whose layout differs from its column - reported as a repair, because a
    funnel that changes its date format should surface here and not as a quietly wrong
    session_day feature.
    """
    aware = pd.to_datetime(values, errors="coerce", utc=True)
    missed = aware.isna() & values.notna()
    fallback = 0
    if missed.any():
        retry = pd.to_datetime(values[missed], errors="coerce", utc=True, format="mixed")
        aware = aware.where(~missed, retry)
        fallback = int((aware.notna() & missed).sum())
    return aware.dt.tz_convert("UTC").dt.tz_localize(None), fallback

def _as_int64(value: object) -> int | None:
    """Parse an id to an exact int64, or None if it cannot be one.

    `pd.to_numeric` is the obvious way to do this and it is wrong here. It picks one
    dtype for the whole column, so a single value that does not fit int64 demotes every
    other value to float64 - and 120227360861540306, a real campaign id, comes back as
    120227360861540304. One bad row silently corrupts the column it shares, which is
    the precise failure this gate exists to prevent.

    Parsing value by value keeps every id that is representable exact, and isolates the
    ones that are not so they can be counted as repairs instead of wrapping to
    INT64_MIN.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    # '120227360861540306.0' is the same id as '120227360861540306', and going via
    # float to find that out would round it. Strip the suffix textually instead.
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        text = text[:-2]
    try:
        parsed = int(text)
    except ValueError:
        try:
            # '1.2e17' still has to go through a float; nothing else can read it.
            as_float = float(text)
        except (ValueError, OverflowError):
            return None
        if as_float != as_float or as_float in (float("inf"), float("-inf")):
            return None
        parsed = int(as_float)
    return parsed if INT64_MIN <= parsed <= INT64_MAX else None


def _exact_int64_column(values: pd.Series) -> pd.Series:
    """Parse a column of ids to exact int64, keeping every representable value.

    Built as an object Series on purpose. `Series.map` infers its own dtype, and a
    single None among large integers is enough for it to choose float64 - which
    reintroduces the very rounding this function exists to avoid, one level up from
    `pd.to_numeric`. Holding Python ints in an object column defers the cast until
    after the nulls have been filled, so nothing ever passes through a float.
    """
    parsed = pd.Series([_as_int64(v) for v in values], index=values.index, dtype=object)
    return parsed

def _as_identifier(value: object) -> str | None:
    """Render an attribution id the way a JSON request would: no trailing '.0'.

    Nulls come back as None - pd.NA, np.nan and None all have to land there, or the
    research code's own `fillna('Other')` misses them and the level becomes the string
    '<NA>'. Leaving them null keeps 'Other' meaning exactly what it meant before.
    """
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    # '1815195.0' from a float-inferred column, and '1815195' from text, are the
    # same id and must produce the same category level.
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        text = text[:-2]
    # A numeric id is rendered the way pandas renders it after a CSV round trip:
    # no leading zeros, no leading '+'. This is what makes the rule *canonical* rather
    # than merely shared. The staged CSV (stage_for_research_code) is re-read with
    # pandas' own inference, and an all-digit column comes back as int64 or uint64 - so
    # '007' becomes 7 and the research code's `.astype(str)` yields '7', while a request
    # carrying '007' would keep it. Normalising here means both sides land on '7'.
    # It does collapse '007' and '7' onto one level, which is correct for an
    # attribution id and is in any case what the round trip already did to training.
    stripped = text[1:] if text[:1] in "+-" else text
    if stripped.isdigit():
        sign = "-" if text[:1] == "-" else ""
        return f"{sign}{int(stripped)}"
    return text


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            "Input extract is missing columns the training code selects: "
            + ", ".join(missing)
        )

    # A repeated header is not a missing column, so the check above waves it through.
    # pandas keeps the first copy under its own name and renames the second to
    # `payout.1`, so the gate then reads whichever copy came first. When that is the
    # empty one, every payout becomes null, rows_in equals rows_out and every repair
    # counter reports zero - the regression label silently wiped with nothing to see.
    duplicated = sorted(
        {
            match.group("base")
            for column in frame.columns
            if (match := _MANGLED_DUPLICATE.fullmatch(str(column)))
            and match.group("base") in set(REQUIRED_COLUMNS)
        }
    )
    if duplicated:
        raise ValueError(
            "Input extract repeats columns the training code selects: "
            + ", ".join(duplicated)
            + ". pandas renames the second copy, so the gate cannot tell which one "
            "carries the real values. Fix the extract."
        )


def stage_for_research_code(frame: pd.DataFrame, run_dir: Path) -> tuple[Path, str]:
    """Materialise a Delta snapshot as the CSV the research code expects.

    bl_models_train.py opens `input_path + input_file` with pd.read_csv. Staging the
    snapshot back to CSV keeps that call untouched, and it is also the more faithful
    option: the research code's dtype inference runs on a CSV, exactly as it did
    during research. The write costs a few seconds in a weekly batch job.

    Returns (directory, filename) so the caller can hand them straight to the
    BLPayoutModelsFit constructor.

    One repair has to survive this round trip, and by default it does not. `sanitise`
    normalises sub1/sub2/sub3 to clean strings and leaves their nulls as null, so the
    research code's own `fillna('Other')` keeps meaning what it always meant. But a
    null written to CSV is an empty cell, and `pd.read_csv` then infers float64 for a
    column whose remaining values are all digits - the exact demotion the gate exists
    to undo. `.astype(str)` turns 7448788 into '7448788.0' in training while a request
    carries '7448788', so the feature misses on every single call.

    Applying the research code's own fill here, before the write, keeps the column
    textual: its `fillna('Other')` then finds nothing to do, and both sides produce the
    same level. No value changes - nulls still become 'Other'.

    The fill alone is not enough, because it only helps a column that *has* a null. A
    column of pure digits with none comes back as int64 whatever we write, so the ids
    themselves are canonicalised in `_as_identifier` - which serving imports, so both
    sides use the one rule. The two together are still an argument about pandas'
    inference, so the round trip is then checked rather than assumed: the staged file is
    re-read the way the research code reads it, and any id that does not survive stops
    the run here instead of becoming a categorical level no request can ever match.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    filename = "bl_full_data.csv"
    staged = frame.copy()
    present = [col for col in SUB_ID_COLUMNS if col in staged.columns]
    for col in present:
        staged[col] = staged[col].fillna(RESEARCH_NULL_CATEGORY).astype(str)
    staged.to_csv(run_dir / filename, index=False)
    _verify_sub_ids_survive_the_round_trip(run_dir / filename, staged, present)
    return run_dir, filename


def _verify_sub_ids_survive_the_round_trip(
    path: Path, staged: pd.DataFrame, columns: list[str]
) -> None:
    """Read the staged file as the research code does and compare the attribution ids.

    Only the three id columns are read, so the check costs a fraction of the write it
    follows. `fillna('Other').astype(str)` is the research code's own expression
    (bl_models_train.py, import_preprocess); applying it to both sides compares what the
    model will actually see, not what we hope pandas did.
    """
    if not columns:
        return
    reread = pd.read_csv(path, usecols=columns, low_memory=False)
    for col in columns:
        after = reread[col].fillna(RESEARCH_NULL_CATEGORY).astype(str)
        before = staged[col].reset_index(drop=True)
        differs = before != after.reset_index(drop=True)
        if bool(differs.any()):
            example = before[differs].iloc[0], after[differs.to_numpy()].iloc[0]
            raise ValueError(
                f"{col} does not survive the staged CSV round trip: pandas reads "
                f"{example[0]!r} back as {example[1]!r} ({int(differs.sum())} rows). "
                f"Training would learn a level no request can match."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest bl_full_data.csv into Delta")
    parser.add_argument("--source", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings.load()
    report = ingest(settings, source=args.source)
    print(f"ingested {report.rows_in:,} rows -> {report.rows_out:,} rows")
    print(f"  table    {resolve(settings.paths.delta_table)}")
    print(f"  version  {report.delta_version}")
    for key, count in report.repairs.items():
        if count:
            print(f"  repaired {key}: {count:,}")


if __name__ == "__main__":
    main()
