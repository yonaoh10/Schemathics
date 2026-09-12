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
        params = {
            "ingest.source": Path(self.source).name,
            "ingest.rows_in": self.rows_in,
            "ingest.rows_out": self.rows_out,
            "ingest.delta_version": self.delta_version,
        }
        params.update({f"ingest.repaired.{k}": v for k, v in self.repairs.items()})
        return params


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
    frame = pd.read_csv(csv_path, low_memory=False, dtype=READ_AS_TEXT)
    report = IngestReport(source=str(csv_path), rows_in=len(frame))

    _require_columns(frame)
    frame, repairs = sanitise(frame)
    report.repairs = repairs
    report.rows_out = len(frame)
    report.delta_version = write_snapshot(frame, settings.paths.delta_table)
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

    # 2. Survey columns must expose the .str accessor. A column that happens to be all
    #    null arrives as float64 and would raise on .str.lower().
    coerced = 0
    for col in SURVEY_COLUMNS:
        if not pd.api.types.is_object_dtype(frame[col]):
            frame[col] = frame[col].astype("object")
            coerced += 1
    repairs["survey_dtype"] = coerced

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



def _to_utc_naive(values: pd.Series) -> pd.Series:
    """Parse timestamps to naive UTC, returning the series and the fallback-row count.

    Two passes, because neither alone is correct. Without `utc=True`, a column mixing
    '...+00:00' with '...-05:00' - which is what a DST transition looks like in an
    export - comes back as an object column of datetimes and the `.dt` accessor raises
    on it. With `utc=True`, pandas refuses to mix aware and naive values in one call
    and quietly turns every naive one into NaT, which the row filter below then drops:
    a crash traded for silent data loss.

    So: parse everything as UTC, then re-parse only what that turned into NaT as naive
    and label it UTC. The warehouse assumption is stated in serving/schemas.py and is
    the same on both sides - a timestamp with no offset is UTC.
    """
    aware = pd.to_datetime(values, errors="coerce", utc=True)
    missed = aware.isna() & values.notna()
    fallback = 0
    if missed.any():
        retry = pd.to_datetime(values[missed], errors="coerce")
        if getattr(retry.dtype, "tz", None) is None:
            retry = retry.dt.tz_localize("UTC")
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
        return text[:-2]
    return text


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            "Input extract is missing columns the training code selects: "
            + ", ".join(missing)
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
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    filename = "bl_full_data.csv"
    staged = frame.copy()
    for col in SUB_ID_COLUMNS:
        if col in staged.columns:
            staged[col] = staged[col].fillna(RESEARCH_NULL_CATEGORY).astype(str)
    staged.to_csv(run_dir / filename, index=False)
    return run_dir, filename


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
