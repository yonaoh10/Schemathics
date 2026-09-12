"""CSV -> Delta ingestion with the data-quality gate the research code depends on.

The research scripts are treated as fixed. They make assumptions about their input that a
raw warehouse dump does not guarantee:

  1. `cellphone` casts cleanly to int        (bl_models_train.py line 100)
  2. the survey columns support `.str`       (line 110)
  3. `fname` and `lname` support `.str`      (lines 209-210)
  4. `payout` is numeric, and has values     (line 74)
  5. the timestamp columns are text          (line 156)
  6. `campaign_id` and sub1/2/3 mean the same thing here as at serve time

Rather than patch the model code, ingestion enforces those invariants once, at the
boundary, and reports how many rows it had to repair. That keeps the contract explicit
and auditable: the counters travel in the Delta commit and are logged as params by the
training run that reads that version, so a sudden jump in repairs is visible instead of
silently changing a feature.

Three of the invariants cannot be repaired, only refused, and the refusals are as much
the point as the repairs: a survey or name column with no text in it at all, a payout
column with no numbers in it at all, and a timestamp column pandas would read as
nanoseconds. Each of those used to pass the gate with every counter at zero and fail
twenty minutes later from inside the vendored code, where the message named neither the
column nor the extract.

The counters answer two different questions and it is worth keeping them apart.
`timestamp_format_fallbacks` counts rows whose layout differs from the rest of their
column - the case pandas itself notices. `timestamp_ambiguous_layout` counts rows where
the reading is a coin flip (06/01/2026), which is the case pandas does *not* notice:
inference picks one order for the whole column and nothing looks unusual.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

from bl_ranking.config import Settings, resolve
from bl_ranking.data import delta
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

# Columns the research code calls `.str` on outside the survey group: additional_features
# takes `.str.len()` of both (bl_models_train.py lines 209-210).
NAME_COLUMNS = ("fname", "lname")
RESEARCH_NULL_CATEGORY = "Other"

# Where the gate's report lives inside the Delta commit it produced.
COMMIT_METADATA_KEY = "bl_ingest_report"

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

    def as_commit_metadata(self) -> dict[str, str]:
        """The counters, as one JSON value to ride along in the Delta commit.

        Ingestion and training are separate jobs: the weekly schedule runs the gate, and
        the trainer then reads a Delta *version*, not the CSV. Without carrying the
        counters across, they die with the ingest process and no training run can report
        the quality of the rows it trained on - which the module docstring above has
        always claimed it does.

        In the commit rather than in a file beside the table, so it is atomic with the
        version it describes and so it works unchanged on object storage and Unity
        Catalog, where a local sibling directory would not exist.
        """
        return {
            COMMIT_METADATA_KEY: json.dumps({
                "source": Path(self.source).name,
                "rows_in": self.rows_in,
                "rows_out": self.rows_out,
                "repairs": self.repairs,
            }, sort_keys=True)
        }


def read_report(table_uri: str | Path, version: int) -> IngestReport | None:
    """The gate's report for one Delta version, or None if that commit carries none.

    None is an ordinary answer: a snapshot written before this existed, or by something
    other than the gate. The training run then simply carries no ingest.* params.
    """
    raw = delta.commit_metadata(table_uri, version, COMMIT_METADATA_KEY)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    repairs = payload.get("repairs")
    return IngestReport(
        source=str(payload.get("source", "")),
        rows_in=int(payload.get("rows_in", 0)),
        rows_out=int(payload.get("rows_out", 0)),
        delta_version=int(version),
        repairs={str(k): int(v) for k, v in (repairs or {}).items()},
    )


def ingest(settings: Settings | None = None, source: Path | None = None) -> IngestReport:
    """Read the raw CSV, enforce the input contract, write a new Delta version."""
    settings = settings or Settings.load()
    csv_path = resolve(source or settings.paths.raw_csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Locally: drop the real bl_full_data.csv there, or run "
            f"`make data` for a synthetic one. On a job cluster there is no repository and "
            f"no make: point paths.raw_dir (BL_PATHS__RAW_DIR) at the volume holding the "
            f"extract - the Databricks bundle sets it to a Unity Catalog volume - and land "
            f"the file there before the ingest task runs."
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

    report.delta_version = write_snapshot(
        frame, settings.paths.delta_table, commit_metadata=report.as_commit_metadata())
    return report


def sanitise(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Enforce the three input invariants. Returns the frame and the repair counts."""
    frame = frame.copy()
    repairs: dict[str, int] = {}

    # 1. cellphone -> int64. The research code takes the first three digits as a
    #    categorical feature, so anything unparseable becomes 0 (prefix '0') rather
    #    than crashing the run or, worse, silently dropping the row.
    raw_phone = frame["cellphone"]
    # The '.0' comes off before the non-digits do. A float-typed source column writes
    # '13055550142.0', whose digits are '130555501420' - twelve, so the country-code rule
    # below does not fire and the prefix becomes '130' where the same number written as an
    # integer gives '305'. Reading the column as text (READ_AS_TEXT) preserves the
    # spelling; it does not interpret it.
    digits = (raw_phone.astype(str)
              .map(_without_float_suffix)
              .str.replace(r"\D", "", regex=True))
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
    repairs["survey_nonstring"] = _keep_only_text(frame, SURVEY_COLUMNS)
    # fname and lname need exactly the same treatment, and for a while did not have it.
    # additional_features calls `.str.len()` on both (bl_models_train.py lines 209-210),
    # so a name column that is numeric or fully redacted upstream passed the gate with
    # every counter at zero and killed the run inside the vendored code.
    repairs["name_nonstring"] = _keep_only_text(frame, NAME_COLUMNS)

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

    # A payout column with no numbers in it at all is refused, for the same reason an
    # empty survey column is: it cannot be repaired into a usable one, and it fails a long
    # way from here. Every repair counter reports zero (there was nothing to coerce),
    # `_typed_for_delta` then types the all-null column as *string* so the table's schema
    # silently changes, and the run dies half an hour later inside CatBoost with "Labels
    # variable is empty" - which names neither the column nor the extract.
    if not bool(payout.notna().any()):
        raise ValueError(
            "payout arrived with no numeric values at all. It is the regression label "
            "and the `payout > 0` mask the research code builds its TabPFN context from, "
            "so there is nothing to train. Check the upstream extract."
        )

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
    ambiguous_layout = 0
    for col in ("session_dt", "conversion_dt", "register_date"):
        # A numeric timestamp column is refused rather than parsed. pandas reads an
        # integer as nanoseconds since the epoch, so a funnel switching to a YYYYMMDD
        # integer date turns every row into 1970-01-01 - one value for the whole column,
        # no row dropped, no repair counted, and session_day and session_day_of_week are
        # model features. There is no way to tell 20260115 (a date) from 20260115 (a
        # nanosecond count) from inside this function, so it says so instead of picking.
        # An all-empty column arrives as float64 too and is not this: it has no values.
        values = frame[col]
        if pd.api.types.is_numeric_dtype(values) and bool(values.notna().any()):
            raise ValueError(
                f"{col} arrived as a numeric column ({values.dtype}). pandas reads a "
                f"number there as nanoseconds since 1970, so every row would become "
                f"1970-01-01 with nothing to show it. Export it as a timestamp string."
            )
        parsed, fallback_rows = _to_utc_naive(values)
        # A row the first pass could not read and the second could is a row whose
        # format differs from the rest of its column, and pandas then infers the
        # layout: '06/01/2026' becomes June 1st or January 6th depending on what it
        # decides. session_day and session_day_of_week are model features, so a
        # misread date is a silently wrong feature rather than an error. Counted like
        # every other repair, so a funnel changing its date format shows up here
        # before it shows up in the model.
        timestamp_fallbacks += fallback_rows
        # And a separate count for the failure the fallback counter cannot see: a *column*
        # written the other way round. `timestamp_format_fallbacks` counts a row whose
        # layout differs from its column's, which is the case pandas notices; when the
        # whole column is DD/MM/YYYY, inference reads all of it as MM/DD and nothing looks
        # unusual. This counts the rows where the reading is a coin flip, which is the
        # thing an operator can actually watch: it is 0 for an ISO column, and jumps to the
        # whole column the week a funnel changes format.
        ambiguous_layout += _count_ambiguous_dates(frame[col])
        frame[col] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S").where(parsed.notna(), None)

    repairs["timestamp_format_fallbacks"] = timestamp_fallbacks
    repairs["timestamp_ambiguous_layout"] = ambiguous_layout

    # A row with no session timestamp cannot be placed on the train/test timeline.
    before = len(frame)
    frame = frame[frame["session_dt"].notna()].reset_index(drop=True)
    repairs["dropped_no_session_dt"] = before - len(frame)

    # Checked on the rows that survive, which is the only set that matters. Evaluated
    # before the drop, a column whose only text sat on rows the gate was about to discard
    # satisfied it - and the research code then died on `.str.lower()` anyway.
    _require_some_text(frame, (*SURVEY_COLUMNS, *NAME_COLUMNS))

    return frame, repairs


def _keep_only_text(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> int:
    """Null every non-string value in `columns`, and return how many there were.

    pandas grants `.str` by the values a column holds, not by its dtype: an object column
    of integers still refuses it. So `astype("object")`, which is what this used to do,
    moved the dtype and left the invariant broken - and the counter reported a repair that
    had not happened.

    Nulled rather than stringified. Research's own `.str` accessor yields NaN for a
    non-string element, and serving's mirror (_lower_or_other) returns 'other' for one, so
    nulling is what both sides already do with such a value. Stringifying would put '12'
    in training against 'other' at serve time - a new skew in place of a crash.
    """
    nonstring = 0
    for col in columns:
        if col not in frame.columns:
            continue
        values = frame[col].astype("object")
        is_text = values.map(lambda value: isinstance(value, str))
        nonstring += int((values.notna() & ~is_text).sum())
        frame[col] = values.where(is_text, other=None)
    return nonstring


def _require_some_text(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Refuse an extract where one of these columns has no text left in it at all.

    It cannot be repaired into a usable one. Its nulls become empty cells in the staged
    CSV, pandas reads those back as float64, and the research code's `.str.lower()` /
    `.str.len()` raises on float64 - twenty minutes into the run, from inside the vendored
    code. Filling a sentinel is not value-preserving either: the research code drops rows
    with fewer than five survey answers *before* it lowers them, so a filled null would
    keep rows the researcher's pipeline discards and quietly change the training set.

    An entire question or name column arriving empty is an upstream outage, not a
    row-level repair, so it is refused here - before a bad snapshot reaches Delta, while
    the previous version is still the one training reads.
    """
    if frame.empty:
        # Every column is empty when there are no rows, and the caller has a better
        # message for that case (naming the file and the counters that explain it).
        return
    empty = [col for col in columns
             if col in frame.columns and not frame[col].notna().any()]
    if empty:
        raise ValueError(
            "These columns arrived with no text in them at all: "
            + ", ".join(empty)
            + ". The research code calls .str on each of them, which pandas refuses on a "
            "column it reads back as numeric. Check the upstream extract."
        )



INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1



# A date written with slashes or dots and a one- or two-digit leading field. '2026-01-06'
# and '2026/01/06' are not in this shape: a four-digit year first is unambiguous.
_AMBIGUOUS_DATE = re.compile(r"^\s*(\d{1,2})[/.](\d{1,2})[/.]\d{2,4}")


def _count_ambiguous_dates(values: pd.Series) -> int:
    """Rows whose date could honestly be read either way round.

    Both of the first two fields at most 12, and different from each other - so pandas'
    choice between DD/MM and MM/DD changes the answer and neither reading is provably
    wrong. Counted rather than refused, because the extract may genuinely be MM/DD and
    the gate has no way to know; session_day and session_day_of_week are model features,
    so what matters is that the ambiguity is *visible* in the run's parameters instead of
    being resolved silently.
    """
    text = values.dropna().astype(str)
    if text.empty:
        return 0
    parts = text.str.extract(_AMBIGUOUS_DATE).dropna()
    if parts.empty:
        return 0
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    return int(((first <= 12) & (second <= 12) & (first != second)).sum())


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

def _without_float_suffix(text: str) -> str:
    """Drop a trailing '.0' from an otherwise-numeric text.

    A float-typed warehouse column writes a phone number as '13055550142.0', and one
    stripped of its non-digits is '130555501420' - twelve digits, so the country-code rule
    does not fire and the cellphone_prefix feature becomes '130' instead of '305'. The
    same value arriving as a JSON float renders identically, so both sides of the system
    need the one rule; this is it, and both import it.

    Only an exact '.0' is removed. '3055550142.5' is left alone: a fractional phone number
    is junk either way, and the two sides have to agree about junk too.
    """
    if text.endswith(".0") and text[:-2].lstrip("+-").isdigit():
        return text[:-2]
    return text


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
    # A blank id is an absent id. An empty tracking parameter reaches the CSV as an empty
    # cell, which pandas reads as null and the research code's own `fillna('Other')` then
    # turns into 'Other' - and serving's mirror does the same with None. Returning '' here
    # instead put an empty level in training against 'Other' at serve time, and once the
    # staged round trip was being checked it stopped the weekly run outright over a blank
    # sub parameter, which is ordinary attribution data.
    if not text:
        return None
    # '1815195.0' from a float-inferred column, and '1815195' from text, are the
    # same id and must produce the same category level.
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        text = text[:-2]
    # A numeric id is rendered the way pandas renders it after a CSV round trip: no
    # leading zeros, no leading '+', no '.0'. This is what makes the rule *canonical*
    # rather than merely shared. The staged CSV (stage_for_research_code) is re-read with
    # pandas' own inference, and an all-digit column comes back as int64 or uint64 - so
    # '007' becomes 7 and the research code's `.astype(str)` yields '7', while a request
    # carrying '007' would keep it. Normalising here means both sides land on '7'.
    # It does collapse '007' and '7' onto one level, which is correct for an attribution id
    # and is in any case what the round trip already did to training.
    #
    # Decimal, not float: it reads every spelling a SQL DECIMAL or float column exports -
    # '7448788.00', '1e3', '1.2e17' - and reads them exactly, where `int(float(text))`
    # would round a 17-digit id. An earlier version handled a single trailing '.0' and
    # nothing else, so '7448788.00' survived the gate and then stopped the weekly retrain
    # at the round-trip check. A value that is numeric but *not* a whole number is left
    # exactly as it came: truncating '7.5' to '7' would silently change an id.
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    # 'nan' and 'Inf' parse as Decimals and are not whole numbers in any useful sense;
    # Infinity also raises on int(). Left as the text they arrived as, which is what both
    # sides then agree on.
    if not number.is_finite() or number != number.to_integral_value():
        return text
    return str(int(number))


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

    # A header that differs from a required one only in case or surrounding whitespace is
    # the same hazard wearing a different hat. pandas keeps both as distinct columns, so
    # neither the missing-column check nor the mangled-duplicate check above sees anything
    # wrong - and the gate then reads whichever copy is spelled exactly right, which in a
    # warehouse view exporting both a snake_case and a display-cased column is as likely as
    # not the empty one. Trailing whitespace in a header is something CSV exporters do
    # routinely, so this is not an exotic input.
    required = set(REQUIRED_COLUMNS)
    lookalikes = sorted(
        {
            f"{column!r} vs {column.strip().casefold()!r}"
            for column in frame.columns
            if (text := str(column)) not in required
            and text.strip().casefold() in required
        }
    )
    if lookalikes:
        raise ValueError(
            "Input extract has column name(s) that differ from a column the training "
            "code selects only in case or whitespace: " + ", ".join(lookalikes)
            + ". The gate would read the exactly-spelled one, which may not be the one "
            "carrying the values. Fix the extract."
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
