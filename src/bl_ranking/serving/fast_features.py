"""Output-identical fast path for the inference feature pipeline.

Why this exists
---------------
Profiling one request (1 user x 15 brands) through the research pipeline gives 60.3 ms
p50 against the production model (81k training rows),
and almost none of it is arithmetic. It is pandas per-operation overhead: a cross join
(3.0 ms), a DataFrame construction from a dict (0.85 ms), four `to_datetime` calls
(0.6 ms each), `Series.apply(lambda: pd.Series(...))` for the gender feature (1.3 ms),
and roughly twenty `.loc[mask, col] = value` assignments across the four band mappings
(0.33 ms each). That cost is per *operation*, not per row, so it does not shrink with
the data - it is simply the price of expressing a 15-row transform as ~100 pandas calls.

On a synchronous funnel endpoint, 60 ms of frame bookkeeping is the entire budget.

What this module is
-------------------
The same transformation, written once over plain Python values and numpy, in the same
order and with the same branch precedence as the research implementation. Every step
below carries the line numbers it mirrors in bl_exp_payout_predictor.py.

It is not an approximation and it is not a reimplementation with "improvements". It is
the same function, and tests/test_feature_equivalence.py asserts that on thousands of
randomised users - including malformed ones - the research pipeline and this one produce
identical feature frames and identical rankings. If they ever diverge, CI fails.

Both paths remain available at runtime (`serving.feature_path`). The research path is
the reference; this one is what production serves.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from bl_ranking.data.schema import TRAIN_COLUMNS

# impute_survey_columns, line 86-87. Order matters for the thresh=5 count.
SURVEY_COLUMNS: tuple[str, ...] = (
    "credit_score", "industry", "loan_amount", "loan_reason",
    "monthly_revenue", "time_in_business", "device_type", "business_type",
)

# At least 5 of the 8 survey answers must be present, or the row is dropped:
# "7 main features - 2 missing answers = 5" (line 89).
SURVEY_MIN_ANSWERS = 5

# import_preprocess, line 60-64.
NEEDED_COLUMNS: tuple[str, ...] = (
    "session_dt", "conversion_dt", "register_date", "campaign_id", "page",
    "auto_city", "auto_country", "auto_state", "device_type", "sub1", "sub2", "sub3",
    "business_type", "credit_score", "industry", "loan_amount", "loan_reason",
    "monthly_revenue", "time_in_business", "fname", "lname", "cellphone",
)

# Columns filled with 'Other' before any other treatment (line 74-75).
FILL_OTHER: tuple[str, ...] = ("country", "state", "city", "sub1", "sub2", "sub3")

# time_in_business_to_num, lines 146-152. Reproduced verbatim, including the revenue
# band that leaked into it upstream.
TIME_IN_BUSINESS_MAP: dict[str, int] = {
    "2+ years": 36,
    "less than 6 months": 3,
    "1-2 years": 18,
    "$75,000 - $99,999": 87500,
    "6-12 months": 8,
}

_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class MissingRegisterDate(ValueError):
    """Mirrors the bare Exception raised at bl_exp_payout_predictor.py line 67."""


class InsufficientSurveyAnswers(ValueError):
    """`dropna(subset=survey_columns, thresh=5)` would drop this user's only row.

    thresh=5 over 8 survey columns means at least 5 answers must be present, i.e. up
    to 3 may be missing. (The comment in the research code says "7 main features - 2
    missing answers = 5"; the subset is actually 8 columns, so the real tolerance is 3.)
    """


def require_survey_answers(user: dict[str, Any]) -> None:
    """Refuse a user the research pipeline could not score, before either path runs."""
    answered = sum(1 for col in SURVEY_COLUMNS if not _is_na(user.get(col)))
    if answered < SURVEY_MIN_ANSWERS:
        raise InsufficientSurveyAnswers(
            f"only {answered} of {len(SURVEY_COLUMNS)} survey answers present, "
            f"{SURVEY_MIN_ANSWERS} required"
        )


def build_features(user: dict[str, Any], brands: np.ndarray,
                   gender_lookup) -> pd.DataFrame:
    """Return the 25 training features for one user against every brand.

    `brands` is the brand universe with 'other' already removed. `gender_lookup` is any
    object exposing `.lookup(name) -> (gender, confidence)`.

    The user-level features are computed once and broadcast, because the research
    pipeline's cross join makes every column except client_name identical across rows -
    a fact worth exploiting, and one that changes nothing about the values produced.
    """
    row = _user_row(user, gender_lookup=gender_lookup)
    n = len(brands)

    # Assemble in TRAIN_COLUMNS order so the frame can be handed straight to the models.
    data: dict[str, Any] = {}
    for column in TRAIN_COLUMNS:
        if column == "client_name":
            data[column] = brands
        else:
            value = row[column]
            data[column] = np.full(n, value, dtype=object if isinstance(value, str) else None)
    return pd.DataFrame(data, columns=list(TRAIN_COLUMNS))


def build_feature_row(user: dict[str, Any], gender_lookup) -> dict[str, Any]:
    """The single-user feature dict, exposed for tests and for the numpy scoring path."""
    return _user_row(user, gender_lookup=gender_lookup)


def _user_row(user: dict[str, Any], gender_lookup=None) -> dict[str, Any]:
    """Every feature except client_name, for one user."""
    missing = [k for k in NEEDED_COLUMNS if k not in user]
    if missing:
        raise KeyError(f"request is missing required fields: {', '.join(missing)}")

    # import_preprocess line 66-67: a user who never submitted the survey cannot be a lead.
    if _is_na(user["register_date"]):
        raise MissingRegisterDate("user cannot be a lead - register_date is absent")

    # -- import_preprocess, lines 72-82 --------------------------------------------
    renamed = {
        "city": user["auto_city"],
        "state": user["auto_state"],
        "country": user["auto_country"],
        "sub1": user["sub1"],
        "sub2": user["sub2"],
        "sub3": user["sub3"],
    }
    filled = {k: ("Other" if _is_na(v) else v) for k, v in renamed.items()}
    country_state = filled["state"] if filled["country"] == "United States" else filled["country"]

    session_dt = _to_datetime(user["session_dt"])
    register_date = _to_datetime(user["register_date"])

    # astype(str) happens after the fillna above, so 'Other' can reach these.
    sub1, sub2, sub3 = (str(filled["sub1"]), str(filled["sub2"]), str(filled["sub3"]))
    cellphone_prefix = str(int(user["cellphone"]))[:3]

    # -- impute_survey_columns, lines 88-95 ----------------------------------------
    require_survey_answers(user)
    survey = {col: _lower_or_other(user[col]) for col in SURVEY_COLUMNS}

    # -- the four band mappings, lines 98-155 --------------------------------------
    credit_score_num = _credit_score_to_numeric(survey["credit_score"])
    loan_amount_num = _loan_amount_to_numeric(survey["loan_amount"])
    monthly_revenue_num = _monthly_revenue_to_numeric(survey["monthly_revenue"])
    # map(...).fillna(0) - an unmapped answer becomes 0, "impute worse case".
    time_in_business_num = float(TIME_IN_BUSINESS_MAP.get(survey["time_in_business"], 0))

    # -- time_features, lines 159-162 ----------------------------------------------
    if pd.isna(session_dt):
        session_day: Any = np.nan
        session_day_of_week: Any = np.nan
        session_hour: Any = np.nan
    else:
        session_day = session_dt.day
        session_day_of_week = _DAY_NAMES[session_dt.weekday()]
        session_hour = session_dt.hour
    # `.value / 1e9`, not `.total_seconds()`. The research code computes this with
    # pandas' `.dt.total_seconds()`, which divides an int64 nanosecond count by 1e9 in
    # float64; datetime.timedelta.total_seconds() computes it exactly. Past about 104
    # days the nanosecond count passes float64's 53-bit mantissa and the two answers
    # separate - 7258118399.000001 against 7258118399.0 on a span the schema accepts.
    # The research path is the reference, so this mirrors its arithmetic rather than
    # improving on it: a feature the two paths compute differently is a broken
    # equivalence contract whichever value is nearer the truth.
    from_start_to_register = (
        np.nan if (pd.isna(session_dt) or pd.isna(register_date))
        else pd.Timedelta(register_date - session_dt).value / 1e9
    )

    # -- additional_features, lines 187-192 ----------------------------------------
    ratio = loan_amount_num / monthly_revenue_num
    fname, lname = user["fname"], user["lname"]
    gender = _gender(fname, gender_lookup)
    return {
        "campaign_id": user["campaign_id"],
        "page": user["page"],
        "city": filled["city"],
        "device_type": survey["device_type"],
        "sub1": sub1,
        "sub2": sub2,
        "sub3": sub3,
        "business_type": survey["business_type"],
        "industry": survey["industry"],
        "loan_reason": survey["loan_reason"],
        "cellphone_prefix": cellphone_prefix,
        "country_state": country_state,
        "credit_score_num": credit_score_num,
        "loan_amount_num": loan_amount_num,
        "monthly_revenue_num": monthly_revenue_num,
        "time_in_business_num": time_in_business_num,
        "session_day": session_day,
        "session_day_of_week": session_day_of_week,
        "session_hour": session_hour,
        "from_start_to_register": from_start_to_register,
        "ratio_loan_amount_revenue": ratio,
        "gender": gender,
        "fname_len": _str_len(fname),
        "lname_len": _str_len(lname),
    }


# --------------------------------------------------------------------------------- #
# Element-wise equivalents of the pandas expressions
# --------------------------------------------------------------------------------- #

def _to_datetime(value: Any) -> Any:
    """`pd.to_datetime(value, errors="coerce")` for one scalar, without the overhead.

    A single scalar through pandas costs ~0.6 ms because it goes through array
    inference. Funnel timestamps are ISO-like, so `datetime.fromisoformat` handles
    almost all of them in microseconds; anything it rejects falls back to pandas so the
    coercion semantics (including NaT) stay identical.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    elif isinstance(value, datetime):
        return value
    return pd.to_datetime(value, errors="coerce")


def _is_na(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # arrays and other non-scalars are never NA here
        return False


def _lower_or_other(value: Any) -> str:
    """`.str.lower()`, then `.where(len > 1, nan)`, then `.fillna('other')`.

    The `.str` accessor yields NaN for any non-string element, so a numeric survey
    answer becomes 'other' rather than being stringified - reproduced here.
    """
    if not isinstance(value, str):
        return "other"
    lowered = value.lower()
    return lowered if len(lowered) > 1 else "other"


def _credit_score_to_numeric(value: str) -> int:
    """credit_score_to_numeric, lines 99-107. Later branches overwrite earlier ones."""
    result = -99
    if "550" in value:
        result = 501
    if "550" in value and "599" in value:
        result = 551
    if "600" in value and "649" in value:
        result = 601
    if "650" in value and "719" in value:
        result = 651
    if "720" in value:
        result = 721
    return result


def _loan_amount_to_numeric(value: str) -> int:
    """credit_loan_amount_to_numeric, lines 113-124."""
    result = -99
    if "10,000" in value and "24,999" in value:
        result = 17500
    if "25,000" in value and "49,999" in value:
        result = 47500
    if "50,000" in value and "74,999" in value:
        result = 57500
    if "75,000" in value and "99,999" in value:
        result = 75000
    if "100,000" in value:
        result = 150000
    if "200,000" in value:
        result = 250000
    return result


def _monthly_revenue_to_numeric(value: str) -> int:
    """monthly_revenue_to_numeric, lines 130-140."""
    result = -99
    if "9,999" in value:
        result = 5000
    if "10,000" in value and "19,999" in value:
        result = 15000
    if "20,000" in value and "49,999" in value:
        result = 35000
    if "50,000" in value and "99,999" in value:
        result = 75000
    if "100,000" in value:
        result = 150000
    if "200,000" in value:
        result = 250000
    return result


def _gender(fname: Any, gender_lookup) -> str:
    """`fname.apply(lambda x: detect(str(x).strip().capitalize()))`, line 189-190.

    `str(x)` is applied before the lookup, so a missing name becomes the literal
    'Nan' or 'None' and is looked up as such. Kept, because the research code does it
    and the table was built over the same key space.
    """
    key = str(fname).strip().capitalize()
    if gender_lookup is None:
        return "unknown"
    return gender_lookup.lookup(key)[0]


def _str_len(value: Any) -> Any:
    """`.str.len()` - NaN for anything that is not a string."""
    return len(value) if isinstance(value, str) else np.nan


def rank_from_scores(brands: np.ndarray, prob_lead: np.ndarray,
                     payout: np.ndarray) -> dict[str, dict[str, float]]:
    """prediction_expected_payout, lines 223-234, in numpy.

    Keeps the same three behaviours: the 0.01 floor, the descending sort, and ranks
    assigned by first occurrence so ties follow sort order.
    """
    expected = payout * prob_lead
    keep = expected > 0.01
    if not keep.any():
        return {}

    kept_brands = brands[keep]
    kept_expected = expected[keep]
    # 'first' ranking over a descending sort == the order produced by a stable argsort
    # of the negated scores.
    order = np.argsort(-kept_expected, kind="stable")
    return {
        str(kept_brands[idx]): {
            "rank": float(position + 1),
            "expected_payout": float(kept_expected[idx]),
        }
        for position, idx in enumerate(order)
    }
