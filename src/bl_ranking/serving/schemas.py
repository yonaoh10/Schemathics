"""Request and response models for the ranking endpoint.

The request schema is the 22-key dictionary the research predictor expects, taken from
the worked example at the bottom of bl_exp_payout_predictor.py. Validating it here does
two things the research code cannot do on its own:

  * turns a malformed payload into a 422 with a useful message, instead of a KeyError
    or a ValueError from deep inside pandas;
  * normalises `cellphone` to an integer. The research feature pipeline does
    `cellphone.astype(int)`, so '(305) 555-0142' from a real funnel would take down the
    request. The same normalisation runs at ingestion for training data, which keeps
    the feature consistent on both sides.

Nothing here changes a feature value for a well-formed request.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

_NON_DIGITS = re.compile(r"\D")
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1

# The shape the research pipeline's pd.to_datetime reads without ambiguity.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# What pandas can actually hold: datetime64[ns] is an int64 count of nanoseconds since
# 1970. A margin is kept off each end so arithmetic on the value cannot overflow either.
# Whole days, so the conversion carries no sub-microsecond remainder to discard.
_TIMESTAMP_MIN = (pd.Timestamp.min + pd.Timedelta(days=1)).floor("D").to_pydatetime()
_TIMESTAMP_MAX = (pd.Timestamp.max - pd.Timedelta(days=1)).floor("D").to_pydatetime()


class RankRequest(BaseModel):
    """One user's post-funnel data."""

    model_config = ConfigDict(extra="ignore")

    session_dt: str
    conversion_dt: str | None = None
    register_date: str = Field(
        description="Survey submission time. A user without one cannot become a lead."
    )
    campaign_id: int | str = Field(validate_default=True)
    page: str
    auto_city: str | None = None
    auto_country: str | None = None
    auto_state: str | None = None
    device_type: str | None = None
    sub1: int | str | None = None
    sub2: int | str | None = None
    sub3: int | str | None = None
    business_type: str | None = None
    credit_score: str | None = None
    industry: str | None = None
    loan_amount: str | None = None
    loan_reason: str | None = None
    monthly_revenue: str | None = None
    time_in_business: str | None = None
    fname: str | None = None
    lname: str | None = None
    # validate_default: a `before` validator does not run when the key is absent,
    # so without this an omitted cellphone stayed None and took down the request
    # inside the research pipeline's `.astype(int)` - a 500 for a field the
    # schema calls optional.
    cellphone: int | str | None = Field(default=None, validate_default=True)

    @field_validator("session_dt", "register_date", mode="before")
    @classmethod
    def normalise_required_timestamp(cls, value: Any) -> str:
        """Reject what the pipeline cannot use, and make offsets explicit.

        Two failures this prevents:

        * An unparseable `session_dt` becomes NaT, which makes `session_day_of_week`
          null, which CatBoost rejects with "cat_features must be integer or string"
          deep inside scoring. In training the same row is silently dropped by
          `split_by_time`, so the two sides disagree about what is even scoreable.
          Better to say so at the door.
        * A timezone-aware `session_dt` with a naive `register_date` raises
          "Cannot subtract tz-naive and tz-aware datetime-like objects" when
          `from_start_to_register` is computed.

        Assumption, stated because it is not verifiable from the code alone: warehouse
        timestamps are naive UTC, so an offset-carrying input is converted to UTC and
        the offset dropped. `session_hour` is a model feature, so a wrong assumption
        here is a systematic shift rather than an error - if the funnel logs local
        time instead, change this one function.
        """
        return _normalise_timestamp(value, required=True)

    @field_validator("conversion_dt", mode="before")
    @classmethod
    def normalise_optional_timestamp(cls, value: Any) -> str | None:
        """conversion_dt is dropped by time_features before any model sees it."""
        if value is None:
            return None
        return _normalise_timestamp(value, required=False)

    @field_validator("campaign_id", mode="before")
    @classmethod
    def normalise_campaign_id(cls, value: Any) -> int:
        """Make it the numeric feature the model was fitted on.

        `campaign_id` is a *numeric* column in training: data/ingest.sanitise does
        `pd.to_numeric(errors="coerce").fillna(0).astype("int64")`. The schema accepts a
        string because a JSON caller may quote a 17-digit id to protect it from a
        float, but an unparseable one used to travel all the way to CatBoost and come
        back as a 500 carrying a library error message. Coercing here applies the same
        rule the training data got, so a junk id degrades to the 0 level on both sides
        instead of failing the request.
        """
        if value is None:
            return 0
        text = str(value).strip()
        try:
            return int(text)
        except ValueError:
            try:
                # '1.2e17' and '120227360861540306.0' both appear in real extracts.
                return int(float(text))
            except (ValueError, OverflowError):
                return 0

    @field_validator("cellphone", mode="before")
    @classmethod
    def normalise_cellphone(cls, value: Any) -> int:
        """Strip formatting and the US country code, matching data/ingest.sanitise.

        Unparseable numbers become 0, which yields the '0' prefix feature rather than
        a 500. The prefix is one weak categorical among 25 features; refusing to rank a
        real user over a badly formatted phone number would cost far more.
        """
        if value is None:
            return 0
        digits = _NON_DIGITS.sub("", str(value))
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if not digits:
            return 0
        number = int(digits)
        # The research pipeline does cellphone.astype(int) into an int64 column, so a
        # number too large for one raised OverflowError there while the vectorised path
        # scored it happily. Treat it as unparseable, which is what the ingestion gate
        # now does with the same value.
        if not (_INT64_MIN <= number <= _INT64_MAX):
            return 0
        return number

    def to_user_data(self) -> dict[str, Any]:
        """The dictionary shape the research predictor consumes."""
        return self.model_dump()


def _normalise_timestamp(value: Any, required: bool) -> str | None:
    """Parse, convert any offset to UTC, and render naive. Raises on unparseable input.

    A list or dict reaching pd.to_datetime comes back as a DatetimeIndex rather than a
    scalar, and `.strftime` on that returns an array. Pydantic then cannot render the
    field, and FastAPI's own validation-error encoder raises while trying to report the
    problem - turning a malformed payload into a 500 that the endpoint's error handler
    never sees. Reject non-scalars at the top so they become an ordinary 422.
    """
    if isinstance(value, list | tuple | dict | set):
        raise ValueError(f"expected a timestamp string, got {type(value).__name__}")
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = pd.to_datetime(value, errors="coerce")
        if hasattr(parsed, "__len__"):
            raise ValueError(f"expected a single timestamp, got {type(parsed).__name__}")
        if parsed is pd.NaT or pd.isna(parsed):
            if required:
                raise ValueError(f"{value!r} is not a parseable timestamp")
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    # pandas represents timestamps as nanoseconds since 1970 in an int64, so anything
    # outside roughly 1677..2262 has no representation. The research feature path dies
    # inside CatBoost on such a row while the vectorised path scores it and returns a
    # confident ranking - the two paths are contractually identical, so the input has
    # to be refused at the door rather than resolved differently by each.
    naive = parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
    if not (_TIMESTAMP_MIN <= naive <= _TIMESTAMP_MAX):
        raise ValueError(
            f"{value!r} is outside the representable timestamp range "
            f"({_TIMESTAMP_MIN:%Y-%m-%d}..{_TIMESTAMP_MAX:%Y-%m-%d})"
        )
    return parsed.strftime(TIMESTAMP_FORMAT)


class RankMeta(BaseModel):
    """Which model produced the ranking, and how long it took."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    payout_backend: str
    payout_exact: bool
    n_brands: int
    latency_ms: float


class RankResponse(BaseModel):
    """`ranking` is exactly what the research predictor returns, unmodified."""

    ranking: dict[str, dict[str, float]]
    meta: RankMeta
