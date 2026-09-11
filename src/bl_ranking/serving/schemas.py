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

# The shape the research pipeline's pd.to_datetime reads without ambiguity.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class RankRequest(BaseModel):
    """One user's post-funnel data."""

    model_config = ConfigDict(extra="ignore")

    session_dt: str
    conversion_dt: str | None = None
    register_date: str = Field(
        description="Survey submission time. A user without one cannot become a lead."
    )
    campaign_id: int | str
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
    cellphone: int | str | None = None

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
        return int(digits) if digits else 0

    def to_user_data(self) -> dict[str, Any]:
        """The dictionary shape the research predictor consumes."""
        return self.model_dump()


def _normalise_timestamp(value: Any, required: bool) -> str | None:
    """Parse, convert any offset to UTC, and render naive. Raises on unparseable input."""
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = pd.to_datetime(value, errors="coerce")
        if parsed is pd.NaT or (hasattr(parsed, "__len__") is False and pd.isna(parsed)):
            if required:
                raise ValueError(f"{value!r} is not a parseable timestamp")
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
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
