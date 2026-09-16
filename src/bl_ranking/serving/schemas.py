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
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_NON_DIGITS = re.compile(r"\D")
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1

# How much of a rejected value an error message may quote.
_SHOWN_CHARS = 80

# The longest string any field may carry, and the tighter bound for the three timestamps.
#
# The request body is capped at serving.max_body_bytes (64 KB), and that bounds the bytes
# but not the work they buy: pd.to_datetime costs about 17 us per byte on a whitespace-
# padded string, so a body sized exactly to the limit with 64 KB of spaces in front of a
# valid date parsed cleanly, returned 200, and held the event loop for ~1 s - four such
# connections stopped a worker answering /healthz. Pydantic runs before the handler, on
# the loop, so the only place to stop it is before the parse. The longest legitimate value
# here is a page path; a timestamp is 19 characters, or 32 with an offset and fractional
# seconds. Both caps are several times that and cost one len() per field.
MAX_TEXT_CHARS = 256
MAX_TIMESTAMP_CHARS = 64


def _shown(value: Any) -> str:
    """A value, rendered short enough to put in an error message.

    Naming the value that was refused is most of what makes a 422 actionable, but the
    value belongs to the caller and its size does not: a 32 MB string in `session_dt`
    became a 32 MB error message and a 33 MB response body, built on the event loop, so
    one request made the worker unavailable for two seconds and cost the sender nothing.
    The size cap on the request body bounds this too now; this keeps the message readable
    and the bound in the one place a reader of the message will look.
    """
    text = repr(value)
    if len(text) <= _SHOWN_CHARS:
        return text
    return f"{text[:_SHOWN_CHARS]}... ({len(text)} characters)"

# The shape the research pipeline's pd.to_datetime reads without ambiguity.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# What pandas can actually hold: datetime64[ns] is an int64 count of nanoseconds since
# 1970. A margin is kept off each end so arithmetic on the value cannot overflow either.
# Funnel timestamps, bounded to a range where every operation on them is defined.
#
# pandas can *hold* 1677..2262, but a timedelta64[ns] spans only ~292 years, so two
# timestamps the schema accepted individually could still overflow when subtracted -
# and `from_start_to_register` subtracts them. The research path raised while the
# vectorised path returned a ranking, which breaks the equivalence contract on input
# neither implementation should have accepted.
#
# The epoch is the floor because these are web-session timestamps: a session before
# 1970 is a malformed field, not a very old lead. The ceiling keeps the whole window
# inside one timedelta span.
# How much of an inverted (register_date, session_dt) pair to forgive. Clock skew between
# two services is ordinary at this scale and is not what the check is for.
_CLOCK_SKEW_TOLERANCE = timedelta(seconds=60)

_TIMESTAMP_MIN = datetime(1970, 1, 1)
_TIMESTAMP_MAX = datetime(2200, 1, 1)


class RankRequest(BaseModel):
    """One user's post-funnel data."""

    model_config = ConfigDict(extra="ignore")

    session_dt: str
    conversion_dt: str | None = None
    # Optional, because "absent" is a documented answer rather than a bad request: a
    # user who never submitted the survey cannot become a lead, and the endpoint says so
    # with a 422 carrying `register_date_absent`. While the field was required, that
    # documented error was unreachable - every such payload came back as a generic
    # validation failure instead, which tells a funnel nothing about why.
    register_date: str | None = Field(
        default=None,
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
    # The default is 0, not None: a `before` validator does not run when the key is
    # absent, so an omitted cellphone stayed None and took down the request inside the
    # research pipeline's `.astype(int)` - a 500 for a field the schema calls optional.
    # `validate_default=True` also fixed that, by running the validator over the default,
    # but pydantic 2.13 warns that the attribute has no effect on a union-typed field
    # wherever it is written, and FastAPI builds exactly such a standalone adapter per
    # field. Defaulting to the value the validator would have produced needs no attribute
    # and no warning. `None` sent explicitly still normalises, because then it is present.
    cellphone: int | str | None = 0

    @model_validator(mode="after")
    def reject_register_before_session(self) -> RankRequest:
        """A user cannot submit the survey materially before the session that showed it.

        `from_start_to_register` is register_date minus session_dt, and every training row
        has it positive: the survey is submitted during the session. An inverted pair
        produced -86,400 seconds behind a 200 - a feature 3.4 million standard deviations
        outside anything the model was fitted on, scored confidently.

        A minute of tolerance rather than zero, because a few seconds of clock skew between
        two services is ordinary and is not what this is for. A day is not skew.
        """
        if self.register_date is None:
            return self
        session = datetime.strptime(self.session_dt, TIMESTAMP_FORMAT)
        register = datetime.strptime(self.register_date, TIMESTAMP_FORMAT)
        if register < session - _CLOCK_SKEW_TOLERANCE:
            raise ValueError(
                f"register_date {self.register_date} is before session_dt {self.session_dt} "
                f"by more than {int(_CLOCK_SKEW_TOLERANCE.total_seconds())}s. The survey "
                f"cannot be submitted before the session that showed it, and the feature "
                f"derived from the two would be negative - which training never sees."
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def reject_unencodable_text(cls, data: Any) -> Any:
        """Refuse a string Python holds but UTF-8 cannot encode.

        `json.loads` accepts the escape `\ud800` and produces a str containing a lone
        UTF-16 surrogate. Nothing downstream can encode it: CatBoost's C++ layer takes
        the whole request down with `SystemError: <class 'UnicodeEncodeError'> returned
        a result with an exception set` - not an exception any handler recognises - and
        FastAPI's own 422 renderer fails the same way while trying to report it, which
        is how a bad string became a 500 on both the valid and the invalid path.

        Checked here, as a model-level `before` validator, so it runs ahead of every
        field validator and covers all 22 fields including the ones only passed through.
        The offending text is named by field and never echoed: repeating it in the error
        body would hit the same encoder that just failed.
        """
        if not isinstance(data, dict):
            return data
        # Only the fields this model reads. `extra="ignore"` means the real warehouse
        # record's other columns - address, business_name, client_id, vertical - are
        # discarded before anything touches them, so scanning those turned a valid
        # 22-field request into a 422 over a value nothing would ever have encoded.
        for key in cls.model_fields:
            value = data.get(key)
            if not isinstance(value, str):
                continue
            # Length first: it is the cheap check, and it is what keeps every later
            # validator's cost bounded. See MAX_TEXT_CHARS.
            if len(value) > MAX_TEXT_CHARS:
                raise ValueError(
                    f"{key} is {len(value):,} characters long; no field here is "
                    f"more than {MAX_TEXT_CHARS}"
                )
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    f"{key} contains text that is not valid UTF-8 "
                    f"(a lone surrogate at position {exc.start})"
                ) from exc
        return data

    @field_validator("session_dt", mode="before")
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

    @field_validator("register_date", mode="before")
    @classmethod
    def normalise_register_date(cls, value: Any) -> str | None:
        """Absent is a data condition; malformed is a bad request.

        None passes through, and `rank` then refuses the user with the documented
        `register_date_absent`. A value that is present and unreadable is a formatting
        problem, and folding it into "absent" would tell the caller their funnel has a
        user who cannot be a lead when what it has is a broken date format.
        """
        if value is None:
            return None
        return _normalise_timestamp(value, required=True)

    @field_validator("conversion_dt", mode="before")
    @classmethod
    def normalise_optional_timestamp(cls, value: Any) -> str | None:
        """conversion_dt is dropped by time_features before any model sees it."""
        if value is None:
            return None
        return _normalise_timestamp(value, required=False)

    @field_validator("sub1", "sub2", "sub3", mode="before")
    @classmethod
    def normalise_attribution_id(cls, value: Any) -> Any:
        """Apply the rule the ingestion gate applies, so both sides agree on the level.

        sub1/sub2/sub3 are categorical features, and the level is whatever string the
        research code's `.astype(str)` produces. data/ingest normalises them so that
        '1815195.0' and '1815195' are the same id; nothing did the same at the request
        boundary, so a caller quoting an id with a trailing '.0' - which is exactly
        what a JSON encoder produces for a float-typed column upstream - scored against
        a level training had never seen.

        Imported from the gate rather than restated here: two copies of a
        normalisation rule are two chances for the sides to drift apart again, which
        is the whole shape of this bug.
        """
        from bl_ranking.data.ingest import _as_identifier

        if value is None:
            return None
        return _as_identifier(value)

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
        # The gate's own parser, imported rather than restated. It reads
        # '120227360861540306.0' exactly - stripping the suffix textually, because the
        # obvious `int(float(text))` rounds it to ...304, which is the precise float64
        # demotion the gate exists to undo, reintroduced at the request boundary. It
        # also applies the int64 bound the training column has, and returns None for a
        # non-finite or unparseable value, which becomes the 0 level here exactly as it
        # does in training. Two copies of this rule would be two chances to drift.
        from bl_ranking.data.ingest import _as_int64
        parsed = _as_int64(value)
        return 0 if parsed is None else parsed

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
        # A JSON caller has no integers: {"cellphone": 13055550142} and
        # {"cellphone": 13055550142.0} are the same payload to a browser, and a
        # float-typed warehouse column writes the same number as the text
        # '13055550142.0'. All three have to give one phone number, so the trailing '.0'
        # comes off with the gate's own rule - imported, not restated, because this is
        # precisely where the two sides last drifted apart.
        from bl_ranking.data.ingest import _without_float_suffix
        digits = _NON_DIGITS.sub("", _without_float_suffix(str(value)))
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
    # A number is refused rather than parsed, which is the same decision the ingestion
    # gate makes about a numeric date column and for the same reason: pandas reads a
    # number here as nanoseconds since 1970, so a funnel sending 20260115 got
    # '1970-01-01 00:00:00' - with session_day, session_day_of_week, session_hour and
    # from_start_to_register all confidently wrong, a 200, and nothing to show it. The
    # epoch is the floor of the accepted range, so the bound below cannot catch it.
    # 20260115 as a date and 20260115 as a nanosecond count are not distinguishable here,
    # so the request is refused instead of guessed at.
    if isinstance(value, bool | int | float):
        raise ValueError(
            f"expected a timestamp string, got the number {_shown(value)}. pandas would read "
            f"it as nanoseconds since 1970; send it as '%Y-%m-%d %H:%M:%S' text."
        )
    # pandas reads 'now' and 'today' as the current time, so a funnel sending either had
    # its own timestamp silently replaced by the server's clock - and session_day,
    # session_day_of_week and session_hour became today's, behind a 200. A timestamp has
    # digits in it; a relative keyword does not.
    if isinstance(value, str) and not any(char.isdigit() for char in value):
        raise ValueError(
            f"{_shown(value)} is not a parseable timestamp. Relative keywords like 'now' and "
            f"'today' are refused for the same reason rather than resolved: pandas would "
            f"read them against this server's clock instead of the funnel's own time."
        )
    # Before pd.to_datetime, which is the expensive call: its cost is linear in the length
    # of the string, whitespace included, and the model-level cap of MAX_TEXT_CHARS is
    # still four times what a timestamp can legitimately need.
    if isinstance(value, str) and len(value) > MAX_TIMESTAMP_CHARS:
        raise ValueError(
            f"{_shown(value)} is {len(value)} characters long; a timestamp is at most "
            f"{MAX_TIMESTAMP_CHARS}"
        )
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = pd.to_datetime(value, errors="coerce")
        if hasattr(parsed, "__len__"):
            raise ValueError(f"expected a single timestamp, got {type(parsed).__name__}")
        if parsed is pd.NaT or pd.isna(parsed):
            if required:
                raise ValueError(f"{_shown(value)} is not a parseable timestamp")
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
            f"{_shown(value)} is outside the representable timestamp range "
            f"({_TIMESTAMP_MIN:%Y-%m-%d}..{_TIMESTAMP_MAX:%Y-%m-%d})"
        )
    return parsed.strftime(TIMESTAMP_FORMAT)


class RankMeta(BaseModel):
    """Which model produced the ranking, and how long it took."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    payout_backend: str
    payout_exact: bool
    # `brands_ranked`, matching the response. The handler renamed it - a brand universe of
    # 15 can rank 14 when one is the sentinel - and this model, which only OpenAPI reads
    # because the handler is `response_model=None`, kept the old name: the published
    # contract required a field no response carried.
    brands_ranked: int
    latency_ms: float


class RankResponse(BaseModel):
    """`ranking` is exactly what the research predictor returns, unmodified."""

    ranking: dict[str, dict[str, float]]
    meta: RankMeta
