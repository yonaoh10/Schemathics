"""The fast path must produce exactly what the research pipeline produces.

This is the test that licenses serving/fast_features.py. The brief says not to change
the given functions and logic; the fast path is a faster expression of the same
function, and the only honest way to make that claim is to check it - on the worked
example, on randomised well-formed users, and on the malformed ones a real funnel sends.

The research path imports bl_exp_payout_predictor, which builds a NameDataset at module
scope (18 s, 2.4 GB). The tests are marked accordingly and skip cleanly where the
library is absent.
"""

from __future__ import annotations

import copy
import random

import pytest

from bl_ranking.data import schema
from bl_ranking.serving.ranker import BrandRanker, InsufficientSurveyData

pytestmark = [pytest.mark.slow, pytest.mark.needs_names_dataset]


@pytest.fixture(scope="module")
def paths(ranker, gender_lookup):
    """The same warm models behind both feature paths, so only the path differs."""
    pytest.importorskip("names_dataset", reason="the research path needs names-dataset")

    research = BrandRanker(ranker.warm, feature_path="research")
    fast = BrandRanker(ranker.warm, feature_path="fast")
    return research, fast


def _rank_or_marker(ranker, user):
    """Collapse both paths' outcomes into something comparable."""
    try:
        return ("ok", ranker.rank(user))
    except InsufficientSurveyData:
        return ("insufficient", None)
    except Exception as exc:  # noqa: BLE001 - the failure mode must match too
        return ("error", type(exc).__name__)


def _assert_same(left, right, label: str) -> None:
    kind_l, value_l = left
    kind_r, value_r = right
    assert kind_l == kind_r, f"{label}: research={left} fast={right}"
    if kind_l != "ok":
        return
    assert list(value_l) == list(value_r), f"{label}: brand order differs"
    for brand in value_l:
        assert value_l[brand]["rank"] == value_r[brand]["rank"], f"{label}/{brand}"
        assert value_l[brand]["expected_payout"] == pytest.approx(
            value_r[brand]["expected_payout"], abs=1e-9
        ), f"{label}/{brand}"


def test_worked_example_matches(paths, example_user):
    research, fast = paths
    _assert_same(_rank_or_marker(research, example_user),
                 _rank_or_marker(fast, example_user), "worked example")


def _random_user(rng: random.Random, base: dict) -> dict:
    user = copy.deepcopy(base)
    user["credit_score"] = rng.choice([b for b, _ in schema.CREDIT_SCORE_BANDS] + [None])
    user["loan_amount"] = rng.choice([b for b, _ in schema.LOAN_AMOUNT_BANDS] + [None])
    user["monthly_revenue"] = rng.choice([b for b, _ in schema.MONTHLY_REVENUE_BANDS])
    user["time_in_business"] = rng.choice(
        [b for b, _ in schema.TIME_IN_BUSINESS_BANDS] + ["unmapped answer", None])
    # 42 is a non-string survey answer: on the serving broadcast the column is
    # homogeneously numeric, so the research .str accessor raises and the fast path must
    # too. Only reachable here because these users bypass RankRequest's str|None fence.
    user["industry"] = rng.choice(schema.INDUSTRIES + [None, "-", "X", 42])
    user["business_type"] = rng.choice(schema.BUSINESS_TYPES + [None])
    user["loan_reason"] = rng.choice(schema.LOAN_REASONS)
    user["device_type"] = rng.choice(schema.DEVICE_TYPES + [None])
    user["page"] = rng.choice([p for p, _ in schema.PAGES])
    user["auto_country"] = rng.choice(["United States", "Canada", None])
    user["auto_state"] = rng.choice(["Florida", "Texas", None])
    user["auto_city"] = rng.choice(["Miami", "Austin", None])
    user["fname"] = rng.choice(["Michael", "Jennifer", "Rigoberto", "Zzzqq", "", None])
    user["lname"] = rng.choice(["Smith", "Rodriguez", None])
    user["cellphone"] = rng.choice([7869914030, 3055550142, 2010000001, 9999999999])
    user["sub1"] = rng.choice([1121993, None])
    user["sub2"] = rng.choice(["01121993 Ad set", None])
    user["sub3"] = rng.choice([1513124082, None])
    user["campaign_id"] = rng.choice([120227360861540306, 999999999])
    day, hour = rng.randint(1, 28), rng.randint(0, 23)
    user["session_dt"] = f"2026-01-{day:02d} {hour:02d}:{rng.randint(0, 59):02d}:11"
    user["register_date"] = f"2026-01-{day:02d} {hour:02d}:{rng.randint(0, 59):02d}:44"

    # One user in eight abandons most of the survey. Nulling fields independently
    # almost never produces 4+ missing at once, and 4+ is the exact threshold where
    # the research pipeline stops being able to score anyone - the case that must
    # behave identically on both paths.
    if rng.random() < 0.125:
        survey = ["credit_score", "industry", "loan_amount", "loan_reason",
                  "monthly_revenue", "time_in_business", "device_type", "business_type"]
        for column in rng.sample(survey, rng.randint(4, 7)):
            user[column] = None
    return user


def test_randomised_users_match(paths, example_user):
    """Includes users the pipeline legitimately refuses, so the failure modes match too."""
    research, fast = paths
    rng = random.Random(20260211)

    outcomes = {"ok": 0, "insufficient": 0, "error": 0}
    for i in range(400):
        user = _random_user(rng, example_user)
        left = _rank_or_marker(research, user)
        _assert_same(left, _rank_or_marker(fast, user), f"random user {i}")
        outcomes[left[0]] += 1

    # A run that only ever produced rankings would not have tested the refusal paths.
    assert outcomes["ok"] > 200, outcomes
    assert outcomes["insufficient"] > 0, (
        "no user tripped the thresh=5 drop; the generator of random users is too kind"
    )


@pytest.mark.parametrize("survey_answers_present", [4, 3, 0])
def test_sparse_surveys_are_refused_identically(paths, example_user, survey_answers_present):
    research, fast = paths
    survey = [
        "credit_score", "industry", "loan_amount", "loan_reason",
        "monthly_revenue", "time_in_business", "device_type", "business_type",
    ]
    user = dict(example_user)
    for column in survey[survey_answers_present:]:
        user[column] = None
    _assert_same(_rank_or_marker(research, user), _rank_or_marker(fast, user),
                 f"{survey_answers_present} answers")


def test_missing_register_date_is_refused_by_both(paths, example_user):
    research, fast = paths
    user = dict(example_user)
    user["register_date"] = None
    left, right = _rank_or_marker(research, user), _rank_or_marker(fast, user)
    assert left[0] == right[0] == "error"


def test_a_numeric_survey_answer_is_refused_by_both(paths, example_user):
    """One user broadcast to N brands makes a numeric survey column homogeneously numeric,
    so the research `.str` accessor raises rather than yielding NaN - the fast path used to
    return 'other' and answer a full ranking instead, diverging. Both refuse now. Reachable
    only by calling the feature path directly: RankRequest's str|None typing turns this into
    a 422 on both HTTP and pyfunc, so it is the fence, not the equivalence, that held before.
    """
    research, fast = paths
    user = {**example_user, "industry": 42}
    left, right = _rank_or_marker(research, user), _rank_or_marker(fast, user)
    assert left[0] == right[0] == "error", (left, right)


def test_the_two_paths_agree_on_a_span_of_years(paths, example_user):
    """`from_start_to_register` is computed by pandas' `.dt.total_seconds()` in the
    research code, which divides an int64 nanosecond count by 1e9 in float64.
    datetime.timedelta.total_seconds() is exact, so past about 104 days the two answers
    separate - 7258118399.000001 against 7258118399.0 on a span the schema accepts."""
    import pandas as pd

    from bl_ranking.serving import fast_features

    session_dt, register_date = "1970-01-01 00:00:01", "2200-01-01 00:00:00"
    row = fast_features.build_feature_row(
        {**example_user, "session_dt": session_dt, "register_date": register_date}, None)

    research = (pd.Series([pd.to_datetime(register_date)])
                - pd.Series([pd.to_datetime(session_dt)])).dt.total_seconds().iloc[0]
    assert row["from_start_to_register"] == research


def test_a_missing_register_date_is_refused_with_the_same_type_by_both(paths, example_user):
    """The research path raises a bare `Exception` for this (import_preprocess line 67,
    verbatim research code), which `except MissingRegisterDate` in the endpoint cannot
    catch - so with `serving.feature_path = research`, a documented setting, the
    documented 422 came back as a 500. Both paths now refuse it up front, typed."""
    import pytest

    from bl_ranking.serving.fast_features import MissingRegisterDate

    research, fast = paths
    user = {**example_user, "register_date": None}
    for ranker in (research, fast):
        with pytest.raises(MissingRegisterDate):
            ranker.rank(user)


def test_the_band_sentinel_signal_is_the_same_on_both_paths(paths, example_user):
    """It is computed from the shared band_values rather than re-derived, so the metric
    cannot disagree with the numbers the model was given - on either feature path."""
    from bl_ranking.serving import fast_features

    renamed = {**example_user, "credit_score": "Reasonably Good", "loan_amount": "a lot"}
    bands = fast_features.band_values(fast_features.survey_answers(renamed))
    assert set(fast_features.band_sentinels(bands)) == {"credit_score_num",
                                                       "loan_amount_num"}

    # And the built row agrees with it, which is what "cannot disagree" means here.
    row = fast_features.build_feature_row(renamed, None)
    assert set(fast_features.band_sentinels(row)) == set(fast_features.band_sentinels(bands))

    # Both rankers still answer; a sentinel is a signal, not a refusal.
    research, fast = paths
    assert research.rank(renamed) and fast.rank(renamed)
