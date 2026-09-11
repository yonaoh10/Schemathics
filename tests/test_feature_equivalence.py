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
    user["industry"] = rng.choice(schema.INDUSTRIES + [None, "-", "X"])
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
