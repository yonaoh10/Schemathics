"""The generated extract must drive the research code the way a real one would.

These tests are the contract between data/generate.py and the research pipeline. If the
generator drifts - a vocabulary changes, a branch stops firing - the models would still
train, silently, on data that exercises less of the code than the real file does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bl_ranking.data import schema
from bl_ranking.serving import fast_features


def test_raw_columns_are_a_superset_of_what_training_selects(raw_csv):
    from bl_ranking.data.ingest import REQUIRED_COLUMNS

    frame = pd.read_csv(raw_csv, low_memory=False, nrows=50)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    assert missing == []


@pytest.mark.parametrize(
    "bands,mapper",
    [
        (schema.CREDIT_SCORE_BANDS, fast_features._credit_score_to_numeric),
        (schema.LOAN_AMOUNT_BANDS, fast_features._loan_amount_to_numeric),
        (schema.MONTHLY_REVENUE_BANDS, fast_features._monthly_revenue_to_numeric),
    ],
)
def test_every_band_maps_to_its_documented_value(bands, mapper):
    """The `str.contains` branches overlap; this pins the precedence.

    'Poor - 550 to 599' matches the '550' rule and then the '550'+'599' rule, and the
    later assignment wins. '$50,000 - $99,999' matches the '9,999' rule (inside
    '99,999') before the correct one overwrites it. Both are load-bearing.
    """
    for label, expected in bands:
        assert mapper(label.lower()) == expected, label


def test_time_in_business_bands_map_exactly():
    for label, expected in schema.TIME_IN_BUSINESS_BANDS:
        assert fast_features.TIME_IN_BUSINESS_MAP.get(label.lower()) == expected, label


def test_research_preprocessing_produces_every_band(ingested, settings):
    """Run the untouched research preprocessing and check each mapping is populated."""
    from bl_ranking.data.delta import read_snapshot
    from bl_ranking.data.ingest import stage_for_research_code
    from bl_ranking.training.trainer import ProductionTrainer, preserve_root_logging

    snapshot = read_snapshot(settings.paths.delta_table)
    run_dir = snapshot.frame.attrs.get("run_dir")
    del run_dir
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path

        stage_dir, stage_file = stage_for_research_code(snapshot.frame, Path(tmp))
        with preserve_root_logging():
            trainer = ProductionTrainer(
                input_path=str(stage_dir) + "/", input_file=stage_file,
                output_predictors_path=tmp + "/", train_test=True,
                payout_cfg=settings.model.payout,
            )
            x_train, y_train, _, _ = trainer.bl_preprocessing()
            trainer.close_log()

    expected_credit = {v for _, v in schema.CREDIT_SCORE_BANDS} | {-99}
    expected_amount = {v for _, v in schema.LOAN_AMOUNT_BANDS}
    expected_revenue = {v for _, v in schema.MONTHLY_REVENUE_BANDS}
    expected_tenure = {float(v) for _, v in schema.TIME_IN_BUSINESS_BANDS} | {0.0}

    assert set(x_train["credit_score_num"].unique()) == expected_credit
    assert set(x_train["loan_amount_num"].unique()) == expected_amount
    assert set(x_train["monthly_revenue_num"].unique()) == expected_revenue
    assert set(x_train["time_in_business_num"].unique()) == expected_tenure

    # Both label classes, and enough of each that CatBoost has something to fit.
    positive_rate = y_train["sold_to_client"].mean()
    assert 0.05 < positive_rate < 0.80, positive_rate
    assert (y_train["payout"] > 0).sum() > 200

    # The 25 model features, in order.
    assert list(x_train.columns) == schema.TRAIN_COLUMNS


def test_labels_exercise_both_client_buy_paths(raw_csv):
    """check_match() and the n_session==1 fallback must both fire on real rows."""
    frame = pd.read_csv(raw_csv, low_memory=False)
    delivered = frame[frame["disposition_source"].notna()]
    sources = delivered["disposition_source"].astype(str)

    assert (sources == delivered["client_name"].astype(str)).any(), "no exact source matches"
    assert sources.str.contains(" api").any(), "no ' api' sources (word-match path)"
    assert (sources == "Lead").any(), "no 'Lead' sources (n_session==1 fallback path)"
    assert sources.isin(["internal", "affiliate network", "partner feed"]).any(), \
        "no unmatched sources - client_buy would never be null"

    n_session = frame.groupby("session_id")["session_dt"].transform("count")
    assert (n_session > 1).any(), "no multi-brand sessions"
    assert (n_session == 1).any()


def test_edge_cases_are_present(raw_csv):
    frame = pd.read_csv(raw_csv, low_memory=False)
    assert frame["register_date"].isna().any(), "no abandoned sessions"
    assert frame["client_name"].isna().any(), "no undelivered rows"
    assert frame["payout"].isna().any()

    survey = ["credit_score", "industry", "loan_amount", "loan_reason",
              "monthly_revenue", "time_in_business", "device_type", "business_type"]
    missing_count = frame[survey].isna().sum(axis=1)
    assert (missing_count.between(1, 2)).any(), "no rows with 1-2 missing answers"
    assert (missing_count >= 3).any(), "no rows that trigger the thresh=5 drop"

    phones = frame["cellphone"].astype(str)
    assert phones.str.contains(r"\D", regex=True).any(), "no unparseable phone numbers"


def test_payout_has_signal_from_credit_and_brand(raw_csv):
    """A generator with no signal would make every evaluation number meaningless."""
    frame = pd.read_csv(raw_csv, low_memory=False)
    paid = frame[frame["payout"].fillna(0) > 0].copy()
    paid["credit_num"] = paid["credit_score"].map(dict(schema.CREDIT_SCORE_BANDS))

    by_credit = paid.groupby("credit_num")["payout"].mean().dropna()
    assert by_credit.index.min() < by_credit.index.max()
    # Better credit must pay materially more, or the classifier has nothing to learn.
    assert by_credit.iloc[-1] > by_credit.iloc[0] * 1.3, by_credit.to_dict()

    by_brand = paid.groupby("client_name")["payout"].mean()
    assert by_brand.max() > by_brand.min() * 2.0, "brands pay indistinguishable amounts"

    assert paid["payout"].between(1.0, 300.0).all(), "payout outside the documented range"


def test_dates_cover_the_window_and_the_research_split_has_seven_buckets(raw_csv):
    frame = pd.read_csv(raw_csv, low_memory=False)
    session_dt = pd.to_datetime(frame["session_dt"])
    assert session_dt.min() >= pd.Timestamp("2025-12-11")
    assert session_dt.max() <= pd.Timestamp("2026-02-12")
    assert (session_dt.max() - session_dt.min()).days >= 55, "window too short"

    # split_by_time (bl_models_train.py lines 213-223) anchors the test window on
    # max(session_dt) rather than on midnight, so its "days" are 24-hour windows offset
    # by the time of the last session. Day 7 is therefore only the minutes between the
    # last session and the same clock time the next day - a sliver by construction, not
    # a data problem. Days 1-6 are the ones that carry weight.
    recent_start = session_dt.max() - pd.Timedelta(days=6)
    split_day = (session_dt - recent_start).dt.days + 1
    split_day = split_day.where(split_day >= 1, 0)
    buckets = split_day[split_day > 0].value_counts()

    assert set(buckets.index) == {1, 2, 3, 4, 5, 6, 7}, sorted(buckets.index)
    assert buckets.loc[[1, 2, 3, 4, 5, 6]].min() >= 20, buckets.to_dict()

    registered = frame["register_date"].notna()
    elapsed = (pd.to_datetime(frame.loc[registered, "register_date"])
               - session_dt[registered]).dt.total_seconds()
    assert (elapsed > 0).all(), "register_date must follow session_dt"
    assert np.isfinite(elapsed).all()
