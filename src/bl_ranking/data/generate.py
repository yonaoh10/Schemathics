"""Synthetic bl_full_data.csv generator.

The real extract lives in a private Drive folder, so the repository ships a generator
instead. It is not a placeholder: the goal is a file that drives every branch of
bl_models_train.py the way the real one does, so that training, evaluation, artifact
production and serving can all be exercised end to end.

Three properties matter and are tested in tests/test_generated_data.py:

  1. Vocabulary fidelity - every survey band maps to the numeric value recorded in
     schema.py, including the overlapping `str.contains` branches.
  2. Label mechanics - check_match(), the n_session==1 fallback, payout_adj and
     sold_to_client all fire on a realistic mix of rows.
  3. Real signal - acceptance and payout genuinely depend on the survey answers and on
     the brand, so a model fitted here has something to learn and the evaluation
     numbers mean something.

If a real extract is present at paths.raw_csv the generator is skipped entirely.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bl_ranking.config import Settings, resolve
from bl_ranking.data import schema


@dataclass(frozen=True)
class Brand:
    """One lender. `share` is traffic weight; the rest define who it accepts and pays.

    accept_logit is a linear model over the standardised user profile: a brand with a
    high credit_w only takes good credit, a brand with a high revenue_w only takes
    large businesses. That is the structure the CatBoost classifier has to recover.
    """

    name: str
    share: float
    base_payout: float
    bias: float
    credit_w: float
    revenue_w: float
    tenure_w: float
    amount_w: float
    preferred_industries: tuple[str, ...] = ()


# Fifteen brands: a realistic universe size for this vertical and, more importantly,
# the number of rows one inference request scores (1 user x N brands).
BRANDS: list[Brand] = [
    Brand("fundera / nerdwallet", 0.130, 78.0, -0.55, 1.15, 0.85, 0.40, 0.30, ("professional services", "technology")),
    Brand("businessloans.com", 0.115, 54.0, -0.35, 0.55, 0.60, 0.25, 0.35),
    Brand("lendio", 0.105, 66.0, -0.7, 0.95, 1.05, 0.55, 0.25, ("construction", "transportation")),
    Brand("national funding", 0.090, 47.0, -0.45, 0.40, 0.75, 0.70, 0.20),
    Brand("ondeck", 0.085, 92.0, -1.0, 1.35, 1.20, 0.80, 0.45),
    Brand("bluevine", 0.075, 61.0, -0.75, 1.00, 0.95, 0.35, 0.30, ("retail", "technology")),
    Brand("credibly", 0.070, 38.0, -0.2, 0.15, 0.35, 0.20, 0.15),
    Brand("rapid finance", 0.065, 44.0, -0.4, 0.30, 0.55, 0.30, 0.25, ("restaurant", "automotive")),
    Brand("xlt", 0.055, 22.0, 0.05, -0.20, 0.10, 0.05, 0.10),
    Brand("forward funding", 0.050, 119.0, -1.45, 1.60, 1.45, 1.05, 0.60),
    Brand("sba central", 0.045, 143.0, -1.75, 1.70, 1.30, 1.55, 0.85, ("real estate", "manufacturing")),
    Brand("clarify capital", 0.040, 57.0, -0.65, 0.70, 0.80, 0.45, 0.30),
    Brand("fora financial", 0.030, 71.0, -0.85, 0.85, 1.10, 0.60, 0.40, ("healthcare", "manufacturing")),
    Brand("uplyft capital", 0.025, 31.0, -0.1, 0.05, 0.25, 0.15, 0.10),
    Brand("smb compass", 0.020, 104.0, -1.3, 1.25, 1.35, 0.95, 0.70, ("agriculture", "construction")),
]

# Deliberately common US names: the research code derives gender and name length from
# them via names-dataset, so made-up strings would collapse that feature to 'unknown'.
FIRST_NAMES: list[str] = [
    "Michael", "Jennifer", "Christopher", "Maria", "David", "Lisa", "James", "Michelle",
    "Robert", "Ashley", "John", "Jessica", "Daniel", "Sarah", "Carlos", "Amanda",
    "Anthony", "Melissa", "Jose", "Stephanie", "William", "Nicole", "Kevin", "Elizabeth",
    "Brian", "Heather", "Jason", "Tiffany", "Rigoberto", "Angela", "Marcus", "Danielle",
    "Steven", "Rebecca", "Eric", "Laura", "Andrew", "Kimberly", "Juan", "Patricia",
    "Tyrone", "Yolanda", "Dmitri", "Svetlana", "Rajesh", "Priya", "Ahmed", "Fatima",
]
LAST_NAMES: list[str] = [
    "Smith", "Johnson", "Williams", "Rodriguez", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
    "Nguyen", "Patel", "Kim", "Okafor", "Petrov", "Al-Rashid", "Oconnell", "Duboisson",
]

# Population-weighted-ish US states, plus a little non-US traffic so the
# country_state branch (state when United States, country otherwise) is exercised.
STATES: list[tuple[str, float]] = [
    ("California", 0.118), ("Texas", 0.098), ("Florida", 0.091), ("New York", 0.070),
    ("Pennsylvania", 0.043), ("Illinois", 0.041), ("Ohio", 0.039), ("Georgia", 0.036),
    ("North Carolina", 0.034), ("Michigan", 0.031), ("New Jersey", 0.029),
    ("Virginia", 0.027), ("Washington", 0.026), ("Arizona", 0.025), ("Tennessee", 0.024),
    ("Massachusetts", 0.023), ("Indiana", 0.021), ("Missouri", 0.020), ("Maryland", 0.019),
    ("Colorado", 0.019), ("Wisconsin", 0.018), ("Minnesota", 0.017), ("Louisiana", 0.016),
    ("Alabama", 0.015), ("South Carolina", 0.015), ("Kentucky", 0.014), ("Oregon", 0.013),
    ("Oklahoma", 0.012), ("Connecticut", 0.011), ("Nevada", 0.011), ("Utah", 0.010),
    ("Arkansas", 0.009), ("Mississippi", 0.009), ("Kansas", 0.008), ("New Mexico", 0.008),
    ("Nebraska", 0.007), ("Idaho", 0.007), ("West Virginia", 0.006), ("Hawaii", 0.006),
    ("Maine", 0.005), ("Rhode Island", 0.005), ("Montana", 0.004), ("Delaware", 0.004),
    ("Alaska", 0.003), ("Vermont", 0.003), ("Wyoming", 0.003),
]

CITIES_BY_STATE: dict[str, list[str]] = {
    "California": ["Los Angeles", "San Diego", "San Jose", "Fresno", "Sacramento", "Oakland"],
    "Texas": ["Houston", "San Antonio", "Dallas", "Austin", "Fort Worth", "El Paso"],
    "Florida": ["Miami", "Jacksonville", "Tampa", "Orlando", "Fort Lauderdale", "Hialeah"],
    "New York": ["New York", "Buffalo", "Rochester", "Yonkers", "Syracuse"],
}
GENERIC_CITIES: list[str] = [
    "Springfield", "Riverside", "Fairview", "Georgetown", "Salem", "Madison",
    "Clinton", "Franklin", "Greenville", "Bristol", "Ashland", "Dover",
]

# Roughly 3% of sessions arrive from outside the US; country_state then falls back
# to the country name.
FOREIGN_COUNTRIES: list[str] = ["Canada", "United Kingdom", "Australia", "India", "Philippines"]

# Hour-of-day weights for a US consumer funnel: quiet overnight, business-hours peak,
# secondary evening bump. session_hour is a model feature, so a flat curve would be wrong.
HOUR_WEIGHTS = np.array([
    0.6, 0.4, 0.3, 0.3, 0.4, 0.7, 1.2, 2.0, 3.2, 4.4, 5.1, 5.4,
    5.3, 5.2, 5.4, 5.3, 4.9, 4.4, 4.0, 3.7, 3.3, 2.6, 1.8, 1.1,
])


def generate(settings: Settings | None = None, sessions: int | None = None,
             seed: int | None = None) -> pd.DataFrame:
    """Build the raw session-by-brand table. One row per (session, brand delivered)."""
    settings = settings or Settings.load()
    cfg = settings.generator
    n_sessions = int(sessions or cfg.sessions)
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)

    sess = _sessions(rng, n_sessions, cfg.start_date, cfg.end_date)
    rows = _explode_to_brand_rows(rng, sess)
    rows = _apply_labels(rng, rows)
    rows = _inject_edge_cases(rng, rows)

    rows = rows.sort_values("session_dt", kind="stable").reset_index(drop=True)
    return rows[schema.RAW_COLUMNS]


# --------------------------------------------------------------------------------- #
# Session-level attributes
# --------------------------------------------------------------------------------- #

def _sessions(rng: np.random.Generator, n: int, start: str, end: str) -> pd.DataFrame:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    span_days = (end_ts - start_ts).days

    # Traffic grows mildly over the window and dips at weekends - enough non-stationarity
    # that the time-based train/test split is a meaningful simulation of a production run.
    day_index = np.arange(span_days + 1)
    trend = 1.0 + 0.004 * day_index
    weekday = (start_ts + pd.to_timedelta(day_index, unit="D")).dayofweek.to_numpy()
    weekend = np.where(weekday >= 5, 0.72, 1.0)
    day_weights = trend * weekend
    day_weights /= day_weights.sum()

    days = rng.choice(day_index, size=n, p=day_weights)
    hours = rng.choice(24, size=n, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
    minutes, seconds = rng.integers(0, 60, n), rng.integers(0, 60, n)
    session_dt = (start_ts + pd.to_timedelta(days, unit="D")
                  + pd.to_timedelta(hours, unit="h")
                  + pd.to_timedelta(minutes, unit="m")
                  + pd.to_timedelta(seconds, unit="s"))

    # Survey answers. Sampled with a shared latent "business strength" so credit,
    # revenue and tenure correlate the way they do in reality.
    strength = rng.normal(0.0, 1.0, n)
    credit = _weighted_band(rng, schema.CREDIT_SCORE_BANDS, strength, slope=0.85)
    revenue = _weighted_band(rng, schema.MONTHLY_REVENUE_BANDS, strength, slope=0.95)
    tenure = _weighted_band(rng, schema.TIME_IN_BUSINESS_BANDS, strength, slope=0.60,
                            tail_weight=0.004)  # the '$75,000 - $99,999' artefact row
    # Loan size tracks revenue but with real spread - a small shop asking for $200k is
    # exactly the kind of mismatch the ranking has to price.
    amount = _weighted_band(rng, schema.LOAN_AMOUNT_BANDS, strength + rng.normal(0, 0.9, n), slope=0.70)

    states = rng.choice([s for s, _ in STATES], size=n,
                        p=_normalised([w for _, w in STATES]))
    is_foreign = rng.random(n) < 0.03
    countries = np.where(is_foreign,
                         rng.choice(FOREIGN_COUNTRIES, size=n),
                         "United States")
    # Vectorised per state rather than a per-row choice: at 60k sessions the loop
    # version dominates generation time for no benefit.
    cities = np.empty(n, dtype=object)
    for state in set(states):
        mask = states == state
        pool = CITIES_BY_STATE.get(state, GENERIC_CITIES)
        cities[mask] = rng.choice(pool, size=int(mask.sum()))

    fname = rng.choice(FIRST_NAMES, size=n)
    lname = rng.choice(LAST_NAMES, size=n)

    # Survey completion takes 45s to ~12 minutes; heavy right tail from distracted users.
    to_register = np.clip(rng.lognormal(mean=4.9, sigma=0.75, size=n), 20, 5400)
    register_date = session_dt + pd.to_timedelta(to_register.round(), unit="s")
    conversion_dt = register_date + pd.to_timedelta(rng.integers(1, 180, n), unit="s")

    # ~11% of sessions abandon before submitting the survey. The training code drops
    # them (register_date is null) but they must exist in the raw file.
    abandoned = rng.random(n) < 0.11
    register_date = pd.Series(register_date).where(~abandoned, pd.NaT)
    conversion_dt = pd.Series(conversion_dt).where(~abandoned, pd.NaT)

    # Real campaign ids exceed 2^53, which is why the ingest layer has to keep the
    # column out of float64 - see data/ingest.sanitise step 5.
    campaign_pool = rng.integers(120_227_000_000_000_000, 120_228_000_000_000_000, 140)
    campaigns = rng.choice(campaign_pool, size=n)
    sub1_pool = rng.integers(1_000_000, 9_999_999, 90)
    sub1 = rng.choice(sub1_pool, size=n)

    return pd.DataFrame({
        "session_id": [f"s{i:08d}" for i in range(n)],
        "session_dt": session_dt,
        "conversion_dt": conversion_dt,
        "register_date": register_date,
        "campaign_id": campaigns,
        "page": rng.choice([p for p, _ in schema.PAGES], size=n,
                           p=_normalised([w for _, w in schema.PAGES])),
        "sub1": sub1,
        "sub2": [f"{s:08d} Ad set" for s in sub1],
        "sub3": rng.integers(1_000_000_000, 1_999_999_999, n),
        "device_type": rng.choice(schema.DEVICE_TYPES, size=n, p=[0.74, 0.21, 0.05]),
        "auto_city": cities,
        "auto_state": np.where(is_foreign, "", states),
        "auto_country": countries,
        "business_type": rng.choice(schema.BUSINESS_TYPES, size=n,
                                    p=[0.30, 0.34, 0.14, 0.10, 0.08, 0.04]),
        "credit_score": credit,
        "industry": rng.choice(schema.INDUSTRIES, size=n,
                               p=_normalised([18, 14, 11, 10, 9, 9, 7, 6, 5, 4, 4, 3])),
        "loan_amount": amount,
        "monthly_revenue": revenue,
        "time_in_business": tenure,
        "loan_reason": rng.choice(schema.LOAN_REASONS, size=n,
                                  p=_normalised([26, 20, 14, 12, 9, 8, 6, 5])),
        "fname": fname,
        "lname": lname,
        "cellphone": rng.integers(2_010_000_000, 9_899_999_999, n),
        # Never read by the models - present because the real extract carries them.
        "address": _addresses(rng, n, "St"),
        "business_address": _addresses(rng, n, "Ave"),
        "business_name": np.char.add(
            np.char.add(lname.astype(str), " "),
            rng.choice(["Holdings", "LLC", "Services", "Group", "Enterprises",
                        "& Sons", "Trading"], size=n).astype(str)),
        "vertical": "business_loans",
        "_strength": strength,
    })


def _addresses(rng: np.random.Generator, n: int, suffix: str) -> np.ndarray:
    numbers = rng.integers(10, 9999, n).astype(str)
    streets = rng.choice(LAST_NAMES, size=n).astype(str)
    return np.char.add(np.char.add(np.char.add(numbers, " "), streets), f" {suffix}")


def _weighted_band(rng: np.random.Generator, bands: list[tuple[str, int]],
                   strength: np.ndarray, slope: float,
                   tail_weight: float = 0.0) -> np.ndarray:
    """Pick a band per row so that higher `strength` shifts mass towards better bands.

    Implemented as an ordered-logit: a single latent score is cut by fixed thresholds,
    which keeps the marginal distribution stable while preserving the correlation.
    """
    labels = [label for label, _ in bands]
    k = len(labels)
    score = slope * strength + rng.normal(0.0, 1.0, len(strength))
    # Even quantile cut points over the latent score, so every band is populated.
    cuts = np.quantile(score, np.linspace(0, 1, k + 1)[1:-1])
    idx = np.searchsorted(cuts, score)
    picked = np.asarray(labels, dtype=object)[idx]

    if tail_weight > 0:
        # Sprinkle the last band (the data-quality artefact) over random rows instead of
        # letting the ordered-logit give it a full share of the distribution.
        picked = np.where(rng.random(len(picked)) < tail_weight, labels[-1], picked)
    return picked


def _normalised(weights: list[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    return arr / arr.sum()


# --------------------------------------------------------------------------------- #
# Session -> (session, brand) rows
# --------------------------------------------------------------------------------- #

def _explode_to_brand_rows(rng: np.random.Generator, sess: pd.DataFrame) -> pd.DataFrame:
    """Each registered session is offered to 1-4 brands; abandoned ones to none.

    Multi-brand sessions are what makes n_session > 1, which in turn disables the
    `disposition_source == 'Lead'` fallback in the training code. Both paths must occur.
    """
    n = len(sess)
    registered = sess["register_date"].notna().to_numpy()

    # 1 brand is by far the common case; the tail exists so the n_session branch matters.
    n_brands = rng.choice([1, 2, 3, 4], size=n, p=[0.70, 0.19, 0.08, 0.03])
    n_brands = np.where(registered, n_brands, 1)

    repeat_idx = np.repeat(np.arange(n), n_brands)
    rows = sess.iloc[repeat_idx].reset_index(drop=True)

    brand_names = [b.name for b in BRANDS]
    brand_p = _normalised([b.share for b in BRANDS])
    # Sampling with replacement can repeat a brand inside a session; drop those so a
    # session never shows the same lender twice.
    chosen = rng.choice(brand_names, size=len(rows), p=brand_p)
    rows["client_name"] = chosen
    rows = rows.drop_duplicates(subset=["session_id", "client_name"], keep="first")

    # Abandoned sessions were never delivered to anyone: no brand, no label.
    was_registered = rows["register_date"].notna()
    rows.loc[~was_registered, "client_name"] = np.nan
    return rows.reset_index(drop=True)


def _apply_labels(rng: np.random.Generator, rows: pd.DataFrame) -> pd.DataFrame:
    """Decide acceptance and payout per (session, brand), then render the label columns.

    Acceptance is a logistic function of the brand's own weights applied to the user's
    standardised profile. That is the relationship the CatBoost classifier is meant to
    recover, and the payout scale is what TabPFN is meant to recover.
    """
    by_brand = {b.name: b for b in BRANDS}
    n = len(rows)

    credit_num = rows["credit_score"].map(dict(schema.CREDIT_SCORE_BANDS)).to_numpy(dtype=float)
    revenue_num = rows["monthly_revenue"].map(dict(schema.MONTHLY_REVENUE_BANDS)).to_numpy(dtype=float)
    tenure_num = rows["time_in_business"].map(dict(schema.TIME_IN_BUSINESS_BANDS)).to_numpy(dtype=float)
    amount_num = rows["loan_amount"].map(dict(schema.LOAN_AMOUNT_BANDS)).to_numpy(dtype=float)

    # Standardise onto a comparable scale; the -99 / 87500 sentinels are clipped so a
    # single bad band does not dominate the logit.
    z_credit = np.clip((credit_num - 620.0) / 80.0, -2.0, 2.0)
    z_revenue = np.clip((np.log10(np.clip(revenue_num, 1, None)) - 4.4) / 0.6, -2.0, 2.0)
    z_tenure = np.clip((np.clip(tenure_num, 0, 60) - 20.0) / 14.0, -2.0, 2.0)
    z_amount = np.clip((np.log10(np.clip(amount_num, 1, None)) - 4.6) / 0.6, -2.0, 2.0)

    has_brand = rows["client_name"].notna().to_numpy()
    names = rows["client_name"].fillna(BRANDS[0].name).to_numpy()

    bias = np.array([by_brand[x].bias for x in names])
    w_credit = np.array([by_brand[x].credit_w for x in names])
    w_revenue = np.array([by_brand[x].revenue_w for x in names])
    w_tenure = np.array([by_brand[x].tenure_w for x in names])
    w_amount = np.array([by_brand[x].amount_w for x in names])
    base_payout = np.array([by_brand[x].base_payout for x in names])

    industry_bonus = np.array([
        0.45 if rows["industry"].iat[i] in by_brand[names[i]].preferred_industries else 0.0
        for i in range(n)
    ])

    # Mobile converts a little worse; a very fast survey completion signals a bot-ish
    # session. Both are real effects the device/timing features can pick up.
    device_effect = np.where(rows["device_type"].to_numpy() == "mobile", -0.12, 0.05)
    elapsed = (rows["register_date"] - rows["session_dt"]).dt.total_seconds().to_numpy()
    speed_effect = np.where(np.nan_to_num(elapsed, nan=200.0) < 45, -0.60, 0.0)

    logit = (bias + w_credit * z_credit + w_revenue * z_revenue
             + w_tenure * z_tenure + w_amount * z_amount
             + industry_bonus + device_effect + speed_effect
             + rng.normal(0.0, 0.45, n))
    accept_p = 1.0 / (1.0 + np.exp(-logit))
    accepted = (rng.random(n) < accept_p) & has_brand

    # Payout: brand scale x user quality x lognormal noise, clipped to the $1-$300
    # range the briefing describes.
    quality = 1.0 + 0.22 * z_credit + 0.30 * z_revenue + 0.16 * z_amount
    payout = base_payout * np.clip(quality, 0.25, 2.4) * rng.lognormal(0.0, 0.28, n)
    payout = np.round(np.clip(payout, 1.0, 300.0), 2)

    # A small share of accepted leads go on to fund. The research code counts only
    # 'Lead' as sold_to_client, so this stays rare by design.
    funded = accepted & (rng.random(n) < 0.06)
    disposition = np.full(n, np.nan, dtype=object)
    disposition[accepted] = "Lead"
    disposition[funded] = "Fund"
    rejected = has_brand & ~accepted
    disposition[rejected] = rng.choice(["Rejected", "Duplicate", "Invalid"],
                                       size=rejected.sum(), p=[0.62, 0.26, 0.12])

    rows["disposition"] = disposition
    rows["disposition_source"] = _disposition_sources(rng, names, has_brand, accepted)
    rows["payout"] = np.where(accepted, payout, np.nan)
    rows["client_id"] = [
        f"c{[b.name for b in BRANDS].index(x) + 101:04d}" if ok else np.nan
        for x, ok in zip(names, has_brand)
    ]
    rows["client_name"] = rows["client_name"].astype(object)
    return rows.drop(columns=["_strength"])


def _disposition_sources(rng: np.random.Generator, names: np.ndarray,
                         has_brand: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    """Reproduce the four shapes of disposition_source that check_match() has to handle.

    check_match (bl_models_train.py lines 50-61) matches the source to the client either
    exactly, or by any whitespace-delimited word other than 'api'. Roughly a tenth of
    rows match neither, which is how client_buy legitimately ends up null.
    """
    out = np.full(len(names), np.nan, dtype=object)
    kind = rng.choice(["exact", "api", "lead", "unmatched"], size=len(names),
                      p=[0.55, 0.20, 0.15, 0.10])
    for i, name in enumerate(names):
        if not has_brand[i]:
            continue
        k = kind[i]
        if k == "exact":
            out[i] = name
        elif k == "api":
            # 'fundera / nerdwallet' -> 'fundera api': one word still matches.
            out[i] = f"{name.split()[0]} api"
        elif k == "lead":
            # Matches nothing by word, so only the n_session==1 fallback can rescue it.
            out[i] = "Lead"
        else:
            out[i] = rng.choice(["internal", "affiliate network", "partner feed"])
    # An unaccepted row still carries a source; the label comes from `disposition`.
    _ = accepted
    return out


def _inject_edge_cases(rng: np.random.Generator, rows: pd.DataFrame) -> pd.DataFrame:
    """Add the malformed rows a real extract always contains.

    Each one exercises a specific defence in the pipeline, so they are added on purpose
    rather than left to chance: missing survey answers (the dropna(thresh=5) path),
    unparseable phone numbers, and blank names.
    """
    n = len(rows)
    survey_cols = ["credit_score", "industry", "loan_amount", "loan_reason",
                   "monthly_revenue", "time_in_business", "device_type", "business_type"]

    # 6% of rows drop 1-2 answers (kept and imputed to 'other'), 1.2% drop 4
    # (removed by the thresh=5 rule).
    n_missing = rng.choice([0, 1, 2, 4], size=n, p=[0.928, 0.042, 0.018, 0.012])
    for count in (1, 2, 4):
        target = np.flatnonzero(n_missing == count)
        for idx in target:
            for col in rng.choice(survey_cols, size=count, replace=False):
                rows.iat[idx, rows.columns.get_loc(col)] = np.nan

    # Single-character answers: the research code turns these into NaN via the
    # str.len() > 1 rule, then imputes 'other'.
    stub = rng.random(n) < 0.004
    rows.loc[stub, "industry"] = "-"

    # Phone numbers that are not clean integers. cellphone.astype(int) in the research
    # code cannot take these, which is exactly why the pipeline sanitises them first.
    # The column has to become object before the assignment or pandas refuses the cast.
    dirty = rng.random(n) < 0.006
    rows["cellphone"] = rows["cellphone"].astype(object)
    rows.loc[dirty, "cellphone"] = rng.choice(
        ["(305) 555-0142", "+1 786 991 4030", "n/a", ""], size=int(dirty.sum()))

    blank_name = rng.random(n) < 0.003
    rows.loc[blank_name, "fname"] = ""

    # Missing attribution parameters. These matter more than they look: a single null
    # makes pandas read the whole sub column as float64, and the research code's
    # `fillna('Other').astype(str)` then yields '1815195.0' in training against
    # '1815195' from a JSON request - a permanent miss on three of the fourteen
    # categorical features. data/ingest.sanitise repairs it; this is what proves it.
    for column, rate in (("sub1", 0.04), ("sub2", 0.03), ("sub3", 0.05)):
        rows[column] = rows[column].astype(object)
        rows.loc[rng.random(n) < rate, column] = np.nan
    return rows


# --------------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------------- #

def write_csv(settings: Settings | None = None, sessions: int | None = None,
              seed: int | None = None, out: Path | None = None,
              overwrite: bool = False) -> Path:
    """Write the generated extract to paths.raw_csv. Never clobbers a real file."""
    settings = settings or Settings.load()
    target = out or settings.paths.raw_csv
    target = resolve(target)
    if target.exists() and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = generate(settings, sessions=sessions, seed=seed)
    frame.to_csv(target, index=False)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic bl_full_data.csv")
    parser.add_argument("--sessions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    settings = Settings.load()
    path = write_csv(settings, sessions=args.sessions, seed=args.seed,
                     out=args.out, overwrite=args.overwrite)
    frame = pd.read_csv(path, low_memory=False)
    print(f"wrote {path}")
    print(f"  rows           {len(frame):,}")
    print(f"  sessions       {frame['session_id'].nunique():,}")
    print(f"  registered     {frame['register_date'].notna().sum():,}")
    print(f"  brands         {frame['client_name'].nunique()}")
    print(f"  accepted       {(frame['disposition'] == 'Lead').sum():,}")
    print(f"  payout > 0     {(frame['payout'].fillna(0) > 0).sum():,}")
    print(f"  date range     {frame['session_dt'].min()} .. {frame['session_dt'].max()}")


if __name__ == "__main__":
    main()
