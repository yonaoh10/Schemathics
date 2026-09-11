"""The bl_full_data.csv contract, derived from the research code's own branches.

The survey vocabularies below are not decorative: bl_models_train.py maps the free-text
bands to numbers with `str.contains` after lower-casing, and several of those branches
overlap. Every string here was checked against the actual conditions, and the expected
numeric result is recorded next to it so a regression in either direction is visible.

Reference (bl_models_train.py):
  credit_score_to_numeric        lines 116-127
  credit_loan_amount_to_numeric  lines 129-144
  monthly_revenue_to_numeric     lines 146-160
  time_in_business_to_num        lines 162-173
"""

from __future__ import annotations

# Columns the real extract carries. The training script selects a subset; the rest are
# present so the generated file is a faithful superset, exactly like a warehouse dump.
RAW_COLUMNS: list[str] = [
    "session_id", "session_dt", "conversion_dt", "register_date",
    "campaign_id", "page", "sub1", "sub2", "sub3",
    "device_type", "auto_city", "auto_state", "auto_country",
    "business_type", "credit_score", "industry", "loan_amount",
    "monthly_revenue", "time_in_business", "loan_reason",
    "fname", "lname", "cellphone", "address", "business_address", "business_name",
    "client_name", "client_id", "payout", "disposition", "disposition_source",
    "vertical",
]

# The 25 features the models are fitted on, in the exact order the research code uses.
TRAIN_COLUMNS: list[str] = [
    "campaign_id", "page", "city", "device_type", "sub1", "sub2", "sub3",
    "business_type", "industry", "loan_reason", "cellphone_prefix",
    "country_state", "credit_score_num", "loan_amount_num", "monthly_revenue_num",
    "time_in_business_num", "session_day", "session_day_of_week",
    "session_hour", "from_start_to_register", "ratio_loan_amount_revenue",
    "gender", "fname_len", "lname_len", "client_name",
]

LABEL_COLUMNS: list[str] = ["sold_to_client", "payout", "payout_adj"]

# Keys of one post-funnel request. Matches the hard-coded example in
# bl_exp_payout_predictor.py lines 249-272 - the endpoint contract is this dict.
REQUEST_FIELDS: list[str] = [
    "session_dt", "conversion_dt", "register_date", "campaign_id", "page",
    "auto_city", "auto_country", "auto_state", "device_type", "sub1", "sub2", "sub3",
    "business_type", "credit_score", "industry", "loan_amount", "loan_reason",
    "monthly_revenue", "time_in_business", "fname", "lname", "cellphone",
]

# --- Survey vocabularies -----------------------------------------------------------
# (label, expected numeric value after the research code's mapping)

# 'Poor - 550 to 599' matches the '550' rule first and the '550'+'599' rule second;
# the later assignment wins, which is why the expected value is 551 and not 501.
CREDIT_SCORE_BANDS: list[tuple[str, int]] = [
    ("Very Poor - Under 550", 501),
    ("Poor - 550 to 599", 551),
    ("Fair - 600 to 649", 601),
    ("Good - 650 to 719", 651),
    ("Excellent - 720 and above", 721),
]

# 'Less than $10,000' contains '10,000' but not '24,999', so no rule fires and the
# -99 sentinel survives. Kept deliberately: real extracts contain it.
LOAN_AMOUNT_BANDS: list[tuple[str, int]] = [
    ("Less than $10,000", -99),
    ("$10,000 - $24,999", 17500),
    ("$25,000 - $49,999", 47500),
    ("$50,000 - $74,999", 57500),
    ("$75,000 - $99,999", 75000),
    ("$100,000 - $199,999", 150000),
    ("$200,000+", 250000),
]

# '$50,000 - $99,999' matches the '9,999' rule (inside '99,999') before the
# '50,000'+'99,999' rule overwrites it - 75000 is correct only because of the order.
MONTHLY_REVENUE_BANDS: list[tuple[str, int]] = [
    ("Less than $5,000", -99),
    ("$5,000 - $9,999", 5000),
    ("$10,000 - $19,999", 15000),
    ("$20,000 - $49,999", 35000),
    ("$50,000 - $99,999", 75000),
    ("$100,000 - $199,999", 150000),
    ("$200,000+", 250000),
]

# time_in_business_to_num maps five literal keys and fills the rest with 0. One of the
# keys is a revenue band - a data-quality artefact in the source system that the
# research code explicitly handles, so the generator reproduces it at a low rate.
TIME_IN_BUSINESS_BANDS: list[tuple[str, int]] = [
    ("Less than 6 months", 3),
    ("6-12 months", 8),
    ("1-2 years", 18),
    ("2+ years", 36),
    ("$75,000 - $99,999", 87500),
]

BUSINESS_TYPES: list[str] = [
    "Sole Proprietorship", "LLC", "S Corporation", "C Corporation",
    "Partnership", "Non-Profit",
]

INDUSTRIES: list[str] = [
    "construction", "retail", "healthcare", "restaurant", "transportation",
    "professional services", "manufacturing", "automotive", "real estate",
    "technology", "agriculture", "personal services",
]

LOAN_REASONS: list[str] = [
    "Working capital", "Equipment purchase", "Expansion", "Inventory",
    "Payroll", "Debt refinance", "Marketing", "Commercial real estate",
]

DEVICE_TYPES: list[str] = ["mobile", "desktop", "tablet"]

# Six landing-page variants, two dominant - as described in the briefing.
PAGES: list[tuple[str, float]] = [
    ("top10us.com/app/business-loans-v2", 0.41),
    ("top10us.com/app/business-loans", 0.33),
    ("top10us.com/app/business-loans-v3", 0.11),
    ("top10us.com/app/sba-loans", 0.07),
    ("top10us.com/app/merchant-cash-advance", 0.05),
    ("top10us.com/app/equipment-financing", 0.03),
]

DISPOSITIONS: list[str] = ["Lead", "Fund", "Duplicate", "Rejected", "Invalid"]
