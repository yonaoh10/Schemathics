"""The research inference pipeline, wired to warm models.

Kept in its own module because importing it is expensive: bl_exp_payout_predictor.py
runs `nd = NameDataset()` at module scope, which costs 9.5 s and 2.1 GB of resident
memory. Production serves the fast path (serving/fast_features.py) and never imports
this, so a serving container starts in ~3.9 s and holds ~290 MB.

It is imported on demand when `serving.feature_path = research`, and by the equivalence
test, which is exactly what it is for: the reference implementation the fast path is
checked against.

BLPayoutModelsPredict is subclassed, not edited. Five methods are overridden: four are
plumbing (logging, warning capture, model loading, gender lookup), and the fifth,
`import_preprocess`, is reproduced with three deviations, each marked inline. Every
feature-engineering method is inherited unchanged.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from bl_ranking.research.bl_exp_payout_predictor import BLPayoutModelsPredict
from bl_ranking.serving.ranker import WarmModels

log = logging.getLogger("bl_ranking.serving")

# The research method ignores `self` - it only consults the module-level NameDataset -
# so it can be called unbound.
_UNBOUND_GENDER = BLPayoutModelsPredict.detect_gender_with_confidence


@lru_cache(maxsize=65536)
def _gender_via_names_dataset(fname: str) -> tuple[str, float]:
    """Memoised live lookup, used when the bundle carries no precomputed table.

    One request scores the same person against every brand, so without the cache the
    identical name is looked up N times. The function is pure, so the cache cannot
    change the answer - it only removes repeated work.
    """
    return _UNBOUND_GENDER(None, fname)


class ServingPredictor(BLPayoutModelsPredict):
    """BLPayoutModelsPredict with per-request I/O lifted out. Feature logic unchanged."""

    def __init__(self, user_data: dict[str, Any], warm: WarmModels) -> None:
        self.warm = warm
        # Deliberately not calling super().__init__: it would create a log file and
        # rebind the global warning handler on every request.
        self.predictors_path = str(warm.bundle_dir) + "/"
        self.user_data = user_data
        self.user_data_file = user_data
        self.logger = log

    # -- plumbing overrides --------------------------------------------------------

    def setup_bl_logger(self, log_dir: str) -> logging.Logger:  # pragma: no cover
        """No per-request log file, and no clearing of the root logger's handlers."""
        return log

    def capture_warnings(self) -> None:  # pragma: no cover
        """Installed once at start-up instead (see install_warning_capture)."""

    def load_models(self):
        """Return the already-warm models rather than rebuilding them."""
        return self.warm.payout, self.warm.columns, self.warm.catboost

    def detect_gender_with_confidence(self, fname: str) -> tuple[str, float]:
        """Same values as the research implementation, from a precomputed table.

        See models/gender_lut.py: the table is the research function materialised over
        the dataset's entire first-name universe, so this is a lookup, not an estimate.
        """
        if self.warm.gender is not None:
            return self.warm.gender.lookup(fname)
        return _gender_via_names_dataset(fname)

    # -- reproduced from the research code -----------------------------------------
    #
    # Three deviations from bl_exp_payout_predictor.py lines 57-83, all marked inline:
    #   1. self.user_data instead of the module global (the original is a NameError
    #      outside __main__, and silently scores the example dict inside it);
    #   2. the cached brand universe instead of re-reading the CSV on every request;
    #   3. the two logger.info calls are logger.debug, because at INFO they would emit
    #      two lines per request for numbers that never vary.
    # Every feature expression below is byte-for-byte the original.

    def import_preprocess(self) -> pd.DataFrame:
        bl_data = pd.DataFrame([self.user_data])          # CHANGED 1: was the global `user_data`
        all_clients = self.warm.all_clients               # CHANGED 2: was pd.read_csv(...) per call
        needed_columns = ['session_dt', 'conversion_dt', 'register_date',
                          'campaign_id', 'page', 'auto_city', 'auto_country', 'auto_state', 'device_type', 'sub1',
                          'sub2', 'sub3',
                          'business_type', 'credit_score', 'industry', 'loan_amount', 'loan_reason', 'monthly_revenue',
                          'time_in_business', 'fname', 'lname', 'cellphone']
        bl_data = bl_data[needed_columns]
        if bl_data['register_date'].isna().any():
            raise Exception("user cannot be a lead - register_date is absent")

        bl_data = bl_data.merge(all_clients, how='cross')
        bl_data = bl_data[bl_data['client_name'] != 'other']
        self.logger.debug(f'raw data rows: {bl_data.shape[0]}')                  # CHANGED 3: was info
        bl_data = bl_data.rename(columns={'auto_city': 'city', 'auto_state': 'state', 'auto_country': 'country'})
        self.logger.debug(f'register date exists - can be leads: {bl_data.shape[0]}')   # CHANGED 3: was info
        bl_data[['country', 'state', 'city', 'sub1', 'sub2', 'sub3']] = (
            bl_data[['country', 'state', 'city', 'sub1', 'sub2', 'sub3']].fillna('Other'))
        bl_data['country_state'] = np.where(bl_data['country'] == 'United States', bl_data['state'], bl_data['country'])
        bl_data['session_dt'] = pd.to_datetime(bl_data['session_dt'], errors="coerce")
        bl_data['register_date'] = pd.to_datetime(bl_data['register_date'], errors="coerce")
        bl_data['sub1'] = bl_data['sub1'].astype(str)
        bl_data['sub2'] = bl_data['sub2'].astype(str)
        bl_data['sub3'] = bl_data['sub3'].astype(str)
        bl_data['cellphone_prefix'] = bl_data['cellphone'].astype(int).astype(str).str[:3].astype(str)
        return bl_data
