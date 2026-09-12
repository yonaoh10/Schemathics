"""Production wrapper around BLPayoutModelsFit.

The research class is subclassed, not edited. Every feature-engineering method, the
CatBoost configuration, the time split and the artifact-writing behaviour are inherited
exactly as written. Four methods are overridden. One of them changes what happens; the
other three only watch.

  tabpfn_regression_payout          the one that changes something: the estimator comes
                                    from the backend registry instead of being hard-wired
                                    to the hosted API. The context is still the last
                                    `CONTEXT_SIZE` rows with payout > 0, in the same
                                    order.

  accuracy_* (two methods)          call super() first, so the researcher log is
                                    written byte for byte as before, then capture the
                                    same numbers so MLflow can compare runs. Nothing
                                    is recomputed differently; the arrays are the ones
                                    the research code already produced.

  split_by_time                     calls super() and records the row counts. The split
                                    itself is untouched - the point is that it happens in
                                    *both* modes, so the fit never sees the last
                                    `days_for_test` days, and the run should say so
                                    rather than report the snapshot size as if it had.

Anything else this file adds is around the edges: run directories, log-handler hygiene,
and a metrics dictionary.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from bl_ranking.config import PayoutSettings
from bl_ranking.models.payout import PayoutBackend, create_backend
from bl_ranking.research.bl_models_train import BLPayoutModelsFit


@contextlib.contextmanager
def preserve_root_logging():
    """Restore the root logger after the research code takes it over.

    setup_bl_logger (bl_models_train.py lines 25-42) calls
    `root_logger.handlers.clear()`, which would otherwise silence MLflow, uvicorn and
    everything else for the rest of the process. Harmless in a script, not in a job.

    It must be entered before the trainer is *constructed*, not before `fit_()`:
    setup_bl_logger runs in `BLPayoutModelsFit.__init__`. Entered after construction this
    saves the already-cleared list and faithfully restores nothing - see the call site in
    training/job.py, which is the only correct ordering.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


class ProductionTrainer(BLPayoutModelsFit):
    """BLPayoutModelsFit with a pluggable payout backend and captured metrics."""

    def __init__(self, input_path: str, input_file: str, output_predictors_path: str,
                 train_test: bool, payout_cfg: PayoutSettings,
                 backend_name: str | None = None,
                 on_event: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.payout_cfg = payout_cfg
        self.backend_name = backend_name or payout_cfg.backend
        self.metrics: dict[str, float] = {}
        # Filled by split_by_time. Empty until bl_preprocessing has run.
        self.split_counts: dict[str, int] = {}
        self.payout_backend: PayoutBackend | None = None
        self._on_event = on_event or (lambda event, payload: None)
        super().__init__(input_path, input_file, output_predictors_path, train_test)

    # -- overridden seam: where the payout estimator comes from ---------------------

    def tabpfn_regression_payout(self, x_train_payout: pd.DataFrame,
                                 y_train_payout: pd.DataFrame):
        """Same context construction as the research code; swappable estimator.

        Research original (bl_models_train.py lines 286-298) hard-codes
        `set_access_token(...)` and `TabPFNRegressor(ignore_pretraining_limits=True)`.
        Here the estimator is resolved from `model.payout.backend`; `tabpfn_client`
        reproduces the original exactly.
        """
        self.logger.info("start tabular transformer")
        context_size = self.payout_cfg.context_size
        tfm_context = {
            "x": x_train_payout.iloc[-context_size:],
            "y": y_train_payout["payout"].iloc[-context_size:],
            "columns": list(x_train_payout.columns),
        }
        backend = create_backend(self.payout_cfg, self.backend_name)
        backend.fit(tfm_context["x"], tfm_context["y"])
        self.payout_backend = backend
        self.logger.info(
            f"finish tabular transformer (backend={backend.name}, "
            f"context_rows={len(tfm_context['x'])})"
        )
        self._on_event("payout_fitted", backend.describe())
        return backend, tfm_context

    # -- observed, not changed: what the time split actually did --------------------

    def split_by_time(self, bl_data, days_for_test):
        """The research split, unchanged. The row counts are recorded on the way past.

        Worth recording because the research code calls this in BOTH modes, so the fit
        never sees the last `days_for_test` days of the window - production mode
        included. The docs described that mode as "fit on everything" and the bundle
        manifest reported `rows_train = len(snapshot)`, which is the snapshot, not what
        was fitted; on the real extract the two differ by about a tenth.

        `days_for_test` is hard-coded to 7 at the call site (bl_models_train.py line 235)
        and `data.days_for_test` only mirrors it, so it is taken from the argument here
        rather than from the config - the number recorded is the one that was used.
        """
        bl_train, bl_test = super().split_by_time(bl_data, days_for_test)
        self.split_counts = {
            "rows_preprocessed": int(len(bl_data)),
            "rows_fitted": int(len(bl_train)),
            "rows_held_out": int(len(bl_test)),
            "days_for_test": int(days_for_test),
        }
        return bl_train, bl_test

    # -- overridden seam: capture the numbers the researcher log prints as text -----

    def accuracy_classification_model(self, CB_model, y_test_all_days, x_test_all_days,
                                      target_col: str) -> None:
        super().accuracy_classification_model(CB_model, y_test_all_days,
                                              x_test_all_days, target_col)
        y_true = y_test_all_days[target_col]
        y_pred = CB_model.predict(x_test_all_days.drop("split_day", axis=1))
        self.metrics.update(_classification_metrics(y_true, y_pred, prefix="clf"))

        # Per-day metrics make weekly drift visible in the MLflow comparison view.
        #
        # The bucket size is always reported, and the scores only when the bucket has both
        # label classes in it. The research split buckets by 24-hour offsets from the
        # window's last *timestamp* (bl_models_train.py line 217), so the final bucket
        # holds only the rows at that instant: one row, in every run so far. Its accuracy,
        # precision, recall and F1 all came out at exactly 1.000 and sat in the comparison
        # view looking like a perfect day, next to six real ones around 0.59. A score
        # computed from one row of one class is noise wearing a metric's name.
        for day in sorted(x_test_all_days["split_day"].unique()):
            mask = x_test_all_days["split_day"] == day
            day_true = y_test_all_days.loc[mask, target_col]
            label = f"clf.day{int(day)}"
            # Reported even when the scores are not, so their absence is explainable
            # rather than just absent. `support` from the report below is the *positive*
            # count, which is not the same question.
            self.metrics[f"{label}.rows"] = float(len(day_true))
            if day_true.nunique() < 2:
                continue
            day_pred = CB_model.predict(x_test_all_days[mask].drop("split_day", axis=1))
            self.metrics.update(_classification_metrics(day_true, day_pred, prefix=label))

    def accuracy_cont_payout_prediction(self, model_tfm, x_test_payout, y_test_payout) -> None:
        super().accuracy_cont_payout_prediction(model_tfm, x_test_payout, y_test_payout)
        # super() left its predictions on the frame; reuse them rather than re-predict.
        actual = y_test_payout["payout"]
        predicted = y_test_payout["pred_payout"]
        self.metrics["payout.mae"] = float(mean_absolute_error(actual, predicted))
        self.metrics["payout.mape"] = float(mean_absolute_percentage_error(actual, predicted))
        self.metrics["payout.n_test"] = float(len(actual))

        # The research code reports the same thresholds; they answer "is the model
        # trustworthy where it matters", i.e. on the brands worth ranking first.
        for threshold in (5, 10, 15, 20):
            subset = y_test_payout[y_test_payout["pred_payout"] >= threshold]
            if subset.empty:
                continue
            self.metrics[f"payout.mae_thr{threshold}"] = float(
                mean_absolute_error(subset["payout"], subset["pred_payout"]))
            self.metrics[f"payout.mape_thr{threshold}"] = float(
                mean_absolute_percentage_error(subset["payout"], subset["pred_payout"]))
            self.metrics[f"payout.n_thr{threshold}"] = float(len(subset))

    # -- helpers -------------------------------------------------------------------

    @property
    def researcher_log(self) -> Path | None:
        """The log file setup_bl_logger created for this run."""
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return Path(handler.baseFilename)
        return None

    def close_log(self) -> None:
        """Flush and detach the file handler so the log can be uploaded as an artifact."""
        for handler in list(self.logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)


def _classification_metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    """Flatten sklearn's report into scalars MLflow can chart across runs."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    positive = report.get("1", report.get(1, {}))
    return {
        f"{prefix}.accuracy": float(report.get("accuracy", 0.0)),
        f"{prefix}.precision": float(positive.get("precision", 0.0)),
        f"{prefix}.recall": float(positive.get("recall", 0.0)),
        f"{prefix}.f1": float(positive.get("f1-score", 0.0)),
        f"{prefix}.support": float(positive.get("support", 0.0)),
        f"{prefix}.positive_rate": float(np.mean(np.asarray(y_true) == 1)),
    }
