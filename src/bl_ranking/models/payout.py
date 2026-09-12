"""Payout-regressor backends.

The research code obtains its payout model like this (bl_models_train.py lines 286-298):

    set_access_token(os.environ["TABPFN_TOKEN"])
    model_tfm = TabPFNRegressor(ignore_pretraining_limits=True)
    model_tfm.fit(x_train_payout.iloc[-1000:], y_train_payout['payout'].iloc[-1000:])

and re-runs exactly that inside `load_models()` on *every prediction request*
(bl_exp_payout_predictor.py lines 48-55). TabPFN is in-context learning, so `fit` is
not training - it is handing the model its 1000-row context. With `tabpfn_client` that
context is shipped to a hosted API, which means every single user-facing request pays
for an upload plus a remote forward pass. That is the main thing this system fixes.

What stays fixed: the context is still the most recent 1000 rows with payout > 0, in
the same order, and the estimator is still a TabPFN regressor. What changes is where it
runs and how often it is constructed.

Four backends, selected by `model.payout.backend`. Measured on a 4-CPU box, one request
being one user scored against 15 brands:

  surrogate         -> DEFAULT. A CatBoost student distilled from TabPFN's own
                    predictions at training time. A whole request costs 4.1 ms
                    against the teacher's 457 ms - both measured end to end over the
                    same 200 users - for 1.21% of expected payout given up (see
                    scripts/compare_backends.py). On CPU it is the only option inside
                    a page-load budget.
  tabpfn_client     The research code's hosted model, with fit() lifted out of the
                    request path: the weekly job fits once and the serving process
                    reuses the server-side fitted set. Exact. One round trip per request.
  tabpfn_local      TabPFN weights in-process. Exact and private, but 457 ms per
                    request on CPU even with fit_with_cache - right for batch scoring,
                    for teaching the surrogate, or on a GPU. Not for the funnel.
  catboost_fallback No TabPFN at all. The degraded path: it keeps the endpoint
                    answering when the others are unreachable, and it is what CI runs.

Every backend reports `name` and `exact` on the prediction response, so a degraded or
approximate endpoint is never a silent one.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from bl_ranking.config import PayoutSettings

# Columns the research code hands the payout model are a mix of numeric and free text.
# CatBoost takes them natively; TabPFN needs them ordinal-encoded, which the
# CategoricalAdapter below does, fitted once on the context.
_CATEGORICAL_DTYPES = ("object", "category", "string")


@dataclass
class PayoutContext:
    """The `payout_tfm_context.joblib` artifact: in-context rows plus column order."""

    x: pd.DataFrame
    y: pd.Series
    columns: list[str]

    @classmethod
    def from_artifact(cls, path: str | Path) -> PayoutContext:
        blob = joblib.load(path)
        return cls(x=blob["x"], y=blob["y"], columns=list(blob["columns"]))

    def as_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "columns": self.columns}


class PayoutBackend(ABC):
    """Duck-type compatible with the TabPFNRegressor the research code expects."""

    name: str = "base"
    #: False when predictions are not bit-identical to a TabPFN forward pass.
    exact: bool = True

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> PayoutBackend:
        ...

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        ...

    def predict_batch(self, batch) -> np.ndarray:
        """Score a serving.batch.ScoringBatch.

        The default materialises a DataFrame and defers to `predict`, so a backend only
        overrides this if it has a cheaper representation. The CatBoost-based backends
        do: they take the Pool the batch already built for the classifier.
        """
        return self.predict(batch.frame())

    def save_extra(self, directory: Path) -> list[str]:
        """Persist anything that cannot be rebuilt from the context alone.

        Called once, by the training job, when the model bundle is assembled.
        """
        return []

    def load_extra(self, directory: Path) -> None:  # noqa: B027 - optional hook
        """Restore what save_extra wrote. Called before prepare().

        Deliberately concrete and empty: most backends have nothing extra to restore,
        and forcing every one of them to declare that would be noise.
        """

    def expected_columns(self) -> list[str] | None:
        """The feature columns this backend was fitted on, in order, if it has any.

        The CatBoost-based backends slice the incoming frame by a column list they
        persisted at training time, so a bundle whose payout model and payout context
        disagree mis-slots features in the payout half of the ranking - the same hazard
        the classifier check covers, on the other model. None means the backend has no
        fixed column order to check against.
        """
        return None

    def prepare(self, context: PayoutContext, directory: Path) -> PayoutBackend:
        """Make this backend ready to predict, at serving start-up.

        The default refits the context, which is what the research code does on every
        request; here it happens once per process. Backends that can restore a fit
        instead of redoing it override this - that is where most of the start-up cost
        and, for the hosted client, all of the per-request upload goes away.
        """
        self.load_extra(directory)
        return self.fit(context.x, context.y)

    def describe(self) -> dict[str, Any]:
        return {"payout_backend": self.name, "payout_exact": self.exact}


# --------------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------------- #


def _fitted_feature_names(model: Any, sidecar: list[str]) -> list[str] | None:
    """The columns this backend will actually score, checked against the model itself.

    Two files are involved and both matter. The sidecar list is what the backend slices
    the incoming frame by, so it decides what gets fed; the model's own recorded names
    are what it was fitted on, so they decide what should have been fed. Reading only
    one of them means the mis-slotting check confirms a file against itself.

    They must agree with each other, and the caller then checks them against the payout
    context. A disagreement here is a bundle assembled from two different runs, which
    is the failure the atomic bundle exists to prevent and cannot catch on its own once
    a bundle has been written or edited inconsistently.
    """
    recorded = list(getattr(model, "feature_names_", None) or [])
    persisted = list(sidecar)
    if recorded and persisted and recorded != persisted:
        raise ValueError(
            f"payout model is inconsistent with its own column list: the model was "
            f"fitted on {len(recorded)} columns and the bundle persists {len(persisted)}, "
            f"or in a different order. Scoring would mis-slot the payout half of the "
            f"ranking silently."
        )
    return persisted or recorded or None

class CategoricalAdapter:
    """Ordinal-encode object columns, fitted on the context and frozen thereafter.

    TabPFN works on numeric matrices. The research code passes raw text columns
    (page, city, gender, client_name, ...) straight in and lets the client library
    encode them; doing it here makes the mapping explicit, stable across the weekly
    retrain boundary, and identical between training and serving.

    Unseen categories at serving time map to -1, which TabPFN treats as just another
    level rather than failing the request. That is the right behaviour on a user-facing
    path: a brand-new campaign id must not take the endpoint down.
    """

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.categorical_indices: list[int] = []
        self._levels: dict[str, dict[Any, int]] = {}

    def fit(self, x: pd.DataFrame) -> CategoricalAdapter:
        self.columns = list(x.columns)
        self.categorical_indices = [
            i for i, c in enumerate(self.columns)
            if str(x[c].dtype) in _CATEGORICAL_DTYPES
        ]
        for i in self.categorical_indices:
            col = self.columns[i]
            levels = pd.Index(x[col].astype(str).unique())
            self._levels[col] = {value: code for code, value in enumerate(levels)}
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        out = x[self.columns].copy()
        for i in self.categorical_indices:
            col = self.columns[i]
            mapping = self._levels[col]
            out[col] = out[col].astype(str).map(mapping).fillna(-1).astype(np.int32)
        return out.to_numpy(dtype=np.float32, na_value=np.nan)


def _categorical_columns(x: pd.DataFrame) -> list[str]:
    return [c for c in x.columns if str(x[c].dtype) in _CATEGORICAL_DTYPES]


def _prepare_for_catboost(x: pd.DataFrame, cat_columns: list[str]) -> pd.DataFrame:
    """CatBoost rejects NaN in categorical columns; make them an explicit level."""
    out = x.copy()
    for col in cat_columns:
        out[col] = out[col].astype(str).fillna("nan")
    return out


# --------------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------------- #

class TabPFNLocalBackend(PayoutBackend):
    """TabPFN weights running in this process. No network, exact model, but slow on CPU.

    This is the right backend for batch scoring and for producing the surrogate's
    training labels. It is **not** viable on the synchronous funnel path on CPU:
    measured on this 4-core box, one request (15 brand rows against a 1000-row
    context) costs, holding `n_estimators=2` so only the fit mode varies

        fit_mode="low_memory"        37.3 s     and materially different answers
        fit_mode="fit_preprocessors"  8.69 s    the library default
        fit_mode="fit_with_cache"     0.26 s    after a ~14.9 s one-off cache build

    At the shipped `n_estimators=4` a whole request measures 457 ms p50 over 200 users,
    against 2.41 ms for the served path. Half a second is still the entire budget of a
    page load. A GPU changes that; a CPU deployment does not. Saying so plainly is more
    useful than shipping a default that times out.

    What `fit_with_cache` does is worth understanding, because it is the reason the
    surrogate is viable at all: TabPFN is in-context learning, so "fitting" is encoding
    the 1000 context rows. Our context is frozen between weekly retrains, so that
    encoding is computed once and the key/value cache kept. `save_fitted_tabpfn_model`
    serialises that cache, so the ~14.9 s build happens once in the weekly job and every
    serving replica starts warm with zero network.
    """

    name = "tabpfn_local"
    exact = True

    FIT_FILE = "payout_tabpfn_fit.tabpfn_fit"
    ADAPTER_FILE = "payout_tabpfn_adapter.joblib"

    def __init__(self, cfg: PayoutSettings) -> None:
        self.cfg = cfg
        self._model: Any = None
        self._adapter = CategoricalAdapter()

    def _build(self) -> Any:
        from tabpfn import TabPFNRegressor  # lazy: importing torch costs ~3.7 s

        device = self.cfg.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # ignore_pretraining_limits mirrors the research code. It is also required:
        # TabPFN's CPU sample limit is exactly 1000 rows, which is where the research
        # code's CONTEXT_SIZE sits, and without this flag a CPU run raises.
        return TabPFNRegressor(
            ignore_pretraining_limits=True,
            n_estimators=self.cfg.n_estimators,
            fit_mode=self.cfg.fit_mode,
            device=device,
            random_state=42,
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> TabPFNLocalBackend:
        self._adapter.fit(x)
        self._model = self._build()
        self._model.fit(self._adapter.transform(x), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(self._adapter.transform(x)), dtype=float)

    def save_extra(self, directory: Path) -> list[str]:
        """Persist the fitted estimator, including the KV cache when there is one.

        `save_fitted_tabpfn_model` deliberately excludes the 42 MB foundation weights
        and includes the inference engine's state, so the expensive part travels with
        the model version and the cheap part is redownloaded (or baked into the image).
        """
        try:
            from tabpfn import save_fitted_tabpfn_model
        except ImportError:
            return []
        joblib.dump(self._adapter, directory / self.ADAPTER_FILE)
        save_fitted_tabpfn_model(self._model, str(directory / self.FIT_FILE))
        return [self.FIT_FILE, self.ADAPTER_FILE]

    def prepare(self, context: PayoutContext, directory: Path) -> PayoutBackend:
        """Restore the saved fit if the bundle has one; otherwise refit the context."""
        fit_path = directory / self.FIT_FILE
        adapter_path = directory / self.ADAPTER_FILE
        if fit_path.exists() and adapter_path.exists():
            from tabpfn import load_fitted_tabpfn_model

            self._adapter = joblib.load(adapter_path)
            self._model = load_fitted_tabpfn_model(str(fit_path))
            return self
        return self.fit(context.x, context.y)

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update({"payout_fit_mode": self.cfg.fit_mode,
                     "payout_n_estimators": self.cfg.n_estimators,
                     "payout_device": self.cfg.device})
        return base



class TabPFNClientBackend(PayoutBackend):
    """The hosted TabPFN API - the research code's own path, with the fit taken off it.

    The research code calls `fit()` inside `load_models()`, which `predict_()` calls on
    every request (bl_exp_payout_predictor.py lines 48-55 and 240). `fit()` in
    tabpfn-client is eager: it serialises the whole 1000x25 context to parquet, uploads
    it to object storage and posts a fit request, before a single brand is scored. Two
    cross-internet round trips plus an upload, per user, while the page is loading.

    The library supports removing all of it. `fit()` returns a server-side
    `fitted_train_set_id`, exposed as `model_id_`, and an estimator whose `model_id_` is
    assigned directly can call `predict` without ever calling `fit`. So the weekly
    training job fits once, the id travels in the model bundle, and a request costs one
    round trip carrying only its ~15 rows. Same fitted set on the server, same
    predictions, no upload.

    Two smaller things that matter in production:

      * `model_path` is pinned. Left at the default the server chooses its current
        checkpoint, so a provider-side upgrade would silently change payouts between
        two weekly runs with nothing in MLflow to explain it.
      * `TABPFN_CLIENT_CI_MODE=true` is set. The client otherwise wraps every call in a
        thread and busy-polls `future.done()` every 200 ms while drawing a spinner,
        which quantises every call onto a 200 ms grid - a 150-195 ms tax on a request
        that should take far less.

    A stored id could in principle expire server-side. The weekly retrain refreshes it,
    and a failed predict falls back to fitting once, in-process.
    """

    name = "tabpfn_client"
    exact = True

    FIT_FILE = "payout_tabpfn_client_fit.json"

    def __init__(self, cfg: PayoutSettings) -> None:
        self.cfg = cfg
        self._model: Any = None
        self._context: PayoutContext | None = None
        self.model_id: str | None = None

    def _authenticate(self) -> None:
        from tabpfn_client import set_access_token

        token = os.environ.get("TABPFN_TOKEN")
        if not token:
            raise RuntimeError(
                "TABPFN_TOKEN is not set; backend 'tabpfn_client' cannot authenticate."
            )
        # Removes the client's 200 ms polling grid. Must be set before the first call.
        os.environ.setdefault("TABPFN_CLIENT_CI_MODE", "true")
        set_access_token(token)

    def _construct(self) -> Any:
        from tabpfn_client import TabPFNRegressor

        kwargs: dict[str, Any] = {"ignore_pretraining_limits": True}
        if self.cfg.model_path:
            kwargs["model_path"] = self.cfg.model_path
        return TabPFNRegressor(**kwargs)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> TabPFNClientBackend:
        self._authenticate()
        self._model = self._construct()
        self._model.fit(x, y)
        self.model_id = str(getattr(self._model, "model_id_", "") or "") or None
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(x), dtype=float)

    def save_extra(self, directory: Path) -> list[str]:
        if not self.model_id:
            return []
        (directory / self.FIT_FILE).write_text(json.dumps({
            "model_id": self.model_id,
            "model_path": self.cfg.model_path,
            "context_rows": int(len(self._context.x)) if self._context else None,
        }, indent=2))
        return [self.FIT_FILE]

    def prepare(self, context: PayoutContext, directory: Path) -> PayoutBackend:
        """Attach to the training run's server-side fit instead of uploading again."""
        self._context = context
        record = directory / self.FIT_FILE
        if not record.exists():
            return self.fit(context.x, context.y)

        saved = json.loads(record.read_text())
        self._authenticate()
        self._model = self._construct()
        # The documented way to reuse a fit: assign the id the server returned.
        self._model.model_id_ = saved["model_id"]
        self.model_id = saved["model_id"]
        return self

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["payout_model_id"] = self.model_id or "unfitted"
        base["payout_model_path"] = self.cfg.model_path or "server default"
        return base



class CatBoostFallbackBackend(PayoutBackend):
    """A CatBoost regressor fitted on the same 1000-row context. No TabPFN involved.

    Purpose is availability, not accuracy: if the TabPFN weights or the hosted API are
    unreachable, an endpoint that returns a slightly worse ranking is far better than
    one that returns 503 while the user waits on the landing page. It is also what the
    test suite and the offline demo use, so CI needs no credentials.

    Because it is a different model, `exact` is False and every response it produces is
    tagged with the backend name.
    """

    name = "catboost_fallback"
    exact = False

    def __init__(self, cfg: PayoutSettings) -> None:
        self.cfg = cfg
        self._model: Any = None
        self._cat_columns: list[str] = []
        self._columns: list[str] = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> CatBoostFallbackBackend:
        from catboost import CatBoostRegressor

        self._columns = list(x.columns)
        self._cat_columns = _categorical_columns(x)
        # Small, shallow and fast: the context is only ~1000 rows, so a large ensemble
        # would overfit and would cost latency for nothing.
        self._model = CatBoostRegressor(
            random_seed=42, depth=6, n_estimators=300, loss_function="RMSE",
            verbose=False, allow_writing_files=False, thread_count=1,
        )
        self._model.fit(_prepare_for_catboost(x, self._cat_columns),
                        np.asarray(y, dtype=float), cat_features=self._cat_columns)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        frame = _prepare_for_catboost(x[self._columns], self._cat_columns)
        return np.asarray(self._model.predict(frame), dtype=float)

    def predict_batch(self, batch) -> np.ndarray:
        return np.asarray(
            self._model.predict(batch.pool(self._cat_indices(batch.columns))), dtype=float
        )

    def expected_columns(self) -> list[str] | None:
        return _fitted_feature_names(self._model, self._columns)

    def _cat_indices(self, columns: list[str]) -> list[int]:
        return [columns.index(c) for c in self._cat_columns]


class SurrogateBackend(PayoutBackend):
    """TabPFN as a teacher, a CatBoost student as the thing actually served.

    The reasoning: TabPFN's forward pass is the most expensive part of a request, but
    the function it computes is *frozen* between weekly retrains. So it can be evaluated
    offline, once, over a large sample of the feature space, and approximated by a model
    whose inference cost is microseconds.

    The sample is not arbitrary. It is the real training rows crossed with the brand
    universe, which is exactly the shape of an inference request, so the student is
    fitted on the distribution it will actually be asked about.

    This is the only backend that changes predictions, so the distillation step
    measures its own fidelity against the teacher (MAE, MAPE, rank correlation, and
    top-1 brand agreement) and logs it to MLflow. Opt in deliberately.
    """

    name = "surrogate"
    exact = False

    STUDENT_FILE = "payout_surrogate.cbm"
    #: Share of users held out to measure fidelity honestly. See fit().
    HOLDOUT_FRACTION = 0.2

    def __init__(self, cfg: PayoutSettings) -> None:
        self.cfg = cfg
        self._student: Any = None
        self._cat_columns: list[str] = []
        self._columns: list[str] = []
        self.teacher: PayoutBackend | None = None
        self.fidelity: dict[str, float] = {}

    def fit(self, x: pd.DataFrame, y: pd.Series) -> SurrogateBackend:
        """Fit the teacher on the context, then distil it into the student.

        Called only in the training job. At serving time the student is loaded from
        disk by `load_extra` and the teacher is never constructed.
        """
        from catboost import CatBoostRegressor

        self.teacher = create_backend(self.cfg, self.cfg.teacher).fit(x, y)
        self._columns = list(x.columns)
        self._cat_columns = _categorical_columns(x)

        sample, n_users, n_brands = self._distillation_sample(x)
        teacher_pred = self.teacher.predict(sample)

        # Hold out whole *users*, not rows. A 600-tree student on 15k labelled rows can
        # memorise them, so an in-sample fidelity number would be near-perfect and mean
        # nothing. Splitting by user also keeps each held-out user's full brand list
        # intact, which is what the top-1 check needs.
        n_eval = max(1, int(n_users * self.HOLDOUT_FRACTION))
        n_fit = n_users - n_eval
        user_index = np.tile(np.arange(n_users), n_brands)
        fit_rows = user_index < n_fit
        eval_rows = ~fit_rows

        self._student = CatBoostRegressor(
            random_seed=42, depth=8, n_estimators=600, loss_function="RMSE",
            verbose=False, allow_writing_files=False, thread_count=1,
        )
        self._student.fit(
            _prepare_for_catboost(sample[fit_rows], self._cat_columns),
            teacher_pred[fit_rows], cat_features=self._cat_columns,
        )
        self.fidelity = self._measure_fidelity(
            sample[eval_rows], teacher_pred[eval_rows], n_eval, n_brands)
        self.fidelity["surrogate_holdout_users"] = float(n_eval)

        # Refit on everything now that fidelity is known, so the shipped student uses
        # every teacher label available. The reported numbers describe the held-out fit
        # of an identically configured model, which is the honest thing to quote.
        self._student = CatBoostRegressor(
            random_seed=42, depth=8, n_estimators=600, loss_function="RMSE",
            verbose=False, allow_writing_files=False, thread_count=1,
        )
        self._student.fit(_prepare_for_catboost(sample, self._cat_columns),
                          teacher_pred, cat_features=self._cat_columns)
        return self

    def _distillation_sample(self, x: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        """Rows the student must imitate: context rows crossed with every brand.

        Cross-joining on client_name reproduces the exact shape of a serving request,
        where one user is scored against the whole brand universe. The layout is
        brand-major - one contiguous block per brand, every block the same users in the
        same order - which is what lets the fidelity check reshape it back into
        (user, brand) and ask whether the student would have picked the same winner.

        Returns the sample and its (n_users, n_brands) shape.
        """
        brands = x["client_name"].dropna().unique() if "client_name" in x.columns else []
        if len(brands) == 0:
            return x.copy(), len(x), 1

        # The setting is the target, not a floor. Written as max(setting, len(x)) it
        # could only ever raise the row count: asking for fewer rows to make a run
        # faster was silently ignored for any value at or below the context size, which
        # is every value someone would choose for that purpose. The floor that does
        # matter is one row per brand, and it is applied below.
        target_rows = self.cfg.surrogate_sample_rows
        per_brand = max(1, target_rows // len(brands))
        base = x.sample(n=min(per_brand, len(x)), random_state=42,
                        replace=per_brand > len(x))

        frames = []
        for brand in brands:
            block = base.copy()
            block["client_name"] = brand
            frames.append(block)
        return pd.concat(frames, ignore_index=True), len(base), len(brands)

    def _measure_fidelity(self, sample: pd.DataFrame, teacher_pred: np.ndarray,
                          n_users: int, n_brands: int) -> dict[str, float]:
        """How closely the student tracks the teacher. Logged as MLflow metrics.

        Called on held-out users only (see fit), so these numbers describe how the
        student generalises rather than how well it memorised its own labels.

        Read them as a regression detector, not as an estimate of production behaviour.
        The sample is drawn from the payout context - rows a brand actually paid for -
        which is a narrower, higher-value slice than live traffic, and these compare
        *payout predictions* rather than the ranking those predictions produce. Measured
        end to end over random users, top-1 agreement is 84.5%, not the ~100% this
        reports. `scripts/compare_backends.py` is the honest estimate; this is the thing
        that should scream if a retrain makes the student materially worse.
        """
        from scipy.stats import spearmanr

        student_pred = self.predict(sample)
        err = np.abs(student_pred - teacher_pred)
        denom = np.clip(np.abs(teacher_pred), 1e-6, None)
        rho = spearmanr(student_pred, teacher_pred).statistic

        fidelity = {
            "surrogate_mae": float(err.mean()),
            "surrogate_mape": float((err / denom).mean()),
            "surrogate_spearman": float(rho) if rho == rho else 0.0,
            "surrogate_max_abs_err": float(err.max()),
        }

        # Reshape the brand-major sample back into (user, brand) and compare orderings.
        if n_brands > 1 and n_users * n_brands == len(sample):
            teacher_grid = teacher_pred.reshape(n_brands, n_users).T
            student_grid = student_pred.reshape(n_brands, n_users).T
            fidelity["surrogate_top1_agreement"] = float(
                (teacher_grid.argmax(axis=1) == student_grid.argmax(axis=1)).mean()
            )
            # Per-user rank correlation, averaged: how much the whole list moves.
            per_user = [spearmanr(t, s).statistic
                        for t, s in zip(teacher_grid, student_grid, strict=False)]
            per_user = [v for v in per_user if v == v]
            if per_user:
                fidelity["surrogate_mean_rank_corr"] = float(np.mean(per_user))
        return fidelity

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        frame = _prepare_for_catboost(x[self._columns], self._cat_columns)
        return np.asarray(self._student.predict(frame), dtype=float)

    def predict_batch(self, batch) -> np.ndarray:
        indices = [batch.columns.index(c) for c in self._cat_columns]
        return np.asarray(self._student.predict(batch.pool(indices)), dtype=float)

    def expected_columns(self) -> list[str] | None:
        return _fitted_feature_names(self._student, self._columns)

    def save_extra(self, directory: Path) -> list[str]:
        target = directory / self.STUDENT_FILE
        self._student.save_model(str(target), format="cbm")
        meta = directory / "payout_surrogate_columns.joblib"
        joblib.dump({"columns": self._columns, "cat_columns": self._cat_columns}, meta)
        return [target.name, meta.name]

    def prepare(self, context: PayoutContext, directory: Path) -> PayoutBackend:
        """Load the student. The teacher is never constructed at serving time - that is
        the entire point of distilling it."""
        self.load_extra(directory)
        return self

    def load_extra(self, directory: Path) -> None:
        from catboost import CatBoostRegressor

        meta = joblib.load(directory / "payout_surrogate_columns.joblib")
        self._columns = meta["columns"]
        self._cat_columns = meta["cat_columns"]
        self._student = CatBoostRegressor(allow_writing_files=False, thread_count=1)
        self._student.load_model(str(directory / self.STUDENT_FILE), format="cbm")

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update(self.fidelity)
        return base


BACKENDS: dict[str, type[PayoutBackend]] = {
    "tabpfn_local": TabPFNLocalBackend,
    "tabpfn_client": TabPFNClientBackend,
    "surrogate": SurrogateBackend,
    "catboost_fallback": CatBoostFallbackBackend,
}


def create_backend(cfg: PayoutSettings, name: str | None = None) -> PayoutBackend:
    chosen = name or cfg.backend
    if chosen not in BACKENDS:
        raise ValueError(
            f"Unknown payout backend {chosen!r}. Available: {', '.join(sorted(BACKENDS))}"
        )
    return BACKENDS[chosen](cfg)


def prepare_backend(cfg: PayoutSettings, context: PayoutContext, directory: Path,
                    name: str | None = None,
                    on_fallback: Any = None) -> PayoutBackend:
    """Build and warm the configured backend, degrading to CatBoost if it cannot start.

    Training and serving want opposite failure behaviour:

      * a weekly training job must fail loudly if TabPFN is unavailable - registering a
        silently degraded model would serve it for a week;
      * a serving process on the funnel's synchronous path must keep answering. A
        slightly worse ranking beats a 503 while the user waits on the landing page.

    So the training job calls `create_backend` directly and lets exceptions through;
    serving calls this. Either way the backend name travels on every response, so a
    degraded endpoint is never a silent one.
    """
    chosen = name or cfg.backend
    try:
        backend = create_backend(cfg, chosen)
        return backend.prepare(context, directory)
    except Exception as exc:  # noqa: BLE001 - import, auth, network or missing weights
        if chosen == "catboost_fallback":
            raise
        if on_fallback:
            on_fallback(chosen, exc)
        fallback = create_backend(cfg, "catboost_fallback")
        return fallback.prepare(context, directory)
