"""Warm, process-local brand ranker.

The research predictor is correct but constructed for a script: every call to
`predict_()` builds a new log file, re-reads all_clients.csv from disk, re-loads the
CatBoost model, and re-fits the TabPFN context (which, on the hosted client, is a
network round trip). On a synchronous path where the user is waiting on the landing
page, that is the whole latency budget spent before any arithmetic happens.

This module keeps the research feature pipeline exactly as written and moves everything
that does not depend on the request to start-up:

  built once, at start-up          paid per request
  ----------------------          ----------------
  CatBoost model load             ~15 rows of feature engineering
  payout context fit              1 predict_proba
  all_clients.csv read            1 payout predict
  gender lookup table             sort + rank
  warning handler install

Two feature paths are available (`serving.feature_path`). The default, `fast`, is
serving/fast_features.py. The reference, `research`, runs the original pipeline and
lives in serving/research_path.py, which is imported on demand - importing it costs
18 s and 2.4 GB because the research module builds a NameDataset at module scope.
"""

from __future__ import annotations

import logging
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier

from bl_ranking.config import Settings
from bl_ranking.models import bundle as bundle_files
from bl_ranking.models.gender_lut import ARTIFACT_NAME as GENDER_ARTIFACT
from bl_ranking.models.gender_lut import UNKNOWN, GenderLookup
from bl_ranking.models.payout import PayoutBackend, PayoutContext, prepare_backend
from bl_ranking.serving import batch, fast_features

log = logging.getLogger("bl_ranking.serving")


@dataclass
class WarmModels:
    """Everything a request needs, already in memory."""

    catboost: CatBoostClassifier
    payout: PayoutBackend
    columns: list[str]
    all_clients: pd.DataFrame
    gender: GenderLookup | None
    manifest: bundle_files.Manifest
    bundle_dir: Path

    @property
    def n_brands(self) -> int:
        return int((self.all_clients["client_name"] != "other").sum())


# The name the serving layer and the API use. It is the same exception the feature
# path raises, aliased here so callers do not have to reach into fast_features.
InsufficientSurveyData = fast_features.InsufficientSurveyAnswers


class BrandRanker:
    """The object the endpoint holds. Thread-safe for concurrent `rank` calls.

    Two feature paths, selected by `serving.feature_path`:

      fast      serving/fast_features.py - the same transformation over plain values,
                ~50 ms of pandas overhead removed. Default.
      research  the research pipeline exactly as written. The reference the fast path
                is tested against, and the escape hatch if anything is ever in doubt.

    tests/test_feature_equivalence.py asserts the two produce identical rankings.
    """

    def __init__(self, warm: WarmModels, feature_path: str = "fast") -> None:
        self.warm = warm
        self.feature_path = feature_path
        # CatBoost and the payout backends are read-only once fitted, but pandas
        # operations on shared frames are not guaranteed re-entrant. The lock keeps a
        # worker honest; parallelism comes from running several worker processes, which
        # is the only kind that helps for GIL-bound work anyway.
        self._lock = threading.Lock()
        # Brand universe as a fixed numpy array, in the order the research cross join
        # would have produced.
        clients = warm.all_clients["client_name"]
        self._brands = clients[clients != "other"].to_numpy()
        # Categorical column positions, taken from the trained classifier so they can
        # never drift from what the model was fitted with.
        self._cat_indices = list(warm.catboost.get_cat_feature_indices())

    # -- construction ---------------------------------------------------------------

    @classmethod
    def load(cls, bundle_dir: str | Path, settings: Settings | None = None) -> BrandRanker:
        settings = settings or Settings.load()
        directory = Path(bundle_dir)
        missing = bundle_files.verify(directory)
        if missing:
            raise FileNotFoundError(
                f"Model bundle at {directory} is incomplete, missing: {', '.join(missing)}"
            )

        manifest = bundle_files.Manifest.read(directory)
        context = PayoutContext.from_artifact(directory / bundle_files.PAYOUT_CONTEXT_FILE)

        catboost = CatBoostClassifier(
            allow_writing_files=False,
            thread_count=settings.serving.threads_per_worker,
        ).load_model(str(directory / bundle_files.CATBOOST_FILE), format="cbm")

        _assert_columns_match_the_classifier(catboost, context.columns, directory)

        # The bundle records which backend produced it; that wins over local config,
        # so a rollback to an older version brings its own backend with it.
        backend_name = manifest.payout_backend or settings.model.payout.backend
        payout = prepare_backend(settings.model.payout, context, directory,
                                 name=backend_name, on_fallback=_log_fallback)

        gender_path = directory / GENDER_ARTIFACT
        if gender_path.exists():
            gender = GenderLookup.load(gender_path)
        else:
            # A bundle built with model.build_gender_lookup off. The fallback used to
            # be logged but not performed: the vectorised path answered 'unknown' for
            # every name while the research path did the real lookup, so the two
            # implementations disagreed on a model feature with nothing to show for it.
            log.warning(
                "%s not in the bundle; using the live names-dataset lookup instead "
                "(18s start-up, 2.4GB resident). Rebuild with model.build_gender_lookup "
                "enabled to avoid both.", GENDER_ARTIFACT,
            )
            try:
                gender = GenderLookup.live()
            except ImportError as exc:
                raise RuntimeError(
                    f"Model bundle at {directory} has no {GENDER_ARTIFACT} and "
                    f"names-dataset is not installed, so the gender feature cannot be "
                    f"computed at all. Rebuild the bundle with "
                    f"model.build_gender_lookup enabled."
                ) from exc

        all_clients = pd.read_csv(directory / bundle_files.CLIENTS_FILE)
        _assert_usable_brand_universe(all_clients, directory)

        install_warning_capture()
        warm = WarmModels(
            catboost=catboost, payout=payout, columns=context.columns,
            all_clients=all_clients, gender=gender, manifest=manifest,
            bundle_dir=directory,
        )
        ranker = cls(warm, feature_path=settings.serving.feature_path)
        ranker.warmup()
        return ranker

    # -- request path ---------------------------------------------------------------

    def rank(self, user: dict[str, Any]) -> dict[str, dict[str, float]]:
        """Score one user against every brand and return the research ranking dict.

        The survey-completeness guard runs before either path. It has to: the research
        pipeline checks for an empty frame only after import_preprocess
        (bl_exp_payout_predictor.py line 197), so a user whose answers are dropped by
        `dropna(thresh=5)` inside impute_survey_columns carries on into
        additional_features and dies there with "Columns must be same length as key" -
        `.apply` on an empty Series returns a Series, not the two-column frame the
        assignment expects. The documented {"expected_payout": 0, "prob_lead": 0}
        sentinel is unreachable from that path.

        Checking up front turns that into one explicit refusal, identical on both
        paths, which the API renders as a 422.
        """
        fast_features.require_survey_answers(user)
        if self.feature_path == "research":
            return self._rank_via_research_pipeline(user)
        return self._rank_fast(user)

    def _rank_fast(self, user: dict[str, Any]) -> dict[str, dict[str, float]]:
        row = fast_features.build_feature_row(user, self.warm.gender)
        scoring = batch.from_row(row, self.warm.columns, self._brands)
        with self._lock:
            # One Pool serves both models: they were fitted on the same columns with
            # the same categorical set.
            pool = scoring.pool(self._cat_indices)
            prob_lead = self.warm.catboost.predict_proba(pool)[:, 1]
            payout = self.warm.payout.predict_batch(scoring)
        return fast_features.rank_from_scores(self._brands, prob_lead, payout)

    def _rank_via_research_pipeline(self, user: dict[str, Any]) -> dict[str, dict[str, float]]:
        # Imported here, not at module scope: the research predictor module runs
        # `nd = NameDataset()` at import, which costs 18 s and 2.4 GB. The fast path
        # never needs it, so a default serving process never pays for it.
        from bl_ranking.serving.research_path import ServingPredictor

        with self._lock:
            predictor = ServingPredictor(user, self.warm)
            result = predictor.predict_()

        # predict_ returns the ranking dict, or the research code's sentinel shape when
        # preprocessing dropped the row. Distinguish them explicitly.
        if set(result) == {"expected_payout", "prob_lead"}:
            raise InsufficientSurveyData(
                "survey answers are too sparse to score this user"
            )
        return result

    def warmup(self) -> None:
        """Pay every first-call cost before the process reports itself ready.

        CatBoost allocates its evaluation buffers lazily and TabPFN's first forward
        pass is slower than the rest; doing this here keeps that cost off a real user.
        """
        try:
            self.rank(WARMUP_USER)
        except InsufficientSurveyData:
            pass
        except Exception as exc:  # noqa: BLE001 - warm-up must never mask a real failure
            raise RuntimeError(f"warm-up scoring failed: {exc}") from exc

    def describe(self) -> dict[str, Any]:
        info = {
            "feature_path": self.feature_path,
            "model_version": self.warm.manifest.mlflow_run_id or "unregistered",
            "trained_at": self.warm.manifest.trained_at,
            "delta_version": self.warm.manifest.delta_version,
            "n_brands": self.warm.n_brands,
            "gender_lookup": "precomputed" if self.warm.gender else "names_dataset",
        }
        info.update(self.warm.payout.describe())
        return info


# A representative post-funnel payload, used for warm-up and by the smoke tests.
# Taken from the example in bl_exp_payout_predictor.py lines 249-272.
WARMUP_USER: dict[str, Any] = {
    "session_dt": "2026-01-06 19:24:22",
    "conversion_dt": "2026-01-06 19:26:10",
    "register_date": "2026-01-06 19:26:07",
    "campaign_id": 120227360861540306,
    "page": "top10us.com/app/business-loans-v2",
    "auto_city": "Fort Lauderdale",
    "auto_country": "United States",
    "auto_state": "Florida",
    "device_type": "mobile",
    "sub1": 1121993,
    "sub2": "01121993 Ad set",
    "sub3": 1513124082,
    "business_type": "C Corporation",
    "credit_score": "Very Poor - Under 550",
    "industry": "construction",
    "loan_amount": "$25,000 - $49,999",
    "loan_reason": "Equipment purchase",
    "monthly_revenue": "$20,000 - $49,999",
    "time_in_business": "2+ years",
    "fname": "Rigoberto",
    "lname": "Rodriguez",
    "cellphone": 7869914030,
}

_warning_capture_installed = False


def install_warning_capture() -> None:
    """Route library warnings into the application logger, once per process.

    The research code does this per instance (`capture_warnings`), which on a serving
    path means rebinding a global on every request.
    """
    global _warning_capture_installed
    if _warning_capture_installed:
        return

    def _warn(message, category, filename, lineno, file=None, line=None):
        log.warning("%s: %s (%s:%s)", category.__name__, message, filename, lineno)

    warnings.showwarning = _warn
    _warning_capture_installed = True




def _assert_usable_brand_universe(all_clients: pd.DataFrame, directory: Path) -> None:
    """Refuse a brand universe that cannot produce an honest ranking.

    Three ways it used to go wrong, all of them quietly:

    * Empty. Every request returned 200 with an empty ranking while /readyz stayed
      green - a total outage that looks like a healthy service from outside.
    * A null or blank name. pandas reads it back as NaN, `str()` makes it the literal
      "nan", and the endpoint offers the funnel a lender by that name.
    * Duplicates. The cross join produces the brand twice, the ranking collapses it to
      one entry, and the response then has fewer brands than the bundle claims - with
      no rank 1 in some orderings.
    """
    column = bundle_files.CLIENT_NAME_COLUMN
    if column not in all_clients.columns:
        raise ValueError(
            f"Model bundle at {directory}: {bundle_files.CLIENTS_FILE} has no "
            f"{column!r} column; columns present are {list(all_clients.columns)}."
        )

    names = all_clients[column]
    blank = names.isna() | (names.astype(str).str.strip() == "")
    if blank.any():
        raise ValueError(
            f"Model bundle at {directory}: {bundle_files.CLIENTS_FILE} has "
            f"{int(blank.sum())} blank brand name(s), which would be served as a "
            f"lender literally called 'nan'."
        )

    usable = names.astype(str).str.strip()
    if usable.empty:
        raise ValueError(
            f"Model bundle at {directory} has an empty brand universe "
            f"({bundle_files.CLIENTS_FILE}); there is nothing to rank."
        )

    duplicated = usable[usable.duplicated()].unique()
    if len(duplicated):
        raise ValueError(
            f"Model bundle at {directory}: {bundle_files.CLIENTS_FILE} lists "
            f"{list(duplicated)[:5]} more than once. Duplicates collapse in the "
            f"ranking, so the response would carry fewer brands than the bundle has."
        )

def _assert_columns_match_the_classifier(
    catboost: CatBoostClassifier, columns: list[str], directory: Path
) -> None:
    """Refuse a bundle whose feature order does not match what the model was fitted on.

    `prediction_expected_payout` reorders the frame by the payout model's column list
    and hands it to the classifier, so if the two disagree the features are silently
    mis-slotted: industry scored as page, city as device_type. There is no error and no
    log line - just a worse ranking, for as long as that version is the champion.

    Registering the five files as one atomic version stops them being mixed across
    runs, but it cannot catch a bundle that was written inconsistently in the first
    place, or edited afterwards. CatBoost stores its own ordered feature names, so the
    check costs nothing and is exact.

    Raising here rather than warning is deliberate: the caller is a worker starting up,
    /readyz stays 503, and the load balancer routes around it. A wrong ranking served
    confidently is the more expensive failure.
    """
    fitted = list(catboost.feature_names_ or [])
    if not fitted:
        # An older model file may carry no names. Nothing to check against.
        return
    if fitted != list(columns):
        raise ValueError(
            f"Model bundle at {directory} is inconsistent: the payout context lists "
            f"{len(columns)} feature columns but the classifier was fitted on "
            f"{len(fitted)}, or in a different order. Scoring would mis-slot features "
            f"silently. First difference at position "
            f"{next((i for i, (a, b) in enumerate(zip(fitted, columns, strict=False)) if a != b), min(len(fitted), len(columns)))}: "
            f"classifier expects {fitted!r}, bundle provides {list(columns)!r}"
        )

def _log_fallback(requested: str, exc: Exception) -> None:
    log.error(
        "payout backend %r could not be constructed (%s); serving degraded to "
        "catboost_fallback - rankings remain available but are not TabPFN's",
        requested, exc,
    )


__all__ = [
    "BrandRanker",
    "InsufficientSurveyData",
    "WarmModels",
    "WARMUP_USER",
    "UNKNOWN",
]
