"""MLflow pyfunc wrapper around the ranker.

Why this exists as well as the FastAPI app: the pyfunc is the *portable* form of a model
version. The same registered artifact can be

  * loaded by the FastAPI container (`mlflow.artifacts.download_artifacts`),
  * served directly with `mlflow models serve`,
  * deployed to a Databricks Model Serving endpoint,

without any of them needing to know how the bundle is laid out. That is what makes
"identify the serving version and roll it back" a one-line operation: move the
`champion` alias.

`load_context` runs once per worker process, which is exactly where the expensive
start-up work belongs - the CatBoost load, the payout context fit and the gender table.

The output column is a JSON string rather than a nested object. MLflow's schema
enforcement has no way to describe a dictionary whose keys are brand names, and the
serving path must survive a brand universe that changes at every retrain. The FastAPI
endpoint returns the native dictionary; only this portable form serialises it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import ModelSignature, infer_signature

from bl_ranking.config import Settings
from bl_ranking.data.schema import REQUEST_FIELDS
from bl_ranking.serving.ranker import WARMUP_USER, BrandRanker, InsufficientSurveyData

BUNDLE_ARTIFACT_KEY = "bundle"


class BrandRankerModel(mlflow.pyfunc.PythonModel):
    """Scores each input row against the brand universe."""

    def load_context(self, context: Any) -> None:
        bundle_dir = Path(context.artifacts[BUNDLE_ARTIFACT_KEY])
        self._ranker = BrandRanker.load(bundle_dir, Settings.load())

    def predict(self, context: Any, model_input: Any, params: dict | None = None) -> pd.DataFrame:
        """One row in, one ranking out, plus which model produced it.

        `payout_backend` and `payout_exact` ride along on every row for the same reason
        the HTTP endpoint returns them: when the payout model is the distilled student
        rather than TabPFN itself, the ranking is an approximation, and a caller must be
        able to see that without consulting a config file. Serving through MLflow rather
        than through our own app must not lose that.
        """
        described = self._ranker.describe()
        backend = str(described["payout_backend"])
        exact = bool(described["payout_exact"])

        frame = _as_frame(model_input)
        rankings: list[str] = []
        for record in frame.to_dict(orient="records"):
            try:
                ranking = self._ranker.rank(record)
            except InsufficientSurveyData:
                ranking = {}
            except Exception as exc:  # noqa: BLE001 - one bad row must not fail a batch
                ranking = {"__error__": str(exc)}
            rankings.append(json.dumps(ranking))
        return pd.DataFrame({
            "ranking": rankings,
            "payout_backend": [backend] * len(rankings),
            "payout_exact": [exact] * len(rankings),
        })


def _as_frame(model_input: Any) -> pd.DataFrame:
    if isinstance(model_input, pd.DataFrame):
        return model_input
    if isinstance(model_input, dict):
        return pd.DataFrame([model_input])
    return pd.DataFrame(model_input)


def request_example() -> pd.DataFrame:
    """One-row input example, stored with the model so the schema is self-documenting."""
    return pd.DataFrame([{k: WARMUP_USER[k] for k in REQUEST_FIELDS}])


def build_signature() -> ModelSignature:
    example = request_example()
    output = pd.DataFrame({
        "ranking": ['{"brand": {"rank": 1.0, "expected_payout": 42.31}}'],
        "payout_backend": ["surrogate"],
        "payout_exact": [False],
    })
    return infer_signature(example, output)
