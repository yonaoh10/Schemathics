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
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, Schema
from pydantic import ValidationError

from bl_ranking.config import Settings
from bl_ranking.data.schema import REQUEST_FIELDS
from bl_ranking.serving.ranker import WARMUP_USER, BrandRanker, InsufficientSurveyData
from bl_ranking.serving.schemas import RankRequest

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
        errors: list[str | None] = []
        for record in frame.to_dict(orient="records"):
            ranking: dict[str, Any] = {}
            error: str | None = None
            try:
                # The same validation the HTTP endpoint applies. Without it this path
                # accepted whatever pandas happened to infer for a column, so a row's
                # features depended on which other rows shared its batch - a null in
                # one row could turn every id in the column into a float. Validating
                # each record on its own removes that coupling and applies one contract
                # to both serving surfaces.
                user = RankRequest(**_scrub(record)).to_user_data()
            except ValidationError as exc:
                errors.append(f"invalid_request: {exc.error_count()} field(s)")
                rankings.append(json.dumps({}))
                continue
            except Exception as exc:  # noqa: BLE001 - one bad row must not fail a batch
                errors.append(f"invalid_request: {type(exc).__name__}")
                rankings.append(json.dumps({}))
                continue
            try:
                ranking = self._ranker.rank(user)
            except InsufficientSurveyData:
                error = "insufficient_survey_answers"
            except Exception as exc:  # noqa: BLE001 - one bad row must not fail a batch
                error = f"{type(exc).__name__}: {exc}"
            rankings.append(json.dumps(ranking))
            errors.append(error)
        # `error` is its own column. It used to be smuggled into the ranking as a brand
        # named __error__ whose value was a string where every real entry is a
        # {rank, expected_payout} object, so any consumer that iterated the ranking
        # either crashed or silently treated the message as a lender.
        return pd.DataFrame({
            "ranking": rankings,
            "error": errors,
            "payout_backend": [backend] * len(rankings),
            "payout_exact": [exact] * len(rankings),
        })



def _scrub(record: dict[str, Any]) -> dict[str, Any]:
    """pandas renders a missing value as NaN; the request schema expects None.

    `pd.isna` returns an *array* for a list or array cell, and `if` on that raises
    "truth value of an array is ambiguous" - which escaped the per-row handler and
    failed the entire batch over one malformed cell. Only scalars are tested; anything
    else is passed through for RankRequest to reject as the single bad row it is.
    """
    return {k: (None if _is_missing(v) else v) for k, v in record.items()}


def _is_missing(value: Any) -> bool:
    if isinstance(value, str):
        return False
    if isinstance(value, list | tuple | dict | set) or hasattr(value, "__len__"):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False

def _as_frame(model_input: Any) -> pd.DataFrame:
    if isinstance(model_input, pd.DataFrame):
        return model_input
    if isinstance(model_input, dict):
        return pd.DataFrame([model_input])
    return pd.DataFrame(model_input)


# What RankRequest insists on. Everything else is optional there, and has to be optional
# here too, or the portable form refuses payloads the HTTP endpoint accepts.
REQUIRED_REQUEST_FIELDS: tuple[str, ...] = (
    "session_dt", "register_date", "campaign_id", "page",
)


def request_example() -> pd.DataFrame:
    """One-row input example, stored with the model so the schema is self-documenting.

    Ids are rendered as text to match the declared schema - see build_signature.
    """
    return pd.DataFrame([{k: _as_text(WARMUP_USER[k]) for k in REQUEST_FIELDS}])


def _as_text(value: Any) -> Any:
    return value if value is None or isinstance(value, str) else str(value)


def build_signature() -> ModelSignature:
    """Declare the input schema rather than inferring it from one example row.

    `infer_signature` read the example and declared campaign_id, sub1, sub3 and
    cellphone as `long`, and every one of the 22 fields as required. Both are narrower
    than the contract the HTTP endpoint honours, and each broke a whole batch:

      * one null `cellphone` - an optional field - demotes its column to float64, which
        enforcement then refuses as an unsafe cast to int64. A single missing phone
        number failed all 500 rows, which is exactly the coupling per-row validation
        exists to remove.
      * a 17-digit `campaign_id` cannot cross a columnar boundary as a number at all.
        One null in the column makes it float64 and 120227360861540306 becomes
        ...304 - the same float64 demotion the ingestion gate reads these columns as
        text to avoid (data/ingest.READ_AS_TEXT), on the other side of the system.

    So: text for every field, and required only for the four RankRequest requires. The
    per-row validator converts each value on its own, which is what makes a row's
    features independent of the rest of its batch. The cost is explicit - a caller that
    sends an id as a JSON number gets an MLflow type error naming the column, instead of
    a silently rounded id - and the HTTP endpoint still accepts either spelling.
    """
    inputs = Schema([
        ColSpec(DataType.string, name, required=name in REQUIRED_REQUEST_FIELDS)
        for name in REQUEST_FIELDS
    ])
    outputs = Schema([
        ColSpec(DataType.string, "ranking"),
        ColSpec(DataType.string, "error", required=False),
        ColSpec(DataType.string, "payout_backend"),
        ColSpec(DataType.boolean, "payout_exact"),
    ])
    return ModelSignature(inputs=inputs, outputs=outputs)
