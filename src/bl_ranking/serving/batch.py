"""The per-request scoring batch: one user x N brands, materialised once.

Both models score the same 25 columns for the same rows, so the row data is built once
and each model takes the representation it wants. Measured on a 15-brand request:

    frame.to_numpy(object)          0.02 ms
    catboost.Pool(matrix, cats)     0.33 ms   <- built once, used by both models
    pandas DataFrame construction   1.40 ms   <- only built if a model needs one

Before this, the two CatBoost calls each rebuilt their own input and the payout backend
re-stringified every categorical column on every request, which alone cost 6.4 ms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# CatBoost will not accept a null in a categorical column. The research code's own
# CatBoost calls have the same constraint; this is the string it would have produced.
NULL_CATEGORY = "nan"


@dataclass
class ScoringBatch:
    """Row data for one request, in whichever form the caller needs."""

    matrix: np.ndarray                  # (n_rows, n_columns), dtype=object
    columns: list[str]
    _frame: pd.DataFrame | None = field(default=None, repr=False)
    _pools: dict[tuple[int, ...], Any] = field(default_factory=dict, repr=False)

    @property
    def n_rows(self) -> int:
        return int(self.matrix.shape[0])

    def frame(self) -> pd.DataFrame:
        """A pandas view, built at most once per request."""
        if self._frame is None:
            self._frame = pd.DataFrame(self.matrix, columns=self.columns)
        return self._frame

    def pool(self, cat_indices: list[int]) -> Any:
        """A CatBoost Pool, cached per categorical-index signature.

        Both the classifier and a CatBoost payout backend derive their categorical
        columns from the same training frame, so in practice this is built once and
        reused by both.
        """
        from catboost import Pool

        key = tuple(cat_indices)
        pool = self._pools.get(key)
        if pool is None:
            pool = Pool(data=self._catboost_safe(cat_indices), cat_features=cat_indices)
            self._pools[key] = pool
        return pool

    def _catboost_safe(self, cat_indices: list[int]) -> np.ndarray:
        """Replace nulls in the categorical columns only, leaving numerics untouched."""
        if not cat_indices:
            return self.matrix
        matrix = self.matrix
        block = matrix[:, cat_indices]
        if pd.isnull(block).any():
            matrix = matrix.copy()
            block = matrix[:, cat_indices]
            block[pd.isnull(block)] = NULL_CATEGORY
            matrix[:, cat_indices] = block
        return matrix


def from_row(row: dict[str, Any], columns: list[str], brands: np.ndarray,
             brand_column: str = "client_name") -> ScoringBatch:
    """Broadcast one user's features across the brand universe.

    The research pipeline cross-joins first and then computes features, which produces
    N identical copies of every user-level value. Computing them once and tiling is the
    same result for a fraction of the work.
    """
    n = len(brands)
    matrix = np.empty((n, len(columns)), dtype=object)
    for j, column in enumerate(columns):
        if column == brand_column:
            matrix[:, j] = brands
        else:
            matrix[:, j] = row[column]
    return ScoringBatch(matrix=matrix, columns=list(columns))
