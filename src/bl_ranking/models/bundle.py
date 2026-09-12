"""The serving bundle: everything one model version needs, versioned as a unit.

The research code writes four loose files into a directory. Loose files are a rollback
hazard - if the CatBoost model and the payout context are copied separately, a serving
process can end up with a classifier from one training run and a payout context from
another, and nothing in the system would notice.

So a bundle is treated as atomic. It is registered to MLflow as a single model version;
rolling back means pointing the `champion` alias at an older version, and the whole set
moves together.

Contents
    CB_bl_lead.cbm              CatBoost classifier, native format (research artifact)
    payout_tfm_context.joblib   in-context rows for the payout model (research artifact)
    all_clients.csv             the brand universe to score against (research artifact)
    gender_lookup.parquet       precomputed gender feature (see models/gender_lut.py)
    manifest.json               provenance: run, data version, backend, row counts
    payout_surrogate.cbm        only when the surrogate backend is used
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Names the research code writes. Kept as constants so nothing drifts silently.
CATBOOST_FILE = "CB_bl_lead.cbm"
PAYOUT_CONTEXT_FILE = "payout_tfm_context.joblib"
CLIENTS_FILE = "all_clients.csv"
MANIFEST_FILE = "manifest.json"
GENDER_LOOKUP_FILE = "gender_lookup.parquet"   # models/gender_lut.ARTIFACT_NAME

# The single column all_clients.csv carries, and the name the research code joins on.
CLIENT_NAME_COLUMN = "client_name"


@dataclass
class Manifest:
    """Provenance for one model version. Written into the bundle and logged as tags."""

    trained_at: str = ""
    mlflow_run_id: str = ""
    delta_version: int = -1
    delta_table: str = ""
    rows_train: int = 0
    rows_payout_context: int = 0
    n_brands: int = 0
    payout_backend: str = ""
    payout_exact: bool = True
    git_sha: str = ""
    research_code_sha: str = ""
    train_columns: list[str] = field(default_factory=list)
    extra_files: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def write(self, directory: Path) -> Path:
        path = directory / MANIFEST_FILE
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return path

    @classmethod
    def read(cls, directory: Path) -> Manifest:
        path = Path(directory) / MANIFEST_FILE
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))

    def as_tags(self) -> dict[str, str]:
        """The subset worth having on the MLflow run and the registered version."""
        return {
            "delta_version": str(self.delta_version),
            "payout_backend": self.payout_backend,
            "payout_exact": str(self.payout_exact),
            "git_sha": self.git_sha,
            "rows_train": str(self.rows_train),
            "n_brands": str(self.n_brands),
        }


def current_git_sha(repo_root: Path | None = None) -> str:
    """Short SHA of the working tree, so a serving version maps back to source."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=5, check=True,
        )
        sha = result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        dirty = ""
    return f"{sha}-dirty" if dirty else sha


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def verify(directory: Path) -> list[str]:
    """Return the list of required files that are missing. Empty list means valid.

    The manifest is required too. The gender table deliberately is not: building it is
    a config flag (`model.build_gender_lookup`), so a bundle without one is legitimate.
    Serving handles its absence by doing the live lookup instead - see ranker.load -
    rather than by quietly filling the feature with a constant.
    """
    required = [CATBOOST_FILE, PAYOUT_CONTEXT_FILE, CLIENTS_FILE, MANIFEST_FILE]
    return [name for name in required if not (Path(directory) / name).exists()]
