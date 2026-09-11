"""Typed configuration loaded from conf/config.yaml plus BL_* environment overrides.

Every knob the pipeline reads goes through Settings so that a training run can log
its own configuration to MLflow and be reproduced from that alone.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# Repository root, resolved from this file so the package works from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    """Where to read conf/config.yaml from.

    Three cases, in order:

      1. `BL_CONFIG` names a file explicitly. This is what a Databricks job task sets,
         because a wheel installed into site-packages has no repository above it.
      2. `./conf/config.yaml` relative to the working directory.
      3. `conf/config.yaml` beside the source tree, which is the repo checkout case.

    If none exists, every setting falls back to its dataclass default and the run still
    works - `Settings.flat()` is logged to MLflow either way, so what was actually used
    is always recoverable from the run.
    """
    override = os.environ.get("BL_CONFIG")
    if override:
        return Path(override)
    local = Path.cwd() / "conf" / "config.yaml"
    if local.exists():
        return local
    return REPO_ROOT / "conf" / "config.yaml"


DEFAULT_CONFIG = _default_config_path()

ENV_PREFIX = "BL_"
NESTING_SEPARATOR = "__"


@dataclass
class PathSettings:
    raw_dir: str = "data/raw"
    raw_file: str = "bl_full_data.csv"
    delta_table: str = "data/delta/bl_sessions"
    run_root: str = "runs"

    @property
    def raw_csv(self) -> Path:
        return _abs(self.raw_dir) / self.raw_file


@dataclass
class DataSettings:
    lookback_days: int | None = None
    days_for_test: int = 7


@dataclass
class CatBoostSettings:
    random_seed: int = 42
    depth: int = 8
    n_estimators: int = 800
    task_type: str = "CPU"
    eval_metric: str = "F1"


@dataclass
class PayoutSettings:
    backend: str = "surrogate"
    # Which TabPFN produces the surrogate's labels. Ignored by the other backends.
    teacher: str = "tabpfn_local"
    context_size: int = 1000
    fit_mode: str = "fit_with_cache"
    n_estimators: int = 4
    device: str = "auto"
    # Pin the checkpoint. Left unset, the hosted API picks its current default and a
    # provider-side upgrade silently changes payouts between two weekly runs.
    model_path: str | None = None
    surrogate_sample_rows: int = 20000


@dataclass
class ModelSettings:
    catboost: CatBoostSettings = field(default_factory=CatBoostSettings)
    payout: PayoutSettings = field(default_factory=PayoutSettings)
    # Materialise the gender feature into the bundle (see models/gender_lut.py).
    # Costs ~47 s at training time and removes names-dataset from the serving image.
    build_gender_lookup: bool = True


@dataclass
class MLflowSettings:
    tracking_uri: str | None = None
    experiment: str = "bl_brand_ranking"
    registered_model: str = "bl_brand_ranker"
    serving_alias: str = "champion"

    def resolved_tracking_uri(self) -> str:
        """Environment wins over the file so containers can be pointed at a server."""
        return os.environ.get("MLFLOW_TRACKING_URI") or self.tracking_uri or f"file://{_abs('mlruns')}"


@dataclass
class ServingSettings:
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 3
    threads_per_worker: int = 1
    request_timeout_ms: int = 1500
    model_uri: str | None = None
    # "fast" (default) or "research". See serving/fast_features.py.
    feature_path: str = "fast"


@dataclass
class ScheduleSettings:
    cron: str = "0 0 5 ? * SUN *"
    timezone: str = "UTC"
    pause_status: str = "UNPAUSED"


@dataclass
class GeneratorSettings:
    sessions: int = 60000
    start_date: str = "2025-12-11"
    end_date: str = "2026-02-11"
    seed: int = 20260211


@dataclass
class Settings:
    paths: PathSettings = field(default_factory=PathSettings)
    data: DataSettings = field(default_factory=DataSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    mlflow: MLflowSettings = field(default_factory=MLflowSettings)
    serving: ServingSettings = field(default_factory=ServingSettings)
    schedule: ScheduleSettings = field(default_factory=ScheduleSettings)
    generator: GeneratorSettings = field(default_factory=GeneratorSettings)

    @classmethod
    def load(cls, config_path: str | Path | None = None, **overrides: Any) -> Settings:
        raw = _read_yaml(Path(config_path) if config_path else _default_config_path())
        _merge(raw, _env_overrides())
        _merge(raw, overrides)
        return _build(cls, raw)

    def flat(self, prefix: str = "") -> dict[str, Any]:
        """Dotted key/value view, used to log the whole config as MLflow params."""
        return _flatten(asdict(self), prefix)


def _abs(value: str | Path) -> Path:
    """Resolve a possibly-relative config path against the repository root."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _env_overrides() -> dict[str, Any]:
    """Turn BL_SERVING__WORKERS=2 into {'serving': {'workers': 2}}."""
    out: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX) or value == "":
            continue
        parts = key[len(ENV_PREFIX):].lower().split(NESTING_SEPARATOR)
        cursor = out
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce(value)
    return out


def _coerce(value: str) -> Any:
    """YAML-parse scalars so BL_..._WORKERS=2 arrives as an int, not '2'."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def _build(cls: type, raw: dict[str, Any]) -> Any:
    """Instantiate a nested dataclass tree from plain dicts, ignoring unknown keys.

    Field types are strings here (`from __future__ import annotations`), so nested
    blocks are detected from each field's *default value* rather than its annotation.
    """
    instance = cls()
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        current = getattr(instance, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            setattr(instance, f.name, _build(type(current), value))
        else:
            setattr(instance, f.name, value)
    return instance


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out[prefix] = node
    return out


def resolve(value: str | Path) -> Path:
    """Public helper: make a config path absolute against the repository root."""
    return _abs(value)
