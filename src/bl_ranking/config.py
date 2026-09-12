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

# The two feature implementations, and the payout backends that exist. Kept here so a
# typo in either is a start-up error naming the alternatives, not a silent default.
FEATURE_PATHS = frozenset({"fast", "research"})
PAYOUT_BACKENDS = frozenset({"surrogate", "tabpfn_client", "tabpfn_local", "catboost_fallback"})

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
        settings = _build(cls, raw)
        settings.validate()
        return settings

    def validate(self) -> None:
        """Refuse a configuration that cannot work, at start-up rather than in traffic.

        Types are already enforced by `_build`; this is about values a correct type can
        still hold. Each of these was reachable and silent: zero or negative workers,
        a port outside the legal range, a payout backend that does not exist, and a
        feature path that is not one of the two implementations - the last of which
        simply meant "fast" while `GET /model` reported the bogus name back as though
        it were in use.
        """
        if self.serving.workers < 1:
            raise ValueError(f"serving.workers must be at least 1, got {self.serving.workers}")
        if self.serving.threads_per_worker < 1:
            raise ValueError(
                f"serving.threads_per_worker must be at least 1, "
                f"got {self.serving.threads_per_worker}"
            )
        if not 1 <= self.serving.port <= 65535:
            raise ValueError(f"serving.port must be 1..65535, got {self.serving.port}")
        if self.serving.feature_path not in FEATURE_PATHS:
            raise ValueError(
                f"serving.feature_path must be one of {sorted(FEATURE_PATHS)}, "
                f"got {self.serving.feature_path!r}"
            )
        if self.model.payout.backend not in PAYOUT_BACKENDS:
            raise ValueError(
                f"model.payout.backend must be one of {sorted(PAYOUT_BACKENDS)}, "
                f"got {self.model.payout.backend!r}"
            )
        if self.model.payout.teacher not in PAYOUT_BACKENDS:
            raise ValueError(
                f"model.payout.teacher must be one of {sorted(PAYOUT_BACKENDS)}, "
                f"got {self.model.payout.teacher!r}"
            )
        if self.model.payout.context_size < 1:
            raise ValueError(
                f"model.payout.context_size must be at least 1, "
                f"got {self.model.payout.context_size}"
            )

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


def env_override_keys() -> frozenset[str]:
    """Dotted key paths currently set as BL_ environment variables.

    Lets a caller tell "this setting holds its default" from "an operator asked for
    this", which the merged Settings object cannot express on its own - both arrive as
    the same plain value. Serving needs the distinction to honour an explicit backend
    override while still letting a rolled-back bundle carry its own.
    """
    keys = set()
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX) or key == "BL_CONFIG" or value == "":
            continue
        keys.add(".".join(key[len(ENV_PREFIX):].lower().split(NESTING_SEPARATOR)))
    return frozenset(keys)


def _env_overrides() -> dict[str, Any]:
    """Turn BL_SERVING__WORKERS=2 into {'serving': {'workers': '2'}}.

    Values stay as strings. They used to be run through `yaml.safe_load`, which is a
    guess rather than a conversion and got three things wrong that all failed silently:
    YAML 1.1 reads a leading zero as octal, so `030` meant 24 days rather than 30; any
    value containing ": " became a dict, so an experiment name like "a: b" replaced a
    string with `{'a': 'b'}`; and anything unrecognised was accepted verbatim, so
    `workers=many` reached uvicorn as the string it was. `_build` now converts each
    value to the type its field actually declares, which is the only place that
    information exists.
    """
    out: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX) or key == "BL_CONFIG":
            continue
        if value == "":
            # An empty value means "not set". Compose files and .env templates carry
            # `BL_X=${BL_X:-}` as a matter of course, and a shell exports that as an
            # empty string rather than omitting the variable. Treating it as a value
            # made an empty BL_SERVING__WORKERS abort the worker at import, which is a
            # far worse outcome than ignoring a variable nobody filled in.
            continue
        parts = key[len(ENV_PREFIX):].lower().split(NESTING_SEPARATOR)
        cursor = out
        for index, part in enumerate(parts[:-1]):
            existing = cursor.get(part)
            if existing is not None and not isinstance(existing, dict):
                prefix = ENV_PREFIX + NESTING_SEPARATOR.join(parts[: index + 1]).upper()
                raise ValueError(
                    f"{key} cannot be applied: {prefix} is also set as a value, so one "
                    f"of the two has to go. Nesting a key inside a scalar used to raise "
                    f"TypeError at import and take the process down."
                )
            cursor = cursor.setdefault(part, {})
        leaf = parts[-1]
        if isinstance(cursor.get(leaf), dict):
            # The same clash seen from the other side: BL_MODEL__PAYOUT__BACKEND was
            # read first and created the block that BL_MODEL__PAYOUT now wants to
            # replace with a string. Environment order decides which side you hit, so
            # both have to be refused or the failure is intermittent.
            raise ValueError(
                f"{key} cannot be applied: settings nested under it are also set "
                f"individually, so one of the two has to go."
            )
        cursor[leaf] = value
    return out


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def _build(cls: type, raw: dict[str, Any], path: str = "") -> Any:
    """Instantiate a nested dataclass tree from plain dicts, ignoring unknown keys.

    Field types are strings here (`from __future__ import annotations`), so nested
    blocks are detected from each field's *default value* rather than its annotation.
    That default is also what says how to read an environment override: this is the one
    place that knows a setting is meant to be an int rather than whatever a string
    happens to parse as.
    """
    instance = cls()
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        current = getattr(instance, f.name)
        where = f"{path}.{f.name}" if path else f.name
        if is_dataclass(current) and isinstance(value, dict):
            setattr(instance, f.name, _build(type(current), value, where))
        else:
            setattr(instance, f.name, _as_field_type(value, current, where, f.type))
    return instance


def _as_field_type(value: Any, default: Any, where: str, annotation: Any = None) -> Any:
    """Convert an override to the type its field declares, or say why it cannot be.

    Only strings are converted, so values read from the YAML file - already typed by
    the parser - pass through untouched. A bad value raises here, naming the setting,
    rather than travelling into the service as the wrong type and failing somewhere
    that gives no clue which variable caused it.
    """
    if not isinstance(value, str) or isinstance(default, str):
        return value
    text = value.strip()
    if isinstance(default, bool):
        if text.lower() in {"1", "true", "yes", "on"}:
            return True
        if text.lower() in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{where}: expected a boolean, got {value!r}")
    if isinstance(default, int):
        try:
            # Base 10 explicitly: a leading zero is a typo, never octal.
            return int(text, 10)
        except ValueError:
            raise ValueError(f"{where}: expected an integer, got {value!r}") from None
    if isinstance(default, float):
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{where}: expected a number, got {value!r}") from None
    if default is None:
        # Nullable settings: an explicit empty value means "unset".
        if text == "":
            return None
        # A default of None says nothing about the type, so read the annotation. With
        # `from __future__ import annotations` it is the source string, e.g.
        # "int | None" - enough to tell a nullable number from a nullable path.
        declared = str(annotation or "")
        if "bool" in declared:
            return _as_field_type(text, True, where)
        if "int" in declared:
            return _as_field_type(text, 0, where)
        if "float" in declared:
            return _as_field_type(text, 0.0, where)
        return text
    return value


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
