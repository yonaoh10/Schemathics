"""The configuration layer: what it accepts, what it converts, and what it refuses.

Every case here was reachable and silent before these tests existed. A service running
on a misread setting is worse than one that will not start, because nothing anywhere
says which variable did it.
"""

from __future__ import annotations

import pytest

from bl_ranking.config import Settings


def _load(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings.load()


def test_a_leading_zero_is_a_typo_not_octal(monkeypatch):
    """YAML 1.1 reads 030 as 24. A retrain window silently 20% short is invisible."""
    assert _load(monkeypatch, BL_SERVING__WORKERS="030").serving.workers == 30


def test_a_value_containing_a_colon_stays_a_string(monkeypatch):
    """`yaml.safe_load('a: b')` returns a dict, which replaced a string setting."""
    settings = _load(monkeypatch, BL_MLFLOW__EXPERIMENT="a: b")
    assert settings.mlflow.experiment == "a: b"


# "" is deliberately absent: an empty variable means "unset", not "invalid".
# See test_an_empty_override_is_ignored_not_fatal.
@pytest.mark.parametrize("value", ["many", "3.7", "1e3", "0x10"])
def test_a_non_integer_for_an_integer_setting_is_refused(monkeypatch, value):
    """The string 'many' used to reach uvicorn as a worker count."""
    with pytest.raises(ValueError, match="serving.workers"):
        _load(monkeypatch, BL_SERVING__WORKERS=value)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_worker_count_below_one_is_refused(monkeypatch, value):
    with pytest.raises(ValueError, match="at least 1"):
        _load(monkeypatch, BL_SERVING__WORKERS=value)


def test_a_port_outside_the_legal_range_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="1..65535"):
        _load(monkeypatch, BL_SERVING__PORT="70000")


def test_an_unknown_payout_backend_is_refused_with_the_alternatives(monkeypatch):
    """It used to be accepted here and fail much later, far from the cause."""
    with pytest.raises(ValueError, match="must be one of"):
        _load(monkeypatch, BL_MODEL__PAYOUT__BACKEND="nope")


def test_an_unknown_feature_path_is_refused(monkeypatch):
    """A typo silently meant 'fast' while GET /model reported the typo back."""
    with pytest.raises(ValueError, match="fast"):
        _load(monkeypatch, BL_SERVING__FEATURE_PATH="bogus")


def test_nesting_a_key_inside_a_scalar_names_both_variables(monkeypatch):
    """This raised TypeError at import and took the process down with no diagnosis."""
    with pytest.raises(ValueError, match="BL_MODEL__PAYOUT"):
        _load(monkeypatch, BL_MODEL__PAYOUT="x", BL_MODEL__PAYOUT__BACKEND="surrogate")


def test_valid_overrides_still_apply(monkeypatch):
    """The guard rails must not block the documented way of switching backends."""
    settings = _load(monkeypatch,
                     BL_MODEL__PAYOUT__BACKEND="tabpfn_local",
                     BL_SERVING__FEATURE_PATH="research",
                     BL_SERVING__WORKERS="7")
    assert settings.model.payout.backend == "tabpfn_local"
    assert settings.serving.feature_path == "research"
    assert settings.serving.workers == 7


def test_the_shipped_config_file_is_valid():
    """conf/config.yaml must satisfy the rules it is the example of."""
    assert Settings.load() is not None

@pytest.mark.parametrize("key", ["BL_SERVING__WORKERS", "BL_MODEL__PAYOUT__BACKEND"])
def test_an_empty_override_is_ignored_not_fatal(monkeypatch, key):
    """`BL_X=${BL_X:-}` is ordinary in compose files and .env templates.

    A shell exports that as an empty string rather than omitting the variable. Treating
    it as a value made an empty BL_SERVING__WORKERS abort the worker at import, which
    is worse than ignoring a variable nobody filled in.
    """
    monkeypatch.setenv(key, "")
    settings = Settings.load()
    assert settings.serving.workers == 3
    assert settings.model.payout.backend in {"surrogate", "catboost_fallback"}

