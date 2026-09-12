"""The configuration layer: what it accepts, what it converts, and what it refuses.

Every case here was reachable and silent before these tests existed. A service running
on a misread setting is worse than one that will not start, because nothing anywhere
says which variable did it.
"""

from __future__ import annotations

from pathlib import Path

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

def test_a_config_path_that_is_not_a_file_says_so(monkeypatch, tmp_path):
    """BL_CONFIG pointing at a directory used to raise a bare IsADirectoryError."""
    directory = tmp_path / "conf"
    directory.mkdir()
    monkeypatch.setenv("BL_CONFIG", str(directory))
    with pytest.raises(IsADirectoryError, match="BL_CONFIG"):
        Settings.load()


def test_a_config_file_that_is_not_a_mapping_says_so(monkeypatch, tmp_path):
    """A YAML list parsed fine and then silently gave every setting its default."""
    path = tmp_path / "config.yaml"
    path.write_text("- a\n- b\n")
    monkeypatch.setenv("BL_CONFIG", str(path))
    with pytest.raises(ValueError, match="mapping of settings"):
        Settings.load()


def test_no_config_file_anywhere_is_still_fine(monkeypatch):
    """Every setting has a default and the effective config is logged either way.

    Expressed through the *search* rather than through BL_CONFIG, which is the distinction
    that matters: a path nobody named may be absent, a path somebody named may not. The
    test used to make that point with BL_CONFIG and so asserted the behaviour that let the
    Databricks bundle read no config at all in silence.
    """
    import bl_ranking.config as config

    monkeypatch.delenv("BL_CONFIG", raising=False)
    monkeypatch.setattr(config, "_default_config_path",
                        lambda: Path("/nonexistent/conf/config.yaml"))
    assert Settings.load().serving.port == 8080


@pytest.mark.parametrize("channel", ["env", "argument"])
def test_a_config_file_somebody_named_must_exist(monkeypatch, tmp_path, channel):
    absent = tmp_path / "absent.yaml"
    if channel == "env":
        monkeypatch.setenv("BL_CONFIG", str(absent))
        with pytest.raises(FileNotFoundError, match="does not exist"):
            Settings.load()
    else:
        monkeypatch.delenv("BL_CONFIG", raising=False)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            Settings.load(config_path=absent)


@pytest.mark.parametrize("variable", ["BL_SERVING", "BL_MODEL"])
def test_overriding_a_whole_block_with_a_scalar_names_the_mistake(monkeypatch, variable):
    """It replaced the dataclass with a string.

    The failure then surfaced far away as "'str' object has no attribute 'workers'",
    which names neither the variable nor what to do about it.
    """
    import os

    # A session fixture elsewhere sets BL_MODEL__PAYOUT__BACKEND, which would trip the
    # nested-collision check first and test a different rule than this one claims to.
    for key in list(os.environ):
        if key.startswith(f"{variable}__"):
            monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv(variable, "x")
    with pytest.raises(ValueError, match="group of settings"):
        Settings.load()



def _write_config(tmp_path, body, monkeypatch):
    path = tmp_path / "probe.yaml"
    path.write_text(body)
    monkeypatch.setenv("BL_CONFIG", str(path))
    return path


@pytest.mark.parametrize(("body", "expected"), [
    ("model:\n  payout:\n    context_size: null\n", "model.payout.context_size"),
    ("model:\n  payout:\n    context_size: [1, 2]\n", "model.payout.context_size"),
    ("serving:\n  threads_per_worker: 1.5\n", "serving.threads_per_worker"),
    ("serving:\n  workers: true\n", "serving.workers"),
    ("paths:\n  raw_dir: 12\n", "paths.raw_dir"),
])
def test_a_wrongly_typed_yaml_value_names_its_setting(tmp_path, monkeypatch, body, expected):
    """The YAML parser types a value; it does not check it against the field.

    Each of these used to reach Settings.validate() and surface as `'<' not supported
    between instances of 'NoneType' and 'int'`, or - for the float - not be objected to
    at all, and reach torch as a fractional thread count.
    """
    _write_config(tmp_path, body, monkeypatch)
    with pytest.raises(ValueError, match=expected):
        Settings.load()


def test_a_config_file_that_is_not_yaml_names_the_file(tmp_path, monkeypatch):
    """PyYAML's own error is a parser trace with a line number and no filename, and the
    filename is the one thing the operator controls through BL_CONFIG."""
    path = _write_config(tmp_path, "this: is: not: yaml:\n\tbad\n", monkeypatch)
    with pytest.raises(ValueError, match=str(path.name)):
        Settings.load()


@pytest.mark.parametrize(("value", "fragment"), [
    ("0", "must be positive"),
    ("-5", "must be positive"),
    # data.days_for_test is 7, and the research code splits that many days off the
    # window before it fits - so a window of 7 leaves the classifier nothing.
    ("7", "must exceed data.days_for_test"),
])
def test_a_lookback_window_that_leaves_nothing_to_fit_is_refused(monkeypatch, value, fragment):
    """Unvalidated, 0 silently meant "the whole table", a negative value produced an empty
    window and a pandas error naming nothing, and anything at or below days_for_test
    failed inside CatBoost twenty minutes into the run."""
    monkeypatch.setenv("BL_DATA__LOOKBACK_DAYS", value)
    with pytest.raises(ValueError, match=fragment):
        Settings.load()


def test_a_usable_lookback_window_is_accepted(monkeypatch):
    monkeypatch.setenv("BL_DATA__LOOKBACK_DAYS", "8")
    assert Settings.load().data.lookback_days == 8
    monkeypatch.setenv("BL_DATA__LOOKBACK_DAYS", "")      # unset means the whole table
    assert Settings.load().data.lookback_days is None


def test_the_config_file_and_the_code_defaults_agree():
    """conf/config.yaml restates the dataclass defaults, and the docs call the file the
    place settings live. That is only true while the two agree - and nothing checked.

    It matters because a missing config file is tolerated: the Databricks bundle pointed
    BL_CONFIG at a path that was never synced, and for months it made no difference
    precisely because of this coincidence. If the file ever carries a value the code does
    not default to, a deployment that fails to read it changes behaviour silently.
    """
    import yaml

    from bl_ranking.config import _build

    root = Path(__file__).resolve().parents[1]
    from_file = yaml.safe_load((root / "conf" / "config.yaml").read_text())
    with_file = _build(Settings, dict(from_file)).flat()
    defaults = Settings().flat()

    differing = {k: (defaults[k], with_file[k]) for k in defaults
                 if defaults[k] != with_file.get(k)}
    assert not differing, (
        "conf/config.yaml now differs from the dataclass defaults, so a deployment that "
        f"cannot read the file behaves differently: {differing}")


def test_a_named_config_file_that_is_missing_is_an_error(tmp_path, monkeypatch):
    """A config file nobody asked for may be absent; one an operator named may not. The
    Databricks bundle named a file that was never synced to the workspace, and because a
    missing file was fine everywhere, the weekly job read no config at all and said nothing.
    """
    monkeypatch.setenv("BL_CONFIG", str(tmp_path / "not-there.yaml"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Settings.load()
