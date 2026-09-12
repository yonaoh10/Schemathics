"""Shared fixtures.

The expensive fixture is `bundle`: it runs the real pipeline end to end (generate ->
ingest -> production training) against a small dataset and hands back a model bundle.
It is session-scoped, so the cost is paid once for the whole suite.

Two deliberate economies keep it under a minute:
  * a few thousand sessions rather than the full window;
  * `build_gender_lookup=false`, with a small hand-built table injected instead. The
    real table takes ~47 s to enumerate, and tests/test_gender_lut.py checks it
    separately against the live dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Serving the CatBoost fallback everywhere: it needs no TabPFN token, no weights and no
# network, so the suite runs anywhere. The backends themselves are covered separately.
os.environ.setdefault("BL_MODEL__PAYOUT__BACKEND", "catboost_fallback")
os.environ.setdefault("BL_MODEL__BUILD_GENDER_LOOKUP", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")

TEST_SESSIONS = 3000


@pytest.fixture(scope="session")
def workspace(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("bl_workspace")


@pytest.fixture(scope="session")
def settings(workspace):
    from bl_ranking.config import Settings

    return Settings.load(
        paths={
            "raw_dir": str(workspace / "raw"),
            "raw_file": "bl_full_data.csv",
            "delta_table": str(workspace / "delta"),
            "run_root": str(workspace / "runs"),
        },
        mlflow={
            "tracking_uri": f"file://{workspace / 'mlruns'}",
            "experiment": "bl_tests",
            "registered_model": "bl_brand_ranker_test",
        },
        # Set here rather than only in the environment, so the suite is hermetic: a
        # developer with BL_MODEL__PAYOUT__BACKEND=surrogate exported would otherwise
        # have the fixture try to reach TabPFN.
        model={
            "build_gender_lookup": False,
            "payout": {"backend": "catboost_fallback"},
        },
        generator={"sessions": TEST_SESSIONS},
    )


@pytest.fixture(scope="session")
def raw_csv(settings) -> Path:
    from bl_ranking.data.generate import write_csv

    # write_csv reports whether it actually wrote, because `make data` used to print
    # "wrote <path>" over an untouched private extract. Here overwrite=True, so it always
    # does; the assertion keeps the fixture honest if that ever changes.
    path, written = write_csv(settings, overwrite=True)
    assert written
    return path


@pytest.fixture(scope="session")
def ingested(settings, raw_csv):
    from bl_ranking.data.ingest import ingest

    return ingest(settings, source=raw_csv)


@pytest.fixture(scope="session")
def bundle(settings, ingested) -> Path:
    """A real, registered model bundle produced by the production training mode."""
    from bl_ranking.models.gender_lut import ARTIFACT_NAME
    from bl_ranking.training.job import run

    result = run(train_test=False, settings=settings, register=True)
    assert result.bundle_dir is not None
    assert not (result.bundle_dir / ARTIFACT_NAME).exists(), (
        "the fixture asked for build_gender_lookup=false but a table was written"
    )
    return result.bundle_dir


@pytest.fixture(scope="session")
def gender_lookup():
    """A small table covering the names the generator uses, plus the null-ish keys.

    'Nan' and 'None' are real keys: the research code does
    `str(x).strip().capitalize()` before looking a name up, so a missing name arrives
    as one of those literals.
    """
    from bl_ranking.models.gender_lut import GenderLookup

    return GenderLookup.from_mapping({
        "Michael": ("male", 0.992),
        "Rigoberto": ("male", 0.993),
        "Jennifer": ("female", 0.987),
        "Maria": ("female", 0.964),
        "Svetlana": ("female", 0.981),
        "Ahmed": ("male", 0.975),
        "Priya": ("female", 0.969),
        "Marcus": ("male", 0.961),
    })


@pytest.fixture(scope="session")
def ranker(bundle, settings, gender_lookup):
    from bl_ranking.serving.ranker import BrandRanker

    built = BrandRanker.load(bundle, settings)
    built.warm.gender = gender_lookup
    return built


@pytest.fixture
def example_user() -> dict:
    from bl_ranking.serving.ranker import WARMUP_USER

    return dict(WARMUP_USER)
