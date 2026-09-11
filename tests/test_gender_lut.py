"""The precomputed gender table must be the research function, not an approximation.

models/gender_lut.py replaces an 18-second, 2.4 GB import with a 4 MB table. That is
only defensible if the table returns exactly what the research implementation returns,
for every name - including the ones it does not know.

These tests need names-dataset installed, so they skip cleanly in the serving image,
which by design does not have it.
"""

from __future__ import annotations

import pytest

from bl_ranking.models.gender_lut import UNKNOWN, GenderLookup

pytestmark = [pytest.mark.slow, pytest.mark.needs_names_dataset]


@pytest.fixture(scope="module")
def live_dataset():
    names_dataset = pytest.importorskip("names_dataset")
    return names_dataset.NameDataset()


@pytest.fixture(scope="module")
def built_table(tmp_path_factory):
    path = tmp_path_factory.mktemp("gender") / "gender_lookup.parquet"
    table = GenderLookup.build(path)
    return table, path


def _research_implementation(dataset, fname):
    """BLPayoutModelsFit.detect_gender_with_confidence, copied verbatim.

    Copied rather than imported: importing the research module costs 18 s and 2.4 GB,
    and this is the definition under test.
    """
    result = dataset.search(fname)
    if result is None:
        return "unknown", 0.0
    first = result.get("first_name")
    if first is None:
        return "unknown", 0.0
    gender_data = first.get("gender")
    if gender_data is None:
        return "unknown", 0.0
    m = gender_data.get("Male", 0)
    f = gender_data.get("Female", 0)
    total = m + f
    if total == 0:
        return "unknown", 0.0
    if m >= f:
        return "male", round(m / total, 3)
    return "female", round(f / total, 3)


def test_table_matches_the_research_function_on_a_random_sample(built_table, live_dataset):
    import random

    table, _ = built_table
    names = list(live_dataset.first_names.keys())
    rng = random.Random(20260211)

    mismatches = []
    for name in rng.sample(names, 4000):
        expected = _research_implementation(live_dataset, name)
        if table.lookup(name) != expected:
            mismatches.append((name, table.lookup(name), expected))
    assert mismatches == [], mismatches[:10]


@pytest.mark.parametrize("name", ["Michael", "Jennifer", "Rigoberto", "Maria", "Svetlana"])
def test_known_names_match(built_table, live_dataset, name):
    table, _ = built_table
    assert table.lookup(name) == _research_implementation(live_dataset, name)


@pytest.mark.parametrize("name", ["Zzzqqxwv", "", "Nan", "None", "12345"])
def test_unknown_names_return_the_same_default(built_table, live_dataset, name):
    """'Nan' and 'None' are real keys: `str(x).strip().capitalize()` produces them for a
    missing name, so the miss path has to behave identically."""
    table, _ = built_table
    assert table.lookup(name) == _research_implementation(live_dataset, name)
    if table.lookup(name) == UNKNOWN:
        assert _research_implementation(live_dataset, name) == UNKNOWN


def test_table_round_trips_through_parquet(built_table):
    table, path = built_table
    reloaded = GenderLookup.load(path)
    assert len(reloaded) == len(table)
    for name in ["Michael", "Jennifer", "Rigoberto"]:
        assert reloaded.lookup(name) == table.lookup(name)


def test_table_is_small_enough_to_ship_in_a_model_bundle(built_table):
    _, path = built_table
    size_mb = path.stat().st_size / 1e6
    assert size_mb < 15, f"{size_mb:.1f} MB - the whole point is that it is small"
