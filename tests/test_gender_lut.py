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


def _serving_key(name):
    """Exactly what the research pipeline passes to the lookup: str.strip().capitalize()."""
    return str(name).strip().capitalize()


def test_table_matches_the_research_function_on_a_random_sample(built_table, live_dataset):
    """Compared through the SERVING key, which is the only lookup that happens.

    This test used to pass the raw dataset spelling to both sides, so it exercised a
    path production never takes and stayed green while 14% of the dataset silently
    missed: `capitalize()` lowercases everything after the first letter, so
    "Anne-Marie" is looked up as "Anne-marie", which was not a key in the table.
    """
    import random

    table, _ = built_table
    names = list(live_dataset.first_names.keys())
    rng = random.Random(20260211)

    mismatches = []
    for name in rng.sample(names, 4000):
        key = _serving_key(name)
        expected = _research_implementation(live_dataset, key)
        if table.lookup(key) != expected:
            mismatches.append((name, key, table.lookup(key), expected))
    assert mismatches == [], mismatches[:10]


def test_names_whose_capitalisation_differs_are_not_silently_unknown(built_table, live_dataset):
    """The 14% of names the old key scheme dropped: hyphenated and multi-word ones.

    Sampled deliberately rather than at random, because a uniform sample of 4000 hits
    few of them and none of the ones a reviewer would think to try.
    """
    import random

    table, _ = built_table
    rng = random.Random(4242)
    awkward = [n for n in rng.sample(list(live_dataset.first_names.keys()), 60000)
               if isinstance(n, str) and n != n.capitalize()][:1500]
    assert awkward, "expected the dataset to contain names that differ under capitalize()"

    mismatches = []
    for name in awkward + ["Anne-Marie", "Jean-Luc", "Mary Ann", "O'Brien"]:
        key = _serving_key(name)
        expected = _research_implementation(live_dataset, key)
        if table.lookup(key) != expected:
            mismatches.append((name, key, table.lookup(key), expected))
    assert mismatches == [], mismatches[:10]


@pytest.mark.parametrize("name", ["Michael", "Jennifer", "Rigoberto", "Maria",
                                  "Svetlana", "Anne-Marie", "Jean-Luc"])
def test_known_names_match(built_table, live_dataset, name):
    table, _ = built_table
    key = _serving_key(name)
    assert table.lookup(key) == _research_implementation(live_dataset, key)


@pytest.mark.parametrize("name", ["Zzzqqxwv", "", "Nan", "None", "12345"])
def test_unknown_names_return_the_same_default(built_table, live_dataset, name):
    """'Nan' and 'None' are real keys: `str(x).strip().capitalize()` produces them for a
    missing name, so the miss path has to behave identically."""
    table, _ = built_table
    key = _serving_key(name)
    assert table.lookup(key) == _research_implementation(live_dataset, key)
    if table.lookup(key) == UNKNOWN:
        assert _research_implementation(live_dataset, key) == UNKNOWN


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

def test_a_table_built_the_old_way_is_detectable(built_table, tmp_path, caplog):
    """A stale table looks perfectly healthy: same rows, same columns, same size.

    It simply answers 'unknown' for every name whose capitalisation differs from its
    own, so the two feature paths disagree with nothing to show for it. The parquet
    therefore carries a key-scheme marker, and an unmarked one says so on load.
    """
    import logging

    import pyarrow.parquet as pq

    from bl_ranking.models.gender_lut import KEY_SCHEME, KEY_SCHEME_FIELD, GenderLookup

    _, path = built_table

    # A freshly built table carries the marker and loads quietly.
    assert (pq.read_table(path).schema.metadata or {}).get(KEY_SCHEME_FIELD) == KEY_SCHEME
    with caplog.at_level(logging.WARNING):
        GenderLookup.load(path)
    assert "predates" not in caplog.text

    # Strip the marker, as every table written before the fix lacks it.
    stale = pq.read_table(path).replace_schema_metadata({})
    stale_path = tmp_path / "stale.parquet"
    pq.write_table(stale, str(stale_path))

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        GenderLookup.load(stale_path)
    assert "predates" in caplog.text



def test_an_unusable_gender_table_is_refused(tmp_path):
    """Nothing validated this file. An empty or truncated one answers 'unknown' for
    every name: the four bundle guards check the two models and the brand list, /readyz
    goes green, and the ranking simply changes."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest

    from bl_ranking.models.gender_lut import GenderLookup

    empty = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"name": pa.array([], type=pa.string()),
                             "gender": pa.array([], type=pa.string()),
                             "confidence": pa.array([], type=pa.float64())}), empty)
    with pytest.raises(ValueError, match="empty gender lookup"):
        GenderLookup.load(empty)

    wrong = tmp_path / "wrong.parquet"
    pq.write_table(pa.table({"first_name": pa.array(["John"])}), wrong)
    with pytest.raises(ValueError, match="missing column"):
        GenderLookup.load(wrong)


def test_a_table_built_the_old_way_reports_itself(tmp_path):
    """A pre-fix table is indistinguishable from a good one by row count or file size,
    and answers 'unknown' for a fifth of all names. GET /model has to say so, because
    an operator comparing a rollback against a champion looks there and not in a log."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from bl_ranking.models.gender_lut import GenderLookup
    from bl_ranking.serving.ranker import _gender_source

    stale = tmp_path / "stale.parquet"
    pq.write_table(pa.table({"name": pa.array(["John"]),
                             "gender": pa.array(["male"]),
                             "confidence": pa.array([0.99])}), stale)   # no marker
    loaded = GenderLookup.load(stale)
    assert loaded.key_scheme_current is False
    assert _gender_source(loaded) == "precomputed_stale_key"

    current = GenderLookup.from_mapping({"John": ("male", 0.99)})
    assert _gender_source(current) == "precomputed"
    assert _gender_source(None) == "names_dataset"


def _write_table(path, names, genders, confidences, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from bl_ranking.models.gender_lut import KEY_SCHEME, KEY_SCHEME_FIELD, ROW_COUNT_FIELD

    table = pa.table({
        "name": pa.array(names),
        "gender": pa.array(genders).dictionary_encode(),
        "confidence": pa.array(confidences, type=pa.float64()),
    }).replace_schema_metadata({
        KEY_SCHEME_FIELD: KEY_SCHEME, ROW_COUNT_FIELD: str(rows).encode(),
    })
    pq.write_table(table, str(path))
    return path


def test_a_truncated_gender_table_is_refused(tmp_path):
    """A short table has the right columns, the right dtypes and a plausible file size, and
    answers 'unknown' for everything it lost - while GET /model reports a healthy
    'precomputed'. The builder's own row count is stamped into the parquet, so this is exact
    and needs no arbitrary "too small" threshold."""
    import pytest

    from bl_ranking.models.gender_lut import GenderLookup

    path = _write_table(tmp_path / "short.parquet", ["John"], ["male"], [0.9], rows=714212)
    with pytest.raises(ValueError, match="truncated"):
        GenderLookup.load(path)

    # The same file, honest about its size, loads.
    path = _write_table(tmp_path / "small.parquet", ["John"], ["male"], [0.9], rows=1)
    assert GenderLookup.load(path).lookup("John") == ("male", 0.9)


def test_a_gender_table_with_the_columns_swapped_is_refused(tmp_path):
    """It loads cleanly, resolves every name to 'unknown', and leaves GET /model reporting
    'precomputed'. detect_gender_with_confidence only ever returns male, female or
    unknown, so anything else in that column was not written by build()."""
    import pytest

    from bl_ranking.models.gender_lut import GenderLookup

    path = _write_table(tmp_path / "swapped.parquet",
                        ["male", "female"], ["John", "Mary"], [0.9, 0.8], rows=2)
    with pytest.raises(ValueError, match="wrong way round"):
        GenderLookup.load(path)


def test_a_table_whose_keys_all_collapse_is_refused(tmp_path):
    """The one check that is about the map load() returns, not the file it read. A parquet
    with the right row count but every key identical passes every file-shape guard and then
    collapses to one entry on the dict build, resolving all but one name to 'unknown' while
    GET /model reports a healthy 'precomputed'."""
    import pytest

    from bl_ranking.models.gender_lut import GenderLookup

    path = _write_table(tmp_path / "collapse.parquet",
                        ["John"] * 100, ["male"] * 100, [0.9] * 100, rows=100)
    with pytest.raises(ValueError, match="distinct name keys"):
        GenderLookup.load(path)


def test_a_gender_table_with_a_numeric_name_column_is_refused(tmp_path):
    """A name column that arrived as int (name and a numeric column transposed) has no null
    keys and the right row count, so it passed every check - and then missed every serving
    lookup, which always passes a string key."""
    import pytest

    from bl_ranking.models.gender_lut import GenderLookup

    path = _write_table(tmp_path / "intname.parquet",
                        [1, 2, 3], ["male", "female", "male"], [0.9, 0.8, 0.7], rows=3)
    with pytest.raises(ValueError, match="not a string one"):
        GenderLookup.load(path)


def test_a_gender_table_with_null_values_is_refused(tmp_path):
    """An all-null gender column slips past the domain check, which subtracts None before
    comparing, and then resolves every name to (None, ...) - a value the research code never
    returns. Null confidence is refused for the same reason."""
    import pytest

    from bl_ranking.models.gender_lut import GenderLookup

    null_gender = _write_table(tmp_path / "nullg.parquet",
                               ["John", "Mary"], ["male", None], [0.9, 0.8], rows=2)
    with pytest.raises(ValueError, match="null gender"):
        GenderLookup.load(null_gender)

    null_conf = _write_table(tmp_path / "nullc.parquet",
                             ["John", "Mary"], ["male", "female"], [0.9, None], rows=2)
    with pytest.raises(ValueError, match="null confidence"):
        GenderLookup.load(null_conf)
