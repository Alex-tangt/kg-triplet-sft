import pytest

from kg_contract import (
    extract_json_array,
    is_valid_triplet,
    validate_triplet,
    validate_triplet_list,
)

VALID_TRIPLET = {
    "source": {"title": "Alan Garner", "type": "entity"},
    "relation": {"type": "developed_by", "weight": 0.8},
    "target": {"title": "The Weirdstone of Brisingamen", "type": "entity"},
}


def test_valid_triplet_passes():
    assert is_valid_triplet(VALID_TRIPLET)
    assert validate_triplet(VALID_TRIPLET) == []


def test_empty_array_is_valid_hard_negative():
    ok, errors = validate_triplet_list([])
    assert ok
    assert errors == []


@pytest.mark.parametrize("mutate,path_fragment", [
    (lambda t: t.pop("source"), "source"),
    (lambda t: t.update({"source": "Alan Garner"}), "source"),
    (lambda t: t["source"].pop("title"), "source.title"),
    (lambda t: t["source"].update({"type": "person"}), "source.type"),
    (lambda t: t.pop("relation"), "relation"),
    (lambda t: t["relation"].update({"type": "invented_by"}), "relation.type"),
    (lambda t: t["relation"].update({"weight": True}), "relation.weight"),
    (lambda t: t["relation"].update({"weight": -0.1}), "relation.weight"),
    (lambda t: t["relation"].update({"weight": 1.5}), "relation.weight"),
    (lambda t: t["relation"].pop("weight"), "relation.weight"),
])
def test_invalid_triplet_reports_an_error(mutate, path_fragment):
    import copy
    broken = copy.deepcopy(VALID_TRIPLET)
    mutate(broken)
    assert not is_valid_triplet(broken)
    errors = validate_triplet(broken)
    assert errors
    if path_fragment is not None:
        assert any(path_fragment in e.path for e in errors)


def test_non_object_triplet_is_invalid():
    for bad in (None, 42, "triplet", ["a"]):
        assert not is_valid_triplet(bad)


def test_list_of_mixed_triplets_reports_indexed_paths():
    ok, errors = validate_triplet_list([VALID_TRIPLET, "not a triplet"])
    assert not ok
    assert any(e.path.startswith("triplets[1]") for e in errors)


def test_extract_json_array_handles_fenced_and_prose_wrapped_output():
    assert extract_json_array([VALID_TRIPLET]) == [VALID_TRIPLET]
    assert extract_json_array('[{"a": 1}]') == [{"a": 1}]
    assert extract_json_array('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert extract_json_array('Sure! Here is the array:\n[{"a": 1}]') == [{"a": 1}]
    assert extract_json_array("no array here") is None
    assert extract_json_array(42) is None


def test_extract_json_array_picks_the_first_parseable_array():
    assert extract_json_array('[{"a": 1}] then [{"b": 2}]') == [{"a": 1}]
    assert extract_json_array('prose [{"a": 1}] trailing [2]') == [{"a": 1}]


def test_extract_json_array_rejects_unparseable_array():
    assert extract_json_array('[{"a": }]') is None
