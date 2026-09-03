"""Semantic label checks: weight anchors and in-chunk grounding."""

from dataset.semantic import check_grounding, check_semantics, check_weight_anchors, normalize, violation_counts

GROUNDED = [
    {
        "source": {"title": "Anarchism", "type": "concept"},
        "relation": {"type": "part_of", "weight": 0.8},
        "target": {"title": "political philosophy", "type": "concept"},
    }
]
PASSAGE = "Anarchism is a political philosophy and movement."


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("  Anarchism,  (a) political-philosophy!  ") == "anarchism a political philosophy"


def test_weight_check_passes_anchors():
    assert check_weight_anchors(GROUNDED) == []


def test_weight_check_flags_off_anchor_values():
    bad = [{"source": {"title": "a", "type": "entity"}, "relation": {"type": "part_of", "weight": 0.3}, "target": {"title": "b", "type": "entity"}}]
    v = check_weight_anchors(bad)
    assert len(v) == 1
    assert v[0].rule == "weight"
    assert v[0].triplet_index == 0
    assert "0.3" in v[0].detail


def test_grounding_passes_titles_in_passage():
    assert check_grounding(GROUNDED, PASSAGE) == []


def test_grounding_flags_invented_entity():
    invented = [
        {
            "source": {"title": "Anarchism", "type": "concept"},
            "relation": {"type": "part_of", "weight": 0.8},
            "target": {"title": "Quantum Physics", "type": "concept"},
        }
    ]
    v = check_grounding(invented, PASSAGE)
    assert len(v) == 1
    assert v[0].rule == "grounding"
    assert "target" in v[0].path


def test_grounding_matches_case_insensitively():
    assert check_grounding(
        [{"source": {"title": "anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.2}, "target": {"title": "POLITICAL PHILOSOPHY", "type": "concept"}}],
        PASSAGE,
    ) == []


def test_check_semantics_combines_both_rules():
    bad = [
        {
            "source": {"title": "Anarchism", "type": "concept"},
            "relation": {"type": "part_of", "weight": 0.3},
            "target": {"title": "Quantum Physics", "type": "concept"},
        }
    ]
    v = check_semantics(bad, PASSAGE)
    assert {x.rule for x in v} == {"weight", "grounding"}


def test_violation_counts_aggregates_by_rule():
    bad = [
        {"source": {"title": "a", "type": "entity"}, "relation": {"type": "part_of", "weight": 0.3}, "target": {"title": "b", "type": "entity"}},
        {"source": {"title": "c", "type": "entity"}, "relation": {"type": "part_of", "weight": 0.9}, "target": {"title": "d", "type": "entity"}},
    ]
    assert violation_counts(check_weight_anchors(bad)) == {"weight": 2}
