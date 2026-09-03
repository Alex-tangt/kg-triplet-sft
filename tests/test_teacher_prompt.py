"""Teacher prompt prose: every section present, key rubric elements stated."""

import re

from kg_contract import RELATION_TYPES, TEACHER_SECTIONS, render_teacher_prompt
from kg_contract.prompts import TEACHER_SECTIONS_PROSE


def test_prose_covers_all_five_sections():
    assert set(TEACHER_SECTIONS_PROSE) == set(TEACHER_SECTIONS)


def test_prose_renders_through_skeleton():
    rendered = render_teacher_prompt(TEACHER_SECTIONS_PROSE)
    for name in TEACHER_SECTIONS:
        assert TEACHER_SECTIONS_PROSE[name] in rendered


def test_relation_dictionary_lists_all_20_relations():
    prose = TEACHER_SECTIONS_PROSE["relation_dictionary"]
    for rel in RELATION_TYPES:
        assert rel in prose, f"missing relation {rel}"
    assert "source" in prose and "relation" in prose and "target" in prose


def test_exhaustiveness_demand_carries_empty_array_rule():
    prose = TEACHER_SECTIONS_PROSE["exhaustiveness_demand"]
    assert "[]" in prose


def test_weight_rubric_pins_three_anchors_only():
    prose = TEACHER_SECTIONS_PROSE["weight_rubric"]
    for anchor in ("0.2", "0.5", "0.8"):
        assert anchor in prose
    assert "never any other number" in prose


def test_entity_naming_rules_state_entity_concept():
    prose = TEACHER_SECTIONS_PROSE["entity_naming_rules"]
    assert "entity" in prose and "concept" in prose


def test_in_chunk_grounding_demands_substring_grounding():
    prose = TEACHER_SECTIONS_PROSE["in_chunk_grounding"]
    assert re.search(r"appear in the chunk", prose) or re.search(r"occur in the text", prose)
    assert "outside the chunk" in prose
