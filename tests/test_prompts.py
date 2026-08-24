import json

import pytest

from kg_contract import (
    TEACHER_SECTIONS,
    TARGET_PROMPT_PREFIX,
    WEIGHT_ANCHORS,
    render_target_prompt,
    render_teacher_prompt,
)

FIXTURES = "tests/fixtures/original_prompt_samples.json"


def _samples():
    with open(FIXTURES, encoding="utf-8") as f:
        return json.load(f)["samples"]


def test_target_prompt_renders_verbatim_original_prompts():
    samples = _samples()
    assert len(samples) >= 5
    for s in samples:
        assert render_target_prompt(s["passage"]) == s["expected_prompt"]


def test_fixture_oracle_is_captured_from_original_data():
    with open(FIXTURES, encoding="utf-8") as f:
        data = json.load(f)
    assert data["_source_entries"], "fixture must record which original entries it was captured from"


def test_target_prompt_prefix_shape():
    assert TARGET_PROMPT_PREFIX.endswith("\n\nText:\n")
    assert "Extract knowledge graph triplets" in TARGET_PROMPT_PREFIX


def test_teacher_skeleton_has_five_fixed_sections():
    assert TEACHER_SECTIONS == (
        "relation_dictionary",
        "entity_naming_rules",
        "exhaustiveness_demand",
        "weight_rubric",
        "in_chunk_grounding",
    )
    assert len(TEACHER_SECTIONS) == 5


def test_weight_anchors_are_fixed():
    assert WEIGHT_ANCHORS == (0.2, 0.5, 0.8)


def test_render_teacher_prompt_joins_sections_in_order():
    texts = {name: name.upper() for name in TEACHER_SECTIONS}
    rendered = render_teacher_prompt(texts)
    assert rendered == "\n\n".join(texts[name] for name in TEACHER_SECTIONS)


def test_render_teacher_prompt_rejects_missing_section():
    with pytest.raises(ValueError, match="relation_dictionary"):
        render_teacher_prompt({name: "x" for name in TEACHER_SECTIONS[1:]})


def test_render_teacher_prompt_rejects_unknown_section():
    with pytest.raises(ValueError, match="unknown"):
        render_teacher_prompt({**{name: "x" for name in TEACHER_SECTIONS}, "extra": "y"})
