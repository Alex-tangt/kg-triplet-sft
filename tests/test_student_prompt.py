"""The student prompt is derived from the official GraphRAG teacher prompt.

The shared task statement (the Goal sentence) is inherited verbatim from
``GRAPH_EXTRACTION_PROMPT`` and must keep matching it — an upstream task-
statement change that is not mirrored here turns the test red, so the two
cannot silently diverge.

The extraction semantics also follow the official prompt, but the field
vocabulary is deliberately unified to the JSON keys the model must emit
(title/type/description; source/target/description/strength) instead of the
official field names (entity_name, relationship_strength, ...) — a small
student must never see two names for one output slot. Those names are asserted
absent. See kg_contract/student_prompt.py.
"""

import kg_contract.graphrag_prompts as gp
from kg_contract.student_prompt import STUDENT_PROMPT

TEACHER = gp.GRAPH_EXTRACTION_PROMPT

GOAL = (
    "Given a text document that is potentially relevant to this activity and a list of entity types, "
    "identify all entities of those types from the text and all relationships among the identified entities."
)


def test_goal_inherited_verbatim_from_teacher():
    assert GOAL in TEACHER, "teacher prompt no longer contains the Goal sentence"
    assert GOAL in STUDENT_PROMPT, "student prompt no longer contains the Goal sentence"


def test_student_prompt_has_no_raw_graphrag_grammar():
    for token in ('("entity"', '"relationship"', "##", "<|COMPLETE|>", "capitalized"):
        assert token not in STUDENT_PROMPT, f"student prompt leaked raw grammar token: {token!r}"


def test_single_field_vocabulary_matches_json_keys():
    # official field names must not appear — one vocabulary only (H4 schema fix)
    for token in (
        "entity_name", "entity_type", "entity_description",
        "source_entity", "target_entity",
        "relationship_description", "relationship_strength",
    ):
        assert token not in STUDENT_PROMPT, f"student prompt reused official field name: {token!r}"
    # every JSON key the model must emit is the name used in -STEPS-
    for key in ('"title"', '"type"', '"description"', '"source"', '"target"', '"strength"'):
        assert key in STUDENT_PROMPT


def test_student_prompt_contract():
    assert '"entities"' in STUDENT_PROMPT and '"relationships"' in STUDENT_PROMPT
    assert "Return a single JSON object" in STUDENT_PROMPT
    assert "PERSON, ORGANIZATION, GEO, EVENT, CONCEPT" in STUDENT_PROMPT
    assert 'return {"entities": [], "relationships": []}' in STUDENT_PROMPT
