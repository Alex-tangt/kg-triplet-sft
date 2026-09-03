"""The student prompt is derived from the official GraphRAG teacher prompt:
the Goal and Step-2 relationship semantics are inherited verbatim. If the
official prompt's prose changes, these tests turn red until the student prompt
mirrors it — the two cannot silently diverge.

Deltas (JSON output, no examples, casing/description wording) are deliberate
and documented in kg_contract/student_prompt.py.
"""

import kg_contract.graphrag_prompts as gp
from kg_contract.student_prompt import STUDENT_PROMPT

TEACHER = gp.GRAPH_EXTRACTION_PROMPT

#: sentences inherited verbatim from the teacher prompt (must exist in both).
SHARED = (
    "Given a text document that is potentially relevant to this activity and a list of entity types, "
    "identify all entities of those types from the text and all relationships among the identified entities.",
    "From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that "
    "are *clearly related* to each other.",
    "- relationship_description: explanation as to why you think the source entity and the target entity "
    "are related to each other",
    "- relationship_strength: a numeric score indicating strength of the relationship between the source "
    "entity and target entity",
)


def test_goal_and_relationship_semantics_inherited_verbatim():
    for sentence in SHARED:
        assert sentence in TEACHER, f"teacher prompt no longer contains: {sentence[:60]}..."
        assert sentence in STUDENT_PROMPT, f"student prompt no longer contains: {sentence[:60]}..."


def test_student_prompt_has_no_raw_graphrag_grammar():
    for token in ('("entity"', '"relationship"', "##", "<|COMPLETE|>", "capitalized"):
        assert token not in STUDENT_PROMPT, f"student prompt leaked raw grammar token: {token!r}"


def test_student_prompt_contract():
    assert '"entities"' in STUDENT_PROMPT and '"relationships"' in STUDENT_PROMPT
    assert "Return a single JSON object" in STUDENT_PROMPT
    assert "PERSON, ORGANIZATION, GEO, EVENT, CONCEPT" in STUDENT_PROMPT
    assert 'return {"entities": [], "relationships": []}' in STUDENT_PROMPT
