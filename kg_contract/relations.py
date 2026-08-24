"""The shared knowledge-graph triplet schema.

Single source of truth for the relation and entity-type vocabularies consumed
by every phase (dataset, finetune, eval, serve). The vendored harness
(``eval/harness/scripts/evaluation.py``) carries its own frozen copy of these
constants; ``tests/test_drift.py`` asserts the two never diverge — that drift
was a historical source of silent mis-scoring.
"""

RELATION_TYPES: frozenset[str] = frozenset({
    "implements", "trained_on", "evaluates", "part_of", "introduces",
    "extends", "depends_on", "contrasts_with", "applied_to", "measured_by",
    "founded_by", "developed_by", "defined_as", "consists_of", "is_type_of",
    "based_on", "used_for", "created_by", "located_in", "predecessor_of",
})

ENTITY_TYPES: frozenset[str] = frozenset({"entity", "concept"})
