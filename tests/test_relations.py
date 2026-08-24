from kg_contract import ENTITY_TYPES, RELATION_TYPES

EXPECTED_RELATIONS = {
    "implements", "trained_on", "evaluates", "part_of", "introduces",
    "extends", "depends_on", "contrasts_with", "applied_to", "measured_by",
    "founded_by", "developed_by", "defined_as", "consists_of", "is_type_of",
    "based_on", "used_for", "created_by", "located_in", "predecessor_of",
}


def test_relation_set_has_exactly_20_members():
    assert len(RELATION_TYPES) == 20


def test_relation_set_matches_documented_ontology():
    assert RELATION_TYPES == EXPECTED_RELATIONS


def test_entity_types_are_entity_and_concept():
    assert ENTITY_TYPES == {"entity", "concept"}


def test_relations_are_clean_identifiers():
    for rel in RELATION_TYPES:
        assert rel.isidentifier()
        assert rel.islower()
