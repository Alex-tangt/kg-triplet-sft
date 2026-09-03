"""Student (SFT) deployment prompt — the only prompt the fine-tuned model ever
sees at training/eval/serving time.

Derived from the official MS GraphRAG extraction prompt
(``kg_contract/graphrag_prompts.py`` ``GRAPH_EXTRACTION_PROMPT``), whose
Goal and Step-2 relationship semantics are inherited verbatim. Deliberate
deltas, per the prompt-strategy decision (deployment is our own JSON
extraction API; the teacher stays on the official prompt as the quality
anchor):
- output serialization is a single JSON object (the ``("entity"<|>...)`` raw
  grammar, the ``##`` list delimiter and ``<|COMPLETE|>`` are gone)
- the three in-prompt examples are dropped (the SFT corpus supplies examples)
- entity casing is stated as canonical uppercase (labels are ~99% ALL-CAPS
  GraphRAG convention, not "capitalized")
- entity_description is stated as concise / one-or-two sentences with a
  numeric-preservation clause (labels median ~90 chars, single sentence)
- the entity_types enum is substituted concretely; relationship strength is
  pinned to 0-10 in the output mapping

The inherited sentences are drift-guarded by tests/test_student_prompt.py:
they must keep appearing verbatim in the teacher prompt, so a change to the
official prompt text that is not mirrored here turns the test red.
"""

STUDENT_PROMPT = """\
-GOAL-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-STEPS-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: name of the entity in canonical uppercase form, consistent across the whole output
- entity_type: one of the following types: [PERSON, ORGANIZATION, GEO, EVENT, CONCEPT]
- entity_description: concise factual description of the entity's attributes and role in the text (one or two sentences directly supported by the text; include numeric details such as dates, counts and measurements when the text states them)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity

-OUTPUT-
Return a single JSON object with exactly two keys:
{"entities": [{"title", "type", "description"}], "relationships": [{"source", "target", "description", "strength"}]}
Map each entity from step 1 to one "entities" entry (title = entity_name, type = entity_type, description = entity_description). Map each related pair to one "relationships" entry (source and target must exactly equal the "title" of an entity listed in "entities"; description = relationship_description; strength = relationship_strength, a number from 0 to 10). Only include entities and relationships explicitly supported by the text; never invent facts, entities or relations. If nothing can be extracted, return {"entities": [], "relationships": []}."""
