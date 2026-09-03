"""Student (SFT) deployment prompt — the only prompt the fine-tuned model ever
sees at training/eval/serving time.

Derived from the official MS GraphRAG extraction prompt
(``kg_contract/graphrag_prompts.py`` ``GRAPH_EXTRACTION_PROMPT``): the task
statement (Goal) is inherited verbatim and the extraction semantics follow the
official prompt. Deliberate deltas, per the prompt-strategy decision
(deployment is our own JSON extraction API; the teacher stays on the official
prompt as the quality anchor):
- output serialization is a single JSON object (the ``("entity"<|>...)`` raw
  grammar, the ``##`` list delimiter and ``<|COMPLETE|>`` are gone)
- the three in-prompt examples are dropped (the SFT corpus supplies examples)
- **one field vocabulary throughout**: ``-STEPS-`` names the fields with the
  exact JSON keys the model must emit (title/type/description for entities,
  source/target/description/strength for relationships). The official prompt's
  names (entity_name, entity_type, entity_description, source_entity,
  target_entity, relationship_description, relationship_strength) are NOT
  reused, so a weak student never sees two names for one slot (field leakage)
- entity casing is stated as canonical uppercase (labels are ~99% ALL-CAPS
  GraphRAG convention, not "capitalized")
- descriptions are stated as concise / one-or-two sentences with a
  numeric-preservation clause (labels median ~90 chars, single sentence)
- the entity_types enum is substituted concretely; strength is pinned to 0-10

The inherited Goal sentence is drift-guarded by tests/test_student_prompt.py:
it must keep appearing verbatim in the teacher prompt, so an upstream task-
statement change that is not mirrored here turns the test red.
"""

STUDENT_PROMPT = """\
-GOAL-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-STEPS-
1. Identify all entities. For each identified entity, record the following:
- title: name of the entity in canonical uppercase form, consistent across the whole output
- type: one of PERSON, ORGANIZATION, GEO, EVENT, CONCEPT
- description: a concise factual description of the entity's attributes and role in the text (one or two sentences directly supported by the text; include numeric details such as dates, counts and measurements when the text states them)

2. From the entities recorded in step 1, identify all pairs of (source, target) that are *clearly related* to each other.
For each related pair, record the following:
- source: title of the source entity, as recorded in step 1
- target: title of the target entity, as recorded in step 1
- description: why you think the source entity and the target entity are related, grounded in the text
- strength: a numeric score from 0 to 10 indicating how strong the relation between the source and target entities is

-OUTPUT-
Return a single JSON object with exactly two keys:
{"entities": [{"title", "type", "description"}], "relationships": [{"source", "target", "description", "strength"}]}
"source" and "target" must exactly equal the "title" of an entity listed in "entities". Only include entities and relationships explicitly supported by the text; never invent facts, entities or relations. If nothing can be extracted, return {"entities": [], "relationships": []}."""
