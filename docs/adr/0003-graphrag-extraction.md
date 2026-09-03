# Data generation diverges from mohar07 reproduction: GraphRAG-style open extraction

We fine-tune an extractor whose output is consumed by an **MS GraphRAG graph-QA
system** over general text (English Wikipedia 60% + arXiv 40%). The original
reproduction pipeline labels every fact with one of mohar07's **20 typed
relations**; that schema drops every quantity/attribute fact (measured: numeric
preservation 0/10 passages vs 7/10) and cannot express much of what our corpus
asserts (Probe A effective pair-level gap 74.4%). A fixed typed vocabulary has
no use value for a retrieval-driven QA consumer.

We therefore generate training data with the **official Microsoft GraphRAG
extraction prompt** (`kg_contract/graphrag_prompts.py`, verbatim, MIT): open
relation descriptions + a 0-10 strength, entities typed from
`PERSON, ORGANIZATION, GEO, EVENT, CONCEPT`. Output maps 1:1 onto the consumer's
graph schema. Three fixes are applied in `dataset/graphrag.py`:
- **CONCEPT** entity type added (arXiv concepts were otherwise dropped entirely);
- aggressive **deduplication** (normalized-title/pair keys, longest description
  wins) plus a **grounding filter** (ungrounded titles dropped) — the teacher's
  continuation loop otherwise triplicates entities (measured 166→33 on one
  passage);
- **thinking ON** — a blind judge over 4 passages showed it materially improves
  description grounding and numeric recall with zero fabrication, while OFF
  produced a mislabeled EVENT and a garbled translation.

The **student prompt** is `STUDENT_PROMPT` (`kg_contract/student_prompt.py`),
teaching the GraphRAG JSON shape; it stays ontology-implicit and single-sourced
for the student (train = eval = serve), preserving ADR-0001's invariant even
though the verbatim-original target prompt is superseded. Its exact design —
derived from this official prompt, and deliberately distinct from it because
deployment is our own JSON API while the teacher remains the quality anchor —
is recorded in ADR-0004. `cutoff_len` grows 2048 → 6144 because the new labels
are 2-6k tokens long (2048 truncates nearly every sample).

The mohar07 reproduction baseline (composite 0.6583) is kept only as a
pipeline-validation gate; its handling is an **open item** (deferred).

_Considered and rejected_: keeping the 20-set and adding a numeric-attribute/
qualifier field (query-KG design; the consumer queries nothing structurally, so
typed relations add no retrieval value); open extraction without
deduplication/grounding (measured noise); thinking-off teacher (blind judge:
quality loss).
