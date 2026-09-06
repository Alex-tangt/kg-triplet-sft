# KG Triplet SFT

Reproduction and extension of the Qwen3-0.6B knowledge-graph triplet extraction pipeline
(data construction → LoRA SFT → evaluation → deployment), built as a resume-grade
end-to-end project. The reproduction target is `mohar07/qwen3-0.6b-kg-triplets`.

## Main line: GraphRAG-style extraction (ADR-0003 / ADR-0004)

The data-generation line pivoted (owner-approved 2026-09) from the typed 20-relation
reproduction to **open GraphRAG-style extraction** whose output feeds an MS GraphRAG
graph-QA consumer. Terms below are the current working vocabulary; the `## Task & Data`
section further down describes the reproduction line (kept as a validation gate).

**label**:
A passage annotated as `{entities: [{title, type, description}], relationships:
[{source, target, description, strength}]}`. Titles are the **canonical uppercase** form
(GraphRAG convention, ~99% ALL-CAPS in the data — not the original casing); types are
`PERSON, ORGANIZATION, GEO, EVENT, CONCEPT`; descriptions are concise factual sentences
(median ~92 chars) grounded in the chunk; `strength` is 0–10.

**teacher prompt**:
The official Microsoft GraphRAG extraction prompt (`kg_contract/graphrag_prompts.py`,
verbatim, MIT) given to qwen3-flash. Batch labeling runs a fixed two-round schedule
(r1 + r2 "continue", because batch inference has no adaptive loop); realtime/pilot runs
the full completeness loop (`LOOP_PROMPT`/`CONTINUE_PROMPT`).

**student prompt**:
`STUDENT_PROMPT` (`kg_contract/student_prompt.py`) — the only prompt the fine-tuned
model sees at train/eval/serve. **Derived** from the teacher prompt: the Goal task
statement inherited verbatim, extraction semantics following it, raw
`("entity"<|>...)` grammar → a single JSON contract, examples stripped, casing/
description wording calibrated to the measured labels, and **one field vocabulary**
throughout — `-STEPS-` uses the exact JSON keys (title/type/description,
source/target/strength), not the official field names, so a small student never
sees two names for one output slot. See ADR-0004 (deployment = our own JSON
extraction API; the teacher/student prompts are deliberately distinct).

**full labels / split**:
3349 labeled passages at `dataset/data/graphrag_labels_full.jsonl`; assignment in
`outputs/graphrag_batch/split.json` is 2575 train / 75 val / 700 test; the 10 tracer-eval
passages live in test only.

**alpaca**:
Rendered from `STUDENT_PROMPT` + label JSON. Full scale:
`outputs/graphrag_full/alpaca_full_train.jsonl` (2575) / `_val.jsonl` (75); tracer:
`outputs/tracer/alpaca_train.jsonl` (95) / `_val.jsonl` (5).

**eval axes (open format)**:
Entity finding P/R/F1, edge (topology) finding P/R/F1, schema compliance, grounding/
hallucination, numeric preservation — `eval/referent_eval.py` vs teacher reference
labels. Matching is **referent-level** (same real-world entity): tier-1 normalized-
exact title, tier-2 thinking-mode LLM with the shared identity rubric (`eval/
referent_pair.py`); scored on a fixed stratified 200-passage sample for future
models (the reference model ran all 699). See ADR-0007. The vendored HGR harness
(`entity_f1`, `composite` below) belongs to the reproduction line only.

**referent pairing**:
The teacher↔student entity correspondence used as the eval's ground truth for
"did the student find this entity". NOT title-string equality: the student garbles
titles, so string matching understates. Two tiers — exact normalized-title pairs
(no LLM) plus every remaining fuzzy pair decided by qwen3.7-flash in thinking mode
with `IDENTITY_RUBRIC` (`eval/referent_pair.py`) and the passage text.

**fixed eval sample**:
`outputs/referent_eval/sample_200.json` — 200 test passages (60/40 wiki/arxiv,
length terciles, fixed seed) that every future candidate model is scored on, so
the eval is comparable across models at a bounded cost (ADR-0007).

## Task & Data

(reproduction line — the typed 20-relation schema superseded for data generation)

**triplet**:
A knowledge-graph edge with `source {title, type}`, `relation {type, weight}`, `target {title, type}`,
where type is `entity` or `concept`.

**target prompt**:
The instruction text embedded verbatim in every training sample AND used verbatim at
inference (evaluation and serving). Reproduced character-for-character from the original
project; the 20-relation ontology is implicit in the labels, never stated in the prompt.
See ADR-0001.

**teacher prompt** (reproduction line):
The instruction given to the label-generating LLM (qwen3-flash) in the typed-relation line.
Independent of the target prompt; it carries the relation dictionary, entity-naming rules,
the exhaustiveness demand, and the weight rubric, so the gold labels are high quality.
(Replaced for data generation by the official GraphRAG teacher prompt above.)

**weight**:
Confidence that a relation holds given the text. Fixed anchors `0.2` (inferred) /
`0.5` (indirectly supported) / `0.8` (explicitly stated), chosen by textual evidence
strength — not world-knowledge truth, not passage salience.

**hard negative**:
A training sample whose gold output is an empty array — teaches the model when to return `[]`.

**entity decontamination**:
Removing any test sample whose gold entities (normalized) intersect the training set's
entity vocabulary, so test scores can't come from memorized entities.

**in-chunk grounding**:
A label rule: every entity in a gold triplet must be grounded inside its passage chunk;
unresolved cross-boundary references are dropped rather than invented.

## Evaluation

**entity_f1**:
The harness's fuzzy endpoint alignment score: Hungarian 1-1 matching with embedding
threshold 0.80 and alignment floor 0.50. NOT exact-string F1 — the thresholds are part
of the metric's definition.

**及格率 (pass line)**:
`entity_f1 ≥ 0.6` under the default harness configuration. Deferred to Round 2.

**归因套件 (attribution suite)**:
The four experiments that split a low entity_f1 across its four suspects: harness
thresholds, label quality, data scale, model capacity.

**composite**:
The harness's five-axis weighted score (schema 0.30 / entity_f1 0.25 / relation_acc 0.20 /
weight 0.10 / grounding 0.15), with inapplicable axes dropped and the rest renormalized.

## Project Structure

**Round 1**:
Faithful 0.6B reproduction + capacity curve (0.6/1.5/3B) + data-scale ablation +
attribution suite. Acceptance: composite in 0.55–0.75 with the same axis shape as the
original's 0.6583, both scaling curves, and an attribution answer for the low entity_f1.

**Round 2**:
(Scoped only after the Round 1 review.) 8B sprint for the pass line, possibly with a
dataset v2.

**复刻线 (reproduction line)**:
The Round 1 0.6B faithful reproduction — original prompt, original hyperparameters,
original schema.

**冲刺线 (sprint line)**:
The pass-line pursuit deferred to Round 2.
