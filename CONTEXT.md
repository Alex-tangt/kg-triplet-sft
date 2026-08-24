# KG Triplet SFT

Reproduction and extension of the Qwen3-0.6B knowledge-graph triplet extraction pipeline
(data construction → LoRA SFT → evaluation → deployment), built as a resume-grade
end-to-end project. The reproduction target is `mohar07/qwen3-0.6b-kg-triplets`.

## Task & Data

**triplet**:
A knowledge-graph edge with `source {title, type}`, `relation {type, weight}`, `target {title, type}`,
where type is `entity` or `concept`.

**target prompt**:
The instruction text embedded verbatim in every training sample AND used verbatim at
inference (evaluation and serving). Reproduced character-for-character from the original
project; the 20-relation ontology is implicit in the labels, never stated in the prompt.
See ADR-0001.

**teacher prompt**:
The instruction given to the label-generating LLM (qwen3-flash). Independent of the target
prompt; it carries the relation dictionary, entity-naming rules, the exhaustiveness demand,
and the weight rubric, so the gold labels are high quality.

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
