# Referent-level evaluation on a fixed sample via thinking-mode LLM pairing

The open-line eval previously scored entity finding by normalized-title string
matching (`eval/tracer_eval.py`); that understates the fine-tuned student
because it garbles titles heavily (entity recall 0.436 title-space). We tried
LLM referent pairing in single-shot non-thinking mode — it over-merges
co-occurring concepts of the same passage (probe 9/12). We therefore evaluate on
**referent identity** (same real-world entity, not same title string) using a
**two-tier matcher**: tier-1 normalized-exact-title pairs (free, certain) and
tier-2 `qwen3.7-flash` in **thinking mode** with a shared identity rubric plus
the passage text deciding every remaining fuzzy pair. On a fixed 12-pair
known-answer probe (spelling-garble rescues as positives, co-occurrence / type-
conflict merges as negatives) thinking mode scores 12/12; non-thinking 9/12.

Future candidate models are scored on a **fixed, stratified 200-passage sample**
(`outputs/referent_eval/sample_200.json`, 60/40 wiki/arxiv, length terciles,
fixed seed) — chosen so the per-axis 95% CI stays ≤ ~±2-3pt per the reference
model's full-699 bootstrap — not on the whole test set. The reference model was
scored on all 699 once (entity recall 0.541 / precision 0.707).

_Considered and rejected_: a separate LLM verify/second-opinion gate and the
exact+double-agreed "intersection" headline — the second judge is the same model
and was tuned over-strict (rejected ~70% of fuzzy pairs, including legitimate
spelling-garble rescues like `GENE TIERNEY`→`GEREN TIERNEY`), so it erases the
referent signal instead of certifying it. Description claim-truth stays out of
scope (no LLM judge in the automated loop); hallucination/grounding is a lexical
two-anchor check.
