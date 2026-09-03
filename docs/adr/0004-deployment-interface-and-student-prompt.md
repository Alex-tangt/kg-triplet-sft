# Deployment interface & the derived student prompt (teacher/student prompts stay distinct)

Status: Accepted (2026-09-03). Refines ADR-0003's "target prompt" paragraph.

## Context

The consumer is an MS GraphRAG graph-QA system over general text; data labels
are produced by the official GraphRAG extraction prompt (`kg_contract/
graphrag_prompts.py`). Open question: should the teacher prompt and the
fine-tuned student's prompt be unified into one text? An earlier hand-written
terse `STUDENT_PROMPT` drifted from label reality (it said titles were
"exactly as written in the text" while labels are ~99% ALL-CAPS GraphRAG
convention; descriptions measured median ~92 chars single-sentence).

## Decision

1. **Deployment is our own JSON extraction API** (Reality 1, not a drop-in
   swap inside stock GraphRAG's build loop): a caller sends `STUDENT_PROMPT` +
   passage text, the model returns the JSON `{entities, relationships}` which
   we feed into the GraphRAG index.
2. **The teacher stays on the official prompt** and is deliberately **not**
   unified with the student prompt. The official prompt is the quality anchor:
   label richness and the eval reference (`outputs/tracer/teacher_ref.jsonl`)
   stay tied to the strongest extractor. Unifying the other way — making the
   teacher label under a weakened spec — would lower label quality and drag the
   eval bar down with it; that option was rejected.
3. **`STUDENT_PROMPT` is derived from the official prompt**
   (`kg_contract/student_prompt.py`): the Goal and Step-2 relationship
   semantics are inherited **verbatim**; the raw `("entity"<|>...)` grammar,
   `##` delimiter and `<|COMPLETE|>` are replaced by a single JSON contract;
   the three in-prompt examples are dropped (the SFT corpus supplies examples);
   casing/description wording is calibrated to the measured labels. Deliberate
   deltas are listed in the module docstring.
4. **Labels were not re-generated**: still official-prompt teacher output
   parsed to JSON (3349 full / 100 tracer).
5. `tests/test_student_prompt.py` guards that the inherited sentences keep
   appearing verbatim in the teacher prompt — an upstream text change that is
   not mirrored turns the test red.

## Evidence (prompt gate)

Teacher run single-shot under the derived `STUDENT_PROMPT` on the 10 tracer-eval
passages (`outputs/prompt_gate/student_prompt.jsonl`): 10/10 outputs parse as
clean JSON, zero raw-grammar leakage. Blind judge vs the official 2-round-loop
labels (`outputs/prompt_gate/compare.jsonl`): per-record quality **EQUIVALENT**
(descriptions at least as text-grounded, numeric preservation higher on
population/astronomy/corporate passages, less noise); verdict **ACCEPT**. Known
properties of single-shot (not of the prompt): recall is lower than the looped
teacher (e.g. one list-heavy passage), and one fabrication slipped through
(wikipedia-01557 Ramsar status).

## Consequences

- Student side is JSON end-to-end: serving "parsing" is a plain JSON parse.
  The raw-grammar parser (`dataset/graphrag.py`) stays teacher-side only.
- Alpaca instruction = the derived `STUDENT_PROMPT` constant for both the
  tracer (95/5) and the full run (2575/75, `outputs/graphrag_full/`).
- Train/serve invariant (ADR-0001) now applies to the student prompt alone —
  identical text at training, evaluation and serving.
