# 01 — Teacher gold is unverified (entity / edge / pairing)

**Status:** needs-triage
**Type:** task
**Related:** ADR-0003 (teacher prompt), ADR-0007 (referent eval), ADR-0009 (base
baseline); `docs/research/llm-label-quality-audit.md`,
`docs/research/multi-agent-label-audit.md`; `eval/referent_eval.py` edge
decomposition.

## Problem

Every number the project reports — the capacity curve, the base baseline, the
LightRAG consumer e2e — is scored against teacher-generated labels
(qwen3-flash, official GraphRAG prompt) that have **never been objectively
verified**. There is no human gold on the open line. The labels are
simultaneously (a) the evaluation ground truth and (b) the SFT training target,
so any label defect propagates into both the measurement and the model. This is
a project-level issue, not a per-model check.

## Evidence

- **Edge metric is dominated by entity pairing.** With the new decomposition
  (`summary.edge_attribution`), only 41.4% (0.6B) / 55.8% (4B) of teacher edges
  have both endpoints referent-matched: ~half the edge-recall shortfall happens
  before relation extraction is ever scored.
- **Low relation fidelity even on scoreable edges.** Among teacher edges whose
  endpoints are matched, the student reproduces only 14% (0.6B) / 35% (4B)
  (`edge_conditional_recall`).
- **Edge precision is low in both directions.** Only 39% (0.6B) / 42% (4B) of
  the student's endpoint-valid edges hit a teacher edge; the cause — student
  spurious vs teacher omission — is unmeasured.
- **Visible teacher label errors** in a 20-passage spot-check
  (`outputs/referent_eval/audit_20.md`): type errors (`SYRIAN GODDESS` typed
  PERSON, `TRACTION` typed GEO), over-splitting (`GAUSSIAN MEASURE` vs
  `GAUSSIAN MEASURE ON THE UNIT CIRCLE`), mis-pairing (`MAXWELL` ->
  `MAXWELL'S FISHEYE`).
- **Proxy metrics.** Grounding / hallucination is a lexical two-anchor proxy,
  not truth; entity `type` is only checked for vocabulary membership, never
  correctness; granularity / duplicate axes are diagnostics only.
- **Pairing is itself an LLM judge** (tier-2 thinking) with only a 20-passage
  spot-audit; a wrong pairing distorts both entity and edge R/P at once.

## Why it matters

- It blocks the "data problem vs model capability" attribution that is supposed
  to gate the next action.
- If labels contain partly unlearnable noise, achievable student fit is bounded
  by target inconsistency — but that bound is currently unknown.
- Absolute precision numbers cannot be trusted until the teacher has a checked
  precision/recall estimate with a confidence interval.

## Scope

Three audit objects — **entity / edge / pairing** — judged with one frozen,
versioned rubric, LLM-ensemble first with a small sealed human anchor for
calibration (PPI) and drift detection. Design is intentionally **deferred**;
this issue records the problem, not the solution.

## Guardrail

The audit sizes the data problem; it must not be used to excuse the model-fit
problem. "Did the model learn its target?" is a separate, teacher-noise-invariant
question (train-set reproduction fidelity + teacher self-consistency as the noise
ceiling). Both legs feed the same gate.

## Comments

- 2026-09-21 (owner): flagged as a project-level issue. Raw-data quality is a
  big problem and the current evaluation design is implicated. Design discussion
  deferred; the model-fit question is to be kept explicitly separate.
- 2026-09-25 (follow-up): Student target-fit and sparse relationship extraction
  are tracked separately in `02-student-target-fit-and-edge-coverage.md`. The
  teacher audit here must not be used as a substitute for measuring whether the
  student reproduces teacher targets on seen training passages.
