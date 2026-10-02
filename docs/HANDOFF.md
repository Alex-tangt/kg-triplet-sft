# HANDOFF — kg-triplet-sft, audit + Round-2 line

Point-in-time handoff so a new session can start without re-deriving. Read with
`OPEN-QUESTIONS.md` (what is *not* decided) and `CONTEXT.md` (vocabulary).

## Where this is

The repo's Round-1 capacity work is done (0.6/1.7/4B masked, published). The
project is now on the **audit + Round-2 line**: attribute the edge/fit behavior,
then improve. The immediate question is whether poor relation extraction is a
data, capacity, optimization, or measurement problem.

## Decisions made (this session)

- Metric layer: **two axes** (fidelity + validity) reported separately (Q1);
  recall via multi-model generation → dedupe → judge + capture-recapture (Q2).
- Bias/variance reading is by **loss**, not seen/unseen F1.
- R1 metrics: `eval/referent_eval.py` gained an **edge decomposition**
  (`edge_recall_ceiling`, `edge_conditional_recall/precision`,
  `edge_student_endpoint_match_rate`, `edge_attribution`); tests 126 → 129.
- Issue tracker moves to **GitHub** (public `Alex-tangt/kg-triplet-sft`).
- Re-org focus: (1) attribution analysis, (2) Round-2 improvement.

## Unresolved (do not assume)

See `OPEN-QUESTIONS.md`. High-impact ones: **Q3** (referent-pairing granularity),
**S1–S5** (L1 reading standard), **task decomposition** (entity vs relation),
teacher protocol (think/no-think, r1 vs r1+r2), gold audit rubric.

## Last experiment — L1 (teacher-forced per-field fit)

- Result + run card + caveats: `docs/evidence/l1-fit.md`.
- Raw: `outputs/l1/l1_{base,masked}.jsonl` (+ log). Kernel `idalextan/kg-l1-loss`.
- **Correction held**: the verbal "underfit is excluded, points to L2" is
  **withdrawn** — cross-field NLL is not comparable (copy vs open-text entropy).
- L1 currently answers only: SFT lowers loss on every field, train≈val. It does
  **not** answer "is it underfit" or "are relations learned".

## Open work products in the tree (uncommitted)

- `eval/referent_eval.py`, `eval/README.md`, `tests/test_referent_eval.py` — R1 metric decomposition.
- `finetune/kaggle/kernel_l1/` — L1 kernel + metadata.
- `docs/research/*.md` (4) — label-quality audit, KG-eval & training signals, distillation & CoT, multi-agent audit.
- `docs/project-brief.md`, `docs/project-status-summary.md` — status material (review before making public).
- `.scratch/kg-audit/issues/01,02` — migrated to GitHub.

## Next actions (suggested order)

1. Freeze the L1 **reading standard** (S1–S5) before any L1-based conclusion.
2. **L2**: free-running on the same seen set + protocol — does the model emit the
   `relationships` array, and how many edges (condR / edge volume)?
3. Teacher **r1 vs r1∪r2** one-variable (artifacts partly exist → nearly free).
4. Decide **Q3** (pairing) so fidelity/validity are not confounded.
5. Gold audit #01 (entity/edge/pairing) with a small sealed human anchor + Jev/PPI.

## Traps already hit

- Kaggle CLI exposes logs/files only after a run completes; write outputs
  incrementally and log dtype/device at start (L1 kernel now does).
- `torch_dtype` may be ignored by newer transformers → force fp16 and assert
  `next(model.parameters()).dtype`; T4 has no native bf16.
- PEFT import dies on Kaggle's torchao 0.10.0 → uninstall torchao first.
