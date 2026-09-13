# Evidence index

Where the numbers behind the README and the resume live. `outputs/` is
gitignored (large/generated), so this folder and `docs/` + `.scratch/` carry the
curated, checkable evidence.

## Digests (verbatim copies of gitignored run reports)

- [`base-baseline.md`](base-baseline.md) — untrained Qwen3-0.6B on the fixed-200
  referent eval (SFT adds +0.34 micro-F1). Source: `outputs/base_eval/RESULT.md`.
- [`consumer-e2e-lightrag.md`](consumer-e2e-lightrag.md) — LightRAG custom-KG
  consumer-impact e2e (T0 graph formation / T1 retrieval / T2 answers). Source:
  `outputs/lightrag_e2e/RESULT.md`.

## Reports (already in repo)

- [`../2026-09-07-capacity-line-repair.md`](../2026-09-07-capacity-line-repair.md)
  — masking root cause, the three-way repair table (§1.4), the final capacity
  curve (§1.4c), 4090 train/infer tuning.
- [`../inference-optimization-brief.md`](../inference-optimization-brief.md) —
  inference throughput/decode-budget study.
- [`../diary/2026-09-11-consumer-pivot-graphrag-to-lightrag.md`](../diary/2026-09-11-consumer-pivot-graphrag-to-lightrag.md)
  — why the consumer pivoted to LightRAG and the claim boundary it created.

## Raw run logs & decisions

- `.scratch/kg-triplet-round1/issues/08-capacity-runs.md` — per-run training/
  inference logs, receipts, the 4B casing finding.
- `docs/adr/0001`–`0009`, `docs/research/kg-ontology-patterns.md` — decisions
  and the schema-defect measurements.
- `eval/README.md` — the referent-eval runbook and the reference baselines
  (full-699 entity R/P/F1 = 0.541/0.707/0.613; fixed-200 = 0.534/0.723/0.614).

## Regenerating

Full commands are in the module READMEs (`dataset/`, `finetune/README_*.md`,
`finetune/kaggle/README_inference.md`, `eval/README.md`). Labeling and judging
need `DASHSCOPE_API_KEY`; training needs a GPU (kernels target Kaggle T4 and a
single RTX 4090).
