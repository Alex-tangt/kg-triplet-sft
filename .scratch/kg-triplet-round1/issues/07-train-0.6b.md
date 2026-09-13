# 07 — 0.6B full training

**What to build:** The full-scale Round-1 0.6B run on the **2575-row** GraphRAG Alpaca
set. Canonical config is **not yet fixed**: per ADR-0005 the tracer's hyperparameters
no longer bind Round-1 automatically; the pipeline (framework/GPU-count/precision) is
chosen from the tracer benches, then frozen as the intra-curve config for 0.6/1.5/3B
and the 500/1300/2575 ablation.

**Blocked by:** 05, 06

**Status:** done (full-scale 0.6B LLaMA-Factory run on the 2575-row GraphRAG set;
this is the reproduction-line "reference 0.6B" — see ADR-0006 and `eval/README.md`).

- [ ] Full run per the chosen canonical pipeline (from ticket-06 benches; see
      `finetune/README_full.md` for the data-side inputs): upload
      `outputs/graphrag_full/alpaca_full_train.jsonl`(2575) + `alpaca_full_val.jsonl`(75)
- [ ] Training completes (or resumes) within the Kaggle quota; loss curves logged
- [ ] Adapter exported/merged; evaluated on a sample of the 700 test passages vs teacher
      reference (open-format 3-axis eval scaled to the test split is still to be built —
      Phase-3 task, cf. ticket 10)
