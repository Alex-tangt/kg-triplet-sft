# 07 — 0.6B full training

**What to build:** The full-scale Round-1 0.6B run on the **2575-row** GraphRAG Alpaca
set. Hyperparameters are byte-identical to the tracer (r=32, α=32, lr 5e-5, 5 epochs,
bs2 × grad-accum 4, **cutoff 6144**, bf16 LoRA); the config differs from the tracer's
only in dataset/output_dir (single-variable invariant).

**Blocked by:** 05, 06

**Status:** ready-for-human (owner runs on Kaggle T4; recommended after the ticket-06
tracer verdict)

- [ ] Full run per `finetune/README_full.md`: upload
      `outputs/graphrag_full/alpaca_full_train.jsonl`(2575) + `alpaca_full_val.jsonl`(75),
      `finetune/qwen3_0.6b_graphrag_full.yaml`, merge `finetune/dataset_info.json`
      (`graphrag_full` → `data/alpaca_full_train.jsonl`, distinct filename so tracer +
      full can coexist); `llamafactory-cli train`
- [ ] Training completes (or resumes) within the Kaggle quota; loss curves logged
- [ ] Adapter exported/merged; evaluated on a sample of the 700 test passages vs teacher
      reference (open-format 3-axis eval scaled to the test split is still to be built —
      Phase-3 task, cf. ticket 10)
