# Full-scale SFT — Kaggle T4 steps (Round-1, 0.6B)

Goal: the full Round-1 run — fine-tune Qwen3-0.6B (LoRA) on all **2575**
GraphRAG-style labels so the model emits the extraction JSON (entities +
relationships with descriptions + strength). Data = `outputs/graphrag_full/`
(built by `dataset/build_alpaca.py --full` from the 3349-passage teacher labels
+ `outputs/graphrag_batch/split.json`; train 2575 / val 75; the 10 tracer-eval
passages are in the test split and never appear here).

This is the scaled-up sibling of the tracer run — see
`finetune/README_tracer.md` for the full LLaMA-Factory mechanics (install,
Qwen3 thinking off during training, export). Only what differs is listed here.

## What to upload to Kaggle (as a dataset or notebook inputs)

- `outputs/graphrag_full/alpaca_full_train.jsonl` (2575 rows)
- `outputs/graphrag_full/alpaca_full_val.jsonl` (75 rows, for reporting)
- `finetune/qwen3_0.6b_graphrag_full.yaml`
- `finetune/dataset_info.json` (this repo: merge into LLaMA-Factory's `data/dataset_info.json`)

The `graphrag_full` dataset_info entry points at `alpaca_full_train.jsonl` (a
distinct filename from the tracer's `alpaca_train.jsonl`, so both can coexist
in LLaMA-Factory's `data/` without a collision).

## Train

```
llamafactory-cli train qwen3_0.6b_graphrag_full.yaml
```

Hyperparameters are byte-identical to the tracer (and to the locked Round-1
values): r=32 α=32, lr 5e-5, 5 epochs, bs2 × grad-accum 4, cutoff 6144, bf16
LoRA. Only the dataset and output_dir differ. Expect a much longer wall-clock
than the tracer's 95 rows.

- On a T4 (16GB), if OOM, lower `cutoff_len` to 5120 or
  `per_device_train_batch_size` to 1 (record it).
- Packed length: p99 ≈ 4.9k tokens under the 6144 cutoff; ~4 of 2575 passages
  slightly exceed it and their JSON tails are simply unlabelled (harmless).

## After training

Export/merge as in README_tracer, then evaluate on a sample of the 700 test
passages against teacher reference labels (open-format 3-axis eval —
`eval/tracer_eval.py` pattern scaled to the test split is a Phase-3 task, not
done yet).
