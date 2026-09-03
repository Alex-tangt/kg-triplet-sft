# Tracer SFT — Kaggle T4 steps

Goal: fine-tune Qwen3-0.6B (LoRA) to emit the GraphRAG-style extraction JSON
(entities + relationships with descriptions + strength). Training data =
`outputs/tracer/alpaca_train.jsonl` (95 rows) / `alpaca_val.jsonl` (5 rows),
built from the GraphRAG-teacher labels in `dataset/data/graphrag_labels_100.jsonl`.

## What to upload to Kaggle (as a dataset or notebook inputs)

- `outputs/tracer/alpaca_train.jsonl`
- `outputs/tracer/alpaca_val.jsonl`
- `finetune/qwen3_0.6b_graphrag_tracer.yaml`
- `finetune/dataset_info.json` (this repo: merge into LLaMA-Factory's `data/dataset_info.json`)

## Notebook steps

1. Install LLaMA-Factory (latest, e.g. `pip install llama-factory` — the config
   targets the current 0.9.x CLI).
2. Put the two alpaca jsonl files into LLaMA-Factory's `data/` directory and add
   the `graphrag_tracer` entry from `dataset_info.json` to
   `LLaMA-Factory/data/dataset_info.json`:
   ```json
   "graphrag_tracer": {
     "file_name": "alpaca_train.jsonl",
     "columns": {"prompt": "instruction", "query": "input", "response": "output"},
     "formatting": "alpaca"
   }
   ```
3. Train:
   ```
   llamafactory-cli train qwen3_0.6b_graphrag_tracer.yaml
   ```
   - On a T4 (16GB), cutoff_len 6144 + LoRA bs2 × grad-accum 4 should fit. If
     OOM, lower `cutoff_len` to 5120 or `per_device_train_batch_size` to 1.
   - Qwen3 defaults to thinking mode in its chat template. The training data
     contains plain JSON answers (no reasoning), so disable thinking during
     training: add `qwen_thinking: false` if your LLaMA-Factory version accepts
     it (otherwise the template param / model config). This prevents the model
     from learning to emit a reasoning block before the JSON.
4. Export / inference (needed for `eval/tracer_eval.py`):
   ```
   llamafactory-cli export --model_name_or_path Qwen/Qwen3-0.6B \
     --adapter_name_or_path saves/qwen3-0.6b-graphrag-tracer \
     --template qwen --finetuning_type lora --export_dir merged-0.6b-graphrag
   ```
   or load the LoRA in the evaluation script directly.

## After training

Report back: the merged model path (or LoRA path) so `eval/tracer_eval.py` can
run the 10 held-out tracer-eval passages (`outputs/graphrag_pilot/*`) through it
and score the three axes (faithfulness / entity coverage / numeric preservation)
against the teacher labels.
