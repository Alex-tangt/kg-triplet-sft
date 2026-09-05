# Kaggle inference assets — read-me for agents

This directory is the **reusable inference stack** for the Qwen3-0.6B GraphRAG
LoRA fine-tune. It answers three questions an agent will ask:
what runs where, what was measured, and how to drive it again.

Scope: this read-me covers **inference only** (train kernels are documented in
`kernel_lffull/lffull.py` and the git history of this repo). Everything here is
the product of the full-699 run; numbers are the measured ones, not guesses.

---

## Pipeline at a glance

```
training kernel output (kernel_sources mount)
        │  kg-tracer-lffull/saves/qwen3-0.6b-graphrag-full-lf/
        ▼
kernel_inferfull (this stack) ── reads passages from trace-test-passages ──► predictions.jsonl
        │  schema-prefix injection + tolerant parser + dual-GPU batch decode
        ▼
local: eval/tracer_eval.py ── vs test_teacher_with_text.jsonl ──► eval_report
```

The adapter never leaves Kaggle: inference mounts the training kernel's OUTPUT
as a `kernel_sources` input at
`/kaggle/input/notebooks/idalextan/kg-tracer-lffull/saves/qwen3-0.6b-graphrag-full-lf/`.
No download/upload round-trip.

## Kernel inventory

| kernel dir | purpose | how to run |
|---|---|---|
| `kernel_inferfull/` | **the production inference kernel** (dual-GPU, batched) | push via `kernel-metadata.json`, runs `/kaggle/src/script.py` |
| `kernel_inferprof/prof_infer.py` | torch.profiler A/B on the adapter → tells you the GEMM/GEMV path & attention impl | separate Kaggle kernel, diagnostic only |
| `kernel_benchbatch/bench_batch.py` | batch-size 1/4/8 decode-rate A/B → the throughput numbers below | separate Kaggle kernel, diagnostic only |
| `kernel_probe/probe.py` | walks `/kaggle/input` to discover mount paths (dataset vs kernel source naming differ) | diagnostic; update it whenever mounts change |

`kernel_inferfull/` contains three files:
- `infer_full.py` — the **source of truth** (template with `# INPUTS_HERE` marker).
- `merge_inferfull.py` — injects `../kernel/inputs.py` (the STUDENT_PROMPT +
  embedded 10 tracer passages) into the template → produces
- `kg_tracer_inferfull.py` — the self-contained file pushed to Kaggle.

Edit `infer_full.py`, then `python merge_inferfull.py`, then push the dir.

## How to drive it (external passages / full test set)

1. Passages are `{id, text}` jsonl lines. Sources, in priority order:
   - `PASSAGES_FILE` env (absolute path on Kaggle), else
   - mounted `trace-test-passages` dataset (`test_passages.jsonl`, 699 rows).
   - fallback: the embedded 10 tracer passages (prints a WARNING + lists
     `/kaggle/input` if it falls back, so a mount change is not silent).
2. `BATCH_SIZE` env (default 4). Tune via `kernel_benchbatch` numbers.
3. `--limit N` CLI / `TRACE_LIMIT` env → smoke a few rows first.
4. Kaggle env knobs are read inside the script (env not settable at push time,
   so for one-off smokes temporarily hardcode in `main()`).

Outputs written to `/kaggle/working`: `predictions.jsonl` (canonical rows
`{id, entities:[{title,type,description}], relationships:[{source,target,
description,strength}]}`), `raw_<id>.txt` (each decoded completion, before
parsing), `summary.csv`.

## Key design decisions (each backed by a measurement)

- **Decode is GEMV-bound, not dtype-bound.** bs=1 decode runs cublas
  `gemv2T`/`gemvx` kernels at ~20 tok/s; fp16 vs bf16 makes no difference
  (kernel_inferprof). The training-era fp16-vs-bf16 finding (3.3x) applies to
  GEMM-heavy training only, NOT to bs=1 decode.
- **Batch turns decode into GEMMs.** Measured aggregate throughput:
  bs=1 → 22 tok/s, bs=4 → 85, bs=8 → 164 (kernel_benchbatch). So: sort
  passages by length, decode in batches, pad within batch.
- **Two GPUs via two processes.** `torch.multiprocessing` spawn, one worker per
  GPU (`CUDA_VISIBLE_DEVICES=rank` set inside the worker before `import torch`).
  Each loads its own fp16 model copy and decodes half the passages. Parent
  merges rows. `if __name__ == "__main__"` guard is REQUIRED or spawn
  re-imports and re-runs `main()`.
- **Schema-prefix injection.** The model, after a long passage, sometimes opens
  with a bullet list instead of `{`; free decoding never recovers. Forcing the
  generation prefix `{"entities": [` (appended to the prompt as input_ids)
  makes every row start inside the JSON. Slice the output from
  `mask.sum() - prefix_len` so the prefix stays in the decoded `raw`.
- **Tolerant parser** (`_parse_answer` in infer_full.py). Residual drift is
  structural, not first-token: the model emits `relationships`/`relations`/
  `relationship` key variants, unterminated arrays, a stray `)` for `}`, and a
  degenerate `{"relationship": [a, b], "strength": n}` shape. The parser
  extracts the two arrays by bracket matching (entities bounded by the rel key),
  repairs chars, salvages per-object, and normalizes the degenerate rel shape.
- **Zero-pip.** Preinstalled transformers 5.0 / peft 0.19.1 load this adapter
  fine. The earlier "pin transformers 4.57" step was the thing that hung on the
  flaky Kaggle pip backend; inference only uninstalls torchao (peft hard-fails
  on it) with a hard timeout.
- **External passages beat embedding.** The tracer's 10 passages were embedded
  via `inputs.py`. The full 699 test is read from a file so the same kernel
  scales without a huge code merge.

## Measured performance (T4, 2x, fp16, bs=4)

| workload | wall time |
|---|---|
| 10 tracer passages | ~4.4 min incl. model load (266 s) |
| 699 full test | ~2 h incl. load (COMPLETE) |

## Dependencies / facts an agent must not re-derive

- Dataset mount paths differ by type:
  - dataset → `/kaggle/input/<dataset-slug>/...`
  - kernel_sources → `/kaggle/input/notebooks/<owner>/<kernel-slug>/...`
  - model → `/kaggle/input/models/<owner>/<model-slug>/.../1`
  Re-run `kernel_probe` if mounts change.
- Raw predictions have residual malformed entities (missing `description`) and
  relationships (missing `strength`) that the eval consumed only after a
  normalization pass (see repo `outputs/full_eval/predictions_full_norm.jsonl`).
- Decode randomness: temperature 0.7 sampling → outputs are not reproducible
  run to run; entity spelling fidelity fluctuates between runs.
