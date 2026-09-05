# Inference optimization brief — Qwen3-0.6B GraphRAG LoRA on Kaggle T4

Status: 2026-09-05. Applies to `kernel_inferfull` (see
`finetune/kaggle/README_inference.md` for the reusable asset).

## TL;DR

Single-stream inference was ~20 tok/s. Batching + dual-GPU took the full 699-
passage test run from an estimated **~12 h down to ~2 h** (~6x). The single
biggest lever was NOT dtype — it was making the decode path a GEMM instead of a
GEMV.

## Problem: slow, opaque single-stream decode

Initial kernel decoded one passage at a time (bs=1) on one T4:

- ~20 tok/s measured, constant across fp16 and bf16.
- Extrapolated: 699 passages x ~1.25k output tokens ≈ 10.6 h generation —
  before prefill and model load. Too slow and quota-hungry.

Three hypotheses were tested before changing anything (each killed with data):

| hypothesis | verdict | evidence |
|---|---|---|
| bf16 load falls back to FP32 (the training 3.3x penalty) | **wrong for decode** | fp16 vs bf16 both ~20 tok/s |
| sampling config (repetition penalty) throttles | wrong | removing it caused repetition loops, not speedup |
| GPU is idle / underutilized | partially true | util mean 58% at bs=1, rose to 75% at bs=4 |

Root cause found with torch.profiler (`kernel_inferprof`): bs=1 decode runs
cublas **GEMV** kernels (`gemv2T_kernel`, `gemvx::kernel`) — matrix x single
vector — which are memory-bandwidth limited on T4. The fast fp16 tensor-core
GEMM kernels only appear when batch > 1.

## The four levers, measured

1. **Batch decode (GEMV → GEMM).** Measured aggregate tok/s (identical model,
   `kernel_benchbatch`):

   | batch | aggregate tok/s | vs bs=1 |
   |---|---|---|
   | 1 | 22 | 1x |
   | 4 | 85 | 3.9x |
   | 8 | 164 | 7.5x |

   Nearly linear — decode became GEMM-bound. Implementation: sort passages by
   length, decode in batches, pad within the batch.

2. **Dual GPU (two processes).** `torch.multiprocessing` spawn, one worker per
   T4, each with its own fp16 model copy, decoding half the passages. ~2x wall.
   (T4 16 GB easily fits a 0.6B fp16 model ~1.5 GB.)

3. **Zero-pip startup.** The Kaggle preinstalled transformers 5.0 / peft 0.19.1
   load the adapter fine. Removing the pip downgrade removed the startup step
   that repeatedly hung on the flaky Kaggle pip backend.

4. **Output-quality fixes that made the speed meaningful** (not raw speed, but
   they are why this kernel is reusable): schema-prefix injection + tolerant
   parser, below.

## Why schema-prefix injection matters

At bs=1 and beyond, the trained model sometimes opens a long completion with a
**bullet list** instead of the required JSON object (`{"entities": [`). Free
decoding never pulls it back. Appending the literal prefix `{"entities": [`
to the prompt input_ids forces every row to start inside the JSON; the model
then completes reliably. This turned a 10/10 format-failure into 10/10 JSON.

A tolerant parser (`_parse_answer`) absorbs the residual structural drift the
model still shows: `relationships`/`relations`/`relationship` key variants,
unterminated arrays, a stray `)` for `}`, and a degenerate two-element-array
relationship shape.

## Results

| workload | before (bs=1, 1 GPU) | after (bs=4, 2 GPUs) |
|---|---|---|
| 10 tracer passages | ~13 min | ~4.4 min |
| 699 full test (est from numbers) | ~12 h | ~2 h (measured COMPLETE) |

699-passage eval vs teacher reference: entity recall 0.362, precision 0.481,
numeric 0.582.

## Residual issues / where the next 10x is

- **rels coverage**: 192/699 rows have zero relationships (27%); relations
  extraction is the model's weak axis, not an inference bug.
- **Sampling non-determinism**: temperature 0.7 makes runs differ; entity
  spelling fidelity fluctuates. For stable eval consider a fixed seed or
  multi-sample voting.
- **Real-time progress is invisible**: Kaggle's API buffers stdout/output of a
  running kernel — nothing is observable until COMPLETE. For very large runs,
  prefer sharding into multiple kernels over one long black box.
- Potential further speedup: bs=8 (164 vs 85 tok/s) if padding waste is small;
  length-bucketed batching; a serving engine (vLLM) for interactive use.

## Files

- `finetune/kaggle/kernel_inferfull/` — production inference kernel.
- `finetune/kaggle/kernel_inferprof/` — kernel-level profiler (GEMV finding).
- `finetune/kaggle/kernel_benchbatch/` — batch-size throughput A/B.
- `finetune/kaggle/kernel_probe/` — mount-path discovery.
- `finetune/kaggle/README_inference.md` — agent-facing runbook.
