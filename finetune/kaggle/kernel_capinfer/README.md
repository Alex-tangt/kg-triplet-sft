# Capacity-line inference — external interface

What runs, what a caller must supply, and what comes back. This is the README for
the **capacity-curve inference kernels** (0.6B / 1.7B / 4B, bf16 Unsloth line). The
reproduction-line kernel (`../kernel_inferfull/`) has its own runbook
(`../README_inference.md`); the two share the same schema-prefix + tolerant-parser +
dual-GPU decode machinery but are NOT interchangeable — see the protocol warning below.

## The one thing to read before running (issue 08 Resolution 2026-09-07)

The capacity adapters were trained in **plain** format but the base they were built on
is the chat-capable Qwen3 main release (`<|im_end|>` EOS). Evaluating them under the
chat protocol **suppresses output**; evaluating under plain protocol **runs away**
(no learned stop; ~5.6 h for 200 passages, 22% of rows never close their braces).
Root cause is a **training defect** (no per-sample EOS), not an inference bug.

**Owner decision (2026-09-07): retrain the capacity line in chat format.** Until the
chat-format adapters exist, these inference kernels must be driven in **chat**
protocol (`PROTOCOL = "chat"`, the default) ONLY as an interim measurement — their
absolute numbers are protocol-suppressed and are NOT comparable to the reference
baseline. Do not silently switch a run to plain to "get better numbers"; plain output
is not deployable and not comparable either.

## What a caller supplies (kernel metadata per size)

| input | where | example (0.6B) |
|---|---|---|
| base model | `model_sources` | `qwen-lm/qwen-3/transformers/0.6b/1` |
| LoRA adapter | dataset `idalextan/cap-upload` | `/kaggle/input/datasets/idalextan/cap-upload/qwen3-0.6b/` (path auto-resolved) |
| passages | `sample_200_passages.jsonl` (in cap-upload or `200-passages`) | `{id, text}` per line; 200 rows |

Generation: `python finetune/kaggle/kernel_capinfer/make_capinfer.py [--only 06|17|4b] [--protocol chat|plain] [--limit N]`.
Each run writes a self-contained push dir `finetune/kaggle/kernel_cap<tag>/` (code +
`kernel-metadata.json`); push with `kaggle kernels push -p <dir>`.

## What comes back (outputs, all under `/kaggle/working`)

- `predictions.jsonl` — canonical deploy rows, one JSON per line:
  `{id, entities: [{title,type,description}], relationships: [{source,target,description,strength}]}`
  — the exact contract the referent eval (`eval/referent_*`) consumes.
- `raw_<id>.txt` — decoded completion per passage (pre-parser), useful for auditing.
- `summary.csv` — `id,entities,relationships` counts.

## Protocol knobs (baked at generation time — Kaggle env is not settable at push)

| var | meaning |
|---|---|
| `PROTOCOL = "chat"` (default) | `apply_chat_template(enable_thinking=False)` — self-consistent with reference line; SUPPRESSES the current plain-trained adapters |
| `PROTOCOL = "plain"` | training-faithful `STUDENT_PROMPT\n<passage>\n\n` — only meaningful for future adapters trained WITH a stop token; today it runs away |
| `EARLY_STOP` | JSON-close StoppingCriteria + truncate (prototype; shelved for the curve) |
| `LIMIT = N` | bake a smoke cap |

## Status of the kernel dirs (as of 2026-09-07)

| dir | protocol | status |
|---|---|---|
| `kernel_cap06/17/4b` | chat | full-200 pushed; cap06 output in `outputs/capacity_eval/qwen3-0.6b/` (suppressed; NOT the real curve point) |
| `kernel_cap06p` | plain | full-200 A/B in `outputs/capacity_eval/qwen3-0.6b-plain/` — protocol-forensic, NOT deployable/comparable |
| `kernel_cap06ps` | plain + early-stop | 6-row smoke (proves early stop truncates closed rows; runaway rows un-salvageable) |

All capacity outputs are currently **interim**; the authoritative curve awaits the
chat-format retrain. See `.scratch/kg-triplet-round1/issues/08-capacity-runs.md`.
