# Capacity-line repair — training root cause + 4090 train/infer loop (2026-09-07)

Scope: why the bf16 capacity curve on the CompShare RTX 4090 was not comparable to
the reference line, what the real fix was, and how to run training **and** inference
efficiently on one 4090. Measured numbers come from the runs documented in
`.scratch/kg-triplet-round1/issues/08-capacity-runs.md` and the artifacts under
`outputs/capacity_eval/`, `outputs/referent_eval/`.

---

## 1. The training problem and its solution

### 1.1 Symptom

All capacity models (0.6B/1.7B/4B, bf16 Unsloth on a 4090) scored ~0.44 referent
recall on the fixed 200-sample eval, vs 0.533 for the reference 0.6B (fp16,
LLaMA-Factory, dual T4). Two fixes we tried made **no** difference:

- plain→chat text wrapper → recall stayed 0.44 (cap06 chat)
- retrain in the Qwen3 stock chat template (empty `<think>` preamble) → 0.44
  (0.6b_chat)

`config` cards said all three capacity runs were identical except `--model`. They
were — to each other. None of them matched the **anchor**.

### 1.2 Root cause: loss coverage, not the wrapper

The audit compared every learning-relevant knob against the anchor kernel
(`finetune/kaggle/kernel_lffull/lffull.py`):

| axis | anchor (reference line) | our capacity runs | verdict |
|---|---|---|---|
| **loss coverage** | response-only (LLaMA-Factory SFT default masks the prompt) | full-sequence LM over a raw `text` field | **the bug** |
| training template | LLaMA-Factory `qwen3_nothink` | stock chat template (empty `<think>` block) | fixed |
| effective batch | 8 | 16 | aligned |
| warmup_ratio / dropout | 0.1 / 0.1 | 0.03 / 0.05 | aligned |
| optimizer | adamw_torch | adamw_8bit | aligned |
| precision | fp16 | bf16 | kept bf16 (4090 native) |

Why masking is the dominant axis: a raw `dataset_text_field` is, in Unsloth, a
**continued-pretraining loss** (every token, prompt included). The prompt is
`instruction + passage` ≈ 10x the JSON answer, so ~85–90% of the gradient went to
predicting prompt continuation. Two very different wrappers scoring identically was
the tell — both shared the same no-mask defect.

Additional confirmed facts (so nobody re-explores them): data file is byte-identical
to the anchor's (`sha256` equal); only 2/2575 rows exceed the 6144 cutoff; the eval
side was already aligned (same `STUDENT_PROMPT`, same chat no-think kernel, same
schema-prefix + sampling).

### 1.3 Fix: canonical masked recipe

`finetune/compshare/train_unsloth.py` (one 4090, run dir carries the evidence):

- render rows through a **no-think chatml** template (messages →
  `apply_chat_template`; stock Qwen3 injects `<think>\n\n</think>`, override +
  restore before saving);
- **response-only loss** via
  `unsloth.chat_templates.train_on_responses_only(trainer, "<|im_start|>user\n",
  "<|im_start|>assistant\n")` — unsloth does **not** mask raw text automatically;
- eff batch 8 (bs 2 × ga 4), warmup 0.1, dropout 0.1, `adamw_torch`, bf16, lr 5e-5
  cosine, 5 epochs, seed 42, cutoff 6144, non-packed, `UNSLOTH_RETURN_LOGITS=1`;
- evidence artifacts: `run_config.json`, `format_check.json`, `mask_check.json`
  (labels not all `-100`), `receipt.json`.

### 1.4 Result

Same 200-sample referent eval (micro):

| run | recall | precision | F1 | schema valid | halluc |
|---|---|---|---|---|---|
| reference (ref200) | 0.533 | 0.723 | 0.614 | 0.931 | 0.142 |
| plain / chat-eval (cap06) | 0.440 | 0.600 | 0.508 | 0.808 | 0.257 |
| stock-chat, no mask (0.6b_chat) | 0.419 | 0.557 | 0.478 | 0.836 | 0.321 |
| **canonical masked 0.6B** | **0.536** | **0.699** | **0.607** | **0.919** | **0.166** |

The masked run is statistically at the reference (n=200, ±~2.7 pt) — root cause
confirmed. 0.6B full run: 1610 steps, final loss 0.355, ~50 min, ~2 CNY.

### 1.5 Lessons

1. **自洽 ≠ 可比.** "Config identical" is only provable against the anchor kernel,
   never between siblings. Always build a reference-vs-candidate table first.
2. Raw text + Unsloth = continued-pretraining. SFT needs explicit response-only
   masking; verify with a `mask_check` that labels are not all `-100`.
3. Qwen3's stock template emits an empty `<think>` block in training renders — the
   reference/eval do not use one.
4. `UNSLOTH_RETURN_LOGITS=1` silently disables packing (fine: non-packed mirrors the
   reference and dodges the fused-loss sparse-mask bug, unsloth#5230).
5. Community-verified: `train_on_responses_only` is the supported masking path
   (unsloth docs); raw-text and long-input masking historically had bugs
   (unsloth#1017); check `grad_norm`/loss curve so a silent zero-gradient
   (unsloth#5230 family) cannot pass unseen.

---

## 2. Running both training and inference on one 4090

### 2.1 Training on the 4090

- Script: `finetune/compshare/train_unsloth.py` — bf16 native on Ampere, single
  card, no DDP. Non-packed masked run is ~50 min / ~2 CNY for 0.6B (1610 steps,
  ~1.9 s/step at bs2 on short sequences). Scaling estimate: 1.7B ~2.5 h, 4B ~5–6 h.
- Do not launch the GPU job through a pipe that can deadlock the agent shell: start
  it detached (`nohup … > run.log 2>&1 &`) and poll the log for the receipt.

### 2.2 Inference on the 4090 (was tuned for a T4 — raise the batch)

The capacity inference kernel (`finetune/kaggle/kernel_capinfer`) bakes
`BATCH_SIZE = 4` for a 16 GB T4. On a 24 GB 4090 this leaves the GPU idle:

| config | measured (0.6B, 200 rows, one 4090) |
|---|---|
| `BATCH_SIZE = 4` | GPU util ~24%, ~3.5 rows/min, first run >30 min for ~120 rows |
| `BATCH_SIZE = 12` | GPU util ~95%, ~17 rows/min → full 200 rows in ~12 min (~5x) |

Guidance:
- `make_capinfer.py --batch N` bakes a per-run batch (default stays 4 for Kaggle
  T4). 0.6B→12, and pick by memory for 1.7B/4B.
- **fp16 == bf16 on Ampere** — keep the kernel's fp16 to stay byte-comparable with
  the reference eval line; do not chase dtype for speed.
- Do not kill a run mid-way: `predictions.jsonl` is only written after the last row,
  and rows are processed in length-sorted batches (longest last, decode up to 8k
  tokens each). Progress on `raw_*.txt` jumps in batch-sized bursts; a flat tail is
  the longest batches working at ~95% util, not a hang.
- If Kaggle GPU quota is tight, run the **same generated kernel** on the 4090 host
  through a `/kaggle/input` symlink tree (base model, adapter dir, passages file) —
  byte-identical code, no Kaggle upload/queue.
- The tolerant parser now drops non-object rows on the strict-parse path too
  (bare-string relationships crashed `referent_eval`); every `predictions.jsonl`
  is schema-clean by construction.

### 2.3 Recommended loop: 4090 train → 4090 infer

```
finetune/compshare/train_unsloth.py   # canonical masked recipe, per model size
  -> run_dir/{run_config,format_check,mask_check,receipt}.json
make_capinfer.py --batch <N> …        # parser-fixed kernel per size
  -> run on the SAME 4090 via /kaggle symlink tree
  -> predictions.jsonl -> outputs/capacity_eval/<tag>/
eval referent pairing + report vs report_ref200   # local, DashScope judge
  -> decide next size / stop
```

Batch training and inference on one continuous 4090 rental to amortize the fixed
cost; measured 0.6B numbers give the curve a ~16–18 CNY budget for 1.7B + 4B.

---

See `docs/adr/0006` (platform), `.scratch/kg-triplet-round1/issues/08-capacity-runs.md`
(evidence trail), `eval/README.md` (eval contracts), AGENTS.md (lessons + do-not-
regress list).
