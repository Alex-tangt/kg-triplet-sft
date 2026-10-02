# Evidence — L1 teacher-forced per-field fit (0.6B base vs masked)

**Status: OBSERVED ONLY.** A reading standard (what counts as underfit / well-fit,
and how to compare fields) is **not yet frozen** (see `OPEN-QUESTIONS`). No
conclusion is drawn here. The interpretation I gave verbally on 2026-09-25
("underfit excluded, points to L2") is recorded below as **WITHDRAWN**.

Claim tags used: `measured` (raw), `hypothesis`, `unresolved`, `withdrawn`.

## Run card (immutable)

| item | value |
|---|---|
| purpose | teacher-forced response-token NLL, split by JSON field, train vs val |
| model (base) | `qwen-lm/qwen-3/transformers/0.6b/1` (no adapter) |
| model (masked) | base + adapter `idalextan/l1-masked-adapter` (= `outputs/adapters/qwen3-0.6b_masked/adapter`) |
| data | `idalextan/trace-full-dataset` → `alpaca_full_train.jsonl` (2575), `alpaca_full_val.jsonl` (75) |
| render | byte-identical to `train_unsloth.render_chat` (no-think chatml, response-only) |
| tokenizer | `tokenizers.Tokenizer.from_file(adapter/tokenizer.json)` (version-independent) |
| cutoff | 6144, truncation, no special tokens |
| dtype | **fp16** (verified: `model_dtype=torch.float16`), T4×2, one model per GPU |
| env fix | `pip uninstall -y torchao` (Kaggle torchao 0.10.0 breaks PEFT import) |
| limit | none (2575+75 per model) |
| runtime | ~74 min wall; measured ~1.0–1.2 s/row |
| kernel | `idalextan/kg-l1-loss` version 3; code `finetune/kaggle/kernel_l1/l1_loss.py` |
| raw outputs | `outputs/l1/l1_base.jsonl`, `outputs/l1/l1_masked.jsonl`, `outputs/l1/kg-l1-loss.log` |

Field spans: `e_title/e_type/e_desc`, `r_source/r_target/r_desc/r_strength`,
`structural` (keys/brackets/quotes/`<|im_end|>`). Span rule: token assigned by its
char-range midpoint against regex spans of JSON values.

## Observed (measured)

Overall mean response-token NLL:

| split | base | masked |
|---|---:|---:|
| train | 0.590 | 0.300 |
| val | 0.590 | 0.333 |

Per-field mean NLL (train; n = token count):

| field | base | masked | Δ | n |
|---|---:|---:|---:|---:|
| e_title | 1.292 | 0.604 | −0.687 | 315,012 |
| e_type | 0.406 | 0.044 | −0.362 | 119,426 |
| e_desc | 1.218 | 0.732 | −0.487 | 1,129,156 |
| r_source | 0.174 | 0.048 | −0.126 | 86,283 |
| r_target | 0.164 | 0.046 | −0.118 | 91,032 |
| r_desc | 0.187 | 0.055 | −0.132 | 405,196 |
| r_strength | 0.178 | 0.054 | −0.125 | 22,406 |
| structural | 0.372 | 0.167 | −0.204 | 3,025,545 |

Masked per-field val NLL: e_title 0.704, e_type 0.056, e_desc 0.798, r_source 0.057,
r_target 0.044, r_desc 0.059, r_strength 0.070, structural 0.181.

Masked overall NLL by response-token decile (train):
`0.400, 0.388, 0.406, 0.432, 0.383, 0.222, 0.190, 0.192, 0.189, 0.202`.

## Interpretation

- `unresolved`: whether the masked model is "underfit" — a reading standard and an
  upper-bound anchor (e.g. a 4B run or a teacher-consistency floor) are required and
  absent. Absolute NLL has no scale on its own.
- `withdrawn`: "underfit is excluded on the relation fields because their NLL is low,
  so the edge gap must be exposure bias / L2." — withdrawn because **cross-field NLL
  is not comparable**: `r_source`/`r_target` are near-copy fields (low intrinsic
  entropy; base is already 0.17), and `e_desc` is open text (high entropy). Low NLL on
  copy-like fields is not evidence of relation capability.
- `hypothesis` (not confirmed): the train≈val gap is small, and loss does not rise
  toward the end of the response. These are consistent with "no overfitting / no
  long-horizon degradation *under teacher forcing*", but neither is a task-level
  statement and neither has a frozen threshold.

## What this does NOT measure

- correctness of the target (teacher audit, #01) or of the student output;
- free-running behavior (whether the model emits the `relationships` array at all);
- relation *semantics* (the edge metric is topology-only).

## Reproduce

```
kaggle kernels pull idalextan/kg-l1-loss -m -p finetune/kaggle/kernel_l1
kaggle kernels push -p finetune/kaggle/kernel_l1
kaggle kernels output idalextan/kg-l1-loss -p outputs/l1/
```
