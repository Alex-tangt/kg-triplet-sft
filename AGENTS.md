# AGENTS.md — kg-triplet-sft

Guidance for AI coding agents working in this repository. Read this before making changes.

## What this project is

Reproduction + extension of the Qwen3-0.6B knowledge-graph triplet extraction pipeline:
data construction → LoRA SFT → evaluation → deployment. Reproduction target:
`mohar07/qwen3-0.6b-kg-triplets` (composite 0.6583, entity_f1 0.179 on its 700-entry test set).

Domain vocabulary lives in `CONTEXT.md`. Read it before naming anything.

## Structure

```
dataset/    Phase 1: corpus → cleaning → teacher labeling → validation → splits → Alpaca
finetune/   Phase 2: LLaMA-Factory yaml configs (0.6B / 1.5B / 3B + data ablation) + dataset card.
            Capacity line: finetune/compshare/train_unsloth.py (canonical masked
            recipe, one CompShare RTX 4090). Inference kernels/runbooks:
            finetune/kaggle/README_inference.md (reproduction inferfull stack),
            finetune/kaggle/kernel_capinfer/README.md (capacity line),
            throughput study docs/inference-optimization-brief.md
eval/       Phase 3: TWO lines — reproduction = vendored HGR harness (zero-change) under
            eval/harness/ + eval/tracer_eval.py; open GraphRAG line = referent-level pipeline
            eval/referent_*.py. Start at eval/README.md.
serve/      Phase 4: LLaMA-Factory export (merge + GGUF), Ollama Modelfile, Gradio demo
docs/adr/   Decisions (0001–0008; eval decisions in 0007/0008)
docs/       Reports: 2026-09-07-capacity-line-repair.md (root cause + 4090 train/infer guidance)
```

## Locked decisions (from the grilling session — do not silently reverse)

> **Data-generation pivot (owner-approved 2026-09) — governs where the bullets below
> conflict.** The student's task changed from mohar07's typed 20-relation triplets to open
> GraphRAG-style extraction for an MS GraphRAG graph-QA consumer: labels are produced by the
> official GraphRAG extraction prompt (entities/relationships + strength 0-10, ADR-0003); the
> student prompt is DERIVED from that prompt, deployment is our own JSON extraction API, and
> the teacher/student prompts are deliberately distinct (ADR-0004); training `cutoff_len` is
> **6144**, not 2048. The bullets below stay as the reproduction line's record (composite /
> harness evaluation) and apply only where not overridden by ADR-0003/0004.

- **Round 1 scope**: 0.6B faithful reproduction + capacity curve (0.6/1.5/3B) + data-scale
  ablation (500/1300/2575 on 0.6B) + attribution suite. 8B and the pass line (entity_f1 ≥ 0.6)
  are **Round 2**, decided after the Round 1 review.
- **Dataset**: English Wikipedia 60% + arXiv 40%; chunks 100–300 words cut at sentence
  boundaries (never mid-sentence); 2575 train / 75 val / 700 test; labels by qwen3-flash
  (DashScope); 15% hard negatives judged by the teacher (gold = `[]`); curriculum ordering
  by (triplet count, word count) ascending; strict entity decontamination train↔test
  (fall back to ≥2-entity rule only if attrition > 25%).
- **Weights**: fixed anchors 0.2 / 0.5 / 0.8 by textual evidence strength.
- **Training**: LLaMA-Factory on Kaggle T4, identical hyperparameters across all sizes
  (r=32, α=32, lr 5e-5, 5 epochs, bs2 × grad-accum 4, ctx 2048, bf16 LoRA) — the only
  variable is the base model. Each run is one yaml config; configs must differ only in
  model/dataset fields. QLoRA only for Round 2's 8B, and that confound must be called out.
- **Prompts**: see ADR-0001 — one template constant renders training/eval/serving prompts;
  the target prompt is verbatim from the original (ontology implicit, never listed).
- **Evaluation**: our own test split only (ADR-0002); default harness config is the reporting
  baseline; the threshold sweep is only 3 configs (default / 0.60+0.30 / --no-embeddings).

### Open-line evaluation (overrides the reproduction bullets where they conflict)

- **Referent-level**: teacher↔student matching on same-real-world-entity identity,
  two-tier — free exact-title pairs + thinking-mode LLM judge (`qwen3.7-flash`,
  shared `IDENTITY_RUBRIC`, ADR-0007). Never treat a single no-thinking pass as
  the headline: it over-merges co-occurring concepts and cannot be tuned to
  thinking parity (hardened/agree/escalation all failed, ADR-0008).
- **Fixed sample**: every future model is scored on the same stratified 200
  passages (`outputs/referent_eval/sample_200.json`, ±2.1pt CI) against the
  reference baseline `report_ref200.json`. `outputs/full_eval/` gold + reference
  predictions are shared inputs — do not regenerate.
- **Consumer axes**: reachability (MiniLM retrieval sim), dup/granularity,
  raw-output redundancy diagnostics. Full commands + data contracts: `eval/README.md`.
- **Cross-protocol caution**: capacity-line models are trained chat OR plain; eval
  must match the training format. Mixing protocols conflates capability and
  protocol penalty (issue 08) — call it out in any comparison. The capacity line's
  validated recipe is **canonical masked** (no-think chatml + response-only
  masking), not plain concat and not the stock chat template.

### Capacity-line lessons (2026-09-07) — do not regress these

- **自洽 ≠ 可比.** "Config identical" may only be asserted **against the anchor
  kernel** (reference line = `finetune/kaggle/kernel_lffull/lffull.py`); a
  diff-cards PASS among sibling capacity runs only proves self-consistency and
  silently missed a structural gap vs the anchor (loss coverage, template,
  packing, eff batch, warmup, dropout). Always produce a reference-vs-candidate
  table before spending GPU on a comparative run.
- **A raw `dataset_text_field` = continued-pretraining loss** (every token,
  prompt included). SFT requires explicit response-only masking via
  `unsloth.chat_templates.train_on_responses_only` (markers
  `<|im_start|>user\n` / `<|im_start|>assistant\n`); verify `mask_check.json`
  (head masked, response ids present, not all `-100`).
- **Qwen3 stock chat template injects an empty `<think>` block** in training
  renders; mirror the reference/eval by overriding `tokenizer.chat_template`
  with no-think chatml, and restore it before `save_pretrained`.
- **`UNSLOTH_RETURN_LOGITS=1` disables unsloth packing** (`packing=True ignored`)
  — acceptable: non-packed mirrors the reference step semantics and dodges the
  fused-loss sparse-mask bug (unsloth#5230).
- **Run-to-run evidence lives in the run dir**: `run_config.json` (card),
  `format_check.json`, `mask_check.json`, `receipt.json`; tick issue checkboxes
  only from those artifacts.
- **4090 inference**: same `kernel_capinfer` code as Kaggle but raise decode
  batch (`make_capinfer.py --batch 12` for 0.6B on one 4090 → ~5x wall vs bs4,
  GPU ~95%). fp16 == bf16 throughput on Ampere; fp16 keeps parity with the
  reference eval line. Run it on the 4090 host via a `/kaggle` symlink tree
  instead of a Kaggle kernel when quota is tight (see the repair report).
- **Mask-check is not the only schema risk**: 4B masked emits ~17% of rows with
  UPPERCASE schema keys (`"TITLE"/"TYPE"/…`), which the tolerant parser drops
  (salvage keys on lowercase only) → those rows land EMPTY. When comparing sizes
  state whether the number is canonical (strict) or case-tolerant ("clean", e.g.
  `qwen3-4b-masked-clean`); deploy must decide on a key-normalizing parser.
- **Decode budget vs gold**: the fixed-200 gold answers are median ~1.9k / p95
  ~3.3k tokens (max 8.7k), so `max_new_tokens=8192` is ~4x over-provisioned; a
  JSON-close early stop or ~5k cap covers ~98% at a fraction of the wall time.

### Round-1 status (2026-09-07 closeout)

Round-1 capacity work is **complete**: the canonical masked recipe ran on all
three sizes (0.6/1.7/4B, CompShare 4090, receipts in each host run dir),
inferred, and referent-evaluated — predictions/reports under
`outputs/capacity_eval/qwen3-{0.6b,1.7b,4b}-masked/` and
`outputs/referent_eval/report_qwen3-…-masked{,-clean}.json`. Final curve reading
+ table: `docs/2026-09-07-capacity-line-repair.md` §1.4c and issue 08 Closeout.
Open/optional items (not Round-1 blockers): HF adapter push, a deploy ADR on
case-tolerant parsing, and the Round-1 review gate; issue-09 ablation and the 8B
pass line stay Round-2.

## Conventions

- Python 3.12. `uv` for the vendored eval harness; plain scripts + requirements for the rest.
- No comments in code unless they explain a non-obvious decision (see ADRs).
- Dataset scripts are CLIs: `python dataset/xxx.py` with deterministic outputs under `dataset/data/`.
- Never write data files outside `data/`, `dataset/data/`, or `outputs/`; all are gitignored.

## Verification

- Eval harness: `uv run python scripts/evaluation.py --no-embeddings --limit 50` must print a
  composite in [0, 1] and exit 0 (offline lexical fallback).
- Open-line eval tooling: `python -m py_compile eval/referent_*.py` must pass; the reachability
  sabotage gate `python eval/referent_reach.py --fixture` must print `PASS` (with
  `HF_HUB_OFFLINE=1`); `referent_eval --ids` must accept a sample subset (bootstrap ns adapts).
- Data pipeline: every script must run end-to-end on a tiny fixture (a few samples), never
  require the full corpus to verify.
- Training scripts: no local GPU — correctness is verified locally by a
  dry-run / config card diff (`train_unsloth.py --write-card` +
  `--diff-cards`); real runs happen on the CompShare 4090 with `format_check` /
  `mask_check` / `receipt` evidence in the run dir.

## Related projects (same machine, reference only)

`../HGR-finetuned-model-evaluation-pipeline` (harness source, 700-entry original test data) and
`../MiniLoRA` (the earlier tutorial, same training stack).

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles map to themselves. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.
