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
finetune/   Phase 2: LLaMA-Factory yaml configs (0.6B / 1.5B / 3B + data ablation) + dataset card
eval/       Phase 3: the vendored HGR evaluation harness (kept zero-change) + report scripts
serve/      Phase 4: LLaMA-Factory export (merge + GGUF), Ollama Modelfile, Gradio demo
docs/adr/   Decisions (prompt single source, own-test evaluation)
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

## Conventions

- Python 3.12. `uv` for the vendored eval harness; plain scripts + requirements for the rest.
- No comments in code unless they explain a non-obvious decision (see ADRs).
- Dataset scripts are CLIs: `python dataset/xxx.py` with deterministic outputs under `dataset/data/`.
- Never write data files outside `data/`, `dataset/data/`, or `outputs/`; all are gitignored.

## Verification

- Eval harness: `uv run python scripts/evaluation.py --no-embeddings --limit 50` must print a
  composite in [0, 1] and exit 0 (offline lexical fallback).
- Data pipeline: every script must run end-to-end on a tiny fixture (a few samples), never
  require the full corpus to verify.
- Training scripts: no local GPU — correctness is verified by a dry-run / config print, real
  runs happen on Kaggle with checkpointing.

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
