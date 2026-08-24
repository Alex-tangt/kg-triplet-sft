Status: ready-for-agent

# Spec: KG-Triplet SFT — Round 1 (faithful reproduction + capacity/data attribution)

## Problem Statement

I want a resume-grade SFT project for autumn recruitment: rebuild the full pipeline behind
`mohar07/qwen3-0.6b-kg-triplets` (KG triplet extraction fine-tune) using my own dataset —
corpus collection through deployment — and, beyond reproducing it, answer with experiments why
the original's entity F1 is only 0.179 (harness thresholds? label quality? data scale? model
capacity?). No local GPU: training runs on Kaggle T4; evaluation and serving run on a CPU-only
laptop (Ryzen 7 7735H, 28GB RAM).

## Solution

A single repo, four phases. Phase 1 builds a ~3350-example English dataset (Wikipedia 60% +
arXiv 40%, labeled by qwen3-flash) through a 10-step pipeline ending in an Alpaca-format
training set. Phase 2 fine-tunes three base sizes (Qwen3-0.6B faithful reproduction, plus 1.5B
and 3B) with identical hyperparameters via LLaMA-Factory on Kaggle, plus a data-scale ablation
(500/1300/2575) on the 0.6B. Phase 3 evaluates every model with the vendored HGR harness on our
own 700-entry test split and produces the attribution report (threshold sweep, gold audit,
scaling curves). Phase 4 exports GGUF via LLaMA-Factory, serves locally with Ollama, and ships
a Gradio demo (text in → triplet list + pyvis graph). Everything is public at the end.

## User Stories

1. As the project owner, I want a corpus-collection script that pulls English Wikipedia and
   arXiv passages through hf-mirror, so that the dataset is reproducible without scraping.
2. As the project owner, I want passages chunked to 100–300 words at sentence boundaries,
   never mid-sentence, so that gold labels are not polluted by truncated or dangling references.
3. As the project owner, I want a quality filter (language, length, boilerplate) with a
   MinHash near-duplicate pass, so that the corpus is clean before I pay for labels.
4. As the project owner, I want the teacher prompt (relation dictionary, entity-naming rules,
   exhaustiveness demand, weight anchors) applied to every passage via qwen3-flash, so that
   gold labels follow one consistent rubric.
5. As the project owner, I want hard negatives (~15%, teacher-judged `[]` outputs) in the
   training set, so that the model learns when to output an empty array.
6. As the project owner, I want schema validation (JSON parseable, relation in the 20-set,
   type in {entity, concept}, weight in [0,1]) with automatic retry, so that invalid labels
   never reach training.
7. As the project owner, I want semantic validation of the in-chunk grounding rule and the
   weight anchors, so that labels violating the rubric are caught early.
8. As the project owner, I want curriculum ordering (triplet count, then word count,
   ascending) in the training file, so that easy examples come first.
9. As the project owner, I want strict entity decontamination between train and test (any
   normalized-entity overlap removes the test sample), degrading to the ≥2-entity rule only
   if attrition exceeds 25%, so that scores cannot come from memorized entities.
10. As the project owner, I want splits of 2575 train / 75 val / 700 test, and an Alpaca
    projection (instruction/input/output) rendered from the single prompt template, so that
    the dataset feeds LLaMA-Factory directly and the template stays the one source of truth.
11. As the project owner, I want one LLaMA-Factory yaml config per training run, all identical
    except base model (and dataset for the ablation points), so that the capacity curve has a
    single controlled variable.
12. As the project owner, I want the 0.6B run to reproduce the original hyperparameters
    (r=32, α=32, lr 5e-5, 5 epochs, bs2×grad4, ctx 2048, bf16 LoRA), so that the reproduction
    claim is honest.
13. As the project owner, I want checkpointing/resume on Kaggle within the 30h/week quota, so
    that a 15h run survives disconnects and quota resets.
14. As the project owner, I want each trained adapter pushed to Hugging Face from the Kaggle
    notebook, so that the models are publicly linkable.
15. As the project owner, I want the vendored evaluation harness to run offline on this laptop
    (--no-embeddings path), so that evaluation never depends on cloud access.
16. As the project owner, I want predictions generated for the base model and every trained
    adapter on our 700-entry test split, so that the comparison table is complete.
17. As the project owner, I want a threshold sweep of exactly three configs (default, relaxed
    0.60/0.30, --no-embeddings), so that the harness-software contribution to the low F1 is
    quantified without over-engineering.
18. As the project owner, I want a gold audit (100 stratified samples; human on 30, LLM on
    all 100, same rubric) that measures label error and omission, so that the data-quality
    contribution is quantified.
19. As the project owner, I want an attribution report answering "which of harness / labels /
    data scale / capacity dominates the low entity F1", with the two scaling curves, so that
    the Round 2 review can decide the 8B plan on data.
20. As the project owner, I want Round 1 acceptance measured as: 0.6B composite in 0.55–0.75
    with the same axis shape as the original 0.6583, plus curves and attribution answer.
21. As the project owner, I want the merged model exported to GGUF Q4 and served via Ollama
    on this laptop, so that I can demonstrate without a GPU.
22. As the project owner, I want a Gradio app (paste text → triplet list + pyvis node graph)
    with the 0.6B as default and 3B selectable, so that interviews get a live demo.
23. As the project owner, I want the dataset, adapters, and repo public (HF blocked → Kaggle
    datasets / GitHub Release fallback), so that the resume carries working links.

## Implementation Decisions

- Single repo with four phase directories and a cross-phase contract module: the 20 relation
  constants, the triplet schema validator, and the prompt template all live in one place and
  are imported by dataset, finetune data prep, eval prediction generation, and serve. The
  vendored harness's constants must stay byte-identical to this module.
- Target prompt is reproduced verbatim from the original project (ontology implicit, never
  listed in the prompt); the teacher prompt is a separate, richer instruction (ADR-0001). The
  template renders the Alpaca projection, the eval predictions, and the serving requests.
- Weights are fixed anchors 0.2 / 0.5 / 0.8 chosen by textual evidence strength (glossary:
  weight). Hard negatives are teacher-judged; corpus-level negatives only fill the 15% quota.
- Chunking is sentence-boundary-aligned at 100–300 words with the in-chunk grounding rule — a
  deliberate improvement over the original's mid-sentence truncation (recorded as a pipeline
  difference in the README).
- Training stack is LLaMA-Factory (replaces raw Unsloth+TRL scripts); hyperparameters stay
  identical to the original model card, mapped 1:1 into yaml configs. bf16 LoRA for all three
  sizes; QLoRA reserved for Round 2's 8B.
- The data-scale ablation trains the 0.6B on 500 / 1300 / 2575 examples with the same config.
- Evaluation uses only our own test split (ADR-0002); the original 700-entry gold is not a
  scoring target. Default harness config is the reporting baseline; the sweep is exactly three
  configs.
- Gold audit: 100 samples stratified by source / polarity / difficulty / relation coverage;
  human scores 30, qwen3-flash scores all 100 with the teacher rubric; inter-rater agreement
  is reported.
- Deployment: LLaMA-Factory export (merge + GGUF Q4) → Ollama Modelfile → Gradio app with
  pyvis visualization. Mini GraphRAG is Round 2.
- Public assets: GitHub repo, HF dataset, HF adapters with model cards; Kaggle datasets and
  GitHub Release are the fallback if HF upload is blocked.

## Testing Decisions

- The one cross-phase seam is the contract module (relations, schema validator, prompt
  template). Tests there are the highest-value: a schema-constants drift test asserts equality
  against the vendored harness's frozen set, since that drift was a historical bug source.
- Each dataset stage is a CLI over files; every stage must run end-to-end on a tiny fixture
  (a handful of samples) and assert file invariants (schema compliance rate, split sizes,
  decontamination property, curriculum ordering).
- Finetune configs are tested without a GPU: a diff test asserts that configs differ only in
  base-model/dataset fields (the single-variable invariant), and the messages-building path is
  exercised locally with the Alpaca fixture.
- Eval is tested by running the vendored harness offline (--no-embeddings --limit 50) and
  asserting a composite in [0,1]; this mirrors the harness's own smoke tests.
- Serve logic (triplet parsing, graph building) is pure functions tested with fixture model
  outputs; the Gradio app is a thin shell.
- Good tests only check external behavior (files in → files out, invariants, printed reports),
  never implementation details. Prior art: the harness's README verification commands.

## Out of Scope

- Round 2: 8B sprint for entity F1 ≥ 0.6, dataset v2, any retraining loops.
- Mini GraphRAG on top of the extracted triplets.
- Chinese extension / bilingual datasets.
- Tuning the harness's scoring weights or ontology (they are frozen for comparability).
- Local GPU training or cloud-pay training.

## Further Notes

- Kaggle free tier: 30h GPU/week, phone-verified. The longest single run (~3B, ~6h) must
  checkpoint; the full Round 1 GPU budget is ~28h across two weeks.
- qwen3-flash label budget: ~3000 teacher calls ≈ ¥20–50 on DashScope; a per-run budget cap
  and retry-with-backoff are required.
- Model downloads go through hf-mirror; uploads to HF happen from Kaggle (no GFW there).
- The attribution report is a deliverable with the same standing as the model scores — it is
  what the Round 2 gate consumes.
