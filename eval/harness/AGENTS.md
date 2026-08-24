# AGENTS.md — KG_triplet_evaluation

Guidance for AI coding agents working in this repository. Read this before making changes.

## What this project is

An **evaluation harness** that scores a model's predicted knowledge-graph triplets against a
gold reference set. It is how the fine-tuned extraction model (from `finetuning_qwen3.5`, served
in `Hybrid-RAG`) is graded. Python 3.12+, managed with `uv`.

A triplet has the shape:

```json
{
  "source":   {"title": "str", "type": "entity|concept"},
  "relation": {"type": "str", "weight": 0.0},
  "target":   {"title": "str", "type": "entity|concept"}
}
```

## Setup & running

```bash
uv sync
uv run python scripts/evaluation.py --input data/predict.jsonl --report report.json
```

The **real entry point is `scripts/evaluation.py`** (an argparse CLI). Note:
- The root `main.py` is a **placeholder stub** and `README.md` is empty — ignore them.
- `generate_predictions.ipynb` is what produces `data/predict.jsonl` (the model's predictions
  paired with gold). Each JSONL entry has keys: `prompt`, `gold_raw`, `gold_parsed`,
  `pred_raw`, `pred_parsed`.

CLI flags:

```
--input, -i             input JSONL (default: data/predict.jsonl)
--report, -r            write the full JSON report to this path
--embedding-model       fastembed model (default: BAAI/bge-small-en-v1.5)
--no-embeddings         disable embedding matching (lexical fallback only)
--embedding-threshold   similarity acceptance (default: 0.80)
--align-accept          Hungarian alignment floor (default: 0.50)
--limit                 only evaluate the first N entries
--quiet                 suppress progress output
```

## How scoring works (in `evaluation.py`)

The pipeline, in order:
1. `validate_schema` — is each predicted triplet well-formed? (highest-weighted axis)
2. `normalize_triplets` — strip whitespace / company suffixes (Inc/Ltd/…) for clean comparison
3. `hungarian_algorithm_for_one_one_matching` — optimal 1-1 pred→gold alignment
4. `embedding_matching` — exact string match first, then fuzzy match via embeddings
   (accepted when similarity ≥ threshold), plus weight-closeness banding
5. `final_hallucination_check` — flag predictions not grounded in gold/prompt; drop
   self-referential (`source == target`) triplets
6. `scoring` — multi-axis composite, weights:
   `schema 0.30`, `entity_f1 0.25`, `relation_acc 0.20`, `weight 0.10`, `grounding 0.15`
   (inapplicable axes are dropped and the rest renormalized, so scores stay in `[0, 1]`)

## Important design properties (preserve these)

- **The embedding backend degrades gracefully and runs offline.** It prefers `fastembed`
  (ONNX), then `sentence-transformers`, then a pure-stdlib lexical similarity. Any change must
  keep the pipeline runnable with **no model and no network** — never make a network/model load
  a hard requirement.
- **Schema is the highest-weighted axis on purpose** — the generating model is small, so schema
  adherence is the cheapest, most reliable signal. Don't rebalance weights casually.
- `evaluation.py` carries comments about historical bugs (e.g. a triple-quoted string that
  silently fused into a `set` literal and dropped a relation). When editing the constants near
  the top, keep relation/entity types as real comments, not adjacent string literals.

## Schema must match the rest of the pipeline

`VALID_RELATION_TYPES` (the **20** relation types) and `VALID_ENTITY_TYPES` (`entity`,
`concept`) are defined at the top of `evaluation.py` and **must stay identical** to the schema
in `hgr-dataset` and `finetuning_qwen3.5`. A relation the model emits that isn't in this set is
treated as a hallucination. If the shared schema changes, update it here too.

## Verification

No unit-test framework is configured. To validate changes:
- Run the CLI on `data/predict.jsonl` (optionally with `--limit 50` for speed) and confirm it
  completes and emits a composite score in `[0, 1]`.
- Test the graceful-degradation path with `--no-embeddings` to confirm the lexical fallback
  still runs.

## Related projects (same workspace)

`hgr-dataset` (dataset + schema) → `finetuning_qwen3.5` (trains the model) → `Hybrid-RAG` (serves
it) → **KG_triplet_evaluation** (this repo, scores its predictions).
