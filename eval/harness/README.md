# KG Triplet Evaluation

An **evaluation harness** that scores a model's predicted knowledge-graph
triplets against a gold reference set. It grades the fine-tuned extraction
model (trained in `finetuning_qwen3.5`, served in `Hybrid-RAG`) on how well its
predicted triplets align with the gold set.

A triplet has the shape:

```json
{
  "source":   {"title": "str", "type": "entity|concept"},
  "relation": {"type": "str", "weight": 0.0},
  "target":   {"title": "str", "type": "entity|concept"}
}
```

Python 3.12+, dependencies managed with [`uv`](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
```

## Running the evaluator

```bash
uv run python scripts/evaluation.py --input data/predict.jsonl --report report.json
```

The default input is `data/predict.jsonl`; `--report` writes the full JSON
report (per-entry metrics plus corpus aggregates). The CLI prints a summary to
stdout and exits `0` on success.

### CLI flags

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

> **Note:** the root `main.py` is a placeholder stub. The real entry point is
> `scripts/evaluation.py`.

## Input format

`predict.jsonl` — one JSON object per line with keys:

- `prompt` — the text the model was asked to extract triplets from
- `gold_raw` / `gold_parsed` — the gold triplets (raw output + parsed list)
- `pred_raw` / `pred_parsed` — the model's triplets (raw output + parsed list)

`generate_predictions.ipynb` produces this file from a chat-format
`test.jsonl` for any Hugging Face model (see the notebook for details).

## How scoring works

The pipeline, in order:

1. **Schema validation** — is each predicted triplet well formed? (highest-weighted axis)
2. **Normalization** — strip whitespace / company suffixes (Inc/Ltd/…) for clean comparison
3. **Hungarian 1-1 matching** — optimal pred→gold alignment (self-contained O(n³) implementation, no SciPy)
4. **Embedding matching** — exact string match first, then fuzzy match via embeddings (accepted when similarity ≥ threshold), plus weight-closeness banding
5. **Hallucination check** — flag predictions not grounded in gold/prompt; drop self-referential (`source == target`) triplets and relations outside the valid set
6. **Multi-axis scoring** — composite with weights:

   | Axis | Weight | Meaning |
   |---|---|---|
   | `schema` | 0.30 | fraction of produced items that are well formed |
   | `entity_f1` | 0.25 | endpoint (source/target) alignment |
   | `relation_acc` | 0.20 | relation correct given matched endpoints |
   | `weight` | 0.10 | confidence weight within 0.10 / 0.20 band |
   | `grounding` | 0.15 | 1 − hallucination rate |

   Inapplicable axes are dropped and the rest renormalized, so the composite
   always stays in `[0, 1]`.

### Embedding backend (graceful degradation)

The similarity backend is pluggable and degrades gracefully. It prefers
`fastembed` (ONNX), then `sentence-transformers`, then a pure-stdlib lexical
similarity — so the pipeline always runs, with **no model and no network**
required. To force the lexical fallback: `--no-embeddings`.

## Data

- `data/predict.jsonl` — 700 entries, **fine-tuned** `Qwen3-0.6B` predictions
- `data/base_qwen_predict.jsonl` — 700 entries, **base** `Qwen3-0.6B` predictions (same prompts/gold)

Both share the same gold set. The fine-tuned model produces schema-valid,
well-grounded triplets but with limited recall/precision of the gold set (see
Methodology note below); the base model largely fails to follow the triplet
schema.

## Methodology note

The composite is deliberately **schema- and grounding-weighted** because the
generating model is small: schema adherence is the cheapest, most reliable
signal. Reported metrics include micro-averaged corpus P/R/F1 (entity- and
triplet-level) and macro-averaged (per-entry mean) axis scores, so the single
composite number can be decomposed. An empty prediction is never rewarded —
schema and grounding are treated as N/A when the model produces nothing.

## Verification

No unit-test framework is configured. To validate changes:

```bash
# Quick smoke test on the first 50 entries
uv run python scripts/evaluation.py --limit 50

# Confirm the lexical fallback still runs
uv run python scripts/evaluation.py --no-embeddings --limit 50
```

Both should complete and emit a composite score in `[0, 1]`.

## Related projects

`hgr-dataset` (dataset + schema) → `finetuning_qwen3.5` (trains the model) →
`Hybrid-RAG` (serves it) → **KG_triplet_evaluation** (this repo, scores its
predictions).

The 20 `VALID_RELATION_TYPES` and `VALID_ENTITY_TYPES` (`entity`, `concept`)
in `scripts/evaluation.py` must stay identical to the schema in `hgr-dataset`
and `finetuning_qwen3.5`. A relation the model emits that isn't in that set is
treated as a hallucination.