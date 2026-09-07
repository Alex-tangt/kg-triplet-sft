# Evaluation pipeline — open GraphRAG line

Score a fine-tuned extractor against the teacher gold on **referent identity**
(same real-world entity), not title strings. Covers finding axes, schema /
grounding, numeric retention, duplication, and consumer reachability. This file
is the single entry point for running the eval and reading its results.

Decisions live in `docs/adr/0007-referent-level-eval.md` and
`docs/adr/0008-consumer-axes-and-nothink-cost.md`; vocabulary in `CONTEXT.md`.

> The **reproduction line** (mohar07 typed triplets) is separate: vendored HGR
> harness under `eval/harness/` and the older `eval/tracer_eval.py`. Do not mix.

## Why this exists (short version)

The student garbles titles, so string matching understates finding. Referent
matching is done in two tiers: free normalized-title-exact pairs, then every
remaining fuzzy pair decided by qwen3.7-flash **in thinking mode** (shared
identity rubric + passage). Single-shot / no-thinking judges over-merge
co-occurring concepts and **cannot** be tuned to parity (ADR-0008), so thinking
is the production judge. Future models run a **fixed 200-passage sample**
(entity-recall 95% CI ≈ ±2.1 pt at n=200).

## Data contracts

Teacher gold (do not regenerate):
`outputs/full_eval/test_teacher_with_text.jsonl` — 699 rows, schema per row:
```json
{"id": "arxiv-00001", "text": "...",
 "entities": [{"title": "X", "type": "CONCEPT", "description": "..."}],
 "relationships": [{"source": "A", "target": "B", "description": "...", "strength": 8.0}]}
```
`type` ∈ {PERSON, ORGANIZATION, GEO, EVENT, CONCEPT}; `strength` 0–10.

Student predictions (produce per model) — a JSONL aligned on the same `id`:
```json
{"id": "arxiv-00001", "entities": [{"title", "type", "description"}, ...],
 "relationships": [{"source", "target", "description", "strength"}, ...]}
```
Example: `outputs/full_eval/predictions_full_norm.jsonl` (reference 0.6B) and
`outputs/capacity_eval/qwen3-0.6b/predictions.jsonl` (capacity run, chat protocol).

Fixed sample + passages for inference (gitignored, archived):
`outputs/referent_eval/sample_200.json` (200 ids) and
`outputs/referent_eval/sample_200_passages.jsonl` (id + text).

## Reference baseline (do not regenerate)

| artifact | meaning |
|---|---|
| `outputs/referent_eval/pairings_think_699.jsonl` | reference model full-699 thinking pairing |
| `outputs/referent_eval/report_think.json` | full-699 headline: entity recall 0.541 / precision 0.707 / F1 0.613 |
| `outputs/referent_eval/report_ref200.json` | same reference restricted to sample_200 (recall 0.534 / precision 0.723) — compare new models to THIS |

## End-to-end: score a new model

Prereqs: `DASHSCOPE_API_KEY` set (`.env`). Steps 4–5 cost ≈ 200 thinking calls
(~9–10 min wall at 16 workers, ~1M tokens ≈ reference think pricing).

```powershell
# 0) export the 200 sample passages for inference (id+text)
$ids = (Get-Content outputs/referent_eval/sample_200.json -Raw | ConvertFrom-Json).ids
python eval/referent_sample.py --n 200 --seed 20260906 --passages-out outputs/referent_eval/sample_200_passages.jsonl

# 1) infer the sample with the candidate model -> <tag>/predictions.jsonl
#    (one {"id", entities[], relationships[]} per line, same ids)
#    Use the pipeline's inference kernels, NOT ad-hoc code:
#      reproduction/full: finetune/kaggle/README_inference.md  (kernel_inferfull)
#      capacity line:     finetune/kaggle/kernel_capinfer/README.md
#    Protocol MUST match the model's training format (chat vs plain, issue 08):
#    chat-suppressed or plain-runaway numbers are not comparable.

# 2) referent pairing (thinking LLM) — resumes existing rows in --out
python eval/referent_pair.py --mode llm --workers 16 --no-passage --desc-len 300 --thinking `
  --student outputs/capacity_eval/<tag>/predictions.jsonl `
  --out outputs/referent_eval/pairings_<tag>.jsonl --ids $ids

# 3) main report (finding / schema / grounding / numeric + bootstrap)
python eval/referent_eval.py --pairs outputs/referent_eval/pairings_<tag>.jsonl `
  --student outputs/capacity_eval/<tag>/predictions.jsonl `
  --out outputs/referent_eval/report_<tag>.json --ids $ids

# 4) consumer axes (API-free)
python eval/referent_dup.py --student outputs/capacity_eval/<tag>/predictions.jsonl `
  --pairs outputs/referent_eval/pairings_<tag>.jsonl --out outputs/referent_eval/dup_<tag>.json --ids $ids

$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1   # local cached MiniLM, no hub
python eval/referent_reach.py --student outputs/capacity_eval/<tag>/predictions.jsonl `
  --pairs outputs/referent_eval/pairings_<tag>.jsonl --out outputs/referent_eval/reach_<tag>.json

# 5) raw-output redundancy / invalid-item diagnostic (no pairing needed)
python eval/referent_redundancy.py --student outputs/capacity_eval/<tag>/predictions.jsonl `
  --out outputs/capacity_eval/redundancy_<tag>.json --ids $ids
```

Compare `report_<tag>.json` side by side with `report_ref200.json`. Keep the
student file you feed to steps 2–5 byte-identical to what inference produced
(never re-parse/dedup between steps).

Notes for large jobs: pairing resumes (rows with `status: ok` are skipped) — safe
to Ctrl-C and re-run. Rate ≈ 0.5 passages/s/… at 16 workers (thinking);
nothink `--variant default|hard|strict` is ~10x faster but **not** production
quality (ADR-0008).

## Script reference

| script | role | key flags |
|---|---|---|
| `referent_sample.py` | draw/archive the fixed sample + export passages | `--n --seed --passages-out` |
| `referent_pair.py` | build referent pairings | `--mode title\|llm`, `--thinking`, `--no-passage`, `--desc-len`, `--variant`, `--ids`, `--student` |
| `referent_eval.py` | main report + bootstrap CI | `--pairs --student --ids --out` |
| `referent_dup.py` | duplication / granularity axis | `--pairs --student --ids` |
| `referent_reach.py` | reachability retrieval sim | `--pairs --student --ids --fixture` (fixture gate) |
| `referent_redundancy.py` | raw-output redundancy / invalid items | `--student --ids --out` |
| `referent_cost.py` | scheme-vs-think experiment table (dev only) | `--scheme NAME=PATH --agree NAME=A,B` |
| `referent_verify.py`, `referent_bench.py` | rejected exploration, read-only | — |

## Axis semantics and caveats

- **Entity / edge finding** (macro + micro, referent-level). Micro = pooled over
  passages; macro = mean of per-passage. Edge recall uses teacher-side matched
  edges, precision student-side; the consumer observes few teacher edges, so
  edge recall sits near a floor — read trend, not level.
- **Schema**: type ∈ 5 vocab, non-empty description, strength in 0–10, endpoints
  resolve inside the student entity list. `placeholder_titles` counts blank / `-`.
- **Grounding / hallucination**: lexical two-anchor (substring or ≥0.6 token
  overlap) on student *unmatched* entities. No LLM claim-checking (ADR-0007).
- **Numeric**: share of passage numbers surviving into descriptions.
- **Dup / granularity** (`referent_dup`): combined-node merges, exact
  normalized-title duplicates, observable referent splits. The 1:1 referent
  matcher hides splits, so the split rate is a lower bound; synonym-level
  redundancy shows up as entity-precision cost, not here. Blank/`-` titles are
  skipped (a blank-title normalize bug inflated dup in early capacity runs —
  fixed; chat 0.6B true dup = 1.65%, not 5.28%).
- **Reachability** (`referent_reach`): MiniLM retrieval sim — gold-description
  query into the deduped student-description index; conditional (of found) and
  unconditional. Only fielded after the sabotage fixture PASSes.
- **Redundancy** (`referent_redundancy`): raw-output volume, exact dup, empty /
  no-letter titles, OOV types, dangling / self / duplicate edges. Use to
  diagnose protocol or decode issues before believing finding axes (capacity
  chat-vs-plain: chat suppresses, plain over-generates — see issue 08).

## Known decisions (do not silently reverse)

- Production judge = thinking pairing (`--thinking`); title floor only for sanity.
- Fixed sample `sample_200.json` for all future models (paired comparisons).
- Description claim-truth is out of scope; no LLM judge in the automated loop.
- Capacity line protocol rule: eval must match the model's **training format**
  (chat vs plain). Cross-protocol referent numbers mix capability + protocol
  penalty; call it out explicitly (issue 08, ADR-0006 pending fold-in).
