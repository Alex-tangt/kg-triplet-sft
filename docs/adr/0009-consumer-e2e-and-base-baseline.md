# Consumer-impact e2e via GraphRAG BYOG + base baseline (resume evidence)

The open-line consumer claims ("output feeds an MS GraphRAG graph-QA consumer",
ADR-0003/0007) were only approximated by API-free axes (reachability, dup).
Owner approved two evidence-gathering legs before the resume is written from the
assets: a base-model lower bound, and a real ms-graphrag terminal test. Both run
on the fixed-200 sample so they are comparable to every existing report.

## Base baseline (SFT-contribution bound)

Decided: evaluate **untrained main-release Qwen3-0.6B** (the exact weights a
training run starts from — NOT the `-base` variant; issue 08 established training
loaded the chat-capable main release) through the same fixed-200 referent eval,
same chat/no-think + schema-prefix protocol and thinking judge as the capacity
line. Reference: `outputs/referent_eval/report_ref200.json` (0.6B masked F1
0.607). The base-vs-masked delta is the resume's "what LoRA added" number.
Run on Kaggle T4 (kernel_capinfer code path, no adapter), judge via DashScope
(~1M tokens, few CNY); expect low schema compliance → tolerant parser empties,
report must state that as the base's own result, not a pipeline failure.

**Outcome (2026-09-08).** Ran via `kernel_capbase` (Kaggle T4, chat/no-think +
schema prefix, batch 4, NO LoRA merge; generator support =
`make_capinfer.py --no-adapter`, verified neutral for the adapter path). Base
micro entity R/P/F1 = 0.159 / 0.854 / 0.268, macro 0.162 / 0.849 / 0.253, edge
F1 0.024, schema-valid 0.666, empty rows 1, parse rate 0.99 (925 entities).
Versus 0.6B masked (micro 0.536 / 0.699 / 0.607): SFT adds +0.38 recall / +0.34
F1 for −0.155 precision — the ~0.60 F1 plateau is learned, not base leakage; the
base's failure mode is omission (its low hallucination 0.031 is a sparse-output
artifact). Judge spend 364k tokens. Artifacts: `outputs/base_eval/` (RESULT.md,
PREP.md, predictions), `outputs/referent_eval/{pairings,report}_qwen3-0.6b-base.*`.

## GraphRAG BYOG consumer-impact e2e

Decided: measure extraction→consumer transmission with **three graphs on one
topical ~30-passage slice drawn from the fixed-200 sample**: gold labels
(control), 0.6B-masked, 4B-clean. Gold + both student prediction sets already
exist for every sample-200 passage → **no new GPU inference**. ms-graphrag runs
in BYOG mode (`workflows: [create_communities, create_community_reports]`, plus
`generate_text_embeddings` for Local search), relationships' `strength` (0-10)
mapped to `weight` (Leiden), text_unit ids = passage ids. Same hand-authored
Local + Global questions per graph; verdicts hit/partial/miss + graph-structure
diagnostics (nodes/edges/communities/empty rows). Judging via DashScope.

Rationale for BYOG over stock indexing: stock would re-extract with graphrag's
own LLM and never test OUR extraction; BYOG is the only mode that isolates the
component we replaced (Phase-3 extraction) and lets gold-vs-student read as an
impact curve. Rationale for staying inside sample_200: predictions are already
produced for all three layers there; widening to train chunks (more realistic)
was considered and rejected for now because it forces new Kaggle inference with
no added claim value for the resume numbers.

Cost guardrails (owner): stop-and-report before exceeding ~15-20 CNY total or
~1M judge+index tokens for this leg; indexer step estimates community count
first and aborts past a token pre-estimate. Every DashScope spend step is
reported before the next runs. Artifacts land under `outputs/graphrag_e2e/` and
`outputs/base_eval/`, scripts under `eval/` (`eval/graphrag_e2e.py`,
`eval/referent_base.py`-style); results feed the resume project column, not a
Round-1 issue.

Rejected alternatives worth remembering: (a) GraphRAG full-article slices over
train+test (realism) — needs 4B inference on ~40 train chunks and adds no claim
the 200-sample slice lacks; (b) Local-search-only verdict — drops the
cross-document Global synthesis that is GraphRAG's flagship; (c) running the
stock extractor on our slice to compare against our models — conflates prompt
and model and costs extraction LLM again.
