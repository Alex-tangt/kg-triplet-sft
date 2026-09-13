# LightRAG consumer-impact e2e — result (ADR-0009 Leg B)

Date: 2026-09-11 · Branch: `exp/lightrag-e2e` · Consumer: `lightrag-hku==1.5.7`

## Setup

- **Slice**: 22 passages, 5 clusters (space 8 / kurosawa 4 / afroasiatic 3 /
  andorra 3 / egypt 4) drawn from the fixed-200 sample; clusters chosen for
  concrete shared entities, not topical adjacency.
- **Layers**: gold (control) / 0.6B-masked / 4B-clean. No new GPU inference.
- **Consumer**: LightRAG `insert_custom_kg` (custom KG, its own extractor never
  runs). Chunks are byte-identical across the three graphs.
- **Embeddings**: local `BAAI/bge-m3` (1024-d, in-process). Reranker:
  `BAAI/bge-reranker-v2-m3`, both arms. KEYWORD `qwen-flash`, QUERY `qwen-max`,
  judge `qwen3.7-flash` thinking. QueryParam fixed (`top_k=60`, `chunk_top_k=20`,
  `max_*_tokens` as in ADR-0009).
- **Questions**: 16 natural-language information needs (single 6 / link 4 /
  agg 4 / theme 2), pre-registered in `eval/lightrag_questions.py` before any run.

## T0 — graph formation (offline, hard gate): GO, strongly differentiated

| layer | nodes | edges | entity F1 | edge F1 | multi-passage | components | largest | isolated |
|---|---|---|---|---|---|---|---|---|
| gold | 555 | 506 | 1.000 | 1.000 | 47 | 140 | 136 | 103 |
| 0.6b | 510 | 169 | 0.393 | 0.077 | 20 | 381 | **26** | **351** |
| 4b | 457 | 320 | 0.569 | 0.279 | 23 | 184 | **72** | 152 |

Extraction quality **does** transmit to graph structure: the 0.6b graph
shatters into 381 components (largest 26, 351 isolated nodes), while 4b keeps a
larger connected core (72) and gold the largest (136). This is the cleanest
signal in the study.

## T1 — retrieval (`only_need_context`, rerank-off): saturated

Mean recall of answer-bearing elements in the retrieved context (16 questions):

| layer | local | global | hybrid | naive |
|---|---|---|---|---|
| gold | 0.812 | 1.000 | 1.000 | 1.000 |
| 0.6b | 1.000 | 0.855 | 1.000 | 1.000 |
| 4b | 0.938 | 0.938 | 0.938 | 1.000 |

`naive` (raw chunks, no graph) already scores **1.000** for every layer, and the
graph modes sit at 0.85-1.0 with no layer ordering. On a 22-passage corpus the
retrieval budget (`top_k=60` entities, `chunk_top_k=20` of 22 chunks) covers the
answer entities regardless of graph quality, so **T1 cannot discriminate the
layers**. The rerank arm changes nothing (0.6b/4b identical; gold differs only on
local 0.812→1.000).

## T2 — answer quality (generation + thinking judge): saturated

Correct-answer rate (192 generations; spend 2.24M query+judge tokens, under the
3M brake):

| layer | local | global | hybrid | naive |
|---|---|---|---|---|
| gold | 0.938 | 0.938 | 1.000 | 0.875 |
| 0.6b | 0.938 | 0.938 | 1.000 | 0.938 |
| 4b | 0.938 | 0.938 | 0.938 | 0.875 |

The three layers are **statistically indistinguishable**. Failures are
question-bound, not layer-bound: the Andorra "institutions" link question (A3)
is partial for nearly every layer/mode, and the Hamito-Semitic single question
(F1) is the only incorrect. By type: agg 1.000, theme 1.000, single 0.944,
link 0.833.

## Reading

- **(a) Feasibility — confirmed.** The open-extraction format feeds a real
  graph-RAG consumer through LightRAG's custom-KG path with zero LLM indexing;
  all three graphs build and answer.
- **(b) Transmission — structural yes, consumer-level null (this case study).**
  Extraction quality clearly reaches the graph (T0), but on this small focused
  corpus it does **not** reach retrieval or answer quality, because both
  saturate: the raw text already contains the answer, and a generous retrieval
  budget surfaces it for every layer.
- **Where the bottleneck is.** The consumer cannot benefit from graph quality it
  does not need — with 22 passages and `top_k=60`, the graph is not load-bearing.
  A discriminating test would need a corpus large enough that retrieval is
  selective (or a much smaller `top_k`), so the graph must carry the connection
  rather than the raw text. That is the honest follow-up, not a Round-1 claim.

## Honest limitations

- Case-study scale (~22 passages, 16 questions); no statistical power, the read
  is directional. The null is a real result, not a failed run.
- T1's element match uses normalized entity titles; `naive` matching raw
  natural-case text only works because matching is case-insensitive.
- Empty relation `keywords` (schema gap) means `global` leans on descriptions.
- The rerank arm was run on T1 only (proved inert there) and omitted from T2 for
  budget; T2 is rerank-off.

## Artifacts

- `outputs/lightrag_e2e/slice.json`, `questions.json`, `{gold,0.6b,4b}/custom_kg.json`
- `outputs/lightrag_e2e/t0_report.json`, `t1_results.json`, `t2_results.json`
- Scripts: `eval/lightrag_slice.py`, `eval/lightrag_questions.py`, `eval/lightrag_e2e.py`
- Deps: `requirements-lightrag.txt`
