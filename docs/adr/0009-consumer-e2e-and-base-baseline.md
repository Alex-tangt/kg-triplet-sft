# Consumer-impact e2e via LightRAG + base baseline (resume evidence)

The open-line consumer claims ("output feeds a graph-RAG QA consumer",
ADR-0003/0007) were only approximated by API-free axes (reachability, dup).
Owner approved two evidence-gathering legs before the resume is written from the
assets: a base-model lower bound (Leg A), and a real graph-RAG terminal test
(Leg B). Both run on the fixed-200 sample so they are comparable to every
existing report. Leg B's consumer pivoted from MS GraphRAG to **LightRAG**
(2026-09-11) — rationale and superseded plan in
`docs/diary/2026-09-11-consumer-pivot-graphrag-to-lightrag.md`.

## Leg A — Base baseline (SFT-contribution bound)

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

## Leg B — LightRAG consumer-impact e2e (custom KG)

Decided: measure extraction→consumer transmission with **three graphs on one
topical slice drawn from the fixed-200 sample** — gold labels (control),
0.6B-masked, 4B-clean. Gold + both student prediction sets already exist for
every sample-200 passage, so **no new GPU inference**. LightRAG is fed through
`LightRAG.insert_custom_kg`, its custom-KG entry point (the analogue of
GraphRAG's BYOG): the caller supplies chunks + entities + relationships, so
LightRAG's own extractor never runs and the only variable is OUR extraction.
This is the same rationale that selected BYOG before — stock indexing would
re-extract with LightRAG's LLM and never test the component we replaced — now
expressed against LightRAG's actual API.

**Why LightRAG over MS GraphRAG.** The pivot is recorded in the diary; the
decision-relevant facts are: (1) LightRAG is a single installable package with a
documented custom-KG write path, versus GraphRAG's parquet-BYOG workflow stack;
(2) it covers two of the three consumer claims the open line was built to
exercise — entity-description retrieval (`local`) and keyword-driven
cross-document relation retrieval (`global`) — but **not** community/regional
summarization: a source audit of `lightrag-hku==1.5.7` finds no
Leiden/community/community-report code, so that GraphRAG capability is explicitly
out of scope for this consumer (a recorded claim boundary, not an oversight);
(3) it exposes the mode set (`local`, `global`, `hybrid`, `naive`, `mix`) that
makes a retrieval-vs-text control arm natural.

### Consumer configuration

- **Modes.** Headline = `local` / `global` / `hybrid` (all graph-dependent).
  `naive` is the extraction-independent **text-sufficiency oracle** — it answers
  from raw chunks, so a graph layer only earns its keep where it beats `naive`.
  `mix` optional. `only_need_context=True` gives the T1 retrieval layer for free.
- **Roles.** KEYWORD extraction `qwen-flash` (non-thinking); QUERY `qwen-max`;
  answer/entailment judge `qwen3.7-flash` thinking (separated from QUERY so the
  judge is never the same call it grades). Embeddings are local **bge-m3**
  (1024-d) supplied as an in-process `EmbeddingFunc` — no embedding server.
- **Reranker.** Local `bge-reranker-v2-m3`; run as **two arms (on / off) × all
  four modes**, so a reranker effect is never conflated with a mode effect.
  Pre-flight must confirm both local models load at the expected dimension.
- **Fixed QueryParam (not tuned).** `top_k=60`, `chunk_top_k=20`,
  `max_entity_tokens=6000`, `max_relation_tokens=8000`, `max_total_tokens=30000`;
  strict-grounding instruction via `user_prompt_prefix`.
- **Isolation.** Three graphs = three `working_dir`s, three `LightRAG` instances
  in one process; every other config field byte-identical across the three.

### Graph construction (LightRAG is a writer, not a builder)

`insert_custom_kg` normalizes identifiers and writes, but does **no merging**: it
keeps the *last* declaration per entity name and the *last* per unordered
endpoint pair (last-wins overwrite), raises on self-loops, and materializes any
endpoint missing from the entity list as an `UNKNOWN` node. So the constructor
must pre-merge, and the mapping is exact:

- **Entities**: merge by normalized title → one row
  `{entity_name: TITLE, entity_type: type-mode, description: " | ".join(...),
  source_id: canonical passage id}`.
- **Relationships**: merge by **unordered endpoint pair** (LightRAG stores edges
  undirected), keep a canonical direction, join descriptions with `" | "`,
  aggregate strength; row `{src_id, tgt_id, description, keywords: "",
  weight: 1.0 + mean(strength)/10, source_id: canonical passage id}`.
- **`keywords` must be an explicit `""`.** LightRAG indexes the relationship row
  by direct subscript (`relationship_data["keywords"]`), so a missing key is a
  `KeyError`, not a default. The empty value is also a recorded **schema gap**:
  our extraction has no keyword field, so `global` mode reaches relations through
  description text only.
- **Weight floor.** LightRAG floors a relation's weight at its number of distinct
  real evidence sources. We supply a single `source_id` per relation (one
  canonical passage), so the floor is `1.0`; `1.0 + mean(strength)/10` carries
  the teacher's 0-10 strength signal on top.
- **Chunks** are byte-identical across the three layers (same text, same ids) so
  `naive` is a true control. Self-loops are dropped before insert (LightRAG
  raises); dangling endpoints are **not** dropped — they become `UNKNOWN` nodes
  and are counted as a T0 diagnostic.
- **No repair.** No disambiguation, no fuzzy merging, no chain-filling: the
  constructor transmits the extraction as-is, because that is what the claim is
  about. Normalization beyond the title-key merge would launder the signal.

### Metrics funnel (T0 → T1 → T2)

- **T0 — graph formation (offline, zero cost, hard gate).** Per layer: linked
  entities that survive into the graph, fragmentation, conflation, `UNKNOWN`
  nodes, empty passages. If T0 is flat across layers there is nothing downstream
  to explain, and the whole leg stops.
- **T1 — retrieval (`only_need_context`, no judge).** Recall of the
  answer-bearing graph/chunk elements for each question, per question × mode ×
  layer. This is the transmission axis, measured before any generation noise.
- **T2 — answer (strict grounding + entailment).** Generated answers graded for
  grounding in the retrieved context and entailment of the reference answer.
- **Questions.** Four types mined mechanically from the gold graph — A
  intersection, B bridge, C neighborhood aggregation, D global theme — each
  double-gated (derivable from gold **and** text-sufficient for `naive`) and
  **pre-registered**; ~2× candidates are mined to land ~10-15 final items.
- **Outcomes, pre-registered.** confirmed / partial / null / non-monotone. `null`
  is a legitimate result: T0 then localizes it (extraction never made it into the
  graph) versus T2 localizing it (it did, but did not help).

### Slice

A ~18-passage multi-cluster slice of the fixed-200 sample, chosen for concrete
shared entities (document frequency ≤ 4, appearing in ≥ 2 passages of the
slice) rather than topical adjacency: `space` (8) + `kurosawa` (4) +
`afroasiatic` (3) + `andorra` (3); `egypt` is an optional appendix and `ankara`
is dropped. The two `Apollo` sub-themes (mythology vs the Apollo program) are
kept as **separate** clusters because the namesake collision would otherwise
manufacture spurious edges. The slice composition and its per-layer referent-F1
spread are recorded as diagnostics, not used to select a favourable subset.
Official corpora (A Christmas Carol / UltraDomain) are deferred to optional
Round-2, not run now.

### Budget guardrails

Indexing spends **zero** LLM tokens (`insert_custom_kg` is LLM-free). Spend is
only KEYWORD + QUERY at query time plus the judge. Owner brake: **stop and
report** before query+judge tokens exceed ~3M or ~¥30; the T0 gate fires before
any query spend. Every DashScope step is reported before the next runs.

### Rejected alternatives

(a) Official corpora (A Christmas Carol / UltraDomain) now — more realistic but
needs new extraction/inference and adds no claim the sample-200 slice lacks;
deferred to optional Round-2. (b) Merging the 2575-row training set into the
main experiment — ~93% overlap with the fixed-200 slice dilutes the inter-layer
signal and lets the student's memorized training chunks contaminate the reading.
(c) Any new GPU inference — the three prediction sets already exist. (d)
Multi-hop success rate as a headline metric — LightRAG has no hop operator, so
the number would be an artifact of prompt phrasing, not a consumer property; the
B-bridge question type captures the underlying cross-passage reach instead.

### Honest limitations

Case-study scale (~22 passages, 16 questions) has no statistical power and
the read is directional. The empty `keywords` field means `global` leans on
descriptions. Student-garbled titles are simultaneously the transmission
mechanism and a measurement-uncertainty source, which is why T0 is a gate rather
than a footnote.

### Outcome (2026-09-11)

Ran on the 22-passage / 5-cluster slice with 16 pre-registered natural questions.
**T0 differentiated strongly** (gate GO): the 0.6B graph shatters into 381
components (largest 26, 351 isolated nodes) vs 4B (largest 72) vs gold (largest
136), so extraction quality clearly reaches graph structure. **T1 and T2 then
saturate**: `naive` (raw chunks, no graph) already recalls every answer-bearing
element (1.000), and the graph modes span 0.85-1.0 with no layer ordering; T2
correct-rate is 0.94-1.0 for all three layers and differences are question-bound
(the Andorra institutions link question is partial everywhere), not layer-bound.
Spend 2.24M query+judge tokens, under the brake. Reading: (a) feasibility
confirmed; (b) transmission is **structural but not consumer-level in this case
study** — with 22 passages and `top_k=60` the raw text already carries the
answer, so the graph is not load-bearing. A discriminating test needs a corpus
large enough that retrieval is selective (or a much smaller `top_k`); that is the
honest follow-up. Full report: `outputs/lightrag_e2e/RESULT.md`.

Artifacts land under `outputs/lightrag_e2e/`; scripts under `eval/`
(`eval/lightrag_e2e.py`, `eval/lightrag_slice.py`, `eval/lightrag_questions.py`);
work on branch `exp/lightrag-e2e`; results feed the resume project column, not a
Round-1 issue.
