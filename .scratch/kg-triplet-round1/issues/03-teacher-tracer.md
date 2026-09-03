# 03 — Teacher labeling tracer

**What to build:** The full teacher-labeling machinery — teacher prompt (relation dictionary,
entity-naming rules, exhaustiveness demand, fixed weight anchors 0.2/0.5/0.8, empty-array rule,
in-chunk grounding), DashScope client (OpenAI-compatible, parallel, retry/backoff, per-run
budget cap, single-call context < 32k) — proven on 100 stratified pilot passages before the
full run spends real money. Includes semantic validation (weight anchors + grounding), a
static human-review report, and verdicts persisted to SQLite with a schema reusable by the
ticket 11 gold audit.

**Blocked by:** 01, 02

**Status:** ready-for-agent

- [ ] Teacher prompt five sections complete; empty-array rule lives inside exhaustiveness
      demand; output JSON schema stated
- [ ] DashScope client: OpenAI-compatible endpoint, injectable transport, network 5xx/429
      backoff retries (1s/2s/4s), schema-invalid responses retried twice then recorded
      `failed+reason` — never silently converted to `[]`; per-run budget hard stop at
      ≤150 total calls (incl. retries); single-call context asserted < 32k
- [ ] Parallel labeling (default 4 workers) with a thread-safe budget; results written in
      deterministic sample order; checkpoint/resume re-runs nothing already labeled
- [ ] 100 passages drawn stratified (Wikipedia/arXiv 60/40 + length) with a fixed seed;
      pilot run in stages (1 → 10 → 100) reusing the same sampled order
- [ ] Schema validation via the contract validator; semantic checks flag weight ∉ {0.2,0.5,0.8}
      and ungrounded entities; violations reported, not silently dropped
- [ ] Hard negatives: teacher `[]` recorded as hard_negative; observed rate reported, no
      target set
- [ ] Static review report `outputs/03_pilot_review.html` (stats bar, per-case cards, flagged
      + ~20 random sample, localStorage, export `verdicts.jsonl`); verdicts ingested into
      SQLite (`outputs/review.sqlite3`) with a schema generalizable to the ticket 11 gold audit
- [ ] Injected-failure tests prove retry and budget-stop behavior; pytest green

## Teacher-prompt convergence (blind-rubric loop, final)

Iteration loop restructured (files as info bus + blind subagent verifier) per review:
results → `outputs/prompt_dev/iterN.jsonl`, rubric+gold → `outputs/prompt_dev/RUBRIC.md`,
held-out → `outputs/prompt_dev/heldout_iterN.jsonl`.

**Final prompt** = the compact imperative `TEACHER_SECTIONS_PROSE` in
`kg_contract/prompts.py` (iter7). Systematic failures fixed across iters 1–7:
- part_of licensing: institutional + structural (component of a structure, constituent
  object of a theory) + ecological (ecoregion → its realm); NOT treaties/participation;
  spatial placement → located_in
- list handling: one triplet per member, connect members to the statement's real subject
  (never an aggregate/category node), split derivative-of-base compounds, never merge
  distinct names into one node
- anti-trivia SKIP is surgical: skips taxonomic/example enumerations, keeps list facts
  that assert a per-member relation (e.g. plants used_for herbs/medicines)
- provenance relation preferred over usage when both verbs apply to one artifact
- derivation chains kept as separate links (X based_on Y, Y extends Z; no collapse)
- self-check: re-read titles against text before finalizing (kills spacing/hyphen drift)
- emit each fact once even if restated

**Verdicts (blind subagent per RUBRIC.md):** iter1 easy PASS / medium FAIL / hard FAIL;
iter4 hard PASS; iter5 easy+medium PASS; iter7 easy+medium PASS, hard FAIL (2 wetlands
dropped); held-out `wikipedia-02266` (Audi, never tuned) core 11/12 gold facts in both
runs, 0 violations.

**Converged — residual is extractor noise, accepted by owner.** Runs cover 24–26/26 gold
facts; the miss is 1–2 *random* facts per run (different member each run: SHOT, realm,
Buna-Shkodër/Karavasta, Audi-name). No run reaches all-PASS because of this stochastic
floor; further prompt edits would chase noise. Decision: accept single-run teacher output;
if strict completeness is later required for training data, add multi-draw union (k=2 +
dedup) to the teacher pipeline (~2× API cost) — flagged, not implemented.
