# Consumer-side eval axes + no-thinking cost experiment verdict

Two follow-ups to ADR-0007 were settled on the reference model.

## Consumer-side axes (ADR-0007's open line, GraphRAG consumer)

**Reachability (retrieval simulation, local MiniLM, no API).** LightRAG (like the
MS GraphRAG consumer it replaced, ADR-0009) routes a query to entities by embedding against indexed entity descriptions; we approximate
"can the user reach a found entity" by querying each gold (teacher) description
against a deduped index of ALL student entity descriptions and asking whether the
referent-matched student node is in the top-k. A planted fixture (correct vs
empty/wrong-topic descriptions) must discriminate before the axis is fielded
(it does: recall@1 1.0 vs 0.5/0.6 sabotaged). On the fixed 200-passage sample the
reference model scores:

    conditional (on the entity being found): recall@1 0.494, @5 0.741, @10 0.804
    unconditional (end-to-end, incl. missing the entity): recall@1 0.264, @5 0.395

So roughly half the found entities' descriptions would not surface the entity to a
teacher-phrased query at top-1; reachability is a real, reportable axis separate
from finding.

**Duplicate / entity-granularity (derived, API-free).** Student outputs on 699:
combined-node merges fold 0.4% of gold entities (58 passages); exact normalized-
title duplicates 0.33% of student entities; observable referent-split is 0 (a
lower bound — the 1:1 referent matcher hides splits). A title-overlap heuristic
for near-synonym redundancy was tried and rejected as too noisy on CONCEPT
clusters; such redundancy shows up as entity-precision cost instead.

## No-thinking cost experiment (ADR-0007: can a cheap judge replace thinking?)

On the ids100 set (20 consistent + 80 disputed, think = gold, plus the 12
known-answer probe) we ran no-thinking single-vote variants and two-vote
agree-only intersections of a hardened rubric (rules + worked exemplars) with a
strict (ambiguous ⇒ NOT SAME) rubric. Against the thinking gold:

    scheme              probe  passage-diff  pair FP  pair FN  guard(20)
    nothink default       9/12       80         307       79       0
    nothink hard          10/12      81         252      108       3
    nothink strict        10/12      81         208      125       2
    agree(default,strict)  7/12      80         165      142       2
    agree(hard,strict)     7/12      79         143      157       3

Cost on the same 100 (total tokens / wall): thinking 480k (~5 min), no-thinking
default 127k, hard 164k (~10-40s). Thinking therefore costs ~2.9-3.8x the
tokens and ~7-25x the wall time.

**Verdict: no-thinking does NOT reach thinking quality, and agree-only is
worse than a single vote.** The hardened/strict prompts cure two of the three
over-merge probe errors but then reject a legitimate garble rescue that both
default-nothink and thinking accept; one hard over-merge (arxiv-00187) survives
every no-thinking prompt. Without reasoning the judge cannot simultaneously
rescue garbled titles and refuse co-occurrence/diff-class merges on these cases.

**Escalation-on-disagreement was then simulated** (two no-thinking votes; the
pairs/passages where they split decided by the thinking gold as oracle): the two
votes share a systematic over-merge bias, so the recoverable errors sit in the
AGREED region, not the disagreement region. def+strict hybrid recovers only
~9/80 passage diffs (probe 11/12, the shared arxiv-00187 over-merge remains)
while needing thinking on 53 of the ~80 fuzzy passages (~2/3 of full-thinking
cost); hard+strict is no better (63 passages, probe 10/12). Not worth it.

Decision: keep thinking-mode pairing as the production judge (ADR-0007). No
cheap no-thinking scheme (single hardened, agree-only, or disagreement
escalation) reaches thinking quality at a meaningful saving; the measured
fallback of a hardened single vote trades ~2 probe errors for ~2.9-3.8x token
savings and is NOT adopted.
