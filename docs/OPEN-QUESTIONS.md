# Open questions (do not treat as decided)

Statuses: `decided`, `unresolved`, `withdrawn`. Anything `unresolved` is **not** a
default — do not build on it.

## Measurement layer (dated-by `eval/referent_eval.py`)

- `decided` — **Q1**: the eval carries two axes, fidelity (teacher as oracle) and
  validity (independent oracle), reported separately; no single score decides.
- `decided` — **Q2**: recall/omission is measured by multi-model candidate
  generation → dedupe → judge, with capture-recapture to estimate the ungenerated
  remainder. Generation on a stratified subsample; judging can be full-corpus.
- `unresolved` — **Q3 (alignment)**: keep 1:1 referent pairing as headline, or add a
  granularity bypass (merge/split not counted in P/R), or move to many-to-many? The
  1:1 matcher structurally counts granularity differences as omission/hallucination.
- `unresolved` — **Q4**: tolerance of the proxies (lexical grounding; directed edge
  match vs the consumer's undirected edges).
- `unresolved` — edge metric is topology-only; relationship `description` and
  `strength` are never scored. Whether/how to score them is undecided.

## L1 reading standard (evidence: `docs/evidence/l1-fit.md`)

- `unresolved` — **S1/S2**: what L1 decides, and the anchor for "well-fit" (Δ vs
  base, absolute, cross-model 4B, or a teacher-consistency floor).
- `unresolved` — **S3**: how to make cross-field NLL comparable (Δ per field vs
  absolute; copy/entropy baselines).
- `unresolved` — **S4**: overfit threshold; **S5**: whether low NLL on copy-like
  fields counts as capability.
- `withdrawn` — "underfit is excluded; the edge gap is exposure bias" (stated before
  the standard existed; see l1-fit.md).

## Training / model line

- `unresolved` — task decomposition (separate entity vs relation extraction; one
  adapter + two passes, two adapters, two models, or relation-as-pair-classification).
- `unresolved` — capacity vs optimization vs target: needs a one-variable ablation
  (masking was confounded with eff-batch/dropout/warmup/optim/template/packing) and a
  capacity axis beyond base size (LoRA r=32).
- `unresolved` — teacher protocol: thinking vs no-think teacher; r1 vs r1+r2 target.

## Data / gold

- `unresolved` — issue #01 three-layer audit (entity / edge / pairing); rubric and
  human-anchor design drafted in discussion, not frozen.

## Decide-model (Jev)

- `unresolved` — substituting a decision model for the LLM judge in eval. Blocked
  practically: `TYPESAFE_API_KEY` is user-level but not visible to the running
  process; `typesafe-sdk` not installed.

## Process

- `decided` — issue tracker moves to GitHub (public `Alex-tangt/kg-triplet-sft`).
- `decided` — re-org focus: (1) attribution analysis, (2) Round-2 improvement.
- `unresolved` — commit strategy for the current uncommitted work.
