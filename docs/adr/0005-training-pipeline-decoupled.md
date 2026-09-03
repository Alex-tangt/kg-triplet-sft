# Round-1 training pipeline is decoupled from the mohar07 reproduction config

Status: Accepted (2026-09-03). Updated (2026-09-03) with the post-tracer
pipeline verdict: LLaMA-Factory on a clean install is the canonical stack
(resolves the silent exit; see issue 06 2026-09-03 llamafactory re-check) and
fp16 is the precision (T4 = Turing, no bf16 tensor cores). See "Post-tracer
verdict" note at the end.
Supersedes, for the GraphRAG line, the
"LLaMA-Factory + fixed hyperparameters + bf16 LoRA on Kaggle T4" training
constraints inherited from the mohar07 reproduction plan.

## Context

ADR-0002 locked evaluation to **our own test split only** and ADR-0003 changed
the data-generation task from mohar07's 20 typed relations to open
GraphRAG-style extraction (JSON entities/relationships + 0-10 strength). Those
two ADRs already cut the numeric-comparability link to the mohar07 baseline
(composite 0.6583): the baseline remains only a pipeline-validation gate.

Yet the training-side locks from the reproduction plan were still being carried
as if the numbers were comparable: LLaMA-Factory as the mandated launcher,
bs2 × grad-accum 4, bf16 LoRA, QLoRA restricted to Round-2's 8B. During the
tracer (issue 06) we discovered these locks are now a liability, not a
guarantee:

- LLaMA-Factory's CLI **silently exits 0 mid-Trainer-init** in the Kaggle image
  (reproduced under both transformers 5.0 and 4.57; no traceback, no training).
  A hand-written `transformers.Trainer` + `peft` pipeline trained the same data
  to completion instead (60/60 steps, loss 0.77 -> 0.61).
- The locked config (bs2 × grad-accum 4, cutoff 6144) **does not fit a 16GB T4**
  at tracer scale: bs2 OOM'd (13.8 GB activations). Working recipe is bs1 ×
  grad-accum 8 + gradient checkpointing + `enable_input_require_grads`.
- A full 2575-row run at the locked config (6144 / 5 epochs / bf16) is
  estimated at **27-40 T4-hours**, more than the weekly GPU quota (~30h,
  ~20h remaining). Efficiency is now a binding constraint, and several levers
  were never exercised (T4 exposes **2 GPUs**; T4 is Turing = no bf16 tensor
  cores, so fp16 may be the native fast path; quantization could buy batch
  headroom).

## Decision

1. **The GraphRAG line no longer inherits the reproduction line's training
   config.** The reproduction record (composite 0.6583, LLaMA-Factory configs)
   stays as documentation but is not the operative target for Round-1 runs.
2. **The only training constraint that survives is intra-line consistency**:
   capacity-curve runs (0.6/1.5/3B) and data-ablation runs (500/1300/2575)
   each use one canonical pipeline, identical across their members. What that
   pipeline is, is now an empirical question, not an inherited one.
3. **Efficiency measures are legitimate Round-1 tools**: framework choice,
   data-parallel GPU count, fp16 vs bf16, and quantization are all open, to be
   decided by measured throughput/VRAM on the T4 — not by the reproduction
   plan's assumptions.
4. **Framework selection is evidence-driven and gated.** A/B benches run first
   (dual-GPU DDP and fp16 vs bf16 on the working hand-written Trainer, smoke
   scale). Only if that does not fit the full run in quota do we pursue
   framework alternatives (LLaMA-Factory in a clean env, unsloth, QLoRA).
   Rationale: every Kaggle bench run spends quota; we spend it to answer
   "can Round-1 run", not to satisfy curiosity about framework speed.
5. **The tracer's quality gate is orthogonal to the training-stack question.**
   Empty inference output (10/10) on the completed v4 adapter is diagnosed and
   fixed first (root adapter selection, decode length, parsing) because the
   eval needs a non-empty `predictions.jsonl` regardless of which stack trains
   Round-1.

## Consequences

- Round-1 runs use whichever canonical pipeline survives the benches; each
  member of a curve shares it. Configs stop mirroring mohar07 by default.
- Newly relevant engineering facts are recorded where they surface (ADR-0005
  here; tracer findings in issue 06 / `.scratch` exploration notes).
- Reproduction-line materials (LLaMA-Factory yamls, README_tracer/full) are
  not deleted, but are flagged as superseded for the GraphRAG line; they may be
  folded into an updated training runbook once the pipeline is chosen.

_Considered and rejected_: keep the reproduction locks and only tune within
them (cannot exercise dual-GPU/quantization/alternative frameworks; the full
run likely overruns quota); switch to a framework a priori (unmeasured; wastes
the working hand-written pipeline and quota on reintegration).

---

## Post-tracer verdict (2026-09-03, adds to the decision, does not reverse it)

The tracer benches (issue 06) resolved the two empirical questions the ADR
left open, and one part of its Context is now corrected:

- **Framework: LLaMA-Factory (clean-install), dual-GPU.** The silent exit the
  ADR blamed on LLaMA-Factory was actually three config omissions on our side
  (CUDA_VISIBLE_DEVICES=0 fighting its native multi-GPU path; yaml missing
  `do_train: true`; yaml missing `report_to: none`). Clean install pins the
  version matrix it needs (accelerate 1.11.0 / peft 0.18.1 / transformers
  4.57.1). Full tracer train ran 95x5 epochs dual-GPU: bf16 56:16, fp16 18:47,
  loss 0.587 / 0.6595. The hand-written Trainer line mentioned in Context is
  superseded — llamafactory is the canonical launcher.
- **Precision: fp16.** T4 (Turing sm_75) has no bf16 tensor cores; bf16 falls
  back to FP32 GEMMs. Profiled step times: bf16 69.7s vs fp16 21.0s (3.3x) on
  native `turing_fp16_s1688gemm` kernels. llamafactory fp16 is full AMP (its
  accelerate GradScaler only exists under `mixed_precision == "fp16"`), so no
  loss-scaling risk. Adopted as the canonical precision for T4.
- **Corrected Context claim:** the ADR's Context said fp16 "may be the native
  fast path" — profiler confirms it is; and dual-GPU was exercised (2x). Full
  2575-row estimate at fp16 dual-GPU now ~6-8h (was 27-40 T4-hours at the old
  bf16 single-GPU assumption).
