# 06 — Kaggle training tracer

**What to build:** Prove the training workflow on Kaggle T4 at tracer scale
before the full run: the 0.6B config, runbook, adapter — on the 95-row GraphRAG
tracer set.

**Blocked by:** 05

**Status:** done (Kaggle tracer train/infer kernels shipped under
`finetune/kaggle/kernel*`; stack verdict ADR-0005).

## Current state (2026-09-03)

**Quality gate: first consistent-pipeline run is a real signal (not empty,
schema-correct, moderate recall).** llamafactory dual-GPU trained the H4-rebuilt
95 rows (loss 0.803→0.587, 56 min); inference on the 10 eval passages used the
SAME H4 prompt + sampling: 8/10 parse as clean JSON with the correct schema
(`title/type/description`, `source/target/description/strength` — the entity_name
leakage is GONE), 2/10 fail on an unescaped control character. Objective eval
(3-axis): avg_entity_recall 0.183, avg_entity_precision 0.412, avg_numeric_ratio
0.25. Diagnosed recall driver: entity-name spelling/casing drift (ARNE KAISER vs
ARNE KAIJSER; SOCIETIE vs SOCIETY; truncated foreign book titles) — the model
extracts similar entities but not exact surface forms; numeric recall is weak.
Real 0.6B@95-row limitation: format learned, exact-name fidelity not.

**Training stack RESOLVED (llamafactory + dual-GPU DDP works).** After the
clean-install route (git clone + `pip install -e ".[torch,metrics]"` →
accelerate 1.11.0 / peft 0.18.1 / transformers 4.57.1) and fixing three config
omissions (`do_train`, `report_to: none`, and not forcing CUDA_VISIBLE_DEVICES),
`llamafactory-cli train` runs dual-GPU DDP on Kaggle T4: torchrun world size 2,
loss 0.837→0.812→0.773 over 6 smoke steps, 38.4 s/step (2.1x the single-GPU
81 s/step). LLaMA-Factory is the Round-1 carrier.

What actually happened — history matters more than the original checklist now:

- **LLaMA-Factory "did not run" was a wrong conclusion, since corrected.** Its
  CLI silently exited 0 under the Kaggle preinstall stack (accelerate 1.13.0
  violates llamafactory's own matrix: accelerate<=1.11.0, peft 0.18.0-0.18.1).
  A clean `git clone` + `pip install -e ".[torch,metrics]"` (community Kaggle
  route, resolves a compatible set) is under verification (`kg-tracer-lfcheck`).
  The hand-written Trainer trained successfully meanwhile, but is now a
  fallback, not the plan (per research-before-hand-rolling discipline).
- **Training on T4 fits only at bs1 × grad-accum 8** for cutoff 6144 (bs2 OOM,
  13.8 GB activations). Working recipe: bs1, grad-accum 8, gradient
  checkpointing, `enable_input_require_grads()`, bf16, lr 5e-5, 5 epochs.
- **Full tracer run (v4) COMPLETED**: 60/60 steps, loss 0.77 -> 0.61; LoRA
  adapter produced (r/α 32, ~80 MB) and downloaded locally
  (`Temp/opencode/full/out4/`). Kaggle status `COMPLETE`; root adapter +
  checkpoints 36/48/60 verified on disk.
- **Inference quality gate: verdict-forming evidence (v4 adapter + sampling).**
  Two decode experiments on the existing adapter:
  - greedy (do_sample=False, 8192 cap): degenerate repetition — emits ~28-39k
    chars of a repeating entity and never stops; all 10 PARSE FAIL.
  - sampling (temp 0.7 / top_p 0.95 / rep 1.15): repetition cured, but the
    output still violates the OUTPUT schema contract:
      * field leakage — `entity_name/entity_type/relationship_description`
        (prompt -STEPS- vocab) instead of `title/type/description`;
      * structural syntax error — `...},\n["relationships": [...]`
        (object/array confusion);
      * inconsistent type casing (CONCEPT/Concept) and wrong labels (EVENT).
  These are model-capacity/data-volume signals — exactly what the tracer exists
  to surface before spending quota on the 2575-row full run.
- **GPU quota (accurate, MCP `get_accelerator_quota`):** used 35,635s / 108,000s
  (~9.9h / 30h) at last check, refresh 2026-09-05 00:00 UTC.

## New plan (per ADR-0005 — training decoupled from mohar07 reproduction config)

The tracer's quality gate no longer depends on which framework eventually
trains Round-1. Order of execution:

- [x] ADR-0005 written (Round-1 pipeline decoupled; intra-curve consistency is
      the only training constraint; efficiency measures legitimate)
- [ ] **Diagnose + fix empty inference** using the existing v4 adapter (no
      retrain): pull the v4 log, confirm PARSE FAIL vs empty-vs-truncation;
      fix infer (root adapter only, decode length, parsing); re-run pure
      inference on the 10 eval passages -> non-empty `predictions.jsonl`
- [ ] **A/B throughput benches** (smoke-scale, on the working hand-written
      Trainer): dual-GPU DDP and fp16 vs bf16, measure per-step time + peak
      VRAM -> decide the Round-1 canonical config and recompute full-run cost
- [ ] Only if benches don't fit the full run in quota: benchmark framework
      alternatives (LLaMA-Factory in a clean env, unsloth, QLoRA)
- [ ] Decision gate: `eval/tracer_eval.py` + 3-axis eval vs
      `outputs/tracer/teacher_ref.jsonl` -> go/no-go for ticket 07
- [ ] Record: export/merge + single-shot inference with the derived
      `STUDENT_PROMPT` (JSON out; decode length >= 8k; serve prompt must mirror
      training join) -> `outputs/tracer/predictions.jsonl`

## Notes

- T4 kernels expose 2 GPUs; earlier runs forced single-GPU
  (`CUDA_VISIBLE_DEVICES=0`). Data-parallel use of both is an open efficiency
  lever under test.
- T4 is Turing (no bf16 tensor cores); fp16 may be the native fast path. This
  is a hypothesis to measure, not a decision (bf16 works fine functionally).

## Comments

- 2026-09-03 (owner + agent, grilling): Round-1 training stack is decoupled
  from the reproduction config (ADR-0005). Tracer quality gate and the stack
  decision are orthogonal; quality gate first (cheap, no retrain), benches
  second (smoke-scale), framework alternatives only if benches don't fit quota.
- 2026-09-03 (agent, root cause v1): empty predictions were NOT truncation-only.
  With decode cap raised 4096->8192 the model STILL emits ~28-39k chars (fills
  the whole 8192 budget) and never stops. Two real defects surfaced:
  (a) **degenerate repetition loop** — greedy decode (do_sample=False) repeats
      entities forever; HF docs: greedy "breaks down ... begins to repeat";
      Qwen3 default is sampling (temp 0.6 / top_p 0.95);
  (b) **schema field leakage** — output uses -STEPS- vocabulary
      (entity_name/entity_type) instead of the -OUTPUT- schema (title/type),
      i.e. the 95-example model didn't bind the OUTPUT contract.
  These are model-quality signals (the tracer's whole point), not inference bugs.
- 2026-09-03 (agent, research): called the community before tuning — greedy is
  wrong for long structured output; fix is sampling + repetition_penalty
  (temp 0.7 / top_p 0.95 / rep 1.15), and constrained decoding (grammar / JSON
  schema, e.g. outlines / llama.cpp GBNF) can force valid keys + closure. The
  serve stack (Ollama GGUF) has grammar support natively.
- 2026-09-03 (agent, bench v1): fp16 == bf16 (~81s/step) on T4; peak VRAM only
  8.86/16GB; Kaggle exposes 2x T4. Trainer-DDP unusable in script kernels
  (bench v2 OOM: parallel_mode != DISTRIBUTED, both ranks hit GPU0).
- 2026-09-03 (owner, discipline): research-before-hand-rolling protocol added
  to global AGENTS.md (3-strike rule). llamafactory was abandoned too early:
  root cause was likely its version matrix (accelerate<=1.11.0; Kaggle has
  1.13.0) never being satisfied, NOT llamafactory being broken on Kaggle.
- 2026-09-03 (agent, infer2 experiment): sampling (temp 0.7 / top_p 0.95 / rep
  1.15, 2 passages) cured the repetition loop (arxiv-00944 produced a parseable
  ~1.9k-char JSON with 13 entities) but confirmed two training-side defects:
  field leakage (entity_name/entity_type) and a JSON syntax error
  (`["relationships":`). wikipedia-02909 (6.4k chars, closed) still failed
  parse at the `["relationships":` syntax break. Verdict so far: 0.6B @ 95 rows
  has NOT learned the OUTPUT schema contract reliably; full run needs either
  more data per task, constrained decoding at serve time, or schema-robust
  evaluation — an owner decision for the go/no-go gate.
- 2026-09-03 (agent, RETRACTION of the infer2 "field leakage" verdict): the
  leakage was an artifact of train/serve prompt INCONSISTENCY (ADR-0001
  violation), not a model-capacity signal. The v4 training data on Kaggle used
  the OLD prompt ("Extract knowledge-graph..."); the inference kernels embedded
  the NEW H4 prompt (whose -STEPS- introduces entity_name etc.). The model was
  shown words it never saw in training and echoed them. H4 fix (commit
  f8b902d) unified the vocabulary and rebuilt alpaca (tracer 95 + full 2575);
  `finetune/kaggle/kernel/inputs.py` was stale and has been regenerated so all
  kernel variants embed the same new prompt. Any quality verdict must come from
  a fresh run where train and infer use the identical new prompt.
- 2026-09-03 (agent, llamafactory re-check): clean-install route
  (git clone + `pip install -e ".[torch,metrics]"`) satisfies its version
  matrix (accelerate 1.11.0 / peft 0.18.1 / transformers 4.57.1). Dual-GPU DDP
  via torchrun starts correctly (world size 2, cuda:0/1). The earlier "silent
  exit" had multiple causes resolved one by one: (1) CUDA_VISIBLE_DEVICES=0
  fighting llamafactory's native multi-GPU path; (2) yaml missing
  `do_train: true` (train() was skipped); (3) yaml missing `report_to: none`
  (wandb init crashed without an API key). All three are config omissions vs
  the official example yaml — not llamafactory bugs.

- 2026-09-03 (agent, fp16 DECIDED for Round-1): T4 has NO bf16 tensor core
  (Ampere+ only — NVIDIA blog + torch is_bf16_supported + flash-attn README all
  agree); bf16 falls back to FP32 GEMMs. Profiler (kg-tracer-prof v3): bf16 one
  optimizer step 69.7s (kernels: magma_sgemmEx<float,__nv_bfloat16> = FP32
  fallback) vs fp16 21.0s (kernels: turing_fp16_s1688gemm + fmha_cutlass_f16_sm75
  = native tensor core) = 3.3x. Full llamafactory tracer run in fp16: loss
  0.6595 vs bf16 0.6667 (equivalent, no NaN), runtime 18:47 vs 56:16, samples/s
  0.421 vs 0.141. fp16 in llamafactory is AMP, NOT bare fp16 — accelerate only
  creates GradScaler when mixed_precision=="fp16". No community third path
  exists (bf16x9 is Blackwell-only, reverse-emulation is mathematically = fp16).
  Decision: fp16 for Round-1; no re-inference needed (fp16/bf16 weights differ
  by <0.01 loss, so the recall 0.183 baseline stands).
