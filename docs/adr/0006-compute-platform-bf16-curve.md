# ADR-0006 — Capacity-curve compute platform: CompShare 4090 + Unsloth (bf16)

Status: Accepted (2026-09-05).

## Context

Round-1's remaining GPU work is the capacity curve: the same 2575-row GraphRAG
Alpaca set trained on three Qwen3 sizes (issue 08). The 0.6B full run already
completed on Kaggle (dual-T4, fp16, llamafactory, ~5h) and anchors the dataset
side. Issue 09's data ablation was **deferred** (owner, 2026-09-05) so no Kaggle
runs are scheduled this cycle.

Platform candidates were ModelScope DSW (explored in depth; handoff
`Temp/opencode/handoff-modelscope-dsw-exploration.md`), Kaggle T4 (proven),
and CompShare/优云智算 (UCloud; official agent CLI + Skills
`compshare-cli`, real SSH, ~2 元/h for 1×RTX 4090 24G). The decisive metric is
**total cost = compute RMB + agent-token friction**, not compute alone: ModelScope's
free GPU is cheap in RMB but its agent interface is unscripted (exec channel is a
binary hermes protocol, gateway SSO + 30-min cookies, captcha on every instance
start, instance id churn) — measured exploration burned far more token value than
2 元/h compute. CompShare exposes lifecycle via API and a real shell (SSH key),
so agent integration friction ≈ 0. Kaggle stays proven but is the fp16/ablation
line, not this cycle.

Also locked here: issue 08 named "1.5B/3B", but **Qwen3 has no 1.5B/3B** — the
family is 0.6/1.7/4/8/... The two extra curve points are therefore
**Qwen3-1.7B and Qwen3-4B (base)**, same GraphRAG dataset, config differing only
in the model field.

## Decision

1. **The capacity curve runs bf16 on a single CompShare RTX 4090 24G via
   Unsloth** (`finetune/compshare/train_unsloth.py`), sizes
   Qwen3-0.6B (bf16 re-run of the fp16 anchor), Qwen3-1.7B, Qwen3-4B. This is a
   *self-contained bf16 line*; it is not numerically compared against the fp16
   Kaggle anchor across lines.
2. **Canonical (intra-line fixed) config**: LoRA r=32 α=32, bf16, cutoff 6144,
   grad-accum 8, lr 5e-5, cosine + warmup_ratio 0.03, 5 epochs, packing to 6144,
   `adamw_8bit`, grad-checkpoint "unsloth". Runs differ **only** in
   `--model / --train-jsonl / --out`.
3. **Data formatting**: uniform plain Alpaca concat
   `instruction [+"\n"+input] + "\n\n" + output` (no chat template — base models
   have none), identical bytes across all three sizes. EOS is the final token of
   the packed sequence.
4. **Artifacts per run** (`/root/kg/out/<run>/` on the instance):
   `checkpoint-NNN` (Peft adapter, cadence `--save-every 50`),
   `adapter/` (final), `merged/` (optional, full bf16 via
   `PeftModel.merge_and_unload`). Local mirror: `outputs/adapters/<model>/`
   (gitignored). Checkpoints are pullable for eval/merge/HF push downstream.
5. **ModelScope DSW is rejected as a training host** for this cycle (integration
   friction; see Context). Its exploration notes are kept; the CPU instance
   remains an occasional free sandbox option. Kaggle T4 (fp16, llamafactory)
   remains the *other* line's runner if ablation resumes.

## Measured results (2026-09-05, single RTX 4090, bf16, seq 6144, 2575×5)

| model | wall | final loss | eval_loss | peak VRAM |
|---|---|---|---|---|
| Qwen3-0.6B | 43.7 min | 0.635 | 0.637 | ~5 GB |
| Qwen3-1.7B | ~88 min | 0.535 | 0.549 | ~12 GB |
| Qwen3-4B | ~3.2 h (est) | in progress | (eval off) | ~13 GB |

Tracer micro-batch throughput at 6144 tokens: 0.6B ~0.46 s/bs1, ~0.41 s/bs2
(unsloth). The external "8B on A10 in 30 min" claim is not reproducible at full
6144 packing (physical floor ~2.4 s/micro on A10 ≈ 4-7 h for 3000 rows); see
`Temp/opencode/research-train-throughput.md`.

## Lessons (enforced in the script)

- **Eval can OOM on big models**: 1.7B died at the step-161 eval because eval
  upcasts logits to fp32 (bs2×6144×151k). Fixes: `per_device_eval_batch_size=1`
  when eval is on, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and
  optionally no mid-run eval on the largest size.
- **Checkpoint early and often** (`--save-every 50`, default): a crash costs
  ≤ ~12 min of wall instead of a whole rerun (the first 1.7B attempt lost 38 min
  because the only save point was also the crash point).
- Do not merge llamafactory's pinned matrix and unsloth in one venv (separate
  envs). Do not stack Liger on top of Unsloth's own fused kernels.

## Consequences

- Issue 08's "1.5B/3B" text is amended to Qwen3-1.7B/4B and framework
  (llamafactory→unsloth, fp16→bf16) with this ADR as the reference.
- Issue 09 (ablation 500/1300) is deferred; its Kaggle fp16 sub-line stays valid
  if resumed.
- The reproduction-line fp16 anchor (Kaggle 0.6B) remains untouched and is the
  cross-check record for the eval thread.
