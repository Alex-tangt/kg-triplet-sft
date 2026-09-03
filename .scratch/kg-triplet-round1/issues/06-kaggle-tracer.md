# 06 — Kaggle training tracer

**What to build:** Prove the LLaMA-Factory workflow on Kaggle T4 at tracer scale before
the full run: the 0.6B config, runbook, adapter/export — on the 95-row GraphRAG tracer
set. Original hyperparameters (r=32, α=32, lr 5e-5, 5 epochs, bs2 × grad-accum 4,
**cutoff 6144** — GraphRAG labels are 2-6k tokens, bf16 LoRA).

**Blocked by:** 05

**Status:** ready-for-human (owner runs on Kaggle T4; tracer SFT was handed out of this
agent's scope)

- [ ] Tracer run on Kaggle T4 per `finetune/README_tracer.md`:
      upload `outputs/tracer/alpaca_train.jsonl`(95) + `alpaca_val.jsonl`(5),
      `finetune/qwen3_0.6b_graphrag_tracer.yaml`, merge `finetune/dataset_info.json`
      (`graphrag_tracer` → `data/alpaca_train.jsonl`); `llamafactory-cli train`
- [ ] Qwen3 thinking disabled during training (`qwen_thinking: false` if supported) — the
      data is plain JSON answers, not reasoning
- [ ] Export/merge completes; then single-shot inference on the 10 tracer-eval passages
      with the derived `STUDENT_PROMPT` (JSON out; `max_tokens` ≥ 8192; serve prompt must
      mirror LLaMA-Factory's instruction+input join) → `outputs/tracer/predictions.jsonl`
- [ ] Decision gate (see `eval/tracer_eval.py` + 3-axis eval vs `outputs/tracer/teacher_ref.jsonl`)
      decides go/no-go for the full run (ticket 07)
