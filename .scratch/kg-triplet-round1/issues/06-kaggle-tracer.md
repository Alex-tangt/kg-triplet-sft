# 06 — Kaggle training tracer

**What to build:** The LLaMA-Factory workflow proven on Kaggle T4 before committing real quota:
environment setup, the 0.6B yaml config (original hyperparameters: r=32, α=32, lr 5e-5,
5 epochs, bs2 × grad-accum 4, ctx 2048, bf16 LoRA), the dataset card, a smoke training on the
pilot data, adapter push to Hugging Face, and GGUF export — every step of the Round 1 training
chain, at minimal scale.

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] The 0.6B config trains the pilot Alpaca set to completion on a T4
- [ ] Checkpoint/resume is configured and demonstrably works
- [ ] Adapter push to Hugging Face succeeds (or the agreed fallback is exercised)
- [ ] GGUF export step succeeds
- [ ] A runbook captures the exact Kaggle setup steps for reuse by later tickets
