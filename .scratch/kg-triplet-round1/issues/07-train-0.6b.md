# 07 — 0.6B full training

**What to build:** The faithful reproduction run: the 0.6B config applied to the full 2575-example
training set with the original hyperparameters, checkpointed against quota resets, adapter pushed
to Hugging Face with a model card.

**Blocked by:** 05, 06

**Status:** ready-for-agent

- [ ] Config differs from the tracer's only in the dataset field (single-variable invariant)
- [ ] Training completes (or resumes) within the Kaggle quota; loss curves logged
- [ ] Adapter is pushed to Hugging Face with a model card
