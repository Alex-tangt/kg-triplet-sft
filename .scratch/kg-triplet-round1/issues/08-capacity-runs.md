# 08 — Capacity runs (1.5B / 3B)

**What to build:** The two additional capacity-curve points: 1.5B and 3B trained on the same
dataset with configs that differ from the 0.6B's only in the base-model field. Both adapters
pushed to Hugging Face.

**Blocked by:** 05, 06

**Status:** ready-for-agent

- [ ] A config-diff test asserts the only difference from the 0.6B config is the base model
- [ ] Both runs complete on T4 within quota; loss curves logged
- [ ] Both adapters pushed to Hugging Face
