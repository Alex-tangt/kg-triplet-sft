# 09 — Data-scale ablation (500 / 1300 / 2575)

**What to build:** The data-scaling curve: three 0.6B runs on 500, 1300, and 2575 training
examples, each with the identical config except the dataset. Adapters pushed to Hugging Face.

**Blocked by:** 05, 06

**Status:** ready-for-agent

- [ ] The three subset Alpaca files exist (500 / 1300 / 2575)
- [ ] Configs differ from the full 0.6B config only in the dataset field
- [ ] All three runs complete; adapters pushed to Hugging Face
