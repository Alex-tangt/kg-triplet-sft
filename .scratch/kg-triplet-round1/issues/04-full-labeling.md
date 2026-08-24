# 04 — Full labeling run

**What to build:** The scale-up of the tracer: every surviving corpus passage labeled by the
teacher with checkpoint/resume (an interruption re-runs nothing already paid for), hard negatives
teacher-judged at ~15% of the final training set, budget cap respected.

**Blocked by:** 02, 03

**Status:** ready-for-agent

- [ ] Checkpoint/resume: a killed run restarts without re-calling already-labeled passages
- [ ] Every target passage is labeled or recorded as failed with a reason
- [ ] Hard negatives land at ~15% of training examples (teacher-judged `[]`)
- [ ] Final spend and per-call cost are logged against the budget cap
