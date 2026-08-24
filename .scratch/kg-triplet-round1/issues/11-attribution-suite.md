# 11 — Attribution suite

**What to build:** The four-experiment attribution report that answers why the entity F1 is low:
the three-config threshold sweep (default / relaxed 0.60+0.30 / --no-embeddings), the gold audit
(100 stratified samples; human scores 30, LLM scores all 100 with the teacher rubric; agreement
reported), the capacity and data-scaling curves, and the final report naming the dominant factor.
This report is the Round 2 gate input.

**Blocked by:** 10

**Status:** ready-for-agent

- [ ] Sweep numbers produced for exactly three configurations
- [ ] Gold audit completed: stratification by source / polarity / difficulty / relation coverage,
      inter-rater agreement reported
- [ ] Both scaling curves plotted from the trained runs
- [ ] Attribution report states which of harness / labels / data scale / capacity dominates,
      with the numbers behind each
