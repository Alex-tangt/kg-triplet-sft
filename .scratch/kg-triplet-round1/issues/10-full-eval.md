# 10 — Full evaluation

**What to build:** Predictions for the base model and every trained adapter on our own 700-entry
test split, then the harness's default-config report for each, producing the Round 1 comparison
table. The 0.6B reproduction is judged here: composite in 0.55–0.75 with the original's axis
shape, plus the base-model contrast (schema gain, ~6x hallucination-rate drop).

**Blocked by:** 07, 08, 09

**Status:** superseded (the reproduction-line harness comparison was dropped when
the project pivoted to the open line; an open-line full-699 referent eval was
delivered instead — see `eval/README.md`, ADR-0002/0007).

- [ ] A predictions file exists for the base model and every adapter
- [ ] Default-config harness reports generated for all models
- [ ] Comparison table: composite, five axes, entity/triplet P/R/F1, hallucination rate per model
- [ ] 0.6B composite lands in 0.55–0.75 with the original's axis shape (ADR-0002: own test only)
