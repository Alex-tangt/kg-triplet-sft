# 05 — Dataset assembly

**What to build:** Validation and split stage: schema validation, semantic validation
(in-chunk grounding, weight anchors), curriculum ordering by (triplet count, word count)
ascending, strict entity decontamination train↔test (degrade to the ≥2-entity rule only if
attrition exceeds 25%), splits of 2575 train / 75 val / 700 test, and the Alpaca projection
rendered from the single prompt template.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] Validation rejects out-of-schema or ungrounded labels; rejection counts reported
- [ ] Training file is ordered ascending by (triplet count, word count)
- [ ] Decontamination removes every test sample sharing any normalized entity title with the
      training set; attrition reported; the ≥2 rule engages only above 25% attrition
- [ ] Splits are exactly 2575 / 75 / 700
- [ ] Alpaca files (instruction/input/output) render from the template constant
- [ ] A 100-sample stratified audit fixture is exported for the attribution suite
