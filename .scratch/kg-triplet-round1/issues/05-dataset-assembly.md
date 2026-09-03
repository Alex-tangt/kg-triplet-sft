# 05 — Dataset assembly

**What to build:** The labeled corpus → splits → Alpaca projection.
> Superseded by the data-generation pivot (ADR-0003/0004): the typed-plan body
> (curriculum ordering by triplet count, strict entity-title decontamination,
> stratified audit fixture) belonged to the mohar07 reproduction line. The GraphRAG
> line executes the split + final cleanup + Alpaca projection below.

**Blocked by:** 04

**Status:** done (2026-09-03, GraphRAG form)

- [x] Split (`dataset/graphrag_batch.py --stage split`): 2575 train / 75 val / 700 test, seed 7; the 10 tracer-eval passages sit in test only (verified: 0 eval ids in train/val)
- [x] Final cleanup pass (`dataset/clean_labels.py`) over the merged labels; split train+val all present in labels (0 missing)
- [x] Alpaca projection renders from the `STUDENT_PROMPT` constant (ADR-0004 derived prompt):
  - full scale: `outputs/graphrag_full/alpaca_full_train.jsonl` (2575) / `alpaca_full_val.jsonl` (75), via `python dataset/build_alpaca.py --full`
  - tracer: `outputs/tracer/alpaca_train.jsonl` (95) / `alpaca_val.jsonl` (5)
- [x] Token check: p99 packed ≈ 4.9k tokens under the 6144 cutoff; ~4/2575 rows slightly exceed it (tails unlabelled, harmless)
- [x] Tests green (125): `tests/test_student_prompt.py` guards verbatim teacher↔student prompt inheritance
- [ ] (superseded) Curriculum ordering, strict decontamination, 100-sample attribution audit fixture — reproduction-line items, not executed here
