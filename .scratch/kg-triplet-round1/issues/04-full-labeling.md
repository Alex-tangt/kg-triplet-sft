# 04 — Full labeling run

**What to build:** Label every split passage (except dropped) with the teacher.
> Superseded by the data-generation pivot (ADR-0003): the original typed-plan body
> (15% teacher-judged hard negatives, single-prompt teacher, checkpoint/resume realtime)
> was replaced by **Aliyun batch inference in a fixed two-round schedule** (r1 + r2
> "continue") over the split passages. The checklist below records what was actually
> executed.

**Blocked by:** 02, 03

**Status:** done (2026-09-03, GraphRAG form)

- [x] Batch Job A (r1) over 3240 not-yet-labeled split passages → `outputs/graphrag_batch/bach_full_r1_success.jsonl` (3238) + `batch_full_r1_error.jsonl`
- [x] Batch Job B (r2, CONTINUE on r1) → `outputs/graphrag_batch/full_r2_success.jsonl` (3238) + `full_r2_error.jsonl`
- [x] merge-full → `dataset/data/graphrag_labels_full.jsonl`: **3349 passages** (entities/relationships after cleanup 80893 / 72955); pre-clean backup at `graphrag_labels_full.pre_clean.jsonl`
- [x] Parser rewritten tolerant of teacher sloppiness (`dataset/graphrag.py`: missing strength, `**##**`/`<|COMPLETE|>`/`>` tails); cleanup + QC (`dataset/clean_labels.py`): 0 corrupted, 0 empty, 0 unresolved endpoints
- [x] Non-compliant passages dropped with reason recorded: `wikipedia-01183` (content moderation); 100 realtime labels at `dataset/data/graphrag_labels_100.jsonl`
- [x] Sample blind QC PASS: second 10-passage sample (qc_sample2) — 0 structural junk, 0 fabrication, all endpoints resolve; residual noise (generic nodes / boilerplate descriptions) accepted
- [ ] (superseded) Hard negatives at ~15% — not executed in the GraphRAG line
