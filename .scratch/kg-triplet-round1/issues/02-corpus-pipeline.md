# 02 — Corpus pipeline

**What to build:** The corpus stage: download English Wikipedia and arXiv passages (60/40) via
hf-mirror, chunk to 100–300 words at sentence boundaries (never mid-sentence — the explicit
improvement over the original's hard truncation), quality-filter (language, length, boilerplate),
and MinHash near-duplicate removal, producing ~5000 clean passages.

**Blocked by:** None — can start immediately

**Status:** done

- [x] Each stage runs end-to-end on a tiny fixture and asserts its output invariants
- [x] Full run produces ~5000 passages with logged stats (source mix, length distribution)
- [x] No produced passage ends mid-sentence (fixture + full-run assertion)
- [x] MinHash dedup removes near-duplicates at the configured threshold; counts reported
- [x] Run is resumable and downloads via hf-mirror only
