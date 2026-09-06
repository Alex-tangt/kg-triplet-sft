# Referent-level eval runbook

How to score a new fine-tuned model on the fixed eval sample. Decision context in
`docs/adr/0007-referent-level-eval.md`; vocabulary in `CONTEXT.md`.

## Pipeline

0. **Export the sample passages for inference** (id + text, aligned with the
   teacher file): `python eval/referent_sample.py --n 200 --seed 20260906 \
   --passages-out outputs/referent_eval/sample_200_passages.jsonl`
1. **Infer the fixed sample** with the candidate model → a JSONL of
   `{"id", "entities": [...], "relationships": [...]}` in the deploy extraction
   format (like `outputs/full_eval/predictions_full_norm.jsonl`).
2. **Referent pairing** against the teacher gold (thinking LLM, ~7 min / model):
   ```
   python eval/referent_pair.py --mode llm --workers 16 --no-passage --desc-len 300 --thinking \
     --out outputs/referent_eval/pairings_<tag>.jsonl \
     --ids (200 sample ids)
   ```
   Requires `DASHSCOPE_API_KEY`. Ids are read from `sample_200.json` (`ids`).
3. **Score**: `python eval/referent_eval.py --pairs outputs/referent_eval/pairings_<tag>.jsonl \
   --out outputs/referent_eval/report_<tag>.json`

   The report prints entity/edge finding P/R/F1 (macro + micro), schema, grounding,
   numeric, and the bootstrap CI table for the sample size.

## Input files (do not regenerate)

- `outputs/full_eval/test_teacher_with_text.jsonl` — 699 teacher gold (labels + text)
- `outputs/referent_eval/sample_200.json` — fixed 200-passage sample (teacher ids)
- `outputs/referent_eval/pairings_think_699.jsonl` + `report_think.json` —
  the reference model's full-699 result (entity recall 0.541 / precision 0.707),
  the baseline to compare new models against on the same sample.

## Notes

- New-model comparisons must use the SAME sample (paired comparison cancels
  passage noise); sample size was set at n=200 so entity-recall 95% CI ≈ ±2.1pt.
- `--mode title` gives the no-LLM string floor for sanity; never treat a single
  non-thinking LLM pass as the headline (it over-merges co-occurring concepts).
- `--thinking` is required for the fuzzy tier; verify/intersection stages were
  tried and abandoned (ADR-0007) — do not reintroduce without re-measuring on a
  known-answer probe.
