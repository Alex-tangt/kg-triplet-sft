# RESULT — Leg A: base Qwen3-0.6B baseline (2026-09-08)

Untrained main-release Qwen3-0.6B (`qwen-lm/qwen-3/transformers/0.6b/1`, the
weights a training run starts from) through the fixed-200 referent eval with the
SAME chat/no-think protocol, schema-prefix injection, tolerant parser and
thinking judge as the capacity line. Kernel `kg-tracer-capbase` (Kaggle T4,
batch 4, no LoRA merge), predictions `outputs/base_eval/qwen3-0.6b-base/`,
pairings `outputs/referent_eval/pairings_qwen3-0.6b-base.jsonl` (200/200 ok,
judge spend 364,273 tokens), report
`outputs/referent_eval/report_qwen3-0.6b-base.json`.

## One-row table (ref200 format)

| run | macro R/P/F1 | micro R/P/F1 | edge F1 | schema-valid | halluc | empty rows | parse rate | student entities |
|---|---|---|---|---|---|---|---|---|
| reference (ref200) | 0.551 / 0.721 / 0.602 | 0.533 / 0.723 / 0.614 | 0.086 | 0.931 | 0.142 | 10 | 0.97 | 3665 |
| 0.6B masked | 0.559 / 0.713 / 0.604 | 0.536 / 0.699 / 0.607 | 0.091 | 0.919 | 0.166 | 2 | 0.97 | 3805 |
| **base (untrained)** | **0.162 / 0.849 / 0.253** | **0.159 / 0.854 / 0.268** | **0.024** | **0.666** | **0.031** | **1** | **0.99** | **925** |

Definitions: entity finding referent-level macro/micro; schema-valid =
`schema.valid_fraction`; halluc = `grounding.hallucination_rate` (student
unmatched entities); empty rows = rows with zero entities AND zero
relationships after the schema-clean parse; parse rate = share of the 200 rows
with ≥1 entity (schema-prefix injection keeps even the base model structurally
compliant). All three runs share the same gold and judge; base used
`--no-passage` like the capacity line.

## Reading (delta vs report_ref200)

SFT's contribution is large and recall-driven, NOT base leakage: the untrained
base sits at micro entity recall 0.159 / F1 0.268, and the masked recipe lifts
it to 0.536 / 0.607 (+0.38 recall, +0.34 F1) at a modest precision cost
(0.854 → 0.699 — the LoRA triples output volume from 925 to 3805 entities, so
it trades a little precision for ~3.4x coverage). Edge F1 rises 0.024 → 0.091
and schema-valid 0.666 → 0.919 (SFT teaches the output contract). The base's
seemingly low hallucination (0.031) is an artifact of sparse output — few
entities, few unmatched; its failure mode is omission, not invention. So the
capacity-curve ~0.60 F1 plateau is a real learned behaviour, not the base model
extracting on its own.
