# Evaluate on our own test split only — no cross-dataset comparison against the original gold

The original 700-entry gold test set sits inside the evaluation-pipeline repo we vendored, and our
models share its schema and prompt — so evaluating our models against it looks possible. We
deliberately do not: our gold labels come from a different teacher LLM than the original's, so a
cross-dataset score would conflate model quality with teacher-style mismatch.

Instead, the reproduction claim rests on structural evidence measured on our own test split
(base vs. fine-tuned on the same gold: schema-axis gain, ~6x hallucination-rate drop, and a
composite axis shape matching the original's 0.6583 profile), not on absolute cross-dataset
numbers.

_Considered and rejected_: cross-dataset comparison against the original 700 gold (nice headline
number, but uninterpretable because the two teachers name and select entities differently).
