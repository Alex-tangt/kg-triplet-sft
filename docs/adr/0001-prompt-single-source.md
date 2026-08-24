# Prompt single source of truth: verbatim target prompt, independent teacher prompt

We are fine-tuning a model that must reproduce the original KG-triplet behaviour, so the
**target prompt** (what the model sees in training samples) is reproduced character-for-character
from the original project, with the 20-relation ontology implicit in the labels and never stated
in the prompt. The **teacher prompt** (what qwen3-flash sees when generating gold labels) is a
separately designed, richer instruction that carries the relation dictionary and weight rubric —
teacher freedom exists for label quality, but the model only ever sees the target prompt.

The invariant is: the prompt embedded in training samples equals the prompt used at inference,
in evaluation and serving alike. A single template constant renders all consumers (Alpaca dataset
projection, evaluation generation, serving), so drift between the three code paths — which would
silently depress scores and misattribute the loss — is structurally impossible.

_Considered and rejected_: storing fully rendered prompts in the dataset (template changes would
require regenerating the whole dataset); listing the ontology in the target prompt (higher scores
possible, but breaks faithful reproduction and loses comparability with the original 0.6583).
