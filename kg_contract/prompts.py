"""Prompt templates: the one source of truth for what the model sees.

The target prompt is reproduced character-for-character from the original
project (ADR-0001): the 20-relation ontology stays implicit, never listed in the
instruction. One template constant renders every consumer — the Alpaca dataset
projection, evaluation prediction generation, and serving — so drift between the
three code paths is structurally impossible.

The teacher prompt (what qwen3-flash sees when generating gold labels) is a
separate, richer instruction. Its skeleton fixes the five elements and their
order; the actual prose is filled in by the labeling tickets.
"""

from __future__ import annotations

TARGET_PROMPT_PREFIX = (
    "Extract knowledge graph triplets from the following text. Return a JSON "
    "array of objects with keys: source (title, type), relation (type, weight), "
    "target (title, type). If no valid triplets can be extracted, return an "
    "empty array.\n\nText:\n"
)

TARGET_PROMPT_TEMPLATE = TARGET_PROMPT_PREFIX + "{text}"


def render_target_prompt(text: str) -> str:
    """Render the verbatim target prompt for one passage."""
    return TARGET_PROMPT_TEMPLATE.format(text=text)


WEIGHT_ANCHORS: tuple[float, ...] = (0.2, 0.5, 0.8)

#: The five required elements of the teacher instruction, in rendering order.
TEACHER_SECTIONS: tuple[str, ...] = (
    "relation_dictionary",
    "entity_naming_rules",
    "exhaustiveness_demand",
    "weight_rubric",
    "in_chunk_grounding",
)


def render_teacher_prompt(sections: dict[str, str]) -> str:
    """Assemble a teacher prompt from its five named sections, in fixed order."""
    missing = [name for name in TEACHER_SECTIONS if name not in sections]
    if missing:
        raise ValueError(f"missing teacher sections: {', '.join(missing)}")
    unknown = [name for name in sections if name not in TEACHER_SECTIONS]
    if unknown:
        raise ValueError(f"unknown teacher sections: {', '.join(unknown)}")
    return "\n\n".join(sections[name] for name in TEACHER_SECTIONS)
