"""Cross-phase contract for the KG-triplet pipeline.

Single home for the triplet schema (relations, validator, prompt templates)
consumed by dataset, finetune, eval, and serve.
"""

from .prompts import (
    TEACHER_SECTIONS,
    TARGET_PROMPT_PREFIX,
    TARGET_PROMPT_TEMPLATE,
    WEIGHT_ANCHORS,
    render_target_prompt,
    render_teacher_prompt,
)
from .relations import ENTITY_TYPES, RELATION_TYPES
from .validator import (
    ValidationError,
    extract_json_array,
    is_valid_triplet,
    validate_triplet,
    validate_triplet_list,
)

__all__ = [
    "ENTITY_TYPES",
    "RELATION_TYPES",
    "TEACHER_SECTIONS",
    "TARGET_PROMPT_PREFIX",
    "TARGET_PROMPT_TEMPLATE",
    "WEIGHT_ANCHORS",
    "ValidationError",
    "extract_json_array",
    "is_valid_triplet",
    "render_target_prompt",
    "render_teacher_prompt",
    "validate_triplet",
    "validate_triplet_list",
]
