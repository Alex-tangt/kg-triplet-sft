"""Triplet schema validation for the cross-phase contract.

The harness's ``is_valid_triplet`` is deliberately permissive (shape-only); this
validator is stricter — it also requires the relation to be in the 20-set and the
source/target ``type`` to be ``entity`` or ``concept`` — because the dataset
pipeline must reject out-of-vocabulary labels before they ever reach training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .relations import ENTITY_TYPES, RELATION_TYPES


@dataclass(frozen=True)
class ValidationError:
    """One schema violation, located by ``path`` (e.g. ``triplets[2].relation.type``)."""

    path: str
    reason: str


def _check_node(sub: Any, key: str, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(sub, dict):
        errors.append(ValidationError(path, f"{key} must be an object"))
        return
    if not isinstance(sub.get("title"), str):
        errors.append(ValidationError(f"{path}.title", f"{key}.title must be a string"))
    value = sub.get("type")
    if value not in ENTITY_TYPES:
        errors.append(ValidationError(f"{path}.type", f"{key}.type must be in {sorted(ENTITY_TYPES)}"))


def validate_triplet(item: Any, path: str = "triplet") -> list[ValidationError]:
    """Return all schema violations for one triplet (empty list = valid)."""
    if not isinstance(item, dict):
        return [ValidationError(path, "triplet must be an object")]

    errors: list[ValidationError] = []
    _check_node(item.get("source"), "source", f"{path}.source", errors)
    _check_node(item.get("target"), "target", f"{path}.target", errors)

    relation = item.get("relation")
    if not isinstance(relation, dict):
        errors.append(ValidationError(f"{path}.relation", "relation must be an object"))
        return errors
    rel_type = relation.get("type")
    if rel_type not in RELATION_TYPES:
        errors.append(
            ValidationError(
                f"{path}.relation.type",
                f"relation must be in the 20-set, got {rel_type!r}",
            )
        )
    weight = relation.get("weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        errors.append(ValidationError(f"{path}.relation.weight", "weight must be a number"))
    elif not 0.0 <= float(weight) <= 1.0:
        errors.append(ValidationError(f"{path}.relation.weight", f"weight must be in [0, 1], got {weight!r}"))
    return errors


def is_valid_triplet(item: Any) -> bool:
    """True when a triplet passes every schema rule (see :func:`validate_triplet`)."""
    return not validate_triplet(item)


def validate_triplet_list(items: Any) -> tuple[bool, list[ValidationError]]:
    """Validate a whole triplets list. ``[]`` is valid (a hard negative)."""
    if not isinstance(items, list):
        return False, [ValidationError("triplets", "must be a JSON array")]
    errors: list[ValidationError] = []
    for i, item in enumerate(items):
        errors.extend(validate_triplet(item, path=f"triplets[{i}]"))
    return not errors, errors


def extract_json_array(raw: Any) -> Any | None:
    """Tolerantly pull the first JSON array out of model output.

    Models wrap the array in markdown fences or prose; this scans for the first
    `[` and decodes exactly one value from there, so arrays inside prose or
    followed by more text are handled. Returns ``None`` when no array is found
    or it does not parse.
    """
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return None
    decoder = json.JSONDecoder()
    start = raw.find("[")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            start = raw.find("[", start + 1)
            continue
        return value if isinstance(value, list) else None
    return None
