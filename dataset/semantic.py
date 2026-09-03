"""Semantic label validation beyond the schema: weight anchors and grounding.

The schema validator (``kg_contract.validator``) rejects structurally invalid
labels; this module applies the rubric rules that are too cheap to defer to a
human — every weight must be exactly one of the fixed anchors, and every entity
title must be grounded (as a normalized substring) inside its passage chunk.
Violations are reported, not silently dropped, so the pilot report can surface
how often the teacher disobeys the rubric.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from kg_contract import WEIGHT_ANCHORS

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for substring checks."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class Violation:
    """One rubric violation, located by ``triplet_index`` and ``rule``."""

    triplet_index: int
    rule: str  # "weight" | "grounding"
    path: str
    detail: str


def check_weight_anchors(triplets: list[dict]) -> list[Violation]:
    violations: list[Violation] = []
    for i, t in enumerate(triplets):
        weight = (t.get("relation") or {}).get("weight")
        if not any(abs(weight - anchor) < 1e-9 for anchor in WEIGHT_ANCHORS):
            violations.append(
                Violation(
                    triplet_index=i,
                    rule="weight",
                    path=f"triplets[{i}].relation.weight",
                    detail=f"weight {weight!r} not in {WEIGHT_ANCHORS}",
                )
            )
    return violations


def _entity_titles(t: dict) -> list[tuple[str, str]]:
    titles: list[tuple[str, str]] = []
    for side in ("source", "target"):
        node = t.get(side)
        if isinstance(node, dict) and isinstance(node.get("title"), str):
            titles.append((side, node["title"]))
    return titles


def check_grounding(triplets: list[dict], passage: str) -> list[Violation]:
    norm_passage = normalize(passage)
    violations: list[Violation] = []
    for i, t in enumerate(triplets):
        for side, title in _entity_titles(t):
            norm_title = normalize(title)
            if not norm_title:
                continue
            if norm_title not in norm_passage:
                violations.append(
                    Violation(
                        triplet_index=i,
                        rule="grounding",
                        path=f"triplets[{i}].{side}.title",
                        detail=f"title {title!r} not grounded in passage",
                    )
                )
    return violations


def check_semantics(triplets: list[dict], passage: str) -> list[Violation]:
    return check_weight_anchors(triplets) + check_grounding(triplets, passage)


def violation_counts(violations: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in violations:
        rule = v["rule"] if isinstance(v, dict) else v.rule
        counts[rule] = counts.get(rule, 0) + 1
    return counts
