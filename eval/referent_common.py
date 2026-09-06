"""Shared helpers for the referent-level eval (eval/referent_pair.py,
eval/referent_eval.py).

A "referent" pairing maps teacher entity indices to student entity indices for
one passage. The teacher gold (outputs/full_eval/test_teacher_with_text.jsonl)
and the student predictions (predictions_full_norm.jsonl) share ``id``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.io import read_jsonl  # noqa: E402

ENTITY_TYPES = frozenset({"person", "organization", "geo", "event", "concept"})

DEFAULT_TEACHER = Path("outputs/full_eval/test_teacher_with_text.jsonl")
DEFAULT_STUDENT = Path("outputs/full_eval/predictions_full_norm.jsonl")
OUT_DIR = Path("outputs/referent_eval")

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (substring-friendly)."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def load_pairs(path: Path) -> dict[str, list[list[int]]]:
    """{id: [[teacher_idx, student_idx], ...]} from a pairings jsonl."""
    out: dict[str, list[list[int]]] = {}
    for row in read_jsonl(path):
        pairs = [[int(a), int(b)] for a, b in row.get("pairs", [])]
        out[row["id"]] = pairs
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def first_index_by_norm(entities: list[dict]) -> dict[str, int]:
    """normalized title -> first list index (entities already deduped upstream)."""
    m: dict[str, int] = {}
    for i, e in enumerate(entities):
        n = normalize(e.get("title"))
        if n and n not in m:
            m[n] = i
    return m


def _split(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize(text)))


def entity_grounded_in_passage(title: str, passage_norm: str, passage_tokens: set[str]) -> bool:
    """Lexical two-anchor grounding for a single title (substring or token overlap)."""
    t = normalize(title)
    if not t:
        return False
    if t in passage_norm:
        return True
    toks = _split(title)
    if not toks:
        return False
    return len(toks & passage_tokens) / len(toks) >= 0.6


def passage_grounding_index(passage: str) -> tuple[str, set[str]]:
    n = normalize(passage)
    return n, _split(passage)
