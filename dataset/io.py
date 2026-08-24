"""Shared file I/O helpers for the dataset stage scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    written = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def word_distribution(counts: list[int]) -> dict:
    if not counts:
        return {"min": 0, "max": 0, "mean": 0}
    return {"min": min(counts), "max": max(counts), "mean": round(sum(counts) / len(counts), 1)}
