"""Tracer evaluation: compare the fine-tuned student's GraphRAG-format output
against the teacher reference on the 10 tracer-eval passages.

Three axes:
1. Entity coverage (objective): normalized-title recall/precision of the
   student's entities against the teacher's.
2. Numeric preservation (objective): which numbers (dates/counts/measurements)
   that appear in the passage survive in the student's descriptions vs the
   teacher's.
3. Faithfulness (LLM-judged): not computed here — the script dumps the paired
   student/teacher outputs to `outdir/pairs/` for a blind judge.

Usage:
  python eval/tracer_eval.py --pred predictions.jsonl --teacher teacher_ref.jsonl --out outputs/tracer/eval_report.json

pred row:   {"id", "entities": [...], "relationships": [...]}   (parsed student output)
teacher row: {"id", "entities": [...], "relationships": [...]}   (teacher reference, fixed config)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.io import read_jsonl  # noqa: E402
from dataset.semantic import normalize  # noqa: E402

_NUM_RE = re.compile(r"\b\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?\b|\b\d+\.\d+\b|\b\d+%\b")


def _numbers_in(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(0).replace(",", "") for m in _NUM_RE.finditer(text)))


def _desc_text(row: dict) -> str:
    parts = [e["description"] for e in row.get("entities", [])]
    parts += [r["description"] for r in row.get("relationships", [])]
    return " ".join(parts)


def _entity_titles(row: dict) -> set[str]:
    return {normalize(e["title"]) for e in row.get("entities", [])}


def _numeric_report(passage: str, row: dict) -> dict:
    nums = _numbers_in(passage)
    desc = _desc_text(row)
    kept = [n for n in nums if n in desc]
    return {"passage_numbers": len(nums), "kept": len(kept), "ratio": len(kept) / len(nums) if nums else 1.0,
            "kept_numbers": kept[:20], "dropped_numbers": [n for n in nums if n not in kept][:20]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/tracer/eval_report.json"))
    parser.add_argument("--pairs-dir", type=Path, default=Path("outputs/tracer/pairs"))
    args = parser.parse_args(argv)

    pred = {r["id"]: r for r in read_jsonl(args.pred)}
    teacher = {r["id"]: r for r in read_jsonl(args.teacher)}
    common = [i for i in teacher if i in pred]

    report = []
    args.pairs_dir.mkdir(parents=True, exist_ok=True)
    for pid in sorted(common):
        t, p = teacher[pid], pred[pid]
        t_titles, p_titles = _entity_titles(t), _entity_titles(p)
        overlap = t_titles & p_titles
        recall = len(overlap) / len(t_titles) if t_titles else 1.0
        precision = len(overlap) / len(p_titles) if p_titles else 0.0
        tn = _numeric_report(t["text"], t)
        pn = _numeric_report(t["text"], p)
        (args.pairs_dir / f"{pid}.json").write_text(
            json.dumps({"id": pid, "teacher": t, "student": p}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        report.append(
            {
                "id": pid,
                "teacher_entities": len(t_titles),
                "student_entities": len(p_titles),
                "entity_recall": round(recall, 3),
                "entity_precision": round(precision, 3),
                "numeric_teacher": tn,
                "numeric_student": pn,
            }
        )

    avg_recall = sum(r["entity_recall"] for r in report) / len(report)
    avg_precision = sum(r["entity_precision"] for r in report) / len(report)
    avg_num = sum(r["numeric_student"]["ratio"] for r in report) / len(report)
    summary = {
        "passages": len(report),
        "avg_entity_recall": round(avg_recall, 3),
        "avg_entity_precision": round(avg_precision, 3),
        "avg_numeric_ratio": round(avg_num, 3),
        "note": "faithfulness not computed here; per-passage student/teacher pairs written for a blind judge",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_passage": report}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"pairs -> {args.pairs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
