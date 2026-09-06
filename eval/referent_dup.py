"""Duplication / entity-granularity metrics from a referent pairing.

Derived, API-free quality axes reported alongside the referent finding axes:

  gold split    a teacher (gold) entity matched to >=2 student entities — the
                model split one referent into several nodes (under-dedup).
                NOTE: the referent matcher forces ~1:1 mapping, so a split only
                shows when the judge found two student nodes for one gold; the
                observable split rate is therefore a lower bound.
  gold merge    one student entity matched to >=2 teacher entities — the model
                merged several distinct gold referents into one node. Allowed
                only when the node is an explicit combined node.
  student dup   student entities within one passage sharing a normalized title
                (post-extraction dedup failure).

Synonym-level redundancy (different titles, one referent) is NOT measured here:
such unmatched nodes surface as entity-precision cost, and a title-overlap
heuristic for them was tried and rejected as too noisy on CONCEPT clusters.

Usage:
  python eval/referent_dup.py --pairs outputs/referent_eval/pairings_think_699.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.io import read_jsonl  # noqa: E402

from eval.referent_common import (  # noqa: E402
    DEFAULT_STUDENT,
    DEFAULT_TEACHER,
    OUT_DIR,
    load_pairs,
    normalize,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    teacher = {r["id"]: r for r in read_jsonl(args.teacher)}
    student = {r["id"]: r for r in read_jsonl(args.student)}
    pairs_by_id = load_pairs(args.pairs)

    per_passage = []
    agg = {"gold_split_entities": 0, "gold_merge_entities": 0, "student_dup_pairs": 0,
           "passages": 0}
    for pid in sorted(teacher):
        st = student.get(pid)
        if not st:
            continue
        te, se = teacher[pid]["entities"], st.get("entities", [])
        t2s: dict[int, list[int]] = {}
        s2t: dict[int, list[int]] = {}
        for ti, si in pairs_by_id.get(pid, []):
            t2s.setdefault(ti, []).append(si)
            s2t.setdefault(si, []).append(ti)
        g_split = sum(1 for v in t2s.values() if len(v) > 1)
        g_merge = sum(1 for v in s2t.values() if len(v) > 1)
        seen_norm: dict[str, list[int]] = {}
        for i, e in enumerate(se):
            n = normalize(e.get("title"))
            if n:
                seen_norm.setdefault(n, []).append(i)
        s_dup = sum(len(v) - 1 for v in seen_norm.values() if len(v) > 1)
        rec = {"id": pid, "n_teacher_entities": len(te), "n_student_entities": len(se),
               "gold_split_entities": g_split, "gold_merge_entities": g_merge,
               "student_dup_pairs": s_dup}
        per_passage.append(rec)
        agg["gold_split_entities"] += g_split
        agg["gold_merge_entities"] += g_merge
        agg["student_dup_pairs"] += s_dup
        agg["passages"] += 1

    n_gold = sum(r["n_teacher_entities"] for r in per_passage)
    n_stu = sum(r["n_student_entities"] for r in per_passage)
    split_pp = sum(1 for r in per_passage if r["gold_split_entities"])
    merge_pp = sum(1 for r in per_passage if r["gold_merge_entities"])
    summary = {
        "mode": args.pairs.stem,
        "passages": agg["passages"],
        "gold_split": {
            "teacher_entities_affected": agg["gold_split_entities"],
            "rate_of_teacher_entities": round(agg["gold_split_entities"] / n_gold, 4) if n_gold else 0.0,
            "passages_affected": split_pp,
            "passage_rate": round(split_pp / agg["passages"], 4),
        },
        "gold_merge": {
            "teacher_entities_affected": agg["gold_merge_entities"],
            "rate_of_teacher_entities": round(agg["gold_merge_entities"] / n_gold, 4) if n_gold else 0.0,
            "passages_affected": merge_pp,
            "passage_rate": round(merge_pp / agg["passages"], 4),
        },
        "student_dup": {
            "extra_dup_entities": agg["student_dup_pairs"],
            "rate_of_student_entities": round(agg["student_dup_pairs"] / n_stu, 4) if n_stu else 0.0,
        },
        "max_per_passage": {k: max((r[k] for r in per_passage), default=0)
                            for k in ("gold_split_entities", "gold_merge_entities", "student_dup_pairs")},
    }

    args.out = args.out or OUT_DIR / f"dup_{args.pairs.stem}.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_passage": per_passage}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
