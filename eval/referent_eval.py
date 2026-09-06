"""Referent-level evaluation of a student extraction against the teacher gold.

Given a referent pairing file (eval/referent_pair.py) this computes, per passage
and pooled across passages:

  entity finding      recall / precision / F1   (teacher vs student nodes)
  edge finding        recall / precision / F1   (relationships, referent endpoints)
  schema compliance   student items valid (type vocab, non-empty, strength in 0-10,
                      endpoints resolve inside the student's own entity list)
  grounding           student extras not grounded in the passage text -> hallucination
  numeric (info)      passage numbers surviving in descriptions

The pairing file indexes the ORIGINAL entity lists of the teacher/student inputs,
so this module also needs those two files. It also bootstraps per-passage micro
metrics to estimate the CI at several sample sizes n (the input to picking the
fixed eval subset for future models).

Usage:
  python eval/referent_eval.py --pairs outputs/referent_eval/pairings_title.jsonl
  python eval/referent_eval.py --pairs outputs/referent_eval/pairings_llm.jsonl --out outputs/referent_eval/report_llm.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.io import read_jsonl  # noqa: E402

from eval.referent_common import (  # noqa: E402
    DEFAULT_STUDENT,
    DEFAULT_TEACHER,
    OUT_DIR,
    ENTITY_TYPES,
    entity_grounded_in_passage,
    first_index_by_norm,
    load_pairs,
    normalize,
    passage_grounding_index,
)

_NUM_RE = re.compile(r"\b\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?\b|\b\d+\.\d+\b|\b\d+%\b")


def _numbers_in(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(0).replace(",", "") for m in _NUM_RE.finditer(text)))


def _desc_text(row: dict) -> str:
    parts = [e.get("description", "") for e in row.get("entities", [])]
    parts += [r.get("description", "") for r in row.get("relationships", [])]
    return " ".join(str(x) for x in parts)


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _resolve(entities: list[dict]) -> dict[str, int]:
    m: dict[str, int] = {}
    for i, e in enumerate(entities):
        n = normalize(e.get("title"))
        if n and n not in m:
            m[n] = i
    return m


def passage_metrics(teacher: dict, student: dict, pairs: list[list[int]]) -> dict:
    te, se = teacher["entities"], student.get("entities", [])
    tr, sr = teacher["relationships"], student.get("relationships", [])

    t2s: dict[int, set[int]] = {}
    s2t: dict[int, set[int]] = {}
    for ti, si in pairs:
        t2s.setdefault(ti, set()).add(si)
        s2t.setdefault(si, set()).add(ti)

    t_matched = len(t2s)
    s_matched = len(s2t)
    n_t = len(te)
    n_s = len(se)

    t_map = _resolve(te)
    s_map = _resolve(se)
    te_edge = [(t_map.get(normalize(r["source"])), t_map.get(normalize(r["target"]))) for r in tr]
    se_edge = [(s_map.get(normalize(r.get("source"))), s_map.get(normalize(r.get("target")))) for r in sr]
    # teacher-side matched edges (numerator of edge recall)
    edge_tp_t = sum(
        1 for ti, tj in te_edge if ti is not None and tj is not None
        and any(si in t2s.get(ti, set()) and sj in t2s.get(tj, set()) for si, sj in se_edge)
    )
    # student-side matched edges (numerator of edge precision)
    edge_tp_s = sum(
        1 for si, sj in se_edge if si is not None and sj is not None
        and any(ti in s2t.get(si, set()) and tj in s2t.get(sj, set()) for ti, tj in te_edge if ti is not None and tj is not None)
    )

    # schema (student side)
    items_total = len(se) + len(sr)
    items_valid = 0
    type_oov = strength_null = empty_desc = placeholder = unresolved_endpoints = 0
    for e in se:
        t = (e.get("title") or "").strip()
        d = (e.get("description") or "").strip()
        typ = normalize(e.get("type") or "")
        ok = True
        if t in ("", "-"):
            placeholder += 1
            ok = False
        if typ not in ENTITY_TYPES:
            type_oov += 1
            ok = False
        if not d:
            empty_desc += 1
            ok = False
        items_valid += ok
    for r in sr:
        d = (r.get("description") or "").strip()
        w = r.get("strength")
        ok = True
        if not isinstance(w, (int, float)) or isinstance(w, bool) or not (0.0 <= float(w) <= 10.0):
            strength_null += 1
            ok = False
        if not d:
            empty_desc += 1
            ok = False
        if normalize(r.get("source")) not in s_map or normalize(r.get("target")) not in s_map:
            unresolved_endpoints += 1
            ok = False
        items_valid += ok

    # grounding: student extras (not paired) checked lexically against the passage
    passage_norm, passage_tokens = passage_grounding_index(teacher["text"])
    ungrounded = novel_grounded = 0
    for i, e in enumerate(se):
        if i in s2t:
            continue
        if entity_grounded_in_passage(e.get("title", ""), passage_norm, passage_tokens):
            novel_grounded += 1
        else:
            ungrounded += 1

    # numeric (info): student vs teacher ratio of passage numbers kept in desc text
    nums = _numbers_in(teacher["text"])
    s_ratio = (sum(1 for n in nums if n in _desc_text(student)) / len(nums)) if nums else 1.0
    t_ratio = (sum(1 for n in nums if n in _desc_text(teacher)) / len(nums)) if nums else 1.0

    entity_recall = t_matched / n_t if n_t else 1.0
    entity_precision = s_matched / n_s if n_s else 0.0
    return {
        "n_teacher_entities": n_t,
        "n_student_entities": n_s,
        "entity_tp": t_matched,
        "entity_recall": round(entity_recall, 4),
        "entity_precision": round(entity_precision, 4),
        "entity_f1": round(_f1(entity_precision, entity_recall), 4),
        "n_teacher_rels": len(tr),
        "n_student_rels": len(sr),
        "edge_tp_t": edge_tp_t,
        "edge_tp_s": edge_tp_s,
        "edge_recall": round(edge_tp_t / len(tr), 4) if tr else 1.0,
        "edge_precision": round(edge_tp_s / len(sr), 4) if sr else 0.0,
        "schema_items": items_total,
        "schema_valid": items_valid,
        "schema_fraction": round(items_valid / items_total, 4) if items_total else None,
        "type_oov": type_oov,
        "strength_null": strength_null,
        "empty_desc": empty_desc,
        "placeholder_titles": placeholder,
        "unresolved_endpoints": unresolved_endpoints,
        "novel_grounded": novel_grounded,
        "ungrounded": ungrounded,
        "hallucination_rate": round(ungrounded / n_s, 4) if n_s else 0.0,
        "numeric_student": round(s_ratio, 4),
        "numeric_teacher": round(t_ratio, 4),
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _safe(a: int, b: int) -> float:
    return a / b if b else 0.0


def _f1c(tp: int, p_den: int, r_den: int) -> float:
    p, r = _safe(tp, p_den), _safe(tp, r_den)
    return _f1(p, r)


def _macro(rows: list[dict], key: str) -> float:
    return round(_mean([r[key] for r in rows]), 4)


def bootstrap_table(rows: list[dict], ns=(50, 100, 150, 200, 300), reps=3000, seed=7) -> dict:
    """Empirical 95% CI half-width of the pooled (micro) entity/edge recall when
    the eval corpus is subsampled to n passages."""
    rng = random.Random(seed)
    t = [r["entity_tp"] for r in rows]
    d = [r["n_teacher_entities"] for r in rows]
    et = [r["edge_tp_t"] for r in rows]
    ed = [r["n_teacher_rels"] for r in rows]
    idx = list(range(len(rows)))
    out: dict[str, list[dict]] = {}
    for n in ns:
        rec_e, rec_g = [], []
        for _ in range(reps):
            s = rng.sample(idx, n)
            rec_e.append(sum(t[i] for i in s) / sum(d[i] for i in s))
            rec_g.append(sum(et[i] for i in s) / sum(ed[i] for i in s))
        import statistics

        out[str(n)] = {
            "entity_recall_ci_half": round(1.96 * statistics.pstdev(rec_e), 4),
            "edge_recall_ci_half": round(1.96 * statistics.pstdev(rec_g), 4),
        }
    return out


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

    rows = []
    for pid in sorted(teacher):
        st = student.get(pid)
        if not st:
            continue
        rec = passage_metrics(teacher[pid], st, pairs_by_id.get(pid, []))
        rec["id"] = pid
        rows.append(rec)

    t_ent = sum(r["n_teacher_entities"] for r in rows)
    s_ent = sum(r["n_student_entities"] for r in rows)
    ent_tp = sum(r["entity_tp"] for r in rows)
    t_rel = sum(r["n_teacher_rels"] for r in rows)
    s_rel = sum(r["n_student_rels"] for r in rows)
    edge_tp_t = sum(r["edge_tp_t"] for r in rows)
    edge_tp_s = sum(r["edge_tp_s"] for r in rows)
    schema_items = sum(r["schema_items"] for r in rows)
    schema_valid = sum(r["schema_valid"] for r in rows)
    hard_hall = sum(r["ungrounded"] for r in rows)
    novel = sum(r["novel_grounded"] for r in rows)
    s_ent_pool = sum(r["n_student_entities"] for r in rows)

    summary = {
        "mode": args.pairs.stem,
        "passages": len(rows),
        "macro": {
            "entity_recall": _macro(rows, "entity_recall"),
            "entity_precision": _macro(rows, "entity_precision"),
            "entity_f1": _macro(rows, "entity_f1"),
            "edge_recall": _macro(rows, "edge_recall"),
            "edge_precision": _macro(rows, "edge_precision"),
        },
        "micro": {
            "entity_recall": round(_safe(ent_tp, t_ent), 4),
            "entity_precision": round(_safe(ent_tp, s_ent), 4),
            "entity_f1": round(_f1c(ent_tp, s_ent, t_ent), 4),
            "edge_recall": round(_safe(edge_tp_t, t_rel), 4),
            "edge_precision": round(_safe(edge_tp_s, s_rel), 4),
            "edge_f1": round(_f1(_safe(edge_tp_s, s_rel), _safe(edge_tp_t, t_rel)), 4),
        },
        "totals": {
            "teacher_entities": t_ent,
            "student_entities": s_ent,
            "teacher_rels": t_rel,
            "student_rels": s_rel,
            "edge_tp_teacher_side": edge_tp_t,
            "edge_tp_student_side": edge_tp_s,
        },
        "schema": {
            "valid_fraction": round(_safe(schema_valid, schema_items), 4),
            "items": schema_items,
            "type_oov": sum(r["type_oov"] for r in rows),
            "strength_null": sum(r["strength_null"] for r in rows),
            "empty_desc": sum(r["empty_desc"] for r in rows),
            "placeholder_titles": sum(r["placeholder_titles"] for r in rows),
            "unresolved_endpoints": sum(r["unresolved_endpoints"] for r in rows),
        },
        "grounding": {
            "hard_hallucinations": hard_hall,
            "novel_grounded": novel,
            "hallucination_rate": round(_safe(hard_hall, s_ent_pool), 4),
        },
        "numeric_info": {
            "student_mean_ratio": round(_mean([r["numeric_student"] for r in rows]), 4),
            "teacher_mean_ratio": round(_mean([r["numeric_teacher"] for r in rows]), 4),
        },
        "bootstrap_ci_halfwidth": bootstrap_table(rows),
    }

    args.out = args.out or OUT_DIR / f"report_{args.pairs.stem}.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_passage": rows}, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
