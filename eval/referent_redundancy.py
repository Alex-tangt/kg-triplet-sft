"""Redundancy / invalid-item diagnostic for a student extraction.

Complements referent_dup (which needs a referent pairing) with API-free raw-output
checks. Per passage over the requested ids:

  volume           entity/relationship counts (mean, p90, max per passage)
  exact dup        student entities sharing a normalized title within one passage
  empty title      entities with a blank or '-' title (a placeholder node, not usable)
  no-letter title  entities whose title has no alphabetic character
  oov type         entities typed outside the ENTITY_TYPES vocabulary
  verbosity        description character stats (median / p95 / max)
  dangling edge    relationships whose source/target normalized title is NOT among
                   the passage's own entity titles (the consumer graph cannot join)
  self loop        source == target
  dup edge         same (source, target) key emitted more than once
  type variety     distinct type strings across the whole run + top generic ones

Usage:
  python eval/referent_redundancy.py --student outputs/capacity_eval/qwen3-0.6b-plain/predictions.jsonl --ids ...ids
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.io import read_jsonl  # noqa: E402

from eval.referent_common import ENTITY_TYPES, OUT_DIR, normalize  # noqa: E402

_JUNK = re.compile(r"^[^a-zA-Z]{1,3}$")


def passage_report(row: dict) -> dict:
    se = row.get("entities", [])
    sr = row.get("relationships", [])
    groups: dict[str, int] = {}
    types = Counter()
    empty_title = no_letter = oov_type = 0
    desc_lens: list[int] = []
    for e in se:
        raw_t = (e.get("title") or "").strip()
        typ = (e.get("type") or "").strip()
        types[typ or "(none)"] += 1
        n = normalize(raw_t) or ""
        if not raw_t or raw_t == "-":
            empty_title += 1
        elif not n or _JUNK.match(n):
            no_letter += 1
        if normalize(typ) not in ENTITY_TYPES:
            oov_type += 1
        if n:
            groups[n] = groups.get(n, 0) + 1
        d = (e.get("description") or "").strip()
        if d:
            desc_lens.append(len(d))
    dup = sum(v - 1 for v in groups.values() if v > 1)
    titles_norm = {normalize(e.get("title")) for e in se if normalize(e.get("title"))}
    dangling = self_loop = dup_edge = 0
    seen_edges: set = set()
    for x in sr:
        s, t = normalize(x.get("source")) or "", normalize(x.get("target")) or ""
        if not s or not t or s not in titles_norm or t not in titles_norm:
            dangling += 1
        if s and t and s == t:
            self_loop += 1
        key = (s, t)
        if key in seen_edges:
            dup_edge += 1
        seen_edges.add(key)
    return {
        "entities": len(se), "rels": len(sr),
        "exact_dup_extra": dup, "empty_titles": empty_title, "no_letter_titles": no_letter,
        "oov_type_entities": oov_type,
        "desc_median": int(statistics.median(desc_lens)) if desc_lens else 0,
        "desc_p95": int(sorted(desc_lens)[int(0.95 * len(desc_lens)) - 1]) if desc_lens else 0,
        "desc_max": max(desc_lens) if desc_lens else 0,
        "dangling_edges": dangling, "self_loops": self_loop, "dup_edges": dup_edge,
        "type_counts": dict(types),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", type=Path, default=None, required=True)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    rows = [r for r in read_jsonl(args.student) if not args.ids or r["id"] in set(args.ids)]
    reps = [passage_report(r) for r in rows]
    n = len(reps)

    def mean(key):
        return round(sum(r[key] for r in reps) / n, 2) if n else 0

    def pct(key, frac):
        vals = sorted(r[key] for r in reps)
        return vals[min(len(vals) - 1, int(frac * len(vals)))]

    tot_e = sum(r["entities"] for r in reps)
    tot_r = sum(r["rels"] for r in reps)
    all_types = Counter()
    for r in reps:
        all_types.update(r["type_counts"])
    summary = {
        "passages": n,
        "volume_per_passage": {"entities_mean": mean("entities"), "entities_p90": pct("entities", 0.9),
                               "entities_max": max(r["entities"] for r in reps),
                               "rels_mean": mean("rels"), "rels_max": max(r["rels"] for r in reps)},
        "exact_dup": {"extra_entities": sum(r["exact_dup_extra"] for r in reps),
                      "rate_of_student_entities": round(sum(r["exact_dup_extra"] for r in reps) / tot_e, 4) if tot_e else 0},
        "empty_titles": {"count": sum(r["empty_titles"] for r in reps),
                         "rate_of_student_entities": round(sum(r["empty_titles"] for r in reps) / tot_e, 4) if tot_e else 0},
        "no_letter_titles": {"count": sum(r["no_letter_titles"] for r in reps),
                             "rate_of_student_entities": round(sum(r["no_letter_titles"] for r in reps) / tot_e, 4) if tot_e else 0},
        "oov_type_entities": {"count": sum(r["oov_type_entities"] for r in reps),
                              "rate_of_student_entities": round(sum(r["oov_type_entities"] for r in reps) / tot_e, 4) if tot_e else 0},
        "verbosity": {"desc_median": int(statistics.median(r["desc_median"] for r in reps if r["desc_median"])),
                      "desc_p95": max(r["desc_p95"] for r in reps)},
        "dangling_edges": {"count": sum(r["dangling_edges"] for r in reps),
                           "rate_of_rels": round(sum(r["dangling_edges"] for r in reps) / tot_r, 4) if tot_r else 0},
        "self_loops": sum(r["self_loops"] for r in reps),
        "dup_edges": sum(r["dup_edges"] for r in reps),
        "type_variety": {"distinct_types_pooled": len(all_types),
                         "top_shared": all_types.most_common(5)},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.out:
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
