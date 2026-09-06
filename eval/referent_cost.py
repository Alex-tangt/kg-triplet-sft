"""Cost-reduction experiment: compare no-thinking pairing schemes against the
thinking-mode gold, and synthesize two-vote agree-only pairings.

Schemes (each a pairings file like eval/referent_pair.py --mode llm produces):
  single      one judge call per passage (variants default / hard / strict)
  agree(A,B)  intersection of two votes' pair sets (1:1 tier-1 pairs are
              identical across votes, so only LLM-tier disagreements are cut)

Reported per scheme, against the gold (thinking-mode) pairings on the same 100
passages:
  passage diff      how many passages' final pair set differs from gold
  pair FP / FN      same-pair decisions that gold does not / gold ones missed
  micro P/R vs gold (of the "is-same" decision, exact title tier counted as hits)
  guard regression  passages where default-nothink == gold must stay == gold
                    under every scheme (ids100 'consistent 20')
  probe accuracy    the 12 known-answer identity pairs in ids100.json
  cost              summed prompt/completion tokens from the runs that captured
                    usage + wall time

Usage:
  python eval/referent_cost.py --gold outputs/referent_eval/pairings_think_v2_100.jsonl \
    --scheme v0_default=outputs/referent_eval/nothink_v2_100.jsonl \
    --scheme hard=outputs/referent_eval/cost_hard_100.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.referent_common import OUT_DIR, load_pairs  # noqa: E402


def _sum_cost(rows: list[dict]) -> dict:
    pt = sum(int(r["tokens"].get("prompt_tokens", 0) or 0) for r in rows if r.get("tokens"))
    ct = sum(int(r["tokens"].get("completion_tokens", 0) or 0) for r in rows if r.get("tokens"))
    attempts = sum(r.get("attempts", 0) for r in rows)
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct, "attempts": attempts}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, default=OUT_DIR / "pairings_think_v2_100.jsonl")
    p.add_argument("--ids", type=Path, default=OUT_DIR / "ids100.json")
    p.add_argument("--scheme", action="append", type=str, metavar="NAME=PATH", default=[])
    p.add_argument("--agree", action="append", type=str, metavar="NAME=PATH1,PATH2", default=[])
    args = p.parse_args(argv)

    gold = load_pairs(args.gold)
    meta = json.loads(args.ids.read_text(encoding="utf-8"))
    probe = meta["probe"]
    base = load_pairs(next(Path(x.split("=", 1)[1]) for x in args.scheme if x.startswith("v0") or x.startswith("default")))

    def _as_set(pairs) -> set:
        return {tuple(x) for x in pairs}

    guard = sorted(i for i in base if i in gold and _as_set(base[i]) == _as_set(gold[i]))

    schemes: dict[str, dict[str, list]] = {}
    for spec in args.scheme:
        name, path = spec.split("=", 1)
        schemes[name] = load_pairs(Path(path))
    for spec in args.agree:
        name, paths = spec.split("=", 1)
        a, b = (Path(x) for x in paths.split(","))
        pa, pb = load_pairs(a), load_pairs(b)
        schemes[name] = {pid: sorted(set(tuple(x) for x in pa.get(pid, [])) & set(tuple(x) for x in pb.get(pid, []))) for pid in gold}

    out: dict = {"guard_n": len(guard), "probe_n": len(probe)}
    for name, pairs in schemes.items():
        diff_passages = 0
        fp = fn = hit_pairs = 0
        for pid in gold:
            g = _as_set(gold[pid])
            s = _as_set(pairs.get(pid, []))
            fp += len(s - g)
            fn += len(g - s)
            hit_pairs += len(s & g)
            if g != s:
                diff_passages += 1
        micro_p = round(hit_pairs / (hit_pairs + fp), 4) if (hit_pairs + fp) else 1.0
        micro_r = round(hit_pairs / (hit_pairs + fn), 4) if (hit_pairs + fn) else 1.0
        guard_regress = sum(1 for pid in guard if _as_set(pairs.get(pid, [])) != _as_set(gold[pid]))
        probe_right = sum(1 for pr in probe if ([pr["a"], pr["b"]] in pairs.get(pr["pid"], [])) == pr["same"])
        out[name] = {
            "passage_diff_vs_gold": diff_passages,
            "pair_fp": fp,
            "pair_fn": fn,
            "pair_micro_precision_vs_gold": micro_p,
            "pair_micro_recall_vs_gold": micro_r,
            "guard_passages": guard_regress,
            "probe": f"{probe_right}/{len(probe)}",
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
