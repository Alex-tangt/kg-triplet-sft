"""Draw and archive the fixed evaluation sample for future candidate models.

Sample size n is decided from the reference model's per-passage variance
(eval/referent_eval.py bootstrap table); this script archives the exact id list
so every future model is scored on the SAME passages.

Selection: test passages (from --teacher, 699 with labels) stratified by corpus
(wikipedia ~60% / arxiv ~40%) and passage-length terciles within each source,
fixed seed.

The default run (no --dry-run) archives the sample. Pass --passages-out to also
export the sample passages (id + text) for running candidate-model inference.

Usage:
  python eval/referent_sample.py --n 200 --seed 20260906 --dry-run
  python eval/referent_sample.py --n 200 --seed 20260906
  python eval/referent_sample.py --n 200 --seed 20260906 --passages-out outputs/referent_eval/sample_200_passages.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.io import read_jsonl  # noqa: E402

from eval.referent_common import DEFAULT_TEACHER, OUT_DIR  # noqa: E402


def _length_strata(items: list[dict], k: int, rng: random.Random) -> list[str]:
    counts = sorted(i["len"] for i in items)
    lo = counts[len(counts) // 3]
    hi = counts[2 * len(counts) // 3]
    buckets: dict[str, list[str]] = {"lo": [], "mid": [], "hi": []}
    for i in items:
        b = "lo" if i["len"] <= lo else "mid" if i["len"] <= hi else "hi"
        buckets[b].append(i["id"])
    drawn: list[str] = []
    per, rem = k // 3, k % 3
    for idx, name in enumerate(("lo", "mid", "hi")):
        want = per + (1 if idx < rem else 0)
        rng.shuffle(buckets[name])
        drawn.extend(buckets[name][:want])
    return drawn


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--passages-out", type=Path, default=None, help="also export sample passages (id,text) for inference")
    args = p.parse_args(argv)

    rows = read_jsonl(args.teacher)
    items = [{"id": r["id"], "src": "wikipedia" if r["id"].startswith("wikipedia") else "arxiv",
              "len": len(r.get("text") or "")} for r in rows]
    rng = random.Random(args.seed)
    by_src: dict[str, list[dict]] = {}
    for it in items:
        by_src.setdefault(it["src"], []).append(it)

    n_wp = round(args.n * 0.6)
    n_ar = args.n - n_wp
    ids = _length_strata(by_src.get("wikipedia", []), min(n_wp, len(by_src.get("wikipedia", []))), rng)
    ids += _length_strata(by_src.get("arxiv", []), min(n_ar, len(by_src.get("arxiv", []))), rng)
    rng.shuffle(ids)
    ids = sorted(ids)

    meta = {
        "n": len(ids),
        "seed": args.seed,
        "method": "stratified: corpus 60/40 + passage-length terciles within source, fixed seed",
        "source_counts": {s: sum(1 for i in ids if i.startswith("wikipedia") == (s == "wikipedia")) for s in ("wikipedia", "arxiv")},
        "teacher_file": str(args.teacher),
    }
    if args.dry_run:
        print(json.dumps({**meta, "first_ids": ids[:10]}, ensure_ascii=False, indent=1))
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"sample_{args.n}.json"
    out.write_text(json.dumps({**meta, "ids": ids}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"archived {len(ids)}-passage sample -> {out}")
    print(f"  {meta['source_counts']}  (requested {n_wp} wiki / {n_ar} arxiv)")
    if args.passages_out:
        by_id = {r["id"]: r for r in rows}
        args.passages_out.parent.mkdir(parents=True, exist_ok=True)
        with args.passages_out.open("w", encoding="utf-8") as fh:
            for i in ids:
                r = by_id[i]
                fh.write(json.dumps({"id": r["id"], "text": r.get("text", "")}, ensure_ascii=False) + "\n")
        print(f"exported {len(ids)} passages for inference -> {args.passages_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
