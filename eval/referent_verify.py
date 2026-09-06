"""Second-opinion verification of the referent tier-2 (fuzzy) pairs.

Every fuzzy pair (teacher entity index, student entity index) the pairing step
decided is re-judged independently: is this the SAME real-world entity? A "no"
verdict marks a likely over-merge (two distinct concepts glued together), which
inflates the referent recall/precision headline.

  python eval/referent_verify.py --sample 145 --seed 1
      -> outputs/referent_eval/verify_results.jsonl  (one row per judged pair)
      -> outputs/referent_eval/verify_report.json    (aggregates + sensitivity)

Caveat: same-model re-judging measures consistency, not ground truth; treat the
"no" rate as an upper bound on tier-2 over-merge error and eyeball samples.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.io import read_jsonl  # noqa: E402
from dataset.teacher import http_transport  # noqa: E402

from eval.referent_common import (  # noqa: E402
    DEFAULT_STUDENT,
    DEFAULT_TEACHER,
    OUT_DIR,
    normalize,
)
from eval.referent_pair import IDENTITY_RUBRIC, _api_key, _entry_line  # noqa: E402

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.7-flash"
SYSTEM_PROMPT = (
    "You are an independent second judge. Two entities were extracted from the "
    "passage below; decide whether they denote the SAME real-world entity.\n"
    "The passage is evidence for DISTINGUISHING referents, never a reason to merge.\n"
    + IDENTITY_RUBRIC + "\n"
    'Reply with ONLY a JSON object: {"same": true}  or  {"same": false}'
)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _parse(raw: str) -> bool | None:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    same = obj.get("same")
    if isinstance(same, bool):
        return same
    return None


def _pair_content(passage: str, t: dict, s: dict, desc_len: int) -> str:
    a = _entry_line(0, t, desc_len).replace("[0] ", "[A] ")
    b = _entry_line(0, s, desc_len).replace("[0] ", "[B] ")
    head = f"PASSAGE:\n{passage[:900]}\n\n" if passage else ""
    return head + f"Entity A (teacher):\n{a}\n\nEntity B (student):\n{b}\n\nReturn the JSON object."


def collect_fuzzy(teacher: dict, student: dict, pairs: list[list[int]]) -> list[tuple[int, int]]:
    se = student.get("entities", [])
    return [
        (a, b) for a, b in pairs
        if normalize(teacher["entities"][a]["title"]) != normalize(se[b].get("title", ""))
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    p.add_argument("--pairs", type=Path, default=OUT_DIR / "pairings_llm.jsonl")
    p.add_argument("--out", type=Path, default=OUT_DIR / "verify_results.jsonl")
    p.add_argument("--report", type=Path, default=OUT_DIR / "verify_report.json")
    p.add_argument("--sample", type=int, default=200, help="max fuzzy pairs to verify (seeded)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--desc-len", type=int, default=300)
    args = p.parse_args(argv)

    teacher = {r["id"]: r for r in read_jsonl(args.teacher)}
    student = {r["id"]: r for r in read_jsonl(args.student)}
    pairs_by_id: dict[str, list[list[int]]] = {}
    for r in read_jsonl(args.pairs):
        if r.get("status") == "ok":
            pairs_by_id[r["id"]] = r["pairs"]

    # gather fuzzy candidates across all passages, seeded sample
    cands: list[tuple[str, int, int]] = []
    for pid in sorted(teacher):
        if pid not in student or pid not in pairs_by_id:
            continue
        t, s = teacher[pid], student[pid]
        for a, b in collect_fuzzy(t, s, pairs_by_id[pid]):
            cands.append((pid, a, b))
    rng = random.Random(args.seed)
    rng.shuffle(cands)
    sample = cands[: args.sample]
    print(f"fuzzy pairs available: {len(cands)}; verifying {len(sample)}")

    if not sample:
        print("nothing to verify")
        return 0

    tr = http_transport(model=MODEL, base_url=BASE_URL, api_key=_api_key(),
                        temperature=0.0, max_tokens=300, timeout=60, enable_thinking=False)
    lock = threading.Lock()
    done = 0
    results: dict[str, list[dict]] = {}
    start = time.time()

    def work(pid: str, a: int, b: int) -> None:
        nonlocal done
        rec = {"pid": pid, "a": a, "b": b}
        for attempt in range(3):
            try:
                raw = tr([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _pair_content(teacher[pid].get("text", ""), teacher[pid]["entities"][a], student[pid]["entities"][b], args.desc_len)},
                ])
                same = _parse(raw)
                if same is None:
                    raise ValueError("unparseable")
                rec.update({"same": same, "attempts": attempt + 1})
                break
            except Exception as exc:
                if attempt == 2:
                    rec.update({"same": None, "reason": str(exc)[:200], "attempts": 3})
        with lock:
            done += 1
            results.setdefault(pid, []).append(rec)
            if rec["same"] is None:
                tag = "ERR"
            else:
                tag = "same" if rec["same"] else "DIFF"
            print(f"[{done}/{len(sample)}] {pid} {a}<->{b} {tag}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, pid, a, b) for pid, a, b in sample]
        for f in futures:
            f.result()

    rows = [r for pid, a, b in sample for r in results.get(pid, [])]
    judged = [r for r in rows if r["same"] is not None]
    n_yes = sum(1 for r in judged if r["same"])
    n_no = sum(1 for r in judged if not r["same"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "fuzzy_available": len(cands),
        "verified": len(rows),
        "judged": len(judged),
        "yes": n_yes,
        "no": n_no,
        "parse_fail": len(rows) - len(judged),
        "no_rate": round(n_no / len(judged), 4) if judged else None,
        "note": "no = likely over-merge (upper bound on tier-2 error); same-model second opinion, not ground truth",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
