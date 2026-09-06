"""A labeled benchmark for the entity-pair identity decision.

Builds a fixed evaluation set of (teacher entity, student entity, passage) pairs
and labels them with a STRONG judge (thinking on + detailed rubric + passage +
reason), then scores a FAST production config (thinking off) against it so the
fast path's accuracy has a measured basis instead of ad-hoc sampling.

Stages:
  python eval/referent_bench.py --stage sample        # tasks.jsonl (kind: proposed/random)
  python eval/referent_bench.py --stage label         # labels.jsonl (thinking judge)
  python eval/referent_bench.py --stage eval-fast     # confusion vs labels + disagreement dump
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
from eval.referent_verify import SYSTEM_PROMPT as VERIFY_PROMPT  # noqa: E402

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.7-flash"

STRONG_PROMPT = (
    "You are a careful reference judge for entity resolution. Decide whether the "
    "two entities below denote the SAME real-world object/individual/concept "
    "within the passage. Reason from the passage and both descriptions.\n"
    "The passage is evidence for DISTINGUISHING referents, never a reason to merge.\n"
    + IDENTITY_RUBRIC + "\n"
    'Reply with ONLY a JSON object: {"same": true|false, "confidence": '
    '"high"|"medium"|"low", "reason": "<one sentence citing the decisive evidence>"}'
)


def _pair_block(passage: str, t: dict, s: dict, desc_len: int) -> str:
    a = _entry_line(0, t, desc_len).replace("[0] ", "Entity A (teacher):\n")
    b = _entry_line(0, s, desc_len).replace("[0] ", "Entity B (student):\n")
    head = f"PASSAGE:\n{passage[:1000]}\n\n" if passage else ""
    return head + a + "\n" + b + "\n\nReturn the JSON object."


def _parse(raw: str) -> dict | None:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(obj.get("same"), bool):
        return obj
    return None


def stage_sample(args) -> None:
    teacher = {r["id"]: r for r in read_jsonl(args.teacher)}
    student = {r["id"]: r for r in read_jsonl(args.student)}
    pairings = {}
    for r in read_jsonl(args.pairs):
        if r.get("status") == "ok":
            pairings[r["id"]] = r["pairs"]
    rng = random.Random(args.seed)
    proposed = []
    random_neg = []
    for pid in sorted(teacher):
        te = teacher[pid]["entities"]
        se = student[pid]["entities"]
        tn = {normalize(e.get("title")): i for i, e in enumerate(te) if normalize(e.get("title"))}
        sn = {normalize(e.get("title")): i for i, e in enumerate(se) if normalize(e.get("title"))}
        prop_set = {(a, b) for a, b in pairings.get(pid, [])}
        for a, b in prop_set:
            if normalize(te[a]["title"]) != normalize(se[b].get("title", "")):
                proposed.append((pid, a, b))
        # random teacher x student pairs that were NOT proposed and titles differ
        cand = [(a, b) for a, b in ((ti, si) for ti in range(len(te)) for si in range(len(se)))
                if (a, b) not in prop_set and normalize(te[a]["title"]) != normalize(se[b].get("title", ""))]
        if cand:
            for _ in range(min(3, len(cand))):
                random_neg.append((pid,) + rng.choice(cand))
    rng.shuffle(proposed)
    rng.shuffle(random_neg)
    n_prop = min(args.n_proposed, len(proposed))
    n_rand = min(args.n_random, len(random_neg))
    tasks = [{"pid": p, "a": a, "b": b, "kind": "proposed"} for p, a, b in proposed[:n_prop]]
    tasks += [{"pid": p, "a": a, "b": b, "kind": "random"} for p, a, b in random_neg[:n_rand]]
    args.tasks.parent.mkdir(parents=True, exist_ok=True)
    with args.tasks.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"bench tasks: {len(tasks)} (proposed {n_prop}, random {n_rand}) -> {args.tasks}")


def _runner(args, prompt, thinking: bool, extract, max_tokens=2048):
    teacher = {r["id"]: r for r in read_jsonl(args.teacher)}
    student = {r["id"]: r for r in read_jsonl(args.student)}
    tasks = [json.loads(l) for l in args.tasks.read_text(encoding="utf-8").splitlines() if l.strip()]
    existing = [json.loads(l) for l in args.labels.read_text(encoding="utf-8").splitlines()] if args.labels.exists() else []
    done_ids = {(r["pid"], r["a"], r["b"]) for r in existing if r.get("status") == "ok"}
    todo = [t for t in tasks if (t["pid"], t["a"], t["b"]) not in done_ids]
    print(f"label: {len(tasks)} tasks, {len(todo)} to run ({len(tasks) - len(todo)} done)")

    tr = http_transport(model=MODEL, base_url=BASE_URL, api_key=_api_key(),
                        temperature=0.0, max_tokens=max_tokens, timeout=120, enable_thinking=thinking)
    lock = threading.Lock()
    done = 0
    start = time.time()

    def work(t: dict) -> None:
        nonlocal done
        pid, a, b = t["pid"], t["a"], t["b"]
        rec = {"pid": pid, "a": a, "b": b, "kind": t.get("kind"), "status": "failed"}
        for attempt in range(3):
            try:
                content = _pair_block(teacher[pid].get("text", ""), teacher[pid]["entities"][a],
                                      student[pid]["entities"][b], args.desc_len)
                raw = tr([{"role": "system", "content": prompt},
                          {"role": "user", "content": content}])
                obj = extract(raw)
                if obj is None:
                    raise ValueError("unparseable")
                rec.update({"status": "ok", "same": obj.get("same"),
                            "confidence": obj.get("confidence"), "reason": obj.get("reason", ""),
                            "attempts": attempt + 1})
                break
            except Exception as exc:
                if attempt == 2:
                    rec["reason"] = str(exc)[:200]
        with lock:
            done += 1
            with args.labels.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["status"] == "ok":
                tag = "same" if rec["same"] else "DIFF"
            else:
                tag = "ERR"
            print(f"[{done}/{len(todo)}] {pid} {a}<->{b} {tag} conf={rec.get('confidence')} "
                  f"elapsed={time.time() - start:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, t) for t in todo]
        for fut in futures:
            fut.result()
    print(f"done {done} rows (elapsed {time.time() - start:.0f}s)")


def stage_label(args) -> None:
    _runner(args, STRONG_PROMPT, thinking=True, extract=_parse)


def stage_eval_fast(args) -> None:
    teacher = {r["id"]: r for r in read_jsonl(args.teacher)}
    student = {r["id"]: r for r in read_jsonl(args.student)}
    tasks = [json.loads(l) for l in args.tasks.read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = {r["pid"]: {} for r in []}
    refs = {}
    for r in [json.loads(l) for l in args.labels.read_text(encoding="utf-8").splitlines() if l.strip()]:
        if r.get("status") == "ok":
            refs[(r["pid"], r["a"], r["b"])] = r
    todo = [t for t in tasks if (t["pid"], t["a"], t["b"]) in refs]
    tr = http_transport(model=MODEL, base_url=BASE_URL, api_key=_api_key(),
                        temperature=0.0, max_tokens=400, timeout=60, enable_thinking=False)
    results = []
    lock = threading.Lock()

    def work(t: dict) -> None:
        pid, a, b = t["pid"], t["a"], t["b"]
        rec = {"pid": pid, "a": a, "b": b, "kind": t.get("kind"), "status": "failed"}
        for attempt in range(3):
            try:
                content = _pair_block(teacher[pid].get("text", ""), teacher[pid]["entities"][a],
                                      student[pid]["entities"][b], args.desc_len)
                raw = tr([{"role": "system", "content": VERIFY_PROMPT},
                          {"role": "user", "content": content}])
                obj = _parse(raw)
                if obj is None:
                    raise ValueError("unparseable")
                rec.update({"status": "ok", "same": obj.get("same"), "attempts": attempt + 1})
                break
            except Exception as exc:
                if attempt == 2:
                    rec["reason"] = str(exc)[:200]
        with lock:
            results.append(rec)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in [pool.submit(work, t) for t in todo]:
            fut.result()

    ok = [r for r in results if r["status"] == "ok"]
    tp = fp = tn = fn = 0
    disagree = []
    for r in ok:
        label = refs[(r["pid"], r["a"], r["b"])]["same"]
        fast = r["same"]
        if label and fast:
            tp += 1
        elif label and not fast:
            fn += 1
            disagree.append(("FN", r, refs[(r["pid"], r["a"], r["b"])]))
        elif not label and fast:
            fp += 1
            disagree.append(("FP", r, refs[(r["pid"], r["a"], r["b"])]))
        else:
            tn += 1
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0
    print(f"\neval-fast vs thinking-label benchmark: n={total}")
    print(f"  confusion: TP(same,same)={tp} FN(same->diff)={fn} FP(diff->same)={fp} TN(diff,diff)={tn}")
    print(f"  accuracy={acc:.3f}  false-reject rate (of same)={fn/(tp+fn):.3f}  false-accept rate (of diff)={fp/(fp+tn):.3f}")
    out = OUT_DIR / "bench_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": total, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                               "accuracy": round(acc, 4),
                               "false_reject_rate": round(fn / (tp + fn), 4) if tp + fn else None,
                               "false_accept_rate": round(fp / (fp + tn), 4) if fp + tn else None},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)

    # dump disagreements with teacher/student titles+descs for diagnosis
    dump = OUT_DIR / "bench_disagreements.txt"
    with dump.open("w", encoding="utf-8") as f:
        for tag, r, ref in sorted(disagree, key=lambda x: x[0]):
            pid, a, b = r["pid"], r["a"], r["b"]
            t = teacher[pid]["entities"][a]
            s = student[pid]["entities"][b]
            f.write(f"{tag} {pid} {a}<->{b} (label_same={ref['same']} conf={ref.get('confidence')})\n")
            f.write(f"  T: ({t.get('type')}) {t['title']} :: {(t.get('description') or '')[:150]}\n")
            f.write(f"  S: ({s.get('type')}) {s.get('title','')} :: {(s.get('description') or '')[:150]}\n")
    print("disagreements ->", dump)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    p.add_argument("--pairs", type=Path, default=OUT_DIR / "pairings_llm.jsonl")
    p.add_argument("--stage", choices=("sample", "label", "eval-fast"), required=True)
    p.add_argument("--tasks", type=Path, default=OUT_DIR / "bench_tasks.jsonl")
    p.add_argument("--labels", type=Path, default=OUT_DIR / "bench_labels.jsonl")
    p.add_argument("--n-proposed", type=int, default=160)
    p.add_argument("--n-random", type=int, default=160)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--desc-len", type=int, default=300)
    args = p.parse_args(argv)
    {"sample": stage_sample, "label": stage_label, "eval-fast": stage_eval_fast}[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
