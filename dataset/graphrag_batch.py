"""Aliyun batch-inference labeling for the GraphRAG teacher (2-stage fixed round).

Batch inference is one-shot per line (no conditional multi-turn), so the
teacher's adaptive continuation loop becomes a fixed two-round schedule:

  Job A (round 1): extraction prompt + passage          -> 100 lines
  Job B (round 2): r1 messages + assistant=r1 output + CONTINUE_PROMPT -> 100 lines

Results of both jobs are merged per passage and deduplicated/grounding-filtered
via ``dedup_graphrag`` into the training labels.

Usage:
  python dataset/graphrag_batch.py --stage sample            # 100 ids -> outputs/graphrag_batch/ids.json
  python dataset/graphrag_batch.py --stage realtime [--workers 8] [--limit 1] [--ids ...]
                                                            # realtime labeling -> dataset/data/graphrag_labels_100.jsonl
  python dataset/graphrag_batch.py --stage split             # full-split train/val/test -> outputs/graphrag_batch/split.json
  python dataset/graphrag_batch.py --stage gen-full-r1       # batch Job A for not-yet-labeled split passages
  python dataset/graphrag_batch.py --stage gen-r1            # (tracer batch path, not used when realtime)
  python dataset/graphrag_batch.py --stage gen-r2 --r1 results_r1.jsonl
  python dataset/graphrag_batch.py --stage merge --r1 results_r1.jsonl --r2 results_r2.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg_contract.graphrag_prompts import CONTINUE_PROMPT, GRAPH_EXTRACTION_PROMPT  # noqa: E402

from dataset.graphrag import dedup_graphrag, parse_graphrag_output  # noqa: E402
from dataset.graphrag_pilot import (  # noqa: E402
    BASE_URL,
    ENTITY_TYPES,
    MODEL,
    MAX_ROUNDS,
    run_graphrag,
)
from dataset.io import read_jsonl  # noqa: E402
from dataset.teacher import Budget, http_transport, sample_passages  # noqa: E402

OUT_DIR = Path("outputs/graphrag_batch")
CORPUS = Path("dataset/data/corpus.jsonl")
LABELS_OUT = Path("dataset/data/graphrag_labels_100.jsonl")
N_LABEL = 100
SEED = 7

N_TRAIN = 2575
N_VAL = 75
N_TEST = 700

#: passages reserved for the tracer eval — never sampled into the training set.
EVAL_IDS = {
    "wikipedia-02909", "arxiv-00944", "wikipedia-01557", "wikipedia-02266",
    "wikipedia-01974", "wikipedia-01420", "wikipedia-01252", "wikipedia-00083",
    "wikipedia-01421", "arxiv-00634",
}

#: Blind judge (4 passages, thinking on vs off, outputs/graphrag_pilot/
#: thinking_compare/): thinking ON is materially better — richer grounded
#: descriptions, more numeric facts preserved, zero fabrication; OFF produced
#: a mislabeled EVENT and a garbled translation. Raw entity counts favoured OFF
#: only because its extras were low-quality variants that dedup/grounding
#: dropped. Thinking tokens cost more but label quality wins for SFT.
ENABLE_THINKING = True
MAX_TOKENS = 4096


def _body(messages: list[dict]) -> dict:
    body: dict = {"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS}
    if ENABLE_THINKING:
        body["enable_thinking"] = True
    return body


def _line(custom_id: str, messages: list[dict]) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": _body(messages),
    }


def _passages(ids: list[str]) -> dict[str, str]:
    by_id = {r["id"]: r for r in read_jsonl(CORPUS)}
    return {i: by_id[i]["text"] for i in ids}


def stage_sample() -> None:
    pool = [r for r in read_jsonl(CORPUS) if r["id"] not in EVAL_IDS]
    drawn = sample_passages(pool, N_LABEL, seed=SEED)
    ids = [r["id"] for r in drawn]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ids.json").write_text(json.dumps(ids, ensure_ascii=False, indent=1), encoding="utf-8")
    n_wp = sum(1 for i in ids if i.startswith("wikipedia"))
    print(f"sampled {len(ids)} ids ({n_wp} wp / {len(ids) - n_wp} arxiv) -> {OUT_DIR / 'ids.json'}")


def stage_gen_r1() -> None:
    ids = json.loads((OUT_DIR / "ids.json").read_text(encoding="utf-8"))
    passages = _passages(ids)
    lines = []
    for pid in ids:
        prompt = GRAPH_EXTRACTION_PROMPT.format(entity_types=ENTITY_TYPES, input_text=passages[pid])
        lines.append(_line(f"{pid}:r1", [{"role": "user", "content": prompt}]))
    out = OUT_DIR / "batch_input_r1.jsonl"
    out.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines)")


def _load_batch_results(path: Path) -> dict[str, str]:
    """Batch output -> {custom_id: assistant content}."""
    out: dict[str, str] = {}
    for line in read_jsonl(path):
        cid = line.get("custom_id")
        if not cid:
            continue
        resp = line.get("response") or {}
        body = resp.get("body") or {}
        choices = body.get("choices") or []
        content = ""
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
        out[cid] = content
    return out


def stage_gen_r2(r1_results: Path) -> None:
    r1 = _load_batch_results(r1_results)
    ids = json.loads((OUT_DIR / "ids.json").read_text(encoding="utf-8"))
    passages = _passages(ids)
    lines = []
    missing = 0
    for pid in ids:
        cid = f"{pid}:r1"
        r1_raw = r1.get(cid)
        if not r1_raw:
            missing += 1
            continue
        prompt = GRAPH_EXTRACTION_PROMPT.format(entity_types=ENTITY_TYPES, input_text=passages[pid])
        lines.append(
            _line(
                f"{pid}:r2",
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": r1_raw},
                    {"role": "user", "content": CONTINUE_PROMPT},
                ],
            )
        )
    out = OUT_DIR / "batch_input_r2.jsonl"
    out.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines, missing r1 results: {missing})")


def stage_merge(r1_results: Path, r2_results: Path) -> None:
    r1 = _load_batch_results(r1_results)
    r2 = _load_batch_results(r2_results)
    ids = json.loads((OUT_DIR / "ids.json").read_text(encoding="utf-8"))
    passages = _passages(ids)
    rows = []
    for pid in ids:
        raw_entities: list[dict] = []
        raw_rels: list[dict] = []

        def absorb(raw: str) -> None:
            parsed = parse_graphrag_output(raw)
            if parsed:
                raw_entities.extend(parsed["entities"])
                raw_rels.extend(parsed["relationships"])

        for suffix in ("r1", "r2"):
            raw = r1.get(f"{pid}:{suffix}") or r2.get(f"{pid}:{suffix}")
            if raw:
                absorb(raw)
        out = dedup_graphrag({"entities": raw_entities, "relationships": raw_rels}, passages[pid])
        rows.append(
            {
                "id": pid,
                "source": "wikipedia" if pid.startswith("wikipedia") else "arxiv",
                "text": passages[pid],
                "teacher": "graphrag",
                "status": "ok",
                "entities": out["entities"],
                "relationships": out["relationships"],
                "raw_entity_count": len(raw_entities),
                "raw_relationship_count": len(raw_rels),
                "dropped": len(raw_entities) - len(out["entities"]),
            }
        )
    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ent = sum(len(r["entities"]) for r in rows)
    n_rel = sum(len(r["relationships"]) for r in rows)
    print(f"wrote {LABELS_OUT} ({len(rows)} passages, {n_ent} entities, {n_rel} relationships)")


def _realtime_record(pid: str, text: str, transport, budget) -> dict:
    try:
        out = run_graphrag(transport, budget, text, ENTITY_TYPES, MAX_ROUNDS)
        out["status"] = "ok"
    except Exception as e:  # BudgetExhausted, RetryableError, NonRetryableError
        out = {"status": "failed", "failure_reason": str(e), "entities": [], "relationships": []}
    out.update({"id": pid, "source": "wikipedia" if pid.startswith("wikipedia") else "arxiv", "text": text})
    return out


def stage_realtime(workers: int, limit: int | None, only_ids: list[str] | None) -> None:
    """Realtime parallel labeling with progress + resume (the batch path queues too slowly)."""
    ids = json.loads((OUT_DIR / "ids.json").read_text(encoding="utf-8"))
    if only_ids:
        ids = [i for i in ids if i in set(only_ids)]
    if limit is not None:
        ids = ids[:limit]

    existing: dict[str, dict] = {}
    if LABELS_OUT.exists():
        existing = {r["id"]: r for r in read_jsonl(LABELS_OUT)}
    remaining = [i for i in ids if i not in existing or existing[i]["status"] != "ok"]
    print(f"[realtime] {len(ids)} requested, {len(remaining)} to run (done {len(ids) - len(remaining)})")

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv

            load_dotenv(Path(".").resolve() / ".env")
            api_key = os.environ.get("DASHSCOPE_API_KEY")
        except ImportError:
            pass
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY not set (see .env.example)")
    passages = _passages(ids)
    budget = Budget(workers * 80 + 100)

    lock = threading.RLock()  # reentrant: log_line() is called while work() holds the lock
    done = 0
    start = time.time()
    log_lines: list[str] = []

    def log_line(s: str) -> None:
        with lock:
            log_lines.append(s)
            sys.stdout.write("\r" + " " * 80 + "\r" + s + "\n")
            sys.stdout.flush()

    def work(pid: str) -> None:
        nonlocal done
        rec = _realtime_record(pid, passages[pid], transport, budget)
        with lock:
            done += 1
            LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
            with LABELS_OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            el = time.time() - start
            n_e = len(rec.get("entities", []))
            n_r = len(rec.get("relationships", []))
            log_line(f"[{done}/{len(remaining)}] {pid}: {rec['status']} {n_e}e/{n_r}r "
                     f"calls={rec.get('calls')} elapsed={el:.0f}s")

    transport = http_transport(
        model=MODEL, base_url=BASE_URL, api_key=api_key,
        temperature=0.0, max_tokens=MAX_TOKENS, timeout=120, enable_thinking=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, remaining))
    failed = [pid for pid in remaining if _status_of(pid) == "failed"]
    print(f"realtime labeling done: {len(remaining) - len(failed)} ok, {len(failed)} failed "
          f"({len(remaining)} total, {time.time() - start:.0f}s)")


def _status_of(pid: str) -> str:
    for l in reversed(LABELS_OUT.read_text(encoding="utf-8").splitlines()):
        if l.strip():
            r = json.loads(l)
            if r["id"] == pid:
                return r["status"]
    return "missing"


def stage_split() -> None:
    """Deterministic full split: 2575 train / 75 val / 700 test (60/40 + length).

    The 10 tracer-eval passages are forced into TEST, the 100 already-labeled
    passages into TRAIN (reused, not re-labeled). Decontamination (entity-level)
    is a separate post-labeling step.
    """
    by_id = {r["id"]: r for r in read_jsonl(CORPUS)}
    rng = random.Random(SEED)

    test_forced = set(EVAL_IDS)
    train_forced = set(json.loads((OUT_DIR / "ids.json").read_text(encoding="utf-8")))
    avail = [r for r in by_id.values() if r["id"] not in test_forced and r["id"] not in train_forced]
    rng.shuffle(avail)

    test_ids = test_forced | {r["id"] for r in sample_passages(avail, N_TEST - len(test_forced), seed=rng.getrandbits(31))}
    avail = [r for r in avail if r["id"] not in test_ids]
    train_ids = train_forced | {r["id"] for r in sample_passages(avail, N_TRAIN - len(train_forced), seed=rng.getrandbits(31))}
    avail = [r for r in avail if r["id"] not in train_ids]
    val_ids = {r["id"] for r in sample_passages(avail, N_VAL, seed=rng.getrandbits(31))}

    split = {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}
    (OUT_DIR / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, ids in split.items():
        n_wp = sum(1 for i in ids if i.startswith("wikipedia"))
        print(f"{name}: {len(ids)} ({n_wp} wp / {len(ids) - n_wp} arxiv)")
    print(f"total: {sum(len(v) for v in split.values())} -> {OUT_DIR / 'split.json'}")


def stage_gen_full_r1() -> None:
    """Batch Job A for the full split, skipping already-labeled passages.

    Already labeled: the 100 realtime records (graphrag_labels_100.jsonl) and the
    10 teacher_ref passages (outputs/tracer/teacher_ref.jsonl). Job B is built
    from Job A's results afterwards (gen-r2), same 2-stage schedule.
    """
    split = json.loads((OUT_DIR / "split.json").read_text(encoding="utf-8"))
    all_ids = split["train"] + split["val"] + split["test"]

    labeled: set[str] = set()
    for path in (LABELS_OUT, Path("outputs/tracer/teacher_ref.jsonl")):
        if path.exists():
            labeled |= {r["id"] for r in read_jsonl(path)}
    todo = [i for i in all_ids if i not in labeled]

    passages = _passages(todo)
    lines = [_line(f"{pid}:r1", [{"role": "user", "content": GRAPH_EXTRACTION_PROMPT.format(
        entity_types=ENTITY_TYPES, input_text=passages[pid])}]) for pid in todo]
    out = OUT_DIR / "batch_input_full_r1.jsonl"
    out.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
    print(f"split total {len(all_ids)}; already labeled {len(labeled)}; batch r1 lines {len(lines)}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def stage_gen_full_r2(r1_results: Path, r1_errors: Path | None) -> None:
    """Job B for the full run: continue-round for every r1 success, fresh-extract
    retry for every r1 failure (content-inspection failures have no output to
    continue from, so they are re-run as :r1retry extraction lines)."""
    success = _load_batch_results(r1_results)  # {cid -> content}
    failed: set[str] = set()
    if r1_errors and r1_errors.exists():
        for line in read_jsonl(r1_errors):
            cid = line.get("custom_id")
            if cid:
                failed.add(cid)
    pids = sorted({cid[:-3] for cid in list(success) + list(failed) if cid.endswith(":r1")})
    passages = _passages(pids)
    lines = []
    for pid in pids:
        prompt = GRAPH_EXTRACTION_PROMPT.format(entity_types=ENTITY_TYPES, input_text=passages[pid])
        r1 = success.get(f"{pid}:r1")
        if r1:
            lines.append(
                _line(f"{pid}:r2",
                      [{"role": "user", "content": prompt},
                       {"role": "assistant", "content": r1},
                       {"role": "user", "content": CONTINUE_PROMPT}])
            )
        else:
            lines.append(_line(f"{pid}:r1retry", [{"role": "user", "content": prompt}]))
    out = OUT_DIR / "batch_input_full_r2.jsonl"
    out.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
    n_retry = sum(1 for l in lines if l["custom_id"].endswith(":r1retry"))
    print(f"r2 lines {len(lines)} ({len(lines) - n_retry} continue, {n_retry} retry) -> {out}")


def stage_merge_full(r1_results: Path, r2_results: Path) -> None:
    """Merge full-run labels with a compliance filter.

    Iterates every split passage. A batch passage is kept only when its merged
    r1/r1retry/r2 output (after dedup/grounding) contains at least one entity —
    truncated or format-malformed outputs (parse-empty) and moderation-blocked
    passages (no results at all) are dropped, not retried. Already-labeled
    passages (100 realtime + 10 teacher_ref) are folded in as-is.
    """
    r1 = _load_batch_results(r1_results)
    r2 = _load_batch_results(r2_results)
    by_id: dict[str, dict] = {}
    for r in read_jsonl(CORPUS):
        by_id[r["id"]] = r
    split = json.loads((OUT_DIR / "split.json").read_text(encoding="utf-8"))
    all_ids = split["train"] + split["val"] + split["test"]

    labeled: dict[str, dict] = {}
    for path in (LABELS_OUT, Path("outputs/tracer/teacher_ref.jsonl")):
        for r in read_jsonl(path):
            labeled[r["id"]] = r

    rows: list[dict] = []
    dropped: list[str] = []
    for pid in all_ids:
        if pid in labeled:
            r = labeled[pid]
            rows.append(
                {
                    "id": pid,
                    "source": r.get("source", "wikipedia" if pid.startswith("wikipedia") else "arxiv"),
                    "text": r["text"],
                    "teacher": "graphrag",
                    "status": "ok",
                    "entities": r["entities"],
                    "relationships": r["relationships"],
                    "raw_entity_count": r.get("raw_entity_count", len(r["entities"])),
                    "raw_relationship_count": r.get("raw_relationship_count", len(r["relationships"])),
                    "dropped": r.get("dropped", 0),
                }
            )
            continue

        raw_entities: list[dict] = []
        raw_rels: list[dict] = []

        def absorb(raw: str) -> None:
            parsed = parse_graphrag_output(raw)
            if parsed:
                raw_entities.extend(parsed["entities"])
                raw_rels.extend(parsed["relationships"])

        for cid in (f"{pid}:r1", f"{pid}:r1retry", f"{pid}:r2"):
            if cid in r1:
                absorb(r1[cid])
            if cid in r2:
                absorb(r2[cid])

        out = dedup_graphrag({"entities": raw_entities, "relationships": raw_rels}, by_id[pid]["text"])
        if not out["entities"]:
            dropped.append(pid)
            continue
        rows.append(
            {
                "id": pid,
                "source": by_id[pid]["source"],
                "text": by_id[pid]["text"],
                "teacher": "graphrag",
                "status": "ok",
                "entities": out["entities"],
                "relationships": out["relationships"],
                "raw_entity_count": len(raw_entities),
                "raw_relationship_count": len(raw_rels),
                "dropped": len(raw_entities) - len(out["entities"]),
            }
        )

    rows.sort(key=lambda r: r["id"])
    out = Path("dataset/data/graphrag_labels_full.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_e = sum(len(r["entities"]) for r in rows)
    n_rel = sum(len(r["relationships"]) for r in rows)
    print(f"full labels: {len(rows)} passages, {n_e} entities, {n_rel} relationships")
    print(f"dropped non-compliant: {len(dropped)} {sorted(dropped)[:20]}")
    print(f"wrote {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sample", "realtime", "split", "gen-full-r1", "gen-full-r2", "merge-full", "gen-r1", "gen-r2", "merge"), required=True)
    parser.add_argument("--r1", type=Path, default=OUT_DIR / "results_r1.jsonl")
    parser.add_argument("--r2", type=Path, default=OUT_DIR / "results_r2.jsonl")
    parser.add_argument("--r1-errors", type=Path, default=OUT_DIR / "results_r1_errors.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", nargs="*", default=None)
    args = parser.parse_args(argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    if args.stage == "sample":
        stage_sample()
    elif args.stage == "realtime":
        stage_realtime(args.workers, args.limit, args.ids)
    elif args.stage == "split":
        stage_split()
    elif args.stage == "gen-full-r1":
        stage_gen_full_r1()
    elif args.stage == "gen-full-r2":
        stage_gen_full_r2(args.r1, args.r1_errors)
    elif args.stage == "merge-full":
        stage_merge_full(args.r1, args.r2)
    elif args.stage == "gen-r1":
        stage_gen_r1()
    elif args.stage == "gen-r2":
        stage_gen_r2(args.r1)
    elif args.stage == "merge":
        stage_merge(args.r1, args.r2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
