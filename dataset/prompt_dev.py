"""Prompt-development harness: run the teacher on difficulty-graded cases and
write raw results to files for rubric-based (subagent) verification.

Usage:
    python dataset/prompt_dev.py --iter N     # run the 3 tuning cases -> outputs/prompt_dev/iterN.jsonl
    python dataset/prompt_dev.py --heldout --iter N   # run the held-out case -> outputs/prompt_dev/heldout_iterN.jsonl

The judge's rubric + gold key-fact lists live in outputs/prompt_dev/RUBRIC.md;
this script only produces the raw model output. Passages and full triplet lists
are never dumped to the console.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg_contract import render_teacher_prompt  # noqa: E402
from kg_contract.prompts import TEACHER_SECTIONS_PROSE  # noqa: E402

from dataset.io import read_jsonl  # noqa: E402
from dataset.teacher import Budget, TeacherClient, http_transport  # noqa: E402

OUT_DIR = Path("outputs/prompt_dev")

CASE_IDS = [
    ("wikipedia-02909", "easy"),
    ("arxiv-00944", "medium"),
    ("wikipedia-01557", "hard"),
]

HELDOUT_CASE_ID = "wikipedia-02266"


def _record(iter_n: int, case_id: str, difficulty: str, passage: str, rec: dict) -> dict:
    return {
        "iter": iter_n,
        "case_id": case_id,
        "difficulty": difficulty,
        "passage": passage,
        "status": rec["status"],
        "schema_errors": rec["schema_errors"],
        "triplets": rec["triplets"],
        "semantic_violations": rec["semantic_violations"],
    }


def _run(ids: list[tuple[str, str]], iter_n: int, heldout: bool) -> Path:
    corpus = read_jsonl(Path("dataset/data/corpus.jsonl"))
    by_id = {r["id"]: r for r in corpus}
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(".").resolve() / ".env")
    except ImportError:
        pass
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    transport = http_transport(
        model="qwen3.7-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        temperature=0.0,
        max_tokens=4096,
        timeout=120,
        enable_thinking=True,
    )
    client = TeacherClient(
        transport, system_prompt=render_teacher_prompt(TEACHER_SECTIONS_PROSE), budget=Budget(50)
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"heldout_iter{iter_n}.jsonl" if heldout else OUT_DIR / f"iter{iter_n}.jsonl"
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["case_id"])
    remaining = [(cid, d) for cid, d in ids if cid not in done]
    if not remaining:
        print(f"{out_path} already complete — nothing to run")
        return out_path
    with out_path.open("a", encoding="utf-8") as fh:
        for case_id, difficulty in remaining:
            passage = by_id[case_id]
            rec = client.label(passage["text"])
            row = _record(iter_n, case_id, difficulty, passage["text"], rec)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[iter{iter_n}] {case_id} ({difficulty}): status={rec['status']} "
                  f"triplets={len(rec['triplets'])} violations={len(rec['semantic_violations'])}")
    print(f"wrote {out_path}")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--heldout", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if args.heldout:
        _run([(HELDOUT_CASE_ID, "heldout")], args.iter, heldout=True)
    else:
        _run(CASE_IDS, args.iter, heldout=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
