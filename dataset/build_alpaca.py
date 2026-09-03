"""Build the Alpaca (LLaMA-Factory) dataset from the GraphRAG teacher labels.

Default (tracer): reads dataset/data/graphrag_labels_100.jsonl and renders each
passage into {instruction, input, output}: instruction = the frozen student
target prompt, input = the passage text, output = the canonical JSON
{entities, relationships}. A small val split (--val-size, default 5) is held
out from the 100 for reporting.

--full: reads the full 3349-passage labels (dataset/data/graphrag_labels_full
.jsonl) and assigns train/val from outputs/graphrag_batch/split.json
(2575 / 75), writing outputs/graphrag_full/. The 10 tracer-eval passages live
in the test split, never in alpaca.

Usage:
    python dataset/build_alpaca.py [--val-size 5]        # tracer (95/5)
    python dataset/build_alpaca.py --full                # full (2575/75)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg_contract.student_prompt import STUDENT_PROMPT  # noqa: E402

from dataset.io import read_jsonl  # noqa: E402

LABELS = Path("dataset/data/graphrag_labels_100.jsonl")
OUT_DIR = Path("outputs/tracer")
FULL_LABELS = Path("dataset/data/graphrag_labels_full.jsonl")
FULL_SPLIT = Path("outputs/graphrag_batch/split.json")
FULL_OUT_DIR = Path("outputs/graphrag_full")


def _output(row: dict) -> str:
    return json.dumps(
        {"entities": row["entities"], "relationships": row["relationships"]},
        ensure_ascii=False,
    )


def _record(r: dict) -> dict:
    return {"instruction": STUDENT_PROMPT, "input": r["text"], "output": _output(r)}


def _write(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _report(out_dir: Path, train: list[dict], val: list[dict], stem: str) -> None:
    avg_out = sum(len(x["output"]) for x in train) // len(train)
    print(f"train: {len(train)} rows, val: {len(val)} rows, avg output len ~{avg_out} chars")
    print(f"wrote {out_dir / (stem + '_train.jsonl')} and {out_dir / (stem + '_val.jsonl')}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-size", type=int, default=5)
    parser.add_argument("--full", action="store_true", help="build full-scale train/val from split.json")
    args = parser.parse_args(argv)

    if args.full:
        by_id = {r["id"]: r for r in read_jsonl(FULL_LABELS)}
        split = json.loads(FULL_SPLIT.read_text(encoding="utf-8"))
        for part in ("train", "val"):
            missing = [i for i in split[part] if i not in by_id]
            if missing:
                raise SystemExit(f"[full] {part} ids missing from labels: {missing[:5]}")
        train = [_record(by_id[i]) for i in split["train"]]
        val = [_record(by_id[i]) for i in split["val"]]
        out_dir = FULL_OUT_DIR
        stem = "alpaca_full"
    else:
        rows = read_jsonl(LABELS)
        rows.sort(key=lambda r: r["id"])
        val_ids = {r["id"] for r in rows[: args.val_size]}
        train = [_record(r) for r in rows if r["id"] not in val_ids]
        val = [_record(r) for r in rows if r["id"] in val_ids]
        out_dir = OUT_DIR
        stem = "alpaca"

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / f"{stem}_train.jsonl", train)
    _write(out_dir / f"{stem}_val.jsonl", val)
    _report(out_dir, train, val, stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
