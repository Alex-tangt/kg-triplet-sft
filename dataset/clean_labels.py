"""Fast regex cleanup of the full GraphRAG labels.

Drops entities/relationships whose descriptions carry structural junk the
teacher embedded mid-record (record separators `##`/`**##**`, the `<|COMPLETE|>`
marker, field separators `<|>`, or a nested record opener), then drops any
passage left with no entities. Cheap, deterministic, low-cost — a final guard
on top of ``parse_graphrag_output``'s tolerant parser.

Usage:
    python dataset/clean_labels.py [--in dataset/data/graphrag_labels_full.jsonl]
Rewrites the file in place (after writing a `.pre_clean` backup).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.io import read_jsonl  # noqa: E402
from dataset.semantic import normalize  # noqa: E402

_JUNK = re.compile(
    r"##|\*\*|<\|>|<\|COMPLETE\|>|\(\"entity\"|\(\"relationship\"|\)\s*##"
)


def _clean(desc: str) -> bool:
    if _JUNK.search(desc):
        return False
    # truncation residues: an unclosed paren or an odd number of double quotes
    if desc.count("(") > desc.count(")"):
        return False
    if desc.count('"') % 2 != 0:
        return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, default=Path("dataset/data/graphrag_labels_full.jsonl"))
    args = parser.parse_args(argv)

    rows = read_jsonl(args.in_path)
    backup = args.in_path.with_suffix(".pre_clean.jsonl")
    if not backup.exists():
        shutil.copy(args.in_path, backup)

    total_e = total_r = 0
    kept_passages = 0
    for r in rows:
        r["entities"] = [e for e in r["entities"] if _clean(e["description"])]
        titles = {normalize(e["title"]) for e in r["entities"]}
        r["relationships"] = [
            rel
            for rel in r["relationships"]
            if _clean(rel["description"])
            and normalize(rel["source"]) in titles
            and normalize(rel["target"]) in titles
        ]
        if not r["entities"]:
            continue
        kept_passages += 1
        total_e += len(r["entities"])
        total_r += len(r["relationships"])

    with args.in_path.open("w", encoding="utf-8") as f:
        for r in rows:
            if not r["entities"]:
                continue
            f.write(__import__("json").dumps(r, ensure_ascii=False) + "\n")

    print(f"passages {len(rows)} -> kept {kept_passages} (dropped {len(rows) - kept_passages})")
    print(f"entities {total_e}, relationships {total_r}")
    print(f"wrote {args.in_path} (backup at {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
