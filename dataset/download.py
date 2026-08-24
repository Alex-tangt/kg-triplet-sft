"""Stage 1 — download raw source text.

Writes JSONL records ``{source, title, text}`` to ``<data-dir>/01_raw/<source>.jsonl``.
Resumable: an existing output file is never rewritten — records already written
are skipped and new ones appended, so a killed run restarts where it stopped.
Downloads honor ``HF_ENDPOINT`` (defaults to the hf-mirror mirror per the locked
decisions; the mirror 308-redirects to huggingface.co, so on machines where that
host is directly reachable the endpoint can be overridden). ``--input`` replaces
the network source with a local JSONL for offline testing of the same
write/resume logic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SOURCES = {
    "wikipedia": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "title": "title",
        "text": "text",
    },
    "arxiv": {
        "dataset": "ccdv/arxiv-summarization",
        "config": None,
        "split": "train",
        "title": None,  # the arXiv abstracts dataset carries no titles
        "text": "abstract",  # the `article` field is the full paper body
    },
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def append_records(rows, path: Path, limit: int | None = None) -> int:
    written = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if limit is not None and written >= limit:
                break
    return written


def _records_from_hub(source: str, skip: int):
    cfg = SOURCES[source]
    from datasets import load_dataset

    dataset = load_dataset(cfg["dataset"], cfg["config"], split=cfg["split"], streaming=True)
    for row in dataset:
        if skip:
            skip -= 1
            continue
        title = row[cfg["title"]] if cfg["title"] else ""
        yield {"source": source, "title": title, "text": row[cfg["text"]]}


def _records_from_input(path: Path, source: str, skip: int):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if skip:
                skip -= 1
                continue
            yield json.loads(line)


def run(args) -> dict:
    data_dir = Path(args.data_dir)
    # Explicit flag always wins; the parser default already falls back to the
    # ambient HF_ENDPOINT then to the mirror.
    os.environ["HF_ENDPOINT"] = args.endpoint
    sources = args.sources if args.sources else list(SOURCES)
    summary: dict[str, dict] = {}
    for source in sources:
        out = data_dir / "01_raw" / f"{source}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        existing = count_lines(out) if out.exists() else 0
        if args.input:
            rows = _records_from_input(Path(args.input), source, existing)
        else:
            rows = _records_from_hub(source, existing)
        written = append_records(rows, out, args.limit)
        summary[source] = {"already_present": existing, "written": written, "total": existing + written}
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--sources", nargs="*", choices=list(SOURCES))
    parser.add_argument("--input", help="local JSONL to copy instead of the Hub (offline/fixture)")
    parser.add_argument("--limit", type=int, default=None, help="max new records per source")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps({"download": summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
