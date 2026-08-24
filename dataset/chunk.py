"""Stage 2 — chunk raw articles into passages capped at 300 words.

Reads ``<data-dir>/01_raw/*.jsonl``, sentence-splits and packs each article into
chunks (never mid-sentence), and writes ``02_chunks.jsonl`` with
``{source, title, chunk_index, text, word_count}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.chunking import chunk_sentences, split_sentences, word_count  # noqa: E402
from dataset.io import read_jsonl, word_distribution  # noqa: E402

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCES = ("wikipedia", "arxiv")


def run(args) -> dict:
    data_dir = Path(args.data_dir)
    out = data_dir / "02_chunks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for source in SOURCES:
        raw = data_dir / "01_raw" / f"{source}.jsonl"
        if raw.exists():
            records.extend(read_jsonl(raw))

    word_counts: list[int] = []
    oversized = 0
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for record in records:
            sentences = split_sentences(record["text"])
            for index, text in enumerate(chunk_sentences(sentences, args.max_words)):
                count = word_count(text)
                word_counts.append(count)
                if count > args.max_words:
                    oversized += 1
                f.write(
                    json.dumps(
                        {
                            "source": record["source"],
                            "title": record["title"],
                            "chunk_index": index,
                            "text": text,
                            "word_count": count,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1

    return {
        "input_records": len(records),
        "chunks_written": written,
        "oversized_chunks": oversized,
        "length_distribution": word_distribution(word_counts),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--max-words", type=int, default=300)
    args = parser.parse_args(argv)
    print(json.dumps({"chunk": run(args)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
