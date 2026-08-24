"""Stage 3 — quality filter + MinHash near-duplicate removal.

Reads ``02_chunks.jsonl``, cleans arXiv annotation artifacts, drops passages that
fail the length / language / boilerplate checks, removes near-duplicates (MinHash
LSH), applies per-source quotas, and writes the final ``corpus.jsonl`` with
``{id, source, title, chunk_index, text, word_count}``. Every passage must end at
a sentence boundary — a violation fails the stage (the locked "never
mid-sentence" invariant).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.chunking import ends_at_sentence_boundary, word_count  # noqa: E402
from dataset.filtering import clean_text, is_boilerplate, is_english  # noqa: E402
from dataset.io import read_jsonl, word_distribution  # noqa: E402
from dataset.minhash import deduplicate  # noqa: E402

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCES = ("wikipedia", "arxiv")


def run(args) -> dict:
    data_dir = Path(args.data_dir)
    out = data_dir / "corpus.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    chunks = read_jsonl(data_dir / "02_chunks.jsonl")

    dropped = {"length": 0, "language": 0, "boilerplate": 0}
    quota = {source: getattr(args, f"max_{source}") for source in SOURCES}
    quota_dropped = {source: 0 for source in SOURCES}
    cleaned_chunks = 0
    kept: list[dict] = []
    for chunk in chunks:
        cleaned = clean_text(chunk["text"])
        if cleaned != chunk["text"]:
            cleaned_chunks += 1
            chunk = dict(chunk, text=cleaned, word_count=word_count(cleaned))
        count = chunk["word_count"]
        if not (args.min_words <= count <= args.max_words):
            dropped["length"] += 1
            continue
        if not is_english(chunk["text"]):
            dropped["language"] += 1
            continue
        if is_boilerplate(chunk["text"]):
            dropped["boilerplate"] += 1
            continue
        kept.append(chunk)

    keep_flags = deduplicate(
        [chunk["text"] for chunk in kept],
        threshold=args.sim_threshold,
        num_hashes=args.num_hashes,
        rows_per_band=args.rows_per_band,
    )
    duplicates = sum(1 for flag in keep_flags if not flag)

    ids: dict[str, int] = {}
    boundaries_violated = 0
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for chunk, keep in zip(kept, keep_flags):
            if not keep:
                continue
            if not ends_at_sentence_boundary(chunk["text"]):
                boundaries_violated += 1
                continue
            source = chunk["source"]
            if quota[source] is not None and quota[source] <= 0:
                quota_dropped[source] += 1
                continue
            if quota[source] is not None:
                quota[source] -= 1
            ids[source] = ids.get(source, 0) + 1
            record = {
                "id": f"{source}-{ids[source]:05d}",
                "source": source,
                "title": chunk["title"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "word_count": chunk["word_count"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    if boundaries_violated:
        raise SystemExit(f"[filter] {boundaries_violated} passages end mid-sentence; refusing to emit corpus")

    corpus = read_jsonl(out)
    return {
        "input_chunks": len(chunks),
        "cleaned_chunks": cleaned_chunks,
        "dropped": dropped,
        "quota_dropped": quota_dropped,
        "near_duplicates_removed": duplicates,
        "passages_written": written,
        "boundaries_violated": boundaries_violated,
        "source_mix": {source: sum(1 for r in corpus if r["source"] == source) for source in SOURCES},
        "length_distribution": word_distribution([r["word_count"] for r in corpus]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--max-words", type=int, default=300)
    parser.add_argument("--sim-threshold", type=float, default=0.85)
    parser.add_argument("--num-hashes", type=int, default=128)
    parser.add_argument("--rows-per-band", type=int, default=8)
    parser.add_argument("--max-wikipedia", type=int, default=None, help="cap passages from wikipedia (None = uncapped)")
    parser.add_argument("--max-arxiv", type=int, default=None, help="cap passages from arxiv (None = uncapped)")
    args = parser.parse_args(argv)
    print(json.dumps({"filter": run(args)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
