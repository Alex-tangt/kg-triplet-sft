"""End-to-end stage tests: each dataset CLI runs on the small fixture and its
output invariants are asserted (download resumability, sentence-boundary
chunking, filter counts and dedup)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from dataset.chunking import ends_at_sentence_boundary, word_count

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "raw_fixture"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_cli(data_dir, *args):
    return subprocess.run(
        [sys.executable, *args, "--data-dir", str(data_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def seed_raw(data_dir):
    raw = data_dir / "01_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("wikipedia.jsonl", "arxiv.jsonl"):
        shutil.copy(FIXTURE / name, raw / name)


# --------------------------------------------------------------------------- #
# Stage 1 — download (offline --input path)
# --------------------------------------------------------------------------- #

def test_download_is_resumable_and_appends(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = data_dir / "01_raw"
    out_dir.mkdir(parents=True)
    out = out_dir / "arxiv.jsonl"

    input_path = tmp_path / "input.jsonl"
    rows = [{"source": "arxiv", "title": "", "text": f"record {i}. " * 30} for i in range(10)]
    with open(input_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    with open(out, "w", encoding="utf-8") as f:
        for row in rows[:2]:
            f.write(json.dumps(row) + "\n")

    result = run_cli(data_dir, "dataset/download.py", "--input", str(input_path), "--sources", "arxiv", "--limit", "4")
    assert result.returncode == 0, result.stderr
    assert read_jsonl(out) == rows[:6]

    result = run_cli(data_dir, "dataset/download.py", "--input", str(input_path), "--sources", "arxiv", "--limit", "10")
    assert result.returncode == 0, result.stderr
    assert read_jsonl(out) == rows

    result = run_cli(data_dir, "dataset/download.py", "--input", str(input_path), "--sources", "arxiv")
    assert result.returncode == 0, result.stderr
    assert read_jsonl(out) == rows


# --------------------------------------------------------------------------- #
# Stage 2 — chunking
# --------------------------------------------------------------------------- #

def test_chunk_stage_invariants(tmp_path):
    data_dir = tmp_path / "data"
    seed_raw(data_dir)

    result = run_cli(data_dir, "dataset/chunk.py")
    assert result.returncode == 0, result.stderr

    chunks = read_jsonl(data_dir / "02_chunks.jsonl")
    assert len(chunks) == 13
    for chunk in chunks:
        assert chunk["word_count"] == word_count(chunk["text"])
        assert ends_at_sentence_boundary(chunk["text"]), chunk["text"][-40:]

    summary = json.loads(result.stdout)
    assert summary["chunk"]["input_records"] == 9
    assert summary["chunk"]["oversized_chunks"] == 0


# --------------------------------------------------------------------------- #
# Stage 3 — filter + dedup
# --------------------------------------------------------------------------- #

def test_filter_stage_invariants(tmp_path):
    data_dir = tmp_path / "data"
    seed_raw(data_dir)
    assert run_cli(data_dir, "dataset/chunk.py").returncode == 0

    result = run_cli(data_dir, "dataset/filter.py")
    assert result.returncode == 0, result.stderr

    corpus = read_jsonl(data_dir / "corpus.jsonl")
    assert len(corpus) == 3
    assert [c["source"] for c in corpus] == ["wikipedia", "wikipedia", "arxiv"]
    assert len({c["id"] for c in corpus}) == 3
    for passage in corpus:
        assert passage["id"].startswith(f'{passage["source"]}-')
        assert 100 <= passage["word_count"] <= 300
        assert ends_at_sentence_boundary(passage["text"]), passage["text"][-40:]

    summary = json.loads(result.stdout)["filter"]
    assert summary["input_chunks"] == 13
    assert summary["dropped"] == {"length": 5, "language": 1, "boilerplate": 1}
    assert summary["quota_dropped"] == {"wikipedia": 0, "arxiv": 0}
    assert summary["near_duplicates_removed"] == 3
    assert summary["passages_written"] == 3
    assert summary["boundaries_violated"] == 0
    assert summary["source_mix"] == {"wikipedia": 2, "arxiv": 1}


def test_filter_stage_applies_source_quotas(tmp_path):
    data_dir = tmp_path / "data"
    seed_raw(data_dir)
    assert run_cli(data_dir, "dataset/chunk.py").returncode == 0

    result = run_cli(
        data_dir,
        "dataset/filter.py",
        "--max-wikipedia",
        "1",
        "--max-arxiv",
        "1",
    )
    assert result.returncode == 0, result.stderr

    corpus = read_jsonl(data_dir / "corpus.jsonl")
    assert [c["source"] for c in corpus] == ["wikipedia", "arxiv"]

    summary = json.loads(result.stdout)["filter"]
    assert summary["quota_dropped"] == {"wikipedia": 1, "arxiv": 0}
    assert summary["passages_written"] == 2
