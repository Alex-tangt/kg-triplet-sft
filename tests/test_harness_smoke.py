"""Smoke-test the vendored harness offline: the lexical-fallback run on its own
data must print a composite in [0, 1] and exit 0 (the harness's own verification
command). Skipped when the gitignored data file is absent (fresh clone)."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parent.parent / "eval" / "harness"
DATA = HARNESS / "data" / "predict.jsonl"

pytestmark = pytest.mark.skipif(
    not DATA.exists(),
    reason="harness data is gitignored; copy from the reference repo to run",
)


def test_harness_offline_smoke():
    result = subprocess.run(
        [sys.executable, "scripts/evaluation.py", "--no-embeddings", "--limit", "50"],
        cwd=HARNESS,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    match = re.search(r"COMPOSITE SCORE\s*:\s*([0-9.]+)", result.stdout)
    assert match, result.stdout
    score = float(match.group(1))
    assert 0.0 <= score <= 1.0
