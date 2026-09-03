"""Generate the self-contained kernel input bundle for the tracer SFT run.

Reads the frozen student prompt and the 10 teacher_ref eval passages, then
emits `finetune/kaggle/kernel/inputs.py` — a Python module the Kaggle kernel
imports at inference time so the eval passages never need a separate upload.

Deterministic: fixed order by passage id.

Usage:
    python finetune/kaggle/gen_inputs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from kg_contract.student_prompt import STUDENT_PROMPT  # noqa: E402

from dataset.io import read_jsonl  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
TEACHER_REF = REPO / "outputs/tracer/teacher_ref.jsonl"
OUT = Path(__file__).resolve().parent / "kernel/inputs.py"

PASSAGE_IDS = [
    "wikipedia-02909",
    "arxiv-00944",
    "wikipedia-01557",
    "wikipedia-02266",
    "wikipedia-01974",
    "wikipedia-01420",
    "wikipedia-01252",
    "wikipedia-00083",
    "wikipedia-01421",
    "arxiv-00634",
]


def main() -> int:
    ref = {r["id"]: r for r in read_jsonl(TEACHER_REF)}
    missing = [i for i in PASSAGE_IDS if i not in ref]
    if missing:
        raise SystemExit(f"missing teacher_ref passages: {missing}")

    passages = [{"id": i, "text": ref[i]["text"]} for i in PASSAGE_IDS]
    payload = {"student_prompt": STUDENT_PROMPT, "passages": passages}
    body = "INPUTS = " + json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body, encoding="utf-8")
    print(f"wrote {OUT} ({len(body)} chars, {len(passages)} passages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
