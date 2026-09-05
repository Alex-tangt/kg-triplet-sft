"""Merge infer_full.py + inputs.py into one self-contained kernel file.

Same mechanism as kernel_infer2/merge_infer2.py, for the full-trained adapter
inference kernel (kernel_inferfull).

Usage:
    python finetune/kaggle/kernel_inferfull/merge_inferfull.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "infer_full.py").read_text(encoding="utf-8")
INPUTS = (HERE.parent / "kernel" / "inputs.py").read_text(encoding="utf-8")

marker = "# INPUTS_HERE"
assert marker in SRC, "INPUTS_HERE marker not found"
assert "INPUTS =" in INPUTS

merged = SRC.replace(marker, INPUTS.strip())
out = HERE / "kg_tracer_inferfull.py"
out.write_text(merged, encoding="utf-8")
print(f"wrote {out} ({len(merged)} chars)")
