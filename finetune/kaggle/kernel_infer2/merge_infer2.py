"""Merge infer_only2.py + inputs.py into one self-contained kernel file.

Same as kernel_infer/merge_infer.py but for the kernel_infer2 variant.

Usage:
    python finetune/kaggle/kernel_infer2/merge_infer2.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "infer_only2.py").read_text(encoding="utf-8")
INPUTS = (HERE.parent / "kernel" / "inputs.py").read_text(encoding="utf-8")

marker = "# INPUTS_HERE"
assert marker in SRC, "INPUTS_HERE marker not found"
assert "INPUTS =" in INPUTS

merged = SRC.replace(marker, INPUTS.strip())
out = HERE / "kg_tracer_infer2.py"
out.write_text(merged, encoding="utf-8")
print(f"wrote {out} ({len(merged)} chars)")
