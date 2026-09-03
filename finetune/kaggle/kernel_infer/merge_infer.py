"""Merge infer_only.py + inputs.py into one self-contained kernel file.

Kaggle script kernels package ONLY the code_file, so sibling modules can't be
imported. inputs.py is a bare `INPUTS = {...}` assignment; we inject its whole
body where the source carries a `# INPUTS_HERE` marker.

Usage:
    python finetune/kaggle/kernel_infer/merge_infer.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "infer_only.py").read_text(encoding="utf-8")
INPUTS = (Path(__file__).resolve().parent.parent / "kernel" / "inputs.py").read_text(encoding="utf-8")

marker = "# INPUTS_HERE"
assert marker in SRC, "INPUTS_HERE marker not found in infer_only.py"
assert "INPUTS =" in INPUTS, "inputs.py does not start with INPUTS = assignment"

merged = SRC.replace(marker, INPUTS.strip())
out = HERE / "kg_tracer_infer.py"
out.write_text(merged, encoding="utf-8")
print(f"wrote {out} ({len(merged)} chars)")
