"""Produce the two deployable kernel variants from the merged single file.

Kaggle script kernels run `python /kaggle/src/script.py`, so top-level code
must invoke main directly (no `__name__ == "__main__"` guard dependency).

Outputs:
  kernel/kg_tracer_full.py     - runs train + infer
  kernel/kg_tracer_verify.py   - runs yaml/env pre-flight only

Usage: python finetune/kaggle/make_variants.py
"""

from __future__ import annotations

from pathlib import Path

KERNEL = Path(__file__).resolve().parent / "kernel"
SRC = (KERNEL / "kg_tracer_single.py").read_text(encoding="utf-8")

# Replace the __main__ guard with a direct call so it fires under `python script.py`.
head, sep, tail = SRC.rpartition('if __name__ == "__main__":\n    sys.exit(main())\n')
assert sep, "guard not found"
base = head + "sys.exit(main())\n"

full = base
verify = base.replace('sys.exit(main())\n', 'sys.exit(main(["--verify"]))\n', 1)
smoke = base.replace('sys.exit(main())\n', 'sys.exit(main(["--smoke"]))\n', 1)

(KERNEL / "kg_tracer_full.py").write_text(full, encoding="utf-8")
(KERNEL / "kg_tracer_verify.py").write_text(verify, encoding="utf-8")
(KERNEL / "kg_tracer_smoke.py").write_text(smoke, encoding="utf-8")
print(f"full  -> kernel/kg_tracer_full.py ({len(full)} chars)")
print(f"verify-> kernel/kg_tracer_verify.py ({len(verify)} chars)")
print(f"smoke -> kernel/kg_tracer_smoke.py ({len(smoke)} chars)")
