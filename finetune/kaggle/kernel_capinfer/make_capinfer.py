"""Generate capacity-curve inference kernels (0.6B / 1.7B / 4B).

Merges `cap_infer.py` (source of truth, `# INPUTS_HERE` marker) with the
STUDENT_PROMPT module from ../kernel/inputs.py, fills per-model placeholders
(MODEL_SLUG, ADAPTER_NAME, PROTOCOL, optional baked LIMIT), and writes one
self-contained push dir per size under finetune/kaggle/:

    kernel_cap06/kg_tracer_cap06.py + kernel-metadata.json   (0.6B)
    kernel_cap17/kg_tracer_cap17.py + kernel-metadata.json   (1.7B)
    kernel_cap4b/kg_tracer_cap4b.py + kernel-metadata.json   (4B)

`--protocol plain` appends a `p` to dir/kernel tags (e.g. kernel_cap06p /
kg-tracer-cap06p) so a plain-protocol A/B does not clobber the chat run.

Usage:
    python finetune/kaggle/kernel_capinfer/make_capinfer.py                    # chat, full
    python finetune/kaggle/kernel_capinfer/make_capinfer.py --limit 6          # smoke
    python finetune/kaggle/kernel_capinfer/make_capinfer.py --protocol plain --only 06
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGLE = HERE.parent
INPUTS_PATH = KAGLE / "kernel" / "inputs.py"

SIZES = [
    ("06", "qwen3-0.6b", "qwen-lm/qwen-3/transformers/0.6b/1"),
    ("17", "qwen3-1.7b", "qwen-lm/qwen-3/transformers/1.7b/1"),
    ("4b", "qwen3-4b", "qwen-lm/qwen-3/transformers/4b/1"),
]


def build(size_tag: str, adapter: str, model_slug: str, limit: int | None,
          protocol: str, early_stop: bool, adapter_dataset: str | None,
          suffix: str, adapter_root: bool, batch: int) -> None:
    src = (HERE / "cap_infer.py").read_text(encoding="utf-8")
    inputs = INPUTS_PATH.read_text(encoding="utf-8")

    marker = "# INPUTS_HERE"
    assert marker in src, "INPUTS_HERE marker not found"
    merged = src.replace(marker, inputs.strip())

    merged = merged.replace('MODEL_SLUG = "%%MODEL_SLUG%%"',
                            f'MODEL_SLUG = "{model_slug}"')
    merged = merged.replace('ADAPTER_NAME = "%%ADAPTER_NAME%%"',
                            f'ADAPTER_NAME = "{adapter}"')
    if batch != 4:
        merged = merged.replace("BATCH_SIZE = 4", f"BATCH_SIZE = {batch}")
    if adapter_dataset is not None:
        merged = merged.replace('DATASET = "cap-upload"',
                                f'DATASET = "{adapter_dataset}"')
    merged = merged.replace('PROTOCOL = "%%PROTOCOL%%"',
                            f'PROTOCOL = "{protocol}"')
    if not early_stop:
        merged = merged.replace("EARLY_STOP = False", "EARLY_STOP = False")
    else:
        merged = merged.replace("EARLY_STOP = False", "EARLY_STOP = True")
    if limit is not None:
        merged = merged.replace("LIMIT = None", f"LIMIT = {limit}")

    tag = size_tag + ("p" if protocol == "plain" else "")
    tag = tag + ("s" if early_stop else "") + suffix
    out = KAGLE / f"kernel_cap{tag}"
    out.mkdir(parents=True, exist_ok=True)
    code_name = f"kg_tracer_cap{tag}.py"
    (out / code_name).write_text(merged, encoding="utf-8")

    dataset_sources = ["idalextan/cap-upload"]
    if adapter_dataset is not None:
        dataset_sources.append(f"idalextan/{adapter_dataset}")
    meta = {
        "id": f"idalextan/kg-tracer-cap{tag}",
        "title": f"kg-tracer-cap{tag}",
        "code_file": code_name,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [model_slug],
    }
    (out / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{out.name}: {model_slug} protocol={protocol} limit={limit} "
          f"adapter={adapter} dataset={adapter_dataset or 'cap-upload'} "
          f"suffix={suffix!r} -> {code_name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--protocol", choices=["chat", "plain"], default="chat")
    ap.add_argument("--early-stop", action="store_true",
                    help="stop each decode when the JSON object closes")
    ap.add_argument("--only", default=None, help="build only this size tag (e.g. 06)")
    ap.add_argument("--adapter-dataset", default=None,
                    help="mount adapters from this extra Kaggle dataset slug "
                         "instead of idalextan/cap-upload (e.g. kg-cap06b-chat-adapter)")
    ap.add_argument("--adapter-name", default=None,
                    help="override the adapter folder name under the dataset")
    ap.add_argument("--adapter-root", action="store_true",
                    help="adapter files sit at the dataset ROOT (ADAPTER_NAME=\"\") "
                         "-- Kaggle CLI uploads cannot keep nested folders")
    ap.add_argument("--batch", type=int, default=4,
                    help="decode batch size (per worker); raise for larger GPUs, "
                         "e.g. 12 on a single 4090 vs 4 on T4")
    ap.add_argument("--suffix", default="",
                    help="extra tag suffix so a variant run does not clobber an existing dir")
    args = ap.parse_args(argv)
    for tag, adapter, slug in SIZES:
        if args.only and tag != args.only:
            continue
        adapter = "" if args.adapter_root else adapter
        if not args.adapter_root and args.adapter_name is not None:
            adapter = args.adapter_name
        build(tag, adapter, slug, args.limit, args.protocol, args.early_stop,
              args.adapter_dataset, args.suffix, args.adapter_root, args.batch)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
