"""Build (and optionally push) Hugging Face model repos for the canonical
masked capacity adapters: qwen3-{0.6b,1.7b,4b} LoRA on the Qwen3 base models.

Reads the local adapters under outputs/adapters/qwen3-<size>_masked/adapter,
rewrites adapter_config.base_model_name_or_path to the HF base id, writes a
model card, and (with --push) uploads each repo via huggingface_hub.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SIZES = {
    "0.6b": {
        "base": "Qwen/Qwen3-0.6B",
        "title": "Qwen3-0.6B KG extraction (LoRA, canonical masked)",
        "metrics": [
            ("entity micro R / P / F1", "0.536 / 0.699 / 0.607"),
            ("schema valid", "0.919"),
            ("hallucination rate", "0.166"),
        ],
    },
    "1.7b": {
        "base": "Qwen/Qwen3-1.7B",
        "title": "Qwen3-1.7B KG extraction (LoRA, canonical masked)",
        "metrics": [
            ("entity micro R / P / F1", "0.590 / 0.597 / 0.594"),
            ("schema valid", "0.844"),
            ("hallucination rate", "0.234"),
        ],
    },
    "4b": {
        "base": "Qwen/Qwen3-4B",
        "title": "Qwen3-4B KG extraction (LoRA, canonical masked)",
        "metrics": [
            ("entity micro R / P / F1 (strict)", "0.595 / 0.720 / 0.651"),
            ("entity micro R / P / F1 (case-tolerant)", "0.664 / 0.743 / 0.701"),
            ("schema valid (strict / case-tolerant)", "0.941 / 0.953"),
            ("hallucination rate (strict / case-tolerant)", "0.099 / 0.084"),
        ],
    },
}

REPO_HINT = "https://github.com/Alex-tangt/kg-triplet-sft"

CARD = """\
---
library_name: peft
base_model: {base}
license: mit
pipeline_tag: text-generation
language: en
tags:
- peft
- lora
- qwen3
- knowledge-graph
- graphrag
- information-extraction
- sft
---

# {title}

LoRA adapter that extracts `{{entities, relationships}}` from open English text in the
[Microsoft GraphRAG](https://github.com/microsoft/graphrag) knowledge-model format.
Part of the **kg-triplet-sft** project: {repo_hint}

- Base model: `{base}`
- Output types: `PERSON` / `ORGANIZATION` / `GEO` / `EVENT` / `CONCEPT`; relationship `strength` is 0–10.
- Entity titles follow the ALL-CAPS GraphRAG convention.

## Output format

```json
{{"entities": [{{"title": "...", "type": "CONCEPT", "description": "..."}}],
 "relationships": [{{"source": "...", "target": "...", "description": "...", "strength": 8.0}}]}}
```

`source`/`target` must equal a `title` in `entities`. The model returns
`{{"entities": [], "relationships": []}}` when nothing is extractable (hard-negative training).

## Prompt

The student prompt lives in the project repo (`kg_contract/student_prompt.py`); it is derived
from the official GraphRAG extraction prompt and uses one field vocabulary
(title/type/description, source/target/description/strength). At inference the project injects
a schema prefix (`{{"entities": [`) to keep small models structurally compliant.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

repo = "<this repo id>"
base = AutoModelForCausalLM.from_pretrained("{base}", torch_dtype=torch.bfloat16, device_map="auto")
tok = AutoTokenizer.from_pretrained(repo)
model = PeftModel.from_pretrained(base, repo)

prompt = STUDENT_PROMPT + "\\n\\n" + passage          # see the repo
msgs = [{{"role": "user", "content": prompt}}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training

Canonical masked recipe (Unsloth, one RTX 4090): Qwen3 in no-think chatml,
**response-only loss** (`train_on_responses_only`), non-packed (unsloth#5230),
r=32 / alpha=32, lr 5e-5 cosine, 5 epochs, cutoff 6144, bs 2 × grad-accum 4
(effective 8), bf16, 2,575 training passages.

## Evaluation

Referent-level (same real-world entity) on a fixed 200-passage sample; judge = Qwen3 thinking.
Reference baseline (same recipe, 0.6B) on this sample: entity R/P/F1 = 0.534 / 0.723 / 0.614.

| metric | value |
|---|---|
{metric_rows}

Entity recall is monotone in size across the capacity line (0.536 → 0.590 → 0.664 case-tolerant 4B).
The 0.6B point matches the reference within the sample's ±2.1 pt CI.

## Limitations

- One fixed-200 sample; single dataset (English Wikipedia 60% + arXiv 40%); no statistical power for small differences.
- ~17% of 4B rows emit UPPERCASE schema keys under some decodes; a case-tolerant key-normalizing parser recovers them (the "case-tolerant" numbers above). Deployment should normalize keys.
- Description factual accuracy is **not** verified by an LLM judge (out of scope); descriptions are grounded lexically only.
- Research artifact, not a production extractor; `serve/` (GGUF / Gradio) is not implemented.

## Evidence & links

- Example node–edge extraction (teacher vs base vs this model, fixed-200): {repo_hint}/blob/master/docs/example-graph.md
- Capacity-line repair report / root-cause (loss coverage): {repo_hint}/blob/master/docs/2026-09-07-capacity-line-repair.md
- Evaluated predictions/reports and full methodology: {repo_hint}

## License & attribution

MIT (adapters). Base models: Qwen3 (Apache-2.0). Teacher labels from the official
Microsoft GraphRAG prompt (MIT). Training data is not redistributed.
"""


def build(size: str, spec: dict, staging: Path) -> Path:
    src = ROOT / "outputs/adapters" / f"qwen3-{size}_masked" / "adapter"
    if not src.is_dir():
        raise SystemExit(f"missing adapter: {src}")
    dest = staging / f"qwen3-{size}-kg-extraction"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    cfg_path = dest / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = spec["base"]
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    rows = "\n".join(f"| {k} | {v} |" for k, v in spec["metrics"])
    (dest / "README.md").write_text(
        CARD.format(base=spec["base"], title=spec["title"], repo_hint=REPO_HINT, metric_rows=rows),
        encoding="utf-8",
    )
    return dest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--staging", type=Path, default=ROOT / "outputs/hf")
    p.add_argument("--namespace", help="HF user or org, e.g. Alex-tangt (required with --push)")
    p.add_argument("--push", action="store_true")
    p.add_argument("--private", action="store_true")
    args = p.parse_args(argv)

    dirs = {size: build(size, spec, args.staging) for size, spec in SIZES.items()}
    for size, d in dirs.items():
        print(f"staged {size}: {d}")

    if not args.push:
        print("dry run (no --push); nothing uploaded")
        return 0
    if not args.namespace:
        raise SystemExit("--namespace is required with --push")

    from huggingface_hub import HfApi

    api = HfApi()
    for size, spec in SIZES.items():
        repo_id = f"{args.namespace}/qwen3-{size}-kg-extraction"
        api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(dirs[size]))
        print(f"pushed {repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
