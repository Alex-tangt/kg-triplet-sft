"""Export the labeled GraphRAG-style KG dataset for Hugging Face.

Source: dataset/data/graphrag_labels_full.jsonl (3,349 passages, teacher labels)
+ outputs/graphrag_batch/split.json (train 2575 / val 75 / test 700).

Writes data/{train,validation,test}.jsonl with the native schema
{id, source, text, entities, relationships} and a dataset card; with --push
uploads to <namespace>/kg-triplet-graphrag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LABELS = ROOT / "dataset/data/graphrag_labels_full.jsonl"
SPLIT = ROOT / "outputs/graphrag_batch/split.json"
SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}

CARD = """\
---
license: other
language:
- en
size_categories:
- 1K<n<10K
task_categories:
- text-generation
- token-classification
tags:
- knowledge-graph
- graphrag
- information-extraction
- ner
- qwen3
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
---

# kg-triplet-graphrag

Open-text passages annotated with `{{entities, relationships}}` in the
[Microsoft GraphRAG](https://github.com/microsoft/graphrag) knowledge-model format.
Part of **kg-triplet-sft**: https://github.com/Alex-tangt/kg-triplet-sft

- **Size**: 3,349 labeled passages — train 2,575 / validation 75 / test 699.
  (The split file lists 700 test ids; one, `wikipedia-01183`, had no teacher labels and is omitted.)
- **Domain / language**: English; Wikipedia 60% + arXiv 40%; sentence-boundary chunks (100–300 words).
- **Labels**: produced by the **official MS GraphRAG extraction prompt** (MIT) via `qwen3-flash`
  (teacher). ~15% are hard negatives judged by the teacher (gold `{{"entities": [], "relationships": []}}`).
- **Schema per row**:

```json
{{"id": "arxiv-00001", "source": "arxiv", "text": "...",
 "entities": [{{"title": "X", "type": "CONCEPT", "description": "..."}}],
 "relationships": [{{"source": "A", "target": "B", "description": "...", "strength": 8.0}}]}}
```

`type` ∈ {{PERSON, ORGANIZATION, GEO, EVENT, CONCEPT}}; `strength` ∈ [0, 10].
Entity titles follow the ALL-CAPS GraphRAG convention.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Alextgt/kg-triplet-graphrag")
print(ds["train"][0])
```

## Uses, limitations, and ethics

- Intended for supervised fine-tuning / evaluation of open information-extraction models.
  The teacher labels are model-generated, not human-annotated — expect noise.
- Decontamination: entity-level train↔test decontamination was applied (see the repo).
- **Licensing is mixed**: passage text comes from Wikipedia (CC BY-SA) and arXiv
  (per-item licenses). The label annotations are released as part of this project;
  downstream users are responsible for respecting the source licenses. See
  https://github.com/Alex-tangt/kg-triplet-sft/blob/master/THIRD_PARTY.md
- Not a factual ground truth: descriptions are grounded in the passage but not
  fact-checked.

## Citation / links

- Project + evaluation: https://github.com/Alex-tangt/kg-triplet-sft
- Models (LoRA adapters): https://huggingface.co/Alextgt/qwen3-0.6b-kg-extraction
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--staging", type=Path, default=ROOT / "outputs/hf_dataset")
    p.add_argument("--repo-name", default="kg-triplet-graphrag")
    p.add_argument("--namespace", help="HF user or org (required with --push)")
    p.add_argument("--push", action="store_true")
    p.add_argument("--private", action="store_true")
    args = p.parse_args(argv)

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    by_id = {}
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_id[r["id"]] = r

    data_dir = args.staging / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for part, out_name in SPLIT_NAMES.items():
        ids = split[part]
        rows, missing = [], []
        for pid in ids:
            r = by_id.get(pid)
            if r is None:
                missing.append(pid)
                continue
            rows.append(
                {
                    "id": r["id"],
                    "source": r.get("source"),
                    "text": r["text"],
                    "entities": r["entities"],
                    "relationships": r["relationships"],
                }
            )
        (data_dir / f"{out_name}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8"
        )
        note = f" (skipped {len(missing)} unlabeled: {', '.join(missing)})" if missing else ""
        print(f"{out_name}: {len(rows)} rows{note}")
    (args.staging / "README.md").write_text(CARD, encoding="utf-8")
    print(f"staged dataset: {args.staging}")

    if not args.push:
        print("dry run (no --push); nothing uploaded")
        return 0
    if not args.namespace:
        raise SystemExit("--namespace is required with --push")

    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = f"{args.namespace}/{args.repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(args.staging))
    print(f"pushed {repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
