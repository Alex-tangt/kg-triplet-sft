# Third-party attribution

This repository is MIT-licensed (see `LICENSE`). It uses or vendors work by
others, listed here. Data files and model weights are **not** redistributed
(gitignored); scripts fetch or document how to obtain them.

## Vendored code

- `eval/harness/` — the HGR evaluation harness (Hungarian entity alignment,
  `composite` scoring), vendored for the original reproduction line's parity.
  Upstream: the `HGR-finetuned-model-evaluation-pipeline` project. No explicit
  license is present in the vendored copy; treat as reference-only until the
  upstream license is confirmed. Not used by the open GraphRAG/referent line.

## Prompts

- `kg_contract/graphrag_prompts.py` — the official Microsoft GraphRAG entity/
  relationship extraction prompt, verbatim from `microsoft/graphrag` (MIT).

## Data

- English Wikipedia (CC BY-SA) and an arXiv abstracts dataset. Raw and derived
  data live under `dataset/data/` and are gitignored; see `dataset/download.py`
  and the dataset scripts for how to rebuild.

## Models

- Qwen3 base models (Qwen3-0.6B/1.7B/4B), Apache-2.0, via Alibaba/Qwen.
- `lightrag-hku==1.5.7` (MIT) for the consumer-impact e2e.
- Local sentence-transformers (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`,
  `all-MiniLM-L6-v2`) under their respective model licenses.
