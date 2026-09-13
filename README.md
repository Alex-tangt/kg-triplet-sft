# kg-triplet-sft

Qwen3 knowledge-graph extraction: data construction → LoRA SFT → referent-level
evaluation → graph-RAG consumer validation. LoRA fine-tuning of Qwen3
(0.6B / 1.7B / 4B) to extract `{entities, relationships}` from open text in the
[Microsoft GraphRAG](https://github.com/microsoft/graphrag) knowledge-model
format, plus a referent-level evaluation harness and a
[LightRAG](https://github.com/HKUDS/LightRAG) consumer-impact study.

**中文摘要**:本项目从复现一个开源 Qwen3-0.6B 知识图谱抽取管线出发,实测其封闭
20 关系 schema 的结构性缺陷(数值类事实丢失、约 74% 真实语义关系无法表达)后,按
"消费者 = 图谱问答"重建为 GraphRAG 式开放抽取;自建 3349 段标注语料,用 LoRA
微调 0.6B/1.7B/4B,自建参照级评测,并做了消费端(LightRAG)端到端验证。

## Results at a glance

| Stage | Result |
|---|---|
| Corpus | 3,349 labeled passages (2,575 train / 75 val / 700 test), English Wikipedia 60% + arXiv 40%, sentence-boundary chunks (100–300 words). |
| Teacher | Official Microsoft GraphRAG extraction prompt (MIT) via `qwen3-flash`. |
| Referent eval | Two-tier matching (normalized-exact + thinking LLM judge); fixed 200-passage sample (entity-recall 95% CI ≈ ±2.1 pt). Full-699 reference: entity **R/P/F1 = 0.541 / 0.707 / 0.613**. |
| Training-defect fix | Raw-text LoRA SFT was a *continued-pretraining loss* (prompt ≈ 10× the answer, ~85–90% of gradient on prompt). Response-only masking lifted 200-sample micro-F1 **0.478–0.508 → 0.607**, at parity with the original-recipe control (0.614). |
| Base baseline | Untrained Qwen3-0.6B scores micro **R/P/F1 0.159 / 0.854 / 0.268**; the masked recipe reaches 0.536 / 0.699 / 0.607 → **SFT adds +0.34 F1** (not base leakage). |
| Capacity curve | Same recipe, 0.6B→4B: recall monotone, micro-F1 0.607 / 0.594 / **0.701** (4B, case-tolerant parse); 4B best on schema (0.953) and hallucination (0.084). Single-4090 cost: 0.6B ≈ 1 h, 4B ≈ 4 h. |
| Consumer e2e (LightRAG) | **Feasibility confirmed**; extraction quality transmits to graph structure (0.6B graph shatters into 381 components, largest 26, vs 4B 72, gold 136). Retrieval/answer quality **saturates at 22 passages** (raw text already answers) → consumer-level null at this scale, with a diagnosed cause. |

## Pipeline

```
         dataset/                     kg_contract/            finetune/                 eval/                       outputs/
Wikipedia/arXiv ──► chunk ──► filter ──► teacher labels ──► Alpaca ──► LoRA SFT ──► referent eval ──► LightRAG e2e
 (100–300 w,          (dedup,       (official GraphRAG      (Qwen3 0.6/1.7/4B,   (two-tier match,   (custom-KG,
  sentence bounds)     quota)        prompt, qwen3-flash)    masked recipe)        fixed 200)         T0/T1/T2)
```

## Repository map

- `dataset/` — corpus → chunk → dedup/filter → teacher labeling → Alpaca. CLIs with outputs under `dataset/data/`.
- `kg_contract/` — prompts and contract (`graphrag_prompts.py` = official GraphRAG prompt; `student_prompt.py` = the derived student prompt), validator, relations.
- `finetune/` — LLaMA-Factory yaml configs, the canonical masked recipe (`compshare/train_unsloth.py`), and the Kaggle kernels (train / infer, incl. `kernel_capinfer/` generator).
- `eval/` — referent-level eval (`referent_*.py`), consumer axes, and the LightRAG consumer-impact e2e (`lightrag_*.py`). Start at `eval/README.md`.
- `docs/` — ADRs (0001–0009), reports, the consumer-pivot diary, research notes, and `docs/evidence/` (curated evidence digests).
- `.scratch/kg-triplet-round1/` — spec and issues (run logs and receipts).

## Reproduction

Requirements: Python 3.12; `DASHSCOPE_API_KEY` in `.env` for labeling/judging
(see `.env.example`); a GPU for training (kernels target Kaggle T4 and a single
RTX 4090).

```bash
pip install -r requirements.txt          # data/eval deps; see module READMEs for training
python dataset/download.py               # → dataset/data/01_raw/
python dataset/chunk.py                  # → 02_chunks.jsonl
python dataset/filter.py                 # → corpus.jsonl (dedup, quota, sentence bounds)
# teacher labeling + split + Alpaca: see dataset/batch_infer.md and dataset/graphrag_batch.py
python finetune/compshare/train_unsloth.py --mask --merge ...   # canonical masked recipe (one 4090)
# referent eval / consumer e2e: see eval/README.md and eval/lightrag_e2e.py
```

Every data-stage script runs end-to-end on tiny fixtures; `pytest` (126 tests)
covers the data contract, prompts, and eval glue.

## Honest limitations

- The **original reproduction line is not complete**: this repo pivoted to open
  GraphRAG-style extraction, so the original's composite/`entity_f1` numbers are
  *its* reported values — not reproduced here. The faithful-reproduction review
  gate in the spec was never run.
- The LightRAG consumer study is a **case study** (~22 passages, 16 questions):
  no statistical power, and the consumer-level result is a **null** by design of
  the scale (the graph is not load-bearing when raw text already answers).
- ~17% of 4B outputs use UPPERCASE schema keys; strict parsing drops those rows
  (deploy needs a case-tolerant parser).
- `serve/` (export/GGUF/Gradio) is **not implemented**.

## Decisions & evidence

Decisions live in `docs/adr/` (see also `docs/diary/` and
`docs/research/`); curated evidence in `docs/evidence/`.

## License & attribution

MIT — see [`LICENSE`](LICENSE) and [`THIRD_PARTY.md`](THIRD_PARTY.md). Data and
model weights are not redistributed.
