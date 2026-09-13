# Extraction example · 抽取示例

**English** | [中文](#中文)

A single fixed-200 passage (`arxiv-01717`) extracted by the teacher (gold) and by
four models. Entity-F1 rises monotonically with model size on this passage:
base 0.47 → 0.6B 0.63 → 1.7B 0.69 → 4B **0.89**.

![Example extraction graph](figures/graph_example.svg)

| run | entities | relationships | entity-F1 |
|---|---|---|---|
| Gold (teacher) | 13 | 10 | — |
| Qwen3-0.6B base (untrained) | 4 | 3 | 0.471 |
| Qwen3-0.6B masked | 6 | 2 | 0.632 |
| Qwen3-1.7B masked | 16 | 11 | 0.690 |
| Qwen3-4B masked (clean) | 14 | 7 | **0.889** |

**Passage** (stored form, lower-cased; from the arXiv 40% of the corpus):

> we approximate the solution of some linear systems of sdes driven by a
> fractional brownian motion with hurst parameter in the wick it sense ,
> including a geometric fractional brownian motion . to this end , we apply a
> donsker - type approximation of the fractional brownian motion by disturbed
> binary random walks due to sottinen . moreover , we replace the rather
> complicated wick products by their discrete counterpart , acting on the binary
> variables , in the corresponding systems of wick difference equations . as the
> solutions of the sdes admit series representations in terms of wick powers ,

**How this example was chosen.** Over the fixed-200 sample, passages were ranked
by the 4B per-passage entity-F1 (tie-break: recall gain over the base), with all
four models producing 3–20 entities and gold 6–16. `arxiv-01717` is the top
result. Nodes are entities colored by type (PERSON / ORGANIZATION / GEO / EVENT /
CONCEPT), arrows are relationships (width ∝ teacher strength).

**Caveats.** This is one illustrative passage, not an aggregate result — the
headline numbers are the fixed-200 micro metrics in the root README. Entity
scoring is referent-level (same real-world entity, LLM judge), so a node title
may differ from the teacher's. Figures are produced by
`eval/make_example_graph.py` (deterministic; requires the local `outputs/`
artifacts).

---

## 中文

[English](#extraction-example--抽取示例) | **中文**

同一段固定 200 样本中的文章(`arxiv-01717`),由教师(gold)与四个模型抽取。
本段上实体 F1 随规模**单调上升**:base 0.47 → 0.6B 0.63 → 1.7B 0.69 → 4B **0.89**。

![示例抽取图](figures/graph_example.svg)

| 运行 | 实体数 | 关系数 | 实体 F1 |
|---|---|---|---|
| Gold(教师) | 13 | 10 | — |
| Qwen3-0.6B base(未微调) | 4 | 3 | 0.471 |
| Qwen3-0.6B masked | 6 | 2 | 0.632 |
| Qwen3-1.7B masked | 16 | 11 | 0.690 |
| Qwen3-4B masked(clean) | 14 | 7 | **0.889** |

**原文段落**(存储形态,已转小写;来自语料中 arXiv 的 40%):

> we approximate the solution of some linear systems of sdes driven by a
> fractional brownian motion with hurst parameter in the wick it sense ,
> including a geometric fractional brownian motion . to this end , we apply a
> donsker - type approximation of the fractional brownian motion by disturbed
> binary random walks due to sottinen . moreover , we replace the rather
> complicated wick products by their discrete counterpart , acting on the binary
> variables , in the corresponding systems of wick difference equations . as the
> solutions of the sdes admit series representations in terms of wick powers ,

**示例如何挑选。** 在固定 200 段上,按 4B 的逐段实体 F1 排序(同分时取相对 base 的召回增益),
并要求四个模型各产生 3–20 个实体、gold 6–16 个。`arxiv-01717` 为最高者。节点=实体,
按类型着色(PERSON / ORGANIZATION / GEO / EVENT / CONCEPT);箭头=关系,线宽 ∝ 教师强度。

**说明与边界。** 这是**单个示例段落**,不是总体结果——头条数字以根 README 的固定 200 段 micro 指标为准。
实体按 referent 级判定(同一真实实体,LLM 判定),故节点标题可能与教师不同。图由
`eval/make_example_graph.py` 确定性生成(依赖本地 `outputs/` 产物)。
