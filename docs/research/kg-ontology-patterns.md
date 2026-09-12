# KG 关系 schema 模式调研 — GraphRAG 提取与图存储

Phase-1 调研文档：决定本项目（kg-triplet-sft）数据生成改用哪种关系模型。
决策链：产品形态 → 消费者 → 必须回答的查询 → schema → 数据 → 评估。

> 2026-09-11 追记：消费端已从 MS GraphRAG 换成 **LightRAG**（ADR-0009 Leg B，
> `docs/diary/2026-09-11-consumer-pivot-graphrag-to-lightrag.md`）。本文的 schema
> 结论不变——LightRAG 同样消费"实体 + 关系描述"，但下文出现的 "MS GraphRAG 索引"
> 请按 LightRAG custom KG 理解；产品名以 ADR 为准。

## TL;DR（决策摘要）

- **产品形态**：可部署的抽取模型/API，产出被一个 **MS GraphRAG 图增强问答系统**消费。
- **消费者要求**：任意文本（WP 60% + arXiv 40%）→ 图索引 → 检索邻居/社区 → 生成回答。吃的是**实体 + 关系描述**，不是 Cypher 查询。
- **结论**：数据生成切换到 **GraphRAG 式开放关系提取**（实体 + 关系描述 + strength），配三项修复（CONCEPT 实体类型、强去重、grounding 过滤）。Phase 0 实测证明：这修复了 20 集 schema 的**数量丢失**（数字保留 7:0），且格式原生适配消费方。
- 与 mohar07 复刻的分叉需新 ADR 记录（Phase 2）。

## 1. 背景：为什么重审本体

原 pipeline 复刻 mohar07 的 20 关系集（implements/trained_on/.../predecessor_of），是**复刻工件**，不是从用途设计的。两个实测缺陷：

1. **数量/属性结构性丢失**：20 集无承载数值的关系（`measured_by` 语义是"由…衡量"）；Phase 0 实测 20 集 teacher 在 10 条里**明确保留数字 0/10**。
2. **覆盖缺口**：Probe A 实测 pair 级有效 gap 74.4%（`outputs/ontology_gap.json`）——participant/designation/quantified 等真实语义类不可表达。

## 2. 图数据库怎么存关系

（以下为教科书级行业规范，原始文档待网络恢复后补抓；GitHub 网络可用，wikidata.org / en.wikipedia.org 数次超时。）

### 属性图（Neo4j）
- 节点 + **有类型的关系** + 节点/边**属性**。
- 关系类型通常来自**小词汇表**（schema 可强制）；数值等属性是**边上的 property**（如 `:POPULATION {value: 12M}`），**不是关系类型**。
- 关系类型是 schema 决策，图数据库本身不强制封闭——Neo4j 允许任意关系类型字符串。

### RDF / Wikidata
- 谓词（predicate）来自受控词汇表；属性/数值/时间用 **qualifier 通道**（如人口属性带 `point in time` qualifier）；n 元事实用 reification。
- 核心模式：**封闭谓词 + 属性修饰**，数字不发明谓词。

### 关键共同点
**关系类型封闭、可查询；数字/属性附着在关系上。** 这直接指向"数量丢失"的正确修法：不是发明"数字关系"，而是给关系加属性通道。

## 3. GraphRAG 系怎么提取（一手源）

### Microsoft GraphRAG（arXiv 2404.16130 + 官方 prompt）
- **实体**：类型**封闭**（prompt 传 `{entity_types}`，如 ORGANIZATION,PERSON,GEO,EVENT）。
- **关系**：**开放自由文本**——`relationship_description` + `relationship_strength`(数值)。无封闭关系词汇表。
- **数值处理**：进实体/关系描述（官方例 "powers 85% of premium smartphones" 在 entity_description 里），非结构化。
- **完整性**：`CONTINUE_PROMPT`/`LOOP_PROMPT` 补漏循环，直到模型答 N。
- 官方 prompt 原文已存于 `kg_contract/graphrag_prompts.py`（MIT，verbatim）。

### LightRAG（HKUDS, EMNLP2025）
- 同款开放关系 + entity/relation description + keywords + strength；两阶段（图索引 + 图检索）。

### 检索方式（为什么开放关系可行）
- Local search：问题→实体嵌入→扩展邻居→收集实体/关系描述+社区摘要→LLM。
- Global search：社区摘要 Map→Reduce。
- **检索吃的是描述和邻居，不依赖关系类型规范化**——这是 GraphRAG 敢用开放关系的原因，也是本项目消费方的直接参考。

## 4. 对我们的含义

| | 可查询 KG（不选） | GraphRAG 式（选择） |
|---|---|---|
| 消费者 | Cypher/SPARQL 结构化查询 | 图增强问答（检索+生成） |
| 关系 | 封闭类型 | 开放描述 + strength |
| 数字 | 结构化 qualifier | 进描述，活着即可 |
| 评估 | 关系对齐 | 忠实度 + 实体覆盖率 + 下游 QA |

架构推论：查询型 KG → 封闭+qualifier；**检索型 QA → 开放关系**。本项目消费者是后者，所以开放关系正确，且**原生格式直接进 MS GraphRAG 索引**。

## 5. Phase 0 试点证据（本项目实测，`outputs/graphrag_pilot/`）

10 条（3 gold + Audi + 6 分层含数字）× GraphRAG teacher vs 20集 teacher，盲评三轴。

| 轴 | GraphRAG teacher | 20集 teacher |
|---|---|---|
| 忠实度 | 大多 GOOD，1 处硬幻觉 + 拼写 | 全 GOOD，0 幻觉 |
| 数字保留 | **7/10** | **0/10** |
| 噪声 | HIGH（166 实体三联重复） | LOW |
| 完整度 | 数据密集段强 | 缺协议/著作/日期 |

修复（`dataset/graphrag.py` dedup_graphrag + `--entity-types` 含 CONCEPT）后重跑 4 条：

| passage | 实体/关系 before | after |
|---|---|---|
| arxiv-00944 | 6/3（只抽人名） | 11/10（物理内容全抽出） |
| wikipedia-00083 | 166/138 | 33/30（drop 94） |
| wikipedia-02266 | 70/54 | 23/24 |
| wikipedia-02909 | 7/7 | 15/19 |

结论：数字轴 7:0 决定性；GraphRAG 的 3 个问题（重复/幻觉/物理漏）全是机械性可修，修复后验证有效。

## 6. 下一步

- Phase 2：schema/格式规范——实体 {title,type,description} + 关系 {source,target,description,strength} 对齐 MS GraphRAG；kg_contract 迁移；新 ADR 记录与 mohar07 分叉。
- 小模型（0.6B/1.5B/3B）学开放格式的能力是本项目核心实验（容量曲线）。
- 评估切换：忠实度（LLM-judge）+ 实体覆盖率（客观）+ 下游 QA 端到端。

## 7. 来源清单

| 来源 | 类型 | 状态 |
|---|---|---|
| GraphRAG paper, arXiv 2404.16130 | 论文 | 已抓（摘要/方法） |
| microsoft/graphrag `extract_graph.py`（官方提取 prompt） | 一手代码 | 已抓全文（GitHub API） |
| HKUDS/LightRAG `lightrag/operate.py` | 一手代码 | 已定位（GitHub search） |
| Neo4j 属性图模型文档 | 官方文档 | 待网络重试 |
| W3C RDF 1.1 Concepts / Primer | 规范 | 待网络重试 |
| Wikidata:Qualifiers | 规范 | 待网络重试（多次超时） |

记录：2026-09-01。后续补抓缺失源时更新本表。
