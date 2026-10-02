# KG Triplet SFT 项目摘要

> 截止：2026-09-21。本文根据 `AGENTS.md`、`CONTEXT.md`、`docs/adr/`、`docs/evidence/`、`docs/2026-09-07-capacity-line-repair.md` 及 `.scratch/kg-triplet-round1/` 整理。

## 一句话结论

项目已经完成了“开放式 GraphRAG 抽取模型”的主要实验闭环：从 Wikipedia/arXiv 语料、教师标注、LoRA SFT，到 referent-level 评估、容量曲线和 LightRAG 消费端验证，并已公开数据集与 0.6B/1.7B/4B adapter。当前项目的真实主线不是最初的 mohar07 typed-triplet 复刻，而是面向 LightRAG 的 `{entities, relationships}` JSON 抽取；原始复刻线、数据规模消融、归因套件和本地 demo 均未完成或已延期。

## 1. 项目是怎样发展到现在的

### 第一阶段：原始复刻计划

最初目标是复现 `mohar07/qwen3-0.6b-kg-triplets`：使用隐含 20 种关系的 typed triplet schema，按原 prompt 和训练配置完成 0.6B 复刻，再做容量曲线、数据规模消融、归因分析和 Ollama/Gradio demo。

这一阶段留下了可运行的契约模块、数据管线和 vendored HGR harness，但没有最终完成原始复刻的 Round-1 review gate。因此原项目报告中的 composite 0.6583 和 `entity_f1` 不能写成“本项目已复现”。

### 第二阶段：从 typed triplet 转向开放 GraphRAG 抽取

研究发现固定 20 关系无法表达数量、属性和大量一般事实，对下游检索型 QA 的价值有限。项目据此转向官方 Microsoft GraphRAG extraction prompt 产生的开放格式：实体包含 `title/type/description`，关系包含 `source/target/description/strength`，strength 为 0–10。

消费端原计划是 MS GraphRAG BYOG，后来因 parquet/workflow 栈复杂且脆弱，改用 LightRAG 的 `insert_custom_kg`。因此当前系统是“自有 JSON 抽取 API → LightRAG custom KG”，不是 stock LightRAG 内置抽取器的直接替换。

### 第三阶段：训练管线和容量曲线修复

早期 Unsloth capacity runs 的问题不是模型规模，而是训练目标错了：把 raw `text` 当作 continued-pretraining loss，导致约 85–90% 梯度用于预测 prompt。对照 anchor 后改成 canonical masked recipe：no-think ChatML、response-only masking、cutoff 6144、bf16、LoRA、5 epochs，并用 `run_config.json`、`format_check.json`、`mask_check.json`、`receipt.json` 留存证据。

该配方在单卡 RTX 4090 上完成 0.6B、1.7B、4B 三个容量点，并统一进入固定 200 条样本的 referent-level 评测。

### 第四阶段：评估、消费者验证和公开发布

项目放弃把字符串标题相等当作唯一实体指标，改用两级 referent matching：先做 normalized-title exact match，剩余模糊匹配由 thinking-mode `qwen3.7-flash` 按统一 rubric 判定。所有后续模型使用固定、分层的 200 条样本。

之后补充了两个证据腿：未训练 base Qwen3-0.6B baseline，以及 LightRAG consumer-impact e2e。最后发布了双语 README、MIT LICENSE、Hugging Face 数据集和三个 adapter/model card。

## 2. 当前项目状态

| 模块 | 状态 | 当前结论 |
|---|---|---|
| 数据集 | 已完成 | 3349 条 GraphRAG labels；2575 train / 75 val / 700 test；Wikipedia 60% + arXiv 40%。 |
| 教师标注 | 已完成 | 官方 GraphRAG prompt + qwen3-flash；完成清洗、端点解析和抽样 QC。 |
| 学生 prompt | 已完成 | `STUDENT_PROMPT` 是由官方 prompt 派生的单一 JSON 接口，训练/评估使用同一格式。 |
| SFT | 已完成 | canonical masked recipe 已跑通 0.6B / 1.7B / 4B。 |
| referent-level 评估 | 已完成 | 固定 200 条样本；thinking judge 是正式口径。 |
| base baseline | 已完成 | 证明 SFT 的收益不是 base model 自带抽取能力造成的。 |
| LightRAG e2e | 可行性完成 | T0 图结构明显区分模型；T1/T2 在当前小规模测试上饱和，消费端结论为 null。 |
| 公开发布 | 已完成 | GitHub、Hugging Face dataset、0.6B/1.7B/4B adapters 均已公开。 |
| 原始 typed-triplet 复刻 | 未完成/已脱离主线 | 原始 composite/entity_f1 review gate 未执行。 |
| 数据规模消融 | 延期 | issue 09，500/1300/2575 子集没有本轮完成。 |
| reproduction attribution suite | 延期 | issue 11；阈值 sweep、gold audit 等随复刻线暂停。 |
| `serve/` 本地部署 | 未实现 | GGUF/Ollama/Gradio 均未交付，目录保持空。 |
| Round 2 / 8B pass line | 未启动 | 等待对当前结果的 review 和范围决策。 |

仓库当前测试基线为 `pytest` 126 项通过；但工作树并非干净，已有若干用户本地修改/未跟踪文件，摘要文件本身也是新增文件。

## 3. 主要实验结果

### 数据和训练

- 训练数据：3349 条开放 GraphRAG 标签，最终 split 为 2575/75/700。
- 标签由官方 GraphRAG extraction prompt 生成，并经过 tolerant parsing、去重、grounding filter 和 QC。
- 新格式标签比原 typed-triplet 更长，因此训练 `cutoff_len` 固定为 6144，而不是原计划的 2048。
- canonical masked recipe 的关键点是 no-think ChatML + response-only loss；裸 `dataset_text_field` 不能视为 SFT。

### 容量曲线

固定 200 条 referent eval 上的 micro entity 指标如下：

| 模型 | Recall | Precision | F1 | 说明 |
|---|---:|---:|---:|---|
| 0.6B masked | 0.536 | 0.699 | 0.607 | 与 reference 0.614 基本同一量级 |
| 1.7B masked | 0.590 | 0.597 | 0.594 | recall 上升，但过生成导致 precision/schema 变差 |
| 4B masked（strict） | 0.595 | 0.720 | 0.651 | 部署到只接受小写 key 的消费者时的保守口径 |
| 4B masked（clean） | 0.664 | 0.743 | 0.701 | 对 schema key 做大小写容错后的能力口径 |

主要阅读方式是：规模增大带来单调的 recall 增长；4B 最强，但约 17% 输出使用了大写 schema key。`clean` 结果不能直接冒充当前严格部署口径，是否采用 case-tolerant parser 仍是部署决策。

### Base baseline

未训练的 Qwen3-0.6B 在同一协议、同一固定样本上为 micro R/P/F1 = 0.159/0.854/0.268；0.6B masked 为 0.536/0.699/0.607。因此 SFT 带来约 +0.34 F1，主要收益来自 recall 和输出契约学习，而不是 base model 已经具备完整抽取能力。base 的低 hallucination 主要是少输出造成的稀疏结果，不能单独解读为更可靠。

### LightRAG 消费端

在约 22 条 passage、16 个问题的 case study 中：

- T0 图形成已通过，抽取质量确实传导到图结构：0.6B 最大连通分量为 26，4B 为 72，gold 为 136；0.6B 图分裂为 381 个 component。
- T1 retrieval 和 T2 answer 没有形成模型层级排序；`naive` 原文模式已经能召回答案，且 retrieval budget 相对 22 条文本过宽。
- 所以当前可声称的是“格式可接入 LightRAG，质量能传导到图结构”；不能声称“抽取质量已提升 QA”。消费端结果是小规模、方向性的 null，后续需要更大的、让 graph 真正承重的 corpus 或更窄的 retrieval budget。

## 4. 已锁定的关键决策

1. **开放 GraphRAG 格式优先于 20 relation typed triplet。** 这是项目主线 pivot，记录在 ADR-0003；原 typed schema 只保留为历史复刻线。
2. **教师和学生 prompt 不合并。** 教师继续使用官方 GraphRAG prompt，保证标签质量；学生使用派生的 `STUDENT_PROMPT`，专门服务自有 JSON API。两者语义相关但字段接口有意不同，记录在 ADR-0004。
3. **单一 prompt source of truth。** 对当前 open line 而言，学生 prompt 在 train/eval/serve 间复用；不能在不同阶段手写变体。ADR-0001 的“原始 prompt 忠实复刻”只适用于 reproduction line 的历史约束。
4. **评估自己的 test split，不跨教师风格比较。** 原始 700-entry gold 不是当前主线的 scoring target；跨教师直接比较会把模型能力和标注风格混在一起，见 ADR-0002。
5. **referent identity 是实体主指标。** 标题字符串会因大小写、拼写和截断造成假阴性，因此 headline 使用 exact-title + thinking LLM judge 的两级配对，并固定 200 条样本，见 ADR-0007。
6. **不采用 no-thinking judge 作为 headline。** 探索显示其会合并同段共现概念，且无法与 thinking 结果调到可靠 parity；no-thinking 只保留为成本/诊断实验，见 ADR-0008。
7. **训练格式必须和评估格式一致。** chat/plain 混用会把协议惩罚误算成模型能力差异；canonical line 使用 no-think chat + response-only masking。
8. **容量线使用单卡 4090 + Unsloth bf16。** 这是独立、内部一致的 capacity line，不应把它和 Kaggle T4/fp16 reference line 的绝对数字无条件混为一谈，见 ADR-0006。
9. **LightRAG 采用 custom KG 写入。** 当前部署边界是“模型输出 JSON，再调用 `insert_custom_kg`”，不是替换 LightRAG 的 stock extractor，见 ADR-0009。

## 5. 当前仍需保持诚实的边界

- “项目完成”只适用于 open GraphRAG extraction line 的数据、训练、评估和公开发布闭环，不适用于最初 typed-triplet reproduction line。
- 4B `clean` 分数是大小写容错后的 capability reading；严格 parser 的 4B F1 是 0.651。若要把 0.701 作为部署结果，必须先实现并测试 key-normalizing parser。
- LightRAG e2e 是 feasibility case study，不是有统计功效的下游收益证明。
- `outputs/` 和许多数据产物被 gitignore；对外可验证数字依赖 `docs/evidence/` 中的 curated reports，而不是直接从仓库重新生成所有原始产物。
- `serve/` 尚未提供 CPU 可运行的 GGUF/Ollama/Gradio 交付物。

## 6. 发展路径和合理的下一步

已经走完的路径是：

```text
typed-triplet reproduction 计划
        ↓ 发现 schema 与消费者不匹配
GraphRAG open extraction 数据与 JSON contract
        ↓ 训练缺陷定位
canonical masked capacity line（0.6B / 1.7B / 4B）
        ↓ 评估协议升级
固定样本 referent eval + base baseline
        ↓ 消费端验证
LightRAG T0/T1/T2（T0 有信号，T1/T2 饱和）
        ↓ 对外固化
README、证据摘要、HF dataset 与 adapters
```

如果继续推进，最稳妥的顺序是：

1. 先做一次 review gate，冻结“open line 已完成、reproduction line 未完成、consumer QA 尚无增益”的对外结论。
2. 决定并测试大小写容错 parser，明确 4B 的 deploy-true 指标；同时保留 strict/clean 两套报告，避免指标口径漂移。
3. 扩大 LightRAG consumer study，使用更大的 corpus、真正 selective 的 retrieval budget 和更有区分度的问题，验证 T0 是否能继续传导到 T1/T2。
4. 只有在 review gate 后，再决定是否做 issue 09 数据规模消融、issue 11 reproduction attribution，或进入 Round 2 的 8B/pass-line。
5. 若目标从“研究证据”转为“可演示产品”，再单独启动 issue 12：GGUF/Ollama/Gradio；这属于新的交付线，不应回填为当前实验已完成。

## 7. 建议的阅读入口

- 项目总览与公开结果：[`README.md`](../README.md) / [`README.zh-CN.md`](../README.zh-CN.md)
- 领域词汇和锁定约束：[`CONTEXT.md`](../CONTEXT.md)
- 决策记录：[`docs/adr/`](adr/)
- 评估入口和命令：[`eval/README.md`](../eval/README.md)
- 容量线修复和结果：[`docs/2026-09-07-capacity-line-repair.md`](2026-09-07-capacity-line-repair.md)
- base baseline：[`docs/evidence/base-baseline.md`](evidence/base-baseline.md)
- LightRAG 结果：[`docs/evidence/consumer-e2e-lightrag.md`](evidence/consumer-e2e-lightrag.md)
- issue 状态与运行记录：[`.scratch/kg-triplet-round1/`](../.scratch/kg-triplet-round1/)
