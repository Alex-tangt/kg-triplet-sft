# LLM 标注质量审计文献调研 — 没有完整 gold 时如何估 teacher label 的 precision/recall/噪声

Phase-1 调研文档：为本项目（kg-triplet-sft）在**不重新造 gold**的前提下，客观审计
teacher（qwen3-flash × 官方 GraphRAG prompt）标签质量提供文献依据。项目现状：teacher
标签即全部评估的 ground truth；teacher 产出 ~0.92 relationship/entity 的高密度边，
student 只复现端点到端点的已匹配 teacher 边中的 ~14%（0.6B）~ ~35%（4B）。在花 GPU 做
scale / distillation / RL 之前，需要先判断这个低 edge 分数属于 (a) student 能力、
(b) teacher 过生成/噪声、还是 (c) referent-matching 指标的伪影。本文只做调研，
不给应用代码；每条论断都附一手来源，未核实的来源不写入结论。

> 追记范围：本文聚焦"无完整 gold 时的标签质量估计"。项目自身的 schema/格式决策见
> `docs/research/kg-ontology-patterns.md` 与 ADR-0003/0004；评估口径见 `eval/README.md`。

## Method（检索与核实方式）

- **检索通道**：arXiv API（`export.arxiv.org/api/query`，`search_query` + `id_list`），
  子主题关键词反复迭代（见下）。ACL Anthology / NeurIPS/ICML/ICLR 未单独抓取，
  凡能定位到 arXiv 的一律以 arXiv 摘要页为准。
- **关键词**（按子主题混用）：`LLM annotation human agreement`、`large language models
  annotators`、`silver labels`、`weak supervision label model`、`data programming`、
  `LLM-as-a-judge meta-evaluation`、`data valuation synthetic`、`synthetic data
  filtering LLM`、`relation extraction LLM evaluation`、`knowledge graph construction
  LLM evaluation`、`label noise estimation`、`audit LLM annotations`、`over-extraction /
  spurious relations`、`stratified sampling evaluation`。
- **核实方式**：每个拟引用条目都实际拉到 arXiv 摘要（`id_list` 批量 + 单条），
  记录精确 title / arXiv ID / 首次提交日期，并只引用摘要中读到的内容与数字。
- **官方工具文档**：Snorkel（`snorkel.readthedocs.io`，HTTP 200）、Cleanlab
  （`docs.cleanlab.ai`，HTTP 200）已确认可达。
- **未能核实的通道（诚实记录）**：Semantic Scholar API 与 OpenAlex 在本机共享 IP 上
  持续 429（预算耗尽）；PNAS 的 Cloudflare 拦截了 Gilardi et al. 的核查请求。
  因此**本机群关于"LLM 标注员优于众包"的原始 PNAS 论文未纳入**（见末节）。
- 说明：本文出现的 arXiv 条目均标注为 preprint；venue 未逐条核实处不臆造，
  Sources 表以 arXiv ID + 年份为准。

## Findings

### 1. LLM 当标注者 vs 人：一致性与系统性偏差

- **高一致 ≠ 可用**。MT-Bench/Chatbot Arena 的一手结论是：强 LLM judge 与人类偏好
  一致性可以很高，但存在 **position bias、verbosity bias、self-enhancement bias**
  与有限推理能力；论文据此提出缓解手段。这说明 teacher 质量审计不能只看与人的
  原始一致率。[arXiv:2306.05685]
- **位置偏差足以翻转结论**。仅改变候选项出现顺序，就能把 Vicuna-13B 在 80 条
  query 中的 66 条判为胜过 ChatGPT（用 ChatGPT 当裁判）。任何成对/排序式审计都必须
  做顺序随机化。[arXiv:2305.17926]
- **独立标注者 vs 辅助者要分开评估**。事件标注的一手实验显示：LLM 自动标注优于
  TF-IDF/Event Set Curation 基线，但相比人类专家仍**不可靠**；把 LLM 放进
  human-in-the-loop 辅助专家则可降低时间/心理负担。即"LLM 当独立标注员"与
  "LLM 当审计助手"是两个结论。[arXiv:2503.06778]
- **专家域需单独验证**。针对需专家知识的标注任务，论文系统比较了单个 LLM 与多
  LLM 组合能否直接替代人类专家，结论是不能想当然。[arXiv:2508.07827]
- **多来源更可靠**。在 keyphrase extraction 上，LLM **ensemble disagreement score**
  可作为人类标注的代理来估计真实误差（覆盖 zero/few-shot 与微调设置）。这直接支持
  "用多次采样/多 prompt 的离散度当逐边置信度"。[arXiv:2309.05619]
- **标注设计本身影响质量**。AnnoLLM 的 explain-then-annotate 两步法让 LLM 成为
  更好的众包标注员；LLMaAA 用主动学习把 LLM 当 annotator 迭代。两者都说明 prompt
  与采样策略是质量旋钮，而非固定噪声。[arXiv:2303.16854, arXiv:2310.19596]
- **人-来源对照研究可作模板**。德语 ABSA 案例用专家重标注建立 ground truth，再用
  IAA 等指标比较专家/学生/众包/LLM 四类标注源的质量——这是"小规模金标准 + 多来源
  对照"的可复制范式。[arXiv:2605.03624]

### 2. 没有完整 gold 时的标签质量估计（weak supervision / silver label）

- **label model 是核心工具**。data programming 把领域启发式表达为 noisy、
  可能冲突的 labeling functions，用生成式模型"去噪"出训练标签，**不需要 gold**；
  这是无 gold 估计标签质量的范式源头。[arXiv:1605.07723] Snorkel 将其工程化为
  rapid training data creation。[arXiv:1711.10160] 综述见
  [arXiv:2202.05433]。
- **闭式解降低门槛**。FlyingSquid 证明对该类 latent variable model 可求闭式解，
  免去 SGD 迭代，使 LF 准确率估计在数据量上可控；Hyper Label Model 进一步一次
  前向即可对任意数据集推断真标签。[arXiv:2002.11955, arXiv:2207.13545]
- **prompt 也能当 LF**。Alfred 是首个"prompted weak supervision"系统：用自然语言
  prompt 作为弱监督源，再交给 label model 消解冲突。对本项目含义：把不同的
  teacher prompt/seed/温度当作多路弱监督源，可估 teacher 的逐边可靠度。[arXiv:2305.18623]
- **逐样本置信度：Confident Learning**。CL 依据 class-conditional noise 假设直接
  估计"给定噪声标签 vs 真实标签"的联合分布，从而排序并识别疑似错标。这是无 gold
  估计 precision 侧噪声的经典一手方法。[arXiv:1911.00068] 配套工具文档见
  [Cleanlab docs]。
- **"算法筛 + 人核"已被证明高效**。Pervasive Label Errors 用 confident learning 筛出
  10 个常用数据集测试集的疑似错标，再众包人工验证：平均 ≥3.3% 标签错误
  （ImageNet val ≥6%），且**被算法标记的候选中平均 51% 确为错标**。这是"有限人工
  预算验证算法标记子集"的强证据，可直接迁移为 teacher 边审计流程。[arXiv:2103.14749]
- **自洽度是廉价信号**。self-consistency 通过在多个采样推理路径上边缘化取最一致
  答案提升正确性；同一思想可用于给 teacher 的每条边打"跨采样稳定度"。[arXiv:2203.11171]
- **子集选择：精度换覆盖**。弱监督训练并非用满所有弱标签最优；结合预训练表示与
  cut statistic 选高质量子集，可在任意 label model/分类器上即插即用。[arXiv:2206.02914]
- **无 gold 的度量本身**。GEM 用生成模型估计候选与参考之间的互信息，在
  machine translation / summarization / peer review 等**无金标准**场景做 benchmark，
  说明"以参考为分布而非唯一真值"的评估可绕过 gold。[arXiv:2411.07127]
- **数据价值/筛选**：Data Shapley 给出个体数据点的公平价值定义，可作为
  "哪些 teacher 标签值得留"的框架。[arXiv:1904.02868] LLM 合成数据的生成/筛选/
  评估统一综述见 [arXiv:2406.15126]。LLM+weak supervision 联合用于社媒标注的
  实证见 [arXiv:2309.06503]。

### 3. 小样本审计的统计策略（audit budget）

- **子集选择可替代全量人评**。Metric Match 从有限标注预算中**选出一个子集**，使
  其上的 judge 相关性指标逼近总体指标；论文报告相对随机子集选择的 win-rate 0.838
  （四个相关性口径）。这正是"如何花有限审计预算"的一手方法。[arXiv:2606.15029]
- **选择性人工监督 = positive-unlabeled**。把审计形式化为 PU 学习：人类只可靠地
  给出少量 positive，其余 unlabeled 且质量混杂；用 Partial Optimal Transport 在
  固定嵌入空间做几何审计，并明确指出 **verbosity bias 与语义质量解耦**。
  [arXiv:2606.19057]
- **探测式 human-in-the-loop 审计**。LLMAuditor 对同一问题生成多版本 probe，
  用不一致暴露 bias/幻觉，并给出可自动化的审计流程。[arXiv:2402.09346]
- **主动学习 + 弱监督协同**。ActiveDP 把主动学习（精确但只覆盖小部分）与 data
  programming（覆盖广但有噪声）结合，兼顾准确率与覆盖——即"审计后修正标签"的
  闭环。[arXiv:2402.06056]
- **分层 + 代理估计**。Model-assisted estimation 用 surrogate 输出配合分层抽样，
  把人工评分只花在少量样本上以估计总体指标。[arXiv:2602.12992] 高风险
  RAG 场景的 human-calibrated 测试/验证框架同理。[arXiv:2411.16391]
- **评估本身会偏**。GraphRAG 的 answer 评估被抓出两个结构性缺陷——**unrelated
  questions** 与 **evaluation bias**——并提出 graph-text-grounded 的问题生成 +
  去偏评估框架。这提示：低分也可能来自评估集/指标设计，而非模型。[arXiv:2506.06331]

### 4. 过生成 / 幻觉边（precision 侧）与漏抽（recall 侧）如何分离

- **传统 P/R 会冤枉生成式抽取**。GenRES 指出：生成式 relation extraction 常产出
  与人工参考**不同但语义正确**的关系，exact-match 的 precision/recall 因此失真；
  论文改为多维评估（topic similarity、uniqueness、granularity、factualness、
  completeness）。这是"低 edge 分可能是指标伪影"的直接证据。[arXiv:2402.10744]
- **漏抽是独立失效模式**。针对关系三元组抽取，LLM 在复杂句上会**漏掉**关系；
  论文用"高精度小模型筛实体对 + LLM 抽取"的 evaluation-filtering 框架补 recall。
  即 recall 侧需要专门机制，与过生成分开测量。[arXiv:2404.09593]
- **过生成/不相关混淆可量化**。CORE 构建 225K 选择题（74 学科，24 种语义关系，
  正/不相关对等量）并做人评验证；人类基线 92.6%（不相关对 95.1%），而 29 个 SOTA
  LLM 仅 48.25–70.9%，且**在有关系对上接近天花板、在不相关对上崩**。这把
  "过生成"具体化为"无法区分'有关系'与'真无关'"。[arXiv:2602.06446]
- **precision 侧的通用估计器**：Confident Learning（§2）给出 principled 的错率
  估计；"算法标记 + 人工验证子集"的分工（51% 命中）是可操作流程。[arXiv:1911.00068,
  arXiv:2103.14749]
- **漏抽（recall）在没有 gold 时更难**：本会话未找到面向"结构化抽取教师漏抽率"
  的成熟无 gold 估计器；可用的替代是弱监督子集/覆盖分析 [arXiv:2206.02914] 或
  多 teacher 并集当上界（见 Uncertainties）。

### 5. 用第二个（更强）LLM 当审计者：何时可信、何时失真

- **裁判必须做 meta-evaluation**。MM-Eval 是面向 LLM-as-a-Judge / reward model 的
  多语言 meta-eval benchmark，核心论点是判官可靠性需按语言/任务单独检验，不能默认
  "强模型 = 好裁判"。[arXiv:2410.17578]
- **原始一致率会系统性高估**。2026 年大规模评估（21 judges × 9 providers ×
  MT-Bench/JudgeBench/RewardBench，118 runs ≈ 541,000 次判断）发现：exact-match
  agreement 不校正 chance，普遍高估区分力，**kappa 与 exact match 的落差在 MT-Bench
  上达 33–41 个百分点**。审计裁判可靠性必须报 Cohen's kappa 等 chance-corrected 指标。
  [arXiv:2606.19544]
- **多裁判/多阶段可降偏**。meta-judge 多智能体框架用"多轮证据 + 裁判选择"来评估
  多个 LLM 判断，并显式承认人类判断本身有偏。[arXiv:2504.17087]
- **领域裁判需要基准化**。生物医学关系抽取上，论文 benchmark 8 个 LLM 当 judge，
  验证 LLM-as-judge 可替代昂贵人工评估的程度与边界。[arXiv:2506.00777]
- **补充（仅标题级核实）**：另有针对 judge 可靠性的压力测试与偏差缓解工作，
  如 [arXiv:2603.05399]、[arXiv:2604.23178]、[arXiv:2607.08535]，可作后续精读入口。

### 6. 关系/三元组特有的标签质量评估

- **Open 关系抽取的质量轴**：GenRES 的多维指标（topic similarity / uniqueness /
  granularity / factualness / completeness）可直接映射为项目关心的
  "过生成（granularity、uniqueness）、漏抽（completeness）、忠实（factualness）"。
  [arXiv:2402.10744]
- **KG 构建的 LLM vs 专用模型对照**。以可持续性文本为用例，比较 ChatGPT 与
  REBEL 的联合 entity+relation extraction，并对自动构建的 KG 做评估——是
  "LLM 抽取器质量"的可复制评测模板。[arXiv:2305.04676]
- **重复实体/关系是独立噪声源**。iText2KG 明确指出未解析与语义重复的
  entity/relation 仍会污染图，需要增量消解。[arXiv:2409.03284] 另有 slot-based 的
  accountable 三元组抽取 [arXiv:2504.12816]（仅标题级核实）。
- **下游效用可当验证信号**。个人 KG 场景用 Qwen/Gemma 抽取 RDF 三元组并链接
  Wikidata，评估**语义抽取保真度 + 下游推荐效用**两个面；这支持"用 consumer 指标
  反查标签质量"。[arXiv:2607.00003]
- **关系"真伪/相关"判断本身是 LLM 弱项**：CORE 显示 LLM 难分"有意义关系"与
  "真无关"。[arXiv:2602.06446]

## Transferable audit methods（按可落地性排序）

| # | 方法 | 成本（人 / LLM） | 测量什么 | 主要 caveat |
|---|---|---|---|---|
| 1 | **分层人工审计 teacher 边 + 置信区间**：按 relation/strength/实体度分层抽边，双人按 rubric 判"真/过生成/无法判"，报 precision 与 CI。 | 小（~150–300 条边 × 2 人），一次性 | teacher 边 precision（直接回答 b）；分层可定位哪类边最脏 | 边匹配与"语义正确但不同表述"会分歧；需先冻结判定 rubric 与匹配规则（GenRES 警告 exact-match 会低估）[2402.10744] |
| 2 | **多源 agreement + label model**：把 teacher 的多次采样/多 prompt/多 seed（或不同模型）当 labeling functions，用 Snorkel/FlyingSquid/Hyper Label Model 估逐边正确率并聚合。 | LLM token 便宜，N 次调用；无人工 | 全量 teacher 边的错误概率分布、噪声水平 | 需要 ≥3 个条件（近似）独立源；源相关会偏乐观 [1605.07723,2002.11955,2207.13545]；Agreement 高不等于对（§5）[2606.19544] |
| 3 | **Confident-learning 筛 + 有界人核**：训一个小的逐边判真模型（或 student 自身），用 Cleanlab 排序疑似错标，只人工核 top-K。 | LLM/GPU 训练 + 小 K 人核 | precision 侧错误率；可复现 Pervasive Label Errors 的 51% 命中范式 | 依赖 class-conditional noise 假设；结构化输出需先降维到"逐边二分类"代理 [1911.00068,2103.14749] |
| 4 | **预算最优子集选择 / PU 审计**：用 Metric Match 式子集选择或 PU + Partial Optimal Transport，把有限人工花在"最能代表总体"的样本上。 | 小人工，需可靠 surrogate 嵌入 | 总体 precision/judge 相关性指标及 CI；能发现 verbosity 等与质量解耦的偏差 | 需可靠的代理信号与无分布漂移假设 [2606.15029,2606.19057] |
| 5 | **自洽度当逐边置信 + 主动式 probe**：同一条边多次采样，离散度高者优先送审；LLMAuditor 式多版本 probe 暴露不一致。 | LLM token，极小人工 | 便宜的全量不确定度排序 | 会漏掉"稳定地错"的边（systematic error）；只能排序不能给绝对 precision [2203.11171,2402.09346] |
| 6 | **第二 LLM 当审计者（kappa 校正 + 顺序随机化）**：用更强 judge 判 edge 真伪，报 Cohen's kappa 而非 exact-match，并随机化边顺序。 | LLM token；微人工抽检校准 | 可扩展的伪标签审计；judge 与人的 chance-corrected 一致性 | 裁判共享 teacher 偏差、自查偏高；必须 meta-eval 校准，且 exact-match 会高估 [2410.17578,2606.19544,2305.17926] |
| 7 | **数据价值 / 影响函数归因**：用 Data Shapley / influence 估计哪些 teacher 标签对 student 贡献为负。 | GPU 高；几乎无人工 | 把"student 复现率低"归因到具体标签的效用 | 噪声大、计算贵，且回答的是"效用"而非"标签真伪" [1904.02868] |

**对项目的直接建议（非执行）**：要分离 (a)(b)(c)，方法 1 给 teacher precision，
方法 2/5 给全量噪声分布，方法 3 把人工预算集中到最可疑的边；同时用 GenRES 的多维
口径重跑 student 评估，检验 (c) referent-matching 是否在惩罚"语义正确但表述不同"的边。

## Uncertainties / not found

- **"LLM 标注员优于众包"的 PNAS 原始论文未能核实**（Cloudflare 拦截；S2/OpenAlex
  在本机共享 IP 上 429），故本文不写入任何相关数字或结论。需要时应在可访问网络下
  补核。
- **未找到面向"结构化抽取教师漏抽率"的无 gold 估计器**。§4 的 recall 侧主要靠
  主动补抽 [2404.09593] 与子集/覆盖分析 [2206.02914] 间接支持；"teacher 漏了多少
  真边"在无 gold 时本质上需要至少部分人工或第二 teacher 上界。
- **未见把 Snorkel 式 label model 直接用于关系三元组（而非分类）的一手论文**。
  LF-over-edges 的做法是把结构化抽取降维为逐边二分类后的合理外推，非原文验证。
- **样本量/功效分析（power analysis）针对 annotator 的专门论文未检索到**；
  最接近的是模型辅助分层估计 [2602.12992] 与子集选择 [2606.15029]。
- 多条 2026 年 judge 可靠性论文仅做了标题级核实（标 [arXiv:...] 于 §5），未读摘要，
  不应据此下强结论。
- 所有 arXiv 条目的正式 venue 未逐条核实；Sources 以 arXiv ID + 年份呈现，
  引用时请以 arXiv 摘要页为准。

## Sources

| # | 标题 | arXiv ID | 年 | 核实程度 |
|---|---|---|---|---|
| 1 | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | 2306.05685 | 2023 | 摘要已读 |
| 2 | Large Language Models are not Fair Evaluators | 2305.17926 | 2023 | 摘要已读 |
| 3 | Large Language Models Are Effective Human Annotation Assistants, But Not Good Independent Annotators | 2503.06778 | 2025 | 摘要已读 |
| 4 | Evaluating Large Language Models as Expert Annotators | 2508.07827 | 2025 | 摘要已读 |
| 5 | Effective Proxy for Human Labeling: Ensemble Disagreement Scores in LLMs for Industrial NLP | 2309.05619 | 2023 | 摘要已读 |
| 6 | AnnoLLM: Making Large Language Models to Be Better Crowdsourced Annotators | 2303.16854 | 2023 | 摘要已读 |
| 7 | LLMaAA: Making Large Language Models as Active Annotators | 2310.19596 | 2023 | 摘要已读 |
| 8 | Annotation Quality in Aspect-Based Sentiment Analysis: Comparing Experts, Students, Crowdworkers, and LLM | 2605.03624 | 2026 | 摘要已读 |
| 9 | Are Large Language Models Reliable Argument Quality Annotators? | 2404.09696 | 2024 | 标题级 |
| 10 | Data Programming: Creating Large Training Sets, Quickly | 1605.07723 | 2016 | 摘要已读 |
| 11 | Snorkel: Rapid Training Data Creation with Weak Supervision | 1711.10160 | 2017 | 摘要已读 |
| 12 | Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods | 2002.11955 | 2020 | 摘要已读 |
| 13 | Learning Hyper Label Model for Programmatic Weak Supervision | 2207.13545 | 2022 | 摘要已读 |
| 14 | Alfred: A System for Prompted Weak Supervision | 2305.18623 | 2023 | 摘要已读 |
| 15 | A Survey on Programmatic Weak Supervision | 2202.05433 | 2022 | 摘要已读 |
| 16 | Confident Learning: Estimating Uncertainty in Dataset Labels | 1911.00068 | 2019 | 摘要已读 |
| 17 | Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks | 2103.14749 | 2021 | 摘要已读 |
| 18 | Self-Consistency Improves Chain of Thought Reasoning in Language Models | 2203.11171 | 2022 | 摘要已读 |
| 19 | Training Subset Selection for Weak Supervision | 2206.02914 | 2022 | 摘要已读 |
| 20 | Benchmarking LLMs' Judgments with No Gold Standard (GEM) | 2411.07127 | 2024 | 摘要已读 |
| 21 | Data Shapley: Equitable Valuation of Data for Machine Learning | 1904.02868 | 2019 | 摘要已读 |
| 22 | On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey | 2406.15126 | 2024 | 摘要已读 |
| 23 | Leveraging Large Language Models and Weak Supervision for Social Media Data Annotation | 2309.06503 | 2023 | 摘要已读 |
| 24 | Metric Match: A Subset Selection Approach to Evaluating LLM Judge Reliability | 2606.15029 | 2026 | 摘要已读 |
| 25 | Quantifying and Auditing LLM Evaluation via Positive-Unlabeled Learning | 2606.19057 | 2026 | 摘要已读 |
| 26 | LLMAuditor: A Framework for Auditing Large Language Models Using Human-in-the-Loop | 2402.09346 | 2024 | 摘要已读 |
| 27 | ActiveDP: Bridging Active Learning and Data Programming | 2402.06056 | 2024 | 摘要已读 |
| 28 | Stratified Sampling for Model-Assisted Estimation with Surrogate Outcomes | 2602.12992 | 2026 | 摘要已读 |
| 29 | Human-Calibrated Automated Testing and Validation of Generative Language Models | 2411.16391 | 2024 | 摘要已读 |
| 30 | How Significant Are the Real Performance Gains? An Unbiased Evaluation Framework for GraphRAG | 2506.06331 | 2025 | 摘要已读 |
| 31 | GenRES: Rethinking Evaluation for Generative Relation Extraction in the Era of LLMs | 2402.10744 | 2024 | 摘要已读 |
| 32 | Improving Recall of Large Language Models: A Model Collaboration Approach for Relational Triple Extraction | 2404.09593 | 2024 | 摘要已读 |
| 33 | CORE: Comprehensive Ontological Relation Evaluation for Large Language Models | 2602.06446 | 2026 | 摘要已读 |
| 34 | MM-Eval: A Multilingual Meta-Evaluation Benchmark for LLM-as-a-Judge and Reward Models | 2410.17578 | 2024 | 摘要已读 |
| 35 | Leveraging LLMs as Meta-Judges: A Multi-Agent Framework for Evaluating LLM Judgments | 2504.17087 | 2025 | 摘要已读 |
| 36 | Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge | 2606.19544 | 2026 | 摘要已读 |
| 37 | Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge | 2506.00777 | 2025 | 摘要已读 |
| 38 | Enhancing Knowledge Graph Construction Using Large Language Models | 2305.04676 | 2023 | 摘要已读 |
| 39 | iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models | 2409.03284 | 2024 | 摘要已读 |
| 40 | From "Strings" to "Things" for Personal Knowledge Graphs: Evaluating LLM Triple Extraction | 2607.00003 | 2026 | 摘要已读 |
| 41 | SMARTe: Slot-based Method for Accountable Relational Triple Extraction | 2504.12816 | 2025 | 标题级 |
| 42 | Judge Reliability Harness: Stress Testing the Reliability of LLM Judges | 2603.05399 | 2026 | 标题级 |
| 43 | Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge | 2604.23178 | 2026 | 标题级 |
| 44 | When the Judge Changes, So Does the Measurement: Auditing LLM-as-Judge Reliability | 2607.08535 | 2026 | 标题级 |

官方工具文档：

| 来源 | 类型 | 状态 |
|---|---|---|
| Snorkel API docs, https://snorkel.readthedocs.io/en/latest/ | 官方文档 | HTTP 200 已确认 |
| Cleanlab docs, https://docs.cleanlab.ai/ | 官方文档 | HTTP 200 已确认 |

记录：2026-09-19。检索受共享 IP 限流影响（S2/OpenAlex 不可用），后续补充时更新本表。
