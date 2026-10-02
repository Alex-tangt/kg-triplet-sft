# 多模型 / 多智能体标签审计与纠错 — 小人工锚点、有效不确定性、判官漂移监测

Phase-1 调研文档：项目（kg-triplet-sft）的 teacher（qwen3-flash × 官方 GraphRAG prompt）
标签即全部评估的 ground truth。owner 想**审计并纠正** teacher 标签，同时把人工工作量压到
最小，并相信当前 LLM 在文本信息处理上已可与人持平或更强。拟建**多模型 / 多智能体审计+纠错
系统**，但要保留一个**小规模人工锚**，用来判断 LLM 判官是否"跑偏"（drift）、并据此知道
判官自身的精度。本文回答两个问题：(1) 多个 LLM 能否组合成带**有效不确定性**的审计与纠正；
(2) 如何用少量人工**校准/监测**判官，从而检测漂移与偏差。本文只做一手文献调研，不写应用
代码；每条论断附一手来源，未核实者标注。

> 与已有调研的分工：无完整 gold 时估 teacher precision/噪声见
> `docs/research/llm-label-quality-audit.md`；抽取评估与训练信号（scale/distill/RL）见
> `docs/research/kg-extraction-eval-and-training-signals.md`。**本文聚焦**：多模型/多智能体
> 审计架构、PPI 小人工锚、判官 meta-eval 与漂移监测、以及生成→验证→纠正闭环。

## Method

- **检索通道**：arXiv API（`export.arxiv.org/api/query`，`search_query` + `id_list`）。
  ACL/NeurIPS/ICML 未单独抓取；凡能定位 arXiv 者以 arXiv 摘要页为准。
- **关键词**（按子主题迭代）：`LLM annotation human agreement`、`correlated errors language
  models`、`Dawid-Skene`、`data programming`、`Snorkel weak supervision`、`FlyingSquid`、
  `hyper label model`、`multi-agent debate factuality`、`self-consistency factuality`、
  `critic verifier`、`prediction-powered inference`、`PPI++`、`LLM judge calibration`、
  `Krippendorff`、`inter-annotator agreement large language models`、`selective annotation`、
  `active learning LLM annotation`、`knowledge graph error detection LLM`、`data cleaning
  large language models`、`majority voting large language models`、`self-preference`。
- **核实方式**：所有拟引用条目用 `id_list=` 批量拉取元数据，逐条核对 exact title /
  arXiv ID / 首次提交日期 / `journal_ref`。**venue 只在元数据明确写出时才标注**，否则只写
  "arXiv:ID (year)"，不凭记忆补。两个已知错误 ID 已丢弃（`2308.06912` 实为 CausalLM
  论文、`2308.02847` 为物理论文，与检索意图无关）。
- **经典方法论文献非 arXiv 通道**：Dawid-Skene(1979)、Cohen's kappa(1960)、Fleiss(1971)、
  Krippendorff alpha(2011) 为领域规范一手方法来源，按规范引用，非 arXiv。
- **局限**：文献几乎没有"开放自由文本关系 + strength、100–300 词 chunk"这一确切设定的
  多模型审计对照实验；下述结论多为邻近任务（分类标注、QA/事实核查、通用 RE、表格清洗）
  的外推，迁移距离随文标注。

## Findings

### 1. LLM 标注者 vs 人：何时追平/超过、何时失效、误差为何相关

- **在部分分类任务上可追平甚至超过人与众包**。Törnberg（`2304.06588`）在 2020 美国大选
  Twitter 政治立场分类上，ChatGPT-4 零样本的**准确率更高、可靠性更高、偏差等于或低于**
  专家与众包，且对照有 ground truth。这是"LLM 标注者可优于人"的 arXiv 一手证据（同一
  结论的 PNAS Gilardi 原文本会话未能核实，见 Uncertainties）。
- **"独立标注员"与"审计助手"是两个结论**。事件标注的端到端实验（`2503.06778`）显示 LLM
  自动标注优于 TF-IDF/Event Set Curation 基线，但相比人类专家**仍不可靠**；而把 LLM 放进
  human-in-the-loop 辅助专家可降低时间与负担。即：可以当助手，不宜当唯一判官。
- **专家域需单独验证**。`2508.07827` 系统比较单 LLM 与多 LLM 组合能否替代专家标注，结论
  是不能想当然；`2506.00777`（生物医学关系抽取）发现 8 个 LLM 当 judge 的准确率**通常
  <50%**，主因是被评输出格式不统一；强制结构化输出后 judge 平均提升约 15%。
- **误差高度相关是核心失效模式**。`2506.07962` 对 350+ LLM 的大规模评测发现：模型错误
  **显著相关**（某 leaderboard 数据集上，两模型都错时仍有 60% 同时同意），共享架构/厂商
  是驱动因素，且**越大越准的模型其错误越相关**。这直接打击"换一个模型就独立"的假设。
- **高一致 ≠ 正确**。exact-match agreement 不校正 chance，会系统性高估判官区分力
  （`2606.19544`，21 judges × 9 providers × 118 runs ≈ 541,000 次判断；kappa 与 exact-match
  的落差在 MT-Bench 上达 33–41 个百分点——本数字取自本仓 `llm-label-quality-audit.md` 对
  该摘要的转述）。原始一致率必须配 chance-corrected 指标。
- **裁判有系统偏差**：position bias（`2305.17926` 翻转眼标即可翻转结论）、verbosity /
  self-enhancement（`2306.05685`）、self-preference（`2404.13076`：判官能识别并偏好自己的
  产出）。`2404.18796` 用**多模型陪审团（PoLL）**替代单个大模型，报告更便宜且 intra-model
  bias 更低。
- **人-来源对照是可复制模板**：`2605.03624` 用专家重标注建立 gold，再比专家/学生/众包/
  LLM 四类来源；`2501.08167`、`2507.14384`、`2501.02532`、`2606.26541` 都在质性编码/内容
  分析上做了人-LLM 对照，均支持"先建小 gold、再校准"。**注意**：`2501.02532` 等多为
  合作者标注而非独立金标准，结论强度受限。

### 2. 多模型集成与无 gold 聚合：label model 能估什么、需要几个源

- **Dawid-Skene 是范式源头**。DS（1979，JRSS-C）用 EM 估计每个标注者的混淆矩阵与每个
  条目的真标签；能**逐条输出后验错误概率**。其可识别性通常要求 **≥3 个（条件）独立标注
  源**——2 个源时人类准确率与标签在无假设下不可分辨。加速版见 Fast Dawid-Skene
  （`1803.02781`，hard-EM），多 schema 扩展见 `1906.01251`。
- **Data Programming / Snorkel 把启发式当"标注函数"**：`1605.07723` 用生成式模型对 noisy、
  可能冲突的 labeling functions 去噪，无需 gold；`1711.10160` 工程化为 Snorkel；综述
  `2202.05433`。
- **闭式解与单次推断**：FlyingSquid（`2002.11955`）证明该类 latent-variable model 有闭式
  解（triplet method），免 SGD；Hyper Label Model（`2207.13545`）一次前向即可对任意数据集
  推断真标签。二者都要求 ≥3 个近似独立源。
- **prompt 也能当标注函数**：Alfred（`2305.18623`）用自然语言 prompt 作弱监督源再交给
  label model 消解冲突。对本项目含义：把 teacher 的多 prompt/多 seed/多模型版本当多路
  弱监督源即可估计逐边可靠度。
- **逐样本置信度与错标排序**：Confident Learning（`1911.00068`，JAIR）按 class-conditional
  noise 估计"给定噪声标签 vs 真标签"的联合分布；Pervasive Label Errors（`2103.14749`，
  NeurIPS 2021 D&B）用 CL 筛疑似错标再人工核，平均 ≥3.3% 测试标签错、ImageNet val ≥6%，
  **被算法标记的候选中平均 51% 确为错标**——"算法筛 + 有界人核"的强证据。
- **集成离散度可当人工标签代理**：`2309.05619` 在 keyphrase extraction 上报告 LLM
  ensemble disagreement score 可作为人类标注代理估计真实误差。
- **聚合方法的共同软肋**：label model 假设各源**条件独立**；LLM 误差相关（`2506.07962`）
  会使逐条正确率估计**偏乐观**。因此必须报告"源相关性敏感性"，不能默认多数投票/label
  model 的置信就是真实置信。

### 3. 多智能体辩论 / critic-verifier / self-consistency：证据与代价

- **辩论的正面证据**：多实例提出答案并多轮辩论可提升事实性与推理（`2305.14325`，数学与
  策略推理、传记生成）；更有说服力的 LLM 辩论者能把对方引向更真实答案（`2402.06782`）。
- **但系统评测给出强烈反证**：`2502.08788` 用 5 种代表性 MAD 方法 × 9 个基准 × 4 个基础
  模型，发现 **MAD 常常打不过单智能体基线（如 Chain-of-Thought）**；`2503.13657`（MAST）
  构建 1600+ 条失败轨迹，归纳出多智能体系统反复出现的失败模式；`2602.09341` 发现审计
  推理树比多数投票与 LLM-judge 更可靠。
- **自纠正的边界**：Self-Refine（`2303.17651`）、CRITIC（`2305.11738`，工具交互式批评）、
  CoVe（`2309.11495`，先规划验证问题再作答）展示"生成→批评→修正"可降幻觉；但
  `2310.01798` 明确指出 **intrinsic self-correction 在没有外部反馈时会降低推理准确率**。
  即"自纠"必须有独立证据通道（工具、检索、另一模型），否则可能越改越糟。
- **自洽度是廉价置信信号**：Self-Consistency（`2203.11171`）在多次采样路径上边缘化取一致
  答案提升正确性；同一思想可给每条边打"跨采样稳定度"。
- **代价与规模**：辩论/多样本显著增加调用数。`2402.05120`（More Agents Is All You Need）
  报告 sampling-and-voting 随 agent 数（在任务难度范围内）提升；但 `2403.02419` 的理论+
  实证给出 compound system 的 scaling 规律，指出增益**非单调**（Filter-Vote 会先升后降）；
  `2406.04692`（Mixture-of-Agents）说明聚合多样模型确有收益。**含义**：辩论不是免费的
  正确性，需按任务与成本设实验，而非默认开启。

### 4. PPI / PPI++：小人工锚 + 全量 LLM 预测的有效估计（本问题的直接答案）

- **PPI 核心**：`2301.09633`（Angelopoulos et al.）给出在实验数据被 ML 预测"增广"时做
  **有效**统计推断的框架，可对均值/分位数/回归系数给出**可证有效的置信区间**，且**不对
  预测算法作任何假设**；预测越准，区间越紧。这正是"少量人工 + 全量 LLM 标签"的统计地基。
- **PPI++**：`2311.01453` 自动适配预测质量，区间**始终不劣于**仅用人工标签的经典区间，计算
  更省。后续变体：FAB-PPI（`2502.02363`，Bayes 辅助）、Calibeating PPI（`2604.21260`）、
  Filtered PPI（`2602.10464`）。
- **已在评测落地**：ARES（`2311.09476`，NAACL 2024）自造合成数据微调轻量 judge，并用
  **PPI + 几百条人工标注**纠偏与验证，在 8 任务与域偏移下仍有效——这是本项目"小人工锚
  验证 LLM 判官"的最直接模板。`2608.26638` 提出 prediction-powered evaluation 与
  **PPSR（human-annotation saving ratio）**，在 6 个 WMT 数据集上量化"省了多少人工"；
  `2606.05308`（排序评测的 PPI）、`2605.29249`（跨任务 PPI）、`2605.31278`（GLIDE 工程库）、
  `2609.20758`（PPI 平滑/验证）构成可落地工具箱。
- **对审计的含义**：把"全量 teacher 边"当 unlabeled、把"人工双标的小样本"当 labeled，
  PPI/PPI++ 可给出**无偏的 teacher precision 估计 + CI**；把同一 PPI 用于"判官 vs 人"即可
  估计判官精度。**代价**是 CI 宽度由人工样本量 n 与预测-人残差方差共同决定；PPI 的价值
  在于同等人力下区间更紧（或同精度下更省人力），而非零人工。**采样必须是概率样本**：
  分层抽样下的 PPI 需用分层/加权变体（`2602.12992`、`2609.20758`、`2605.29249`）。

### 5. 判官校准与漂移监测：chance-corrected、探针集、主动送审

- **必须报 chance-corrected 一致性**：Cohen's kappa（1960）、Fleiss' kappa（1971）、
  Krippendorff's alpha（2011）是规范一手方法来源。`2606.19544` 用 21 个 judge × 9 家供应商
  × 118 runs（≈541,000 判断）证明 **exact-match agreement 不校正 chance、系统性高估**，
  且这种高估跨供应商稳定存在。`2410.17578`（MM-Eval）强调判官可靠性须按语言/任务单独检验。
- **判官 meta-eval 与"多判官选判官"**：`2504.17087` 用多智能体 meta-judge 评估 LLM 判断，
  并承认人类判断本身有偏；`2404.18796`（PoLL）用多样化小模型陪审团降 intra-model bias，
  且更便宜。生物医学 RE 的 judge 基准化见 `2506.00777`。
- **小人工锚 = 校准 + 漂移探针**：把 PPI/ARES 式人工样本同时当作**密封的 canary/probe set**
  （固定一段人工标注条目），周期性重跑全部判官并追踪 kappa/alpha 与校准误差。本会话**未
  找到**专门"LLM 判官漂移监测"的一手论文；最接近的是 human-calibrated 自动化测试框架
  （`2411.16391`）与 PRISM 的迭代仿真+监控（`2605.15665`）。把 probe set 用于漂移检测属
  **工程判断**（见下节）。
- **主动/选择性送审（把人工花在刀刃上）**：Selective Annotation（`2209.01975`）与
  Active Learning Principles for ICL（`2305.14264`）证明选样本可大幅省标注；ActiveDP
  （`2402.06056`）把主动学习（精确但覆盖小）与 data programming（覆盖广但有噪）结合；
  Metric Match（`2606.15029`）从有限预算中选子集使 judge 相关性指标逼近总体；PU 几何审计
  （`2606.19057`）把审计形式化为 positive-unlabeled 并显式解耦 verbosity bias 与语义质量；
  LLMAuditor（`2402.09346`）用多版本 probe 暴露不一致。**含义**：低一致/高不确定条目优先
  送审，可与 label model 的逐条后验错误概率直接对接。

### 6. LLM 生成 KG/关系标签的验证与清洗闭环（generate → verify → correct）

- **多智能体 KG 纠错有直接一手工作**：MAKGED（`2501.15791`）用多个 LLM 协作、拼接细粒度
  子图信息做 KG **错误检测**，并强调决策透明；`2503.07282` 给出基于图的 fact-checking
  验证框架；`2409.11283` 用图上下文知识三元组建模做零资源幻觉检测。这些支持"关系边可由
  独立验证器判定"。
- **清洗闭环可行但复杂**：Cocoon（`2410.15547`）用 LLM 生成语义清洗规则 + 统计错误检测，
  并因"单次做不好"而**分解为多步**；`2502.16790` 系统评测 LLM 作数据预处理器的能力边界；
  `2509.23564`（Clean First, Align Later）基准化偏好数据清洗对下游对齐的可靠影响。
  `2604.02617`（AutoVerifier）是 agentic 自动验证框架。
- **重复/未消解是独立噪声源**：iText2KG（`2409.03284`）指出未解析与语义重复的
  entity/relation 会污染图，需增量消解；SMARTe（`2504.12816`）用 slot 结构做可问责三元组
  抽取。
- **纠正是否改善下游？** 有正面间接证据：Distill-SynthKG（`2410.16597`）报告合成+蒸馏的
  KG 质量胜 8× 大模型并改善后续检索与 QA；`2509.23564` 报告清洗改善对齐可靠性。但本会话
  **未找到**直接证明"对 teacher 标签做 LLM 审计纠正后，小模型 SFT 的边复现/下游 QA 变好"
  的一手对照实验（见 Uncertainties）。
- **评估口径警告**：GenRES（`2402.10744`）证明 exact-match 会冤枉"语义正确但表述不同"的
  生成边；CORE（`2602.06446`）显示 LLM 在"有意义关系 vs 真无关"上会崩。审计的判定 rubric
  必须先冻结，并区分"过生成 / 漏抽 / 表述差异"。

## A minimal-human audit architecture

目标：对 teacher 的 entity/edge 做**审计 + 纠正**，报告**带 CI 的**质量指标，用**尽可能小且
可复用的**人工锚验证判官并监测漂移。以下每个组件标注 `[来源]` 或 `[工程判断]`。

### 管道

```
teacher labels (entities, edges, source passage)
        │
        ├─(B) 多判官独立性检查：≥3 个异构模型，逐条判
        │      entity-real / edge-supported+faithful / missed-edge(recall probe)
        │
        ├─(C) label model 聚合 → 每条 P(error)（不用裸多数投票）
        │
        ├─(D) 小人工锚：分层概率抽样 ~150–300 条 × 2 人双标 + 仲裁
        │      → PPI/PPI++ 纠偏全量估计 + CI；同时算每个判官的 kappa/alpha
        │
        ├─(E) 选择性送审：低一致 / 高 P(error) 条目 → critic/debate → 仍不定才给人
        │
        ├─(F) 写回：修正后的 label 版本 + 逐条置信度 + 审计报告
        │
        └─(G) 漂移监测：密封 canary probe set 周期重跑，追踪 kappa/alpha/校准
```

| 组件 | 具体做法 | 依据 | 备注 |
|---|---|---|---|
| **B 判官数/异构性** | **≥3 个**（最好跨厂商）模型独立作答；顺序随机化；强制结构化输出 | `2506.07962`（误差相关、越大越相关）、`2404.18796`（多样陪审团）、`2305.17926`（顺序偏置）、`2506.00777`（结构化提升 judge） | 同源模型≈同一标注者，第 4 个同族模型边际价值低；"更多调用"非单调 `2403.02419` |
| **B 三类任务** | (i) 标题是否真实体；(ii) 边是否被 passage 支持、描述是否忠实；(iii) 漏边检测（先让模型生成候选再验证） | `2402.10744`/`2602.06446`（过生成 vs 无关）、`2404.09593`（漏抽需专门机制）、`2309.11495`（先规划验证） | recall 侧无 gold 时最难；用第二 teacher 并集当上界 `[工程判断]` |
| **C 聚合** | label model（Hyper Label Model `2207.13545` / FlyingSquid `2002.11955`）估逐条 P(error)，而非多数投票 | `1605.07723`、`1711.10160`、DS(1979) | 需把"逐边"降维成二分类 LF；**条件独立假设被 LLM 相关性违背** → 报敏感性 `[工程判断]` |
| **D 小人工锚** | 概率/分层抽样（按 edge 类型 × strength × 判官分歧分层）；2 人独立 + 1 人仲裁；冻结 rubric | `2605.03624`（专家重标建 gold）、`2305.14264`/`2209.01975`（选样本省标注）、`2606.15029`（子集选择） | n 由 CI 目标定；样本须可代表总体 |
| **D 有效性** | 用 **PPI/PPI++** 把小样本 gold 与全量 LLM 判官输出结合，报**无偏** precision/recall 估计 + CI；报 PPSR | `2301.09633`、`2311.01453`、`2311.09476`、`2608.26638`、`2606.05308` | PPI 对预测算法无假设，但要求 labeled 抽样正确；分层抽样用 `2602.12992`/`2609.20758` |
| **D 判官精度** | 在锚样本上算每个判官 vs 人的 kappa/alpha、混淆矩阵、位置/长度分层误差 | `2606.19544`、Cohen/Fleiss/Krippendorff、`2404.13076`（self-preference）、`2305.17926` | 报 chance-corrected，不报 exact-match |
| **E 选择性送审** | 高 P(error) / 多判官分歧的条目先送 critic（独立证据或辩论），仍不决才送人 | `2402.06056`、`2606.19057`、`2606.15029`、`2402.09346` | 辩论收益不稳定：`2502.08788`、`2503.13657`、`2310.01798` |
| **F 写回/纠正** | 生成修正版 teacher 标签 + 逐条置信度；保留原始版以便回滚与 A/B | `2410.15547`（多步清洗）、`2509.23564`（清洗改善对齐）、`2501.15791`（KG 纠错） | teacher 纠正是否改善 student 无直接证据 `[见 Uncertainties]` |
| **G 漂移监测** | 密封 canary probe 集（人工 gold，定期重跑）：单向追踪 kappa/alpha、P(error) 分布、各模型判官一致率；用 CI/显著性判"漂移" | `2411.16391`（human-calibrated 验证）、`2605.15665`（迭代监控）；canary 设计=`[工程判断]` | 只换模型/更新 API 时就应重跑 probe；漂移=与历史锚点显著不同 |

### 这套系统产出的统计量

1. **逐条**：label model 的 P(teacher edge 正确)、判官一致率/分歧、是否需要人审。`[2207.13545]`
2. **语料级**：PPI 纠偏后的 teacher precision（entity-real / edge-supported 两口径）、
   **带 CI**；以及 PPSR（相对纯人工省了多少）。`[2301.09633, 2311.01453, 2608.26638]`
3. **判官级**：每个判官 vs 人工锚的 kappa/alpha、准确率、偏差分层（位置/长度/自偏好）。
   `[2606.19544, 2404.13076, 2305.17926]`
4. **时间序列**：canary probe 上的上述指标随时间的移动，用于漂移告警。`[2411.16391]`
5. **recall 侧上界**：多 teacher 并集/多判官生成候选，给出漏边率的**上界**而非点估计；
   点估计仍需人工或第二独立抽取器。`[工程判断]`

### 人工预算的直觉（工程判断，非文献数字）

- label model + 多判官在**全量**上给排序与逐条概率（LLM token 成本，无人工）；
- PPI 只要求一个**概率小样本**（量级 ~150–300 条，双人双标）即可把全量指标纠偏并给 CI；
  样本量与 CI 宽度由残差方差决定，需先做一次 pilot 估方差再定 n；
- canary probe 集**复用**同一批锚样本，增量人工≈0（只需在模型/API 变更时重跑）。

## Uncertainties / not found

- **没有找到**把 Snorkel/Hyper Label Model 式 label model **直接用于开放自由文本关系边**
  的一手论文；"逐边二分类 + LF"是合理外推，非原文验证。这与
  `llm-label-quality-audit.md` 的结论一致。
- **没有找到**专门做"**LLM 判官漂移监测**（canary/probe 随时间）"的一手论文；最接近的是
  人校准验证框架 `2411.16391` 与 PRISM 监控 `2605.15665`。probe-set 漂移检测属工程判断，
  其统计阈值（CI 重叠 / McNemar / 顺序检验）本文未找到直接来源。
- **"LLM 标注者优于众包"的 PNAS 原文（Gilardi et al.）本会话未核实**（此前记录为
  Cloudflare 拦截）。`2304.06588` 是 arXiv 上的同类一手证据，可替代引用。
- **未找到**直接证明"对 teacher 标签做 LLM 审计纠正后，小模型 SFT 的边复现率或下游 QA
  变好"的一手对照实验。正面证据均为间接（`2410.16597` KG 质量/检索改善、`2509.23564`
  对齐改善）。这是本项目可自行补的空白。
- **PPI 的适用前提**：要求人工样本是总体概率样本（或正确分层加权）；teacher 边是分层/
  聚类结构（同一 passage 内相关），若不建模聚类，CI 会偏窄。本文未找到"PPI + 聚类/分层
  KG 边"的专项论文；应使用分层/加权 PPI 变体并做敏感性分析 `[2602.12992, 2609.20758]`。
- **判官数的最优值**：文献支持"≥3 源"（可识别性）与"多样异构更好"，但没有"多少模型"
  的定量最优点；`2403.02419` 指出 compound system 增益非单调。
- **venue 未逐条核实**：除 `2512.00047`（EMNLP 2025）、`1605.07723`/`2103.14749`/
  `1911.00068` 等元数据明确者外，Sources 以 arXiv ID + 年份呈现；引用以 arXiv 摘要页为准。
- 少数条目本会话仅核到标题（`2304.10145`、`2309.05619` 等），已在 Sources 标注核实程度。

## Sources

| # | 标题 | arXiv ID | 年 | 核实程度 |
|---|---|---|---|---|
| 1 | ChatGPT-4 Outperforms Experts and Crowd Workers in Annotating Political Twitter Messages with Zero-Shot Learning | 2304.06588 | 2023 | 摘要已读 |
| 2 | Can ChatGPT Reproduce Human-Generated Labels? A Study of Social Computing Tasks | 2304.10145 | 2023 | 标题级 |
| 3 | Large Language Models Are Effective Human Annotation Assistants, But Not Good Independent Annotators | 2503.06778 | 2025 | 摘要已读（ACL 2026 Findings） |
| 4 | Evaluating Large Language Models as Expert Annotators | 2508.07827 | 2025 | 摘要已读 |
| 5 | Correlated Errors in Large Language Models | 2506.07962 | 2025 | 摘要已读 |
| 6 | Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias | 2606.19544 | 2026 | 摘要已读 |
| 7 | Annotation Quality in Aspect-Based Sentiment Analysis: Comparing Experts, Students, Crowdworkers, and LLM | 2605.03624 | 2026 | 摘要已读 |
| 8 | Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge | 2506.00777 | 2025 | 摘要已读 |
| 9 | Potential and Perils of Large Language Models as Judges of Unstructured Textual Data | 2501.08167 | 2025 | 标题级 |
| 10 | Evaluating LLMs Against Human Annotators in Latent Content Analysis | 2501.02532 | 2025 | 标题级 |
| 11 | Assessing the Reliability of LLMs for Deductive Qualitative Coding | 2507.14384 | 2025 | 标题级 |
| 12 | Can LLMs Reliably Code Qualitative Humanitarian Data? | 2606.26541 | 2026 | 标题级 |
| 13 | Bias in, Bias out: Annotation Bias in Multilingual Large Language Models | 2511.14662 | 2025 | 标题级 |
| 14 | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (MT-Bench) | 2306.05685 | 2023 | 摘要已读 |
| 15 | Large Language Models are not Fair Evaluators | 2305.17926 | 2023 | 摘要已读 |
| 16 | LLM Evaluators Recognize and Favor Their Own Generations | 2404.13076 | 2024 | 摘要已读 |
| 17 | Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models | 2404.18796 | 2024 | 摘要已读 |
| 18 | Fast Dawid-Skene: A Fast Vote Aggregation Scheme for Sentiment Classification | 1803.02781 | 2018 | 摘要已读 |
| 19 | The Extended Dawid-Skene Model: Fusing Information from Multiple Data Schemas | 1906.01251 | 2019 | 标题级 |
| 20 | Data Programming: Creating Large Training Sets, Quickly | 1605.07723 | 2016 | 摘要已读（NeurIPS 2016） |
| 21 | Snorkel: Rapid Training Data Creation with Weak Supervision | 1711.10160 | 2017 | 摘要已读（VLDB 2018） |
| 22 | A Survey on Programmatic Weak Supervision | 2202.05433 | 2022 | 摘要已读 |
| 23 | Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods (FlyingSquid) | 2002.11955 | 2020 | 摘要已读 |
| 24 | Learning Hyper Label Model for Programmatic Weak Supervision | 2207.13545 | 2022 | 摘要已读 |
| 25 | Alfred: A System for Prompted Weak Supervision | 2305.18623 | 2023 | 摘要已读 |
| 26 | Effective Proxy for Human Labeling: Ensemble Disagreement Scores in LLMs for Industrial NLP | 2309.05619 | 2023 | 标题级 |
| 27 | Confident Learning: Estimating Uncertainty in Dataset Labels | 1911.00068 | 2019 | 摘要已读（JAIR 2021） |
| 28 | Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks | 2103.14749 | 2021 | 摘要已读（NeurIPS 2021 D&B） |
| 29 | Improving Factuality and Reasoning in Language Models through Multiagent Debate | 2305.14325 | 2023 | 摘要已读 |
| 30 | Debating with More Persuasive LLMs Leads to More Truthful Answers | 2402.06782 | 2024 | 摘要已读 |
| 31 | Stop Overvaluing Multi-Agent Debate — We Must Rethink Evaluation and Embrace Model Heterogeneity | 2502.08788 | 2025 | 摘要已读 |
| 32 | Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs | 2311.17371 | 2023 | 标题级 |
| 33 | Improving Multi-Agent Debate with Sparse Communication Topology | 2406.11776 | 2024 | 标题级 |
| 34 | Why Do Multi-Agent LLM Systems Fail? (MAST) | 2503.13657 | 2025 | 摘要已读 |
| 35 | Auditing Multi-Agent LLM Reasoning Trees Outperforms Majority Vote and LLM-as-Judge | 2602.09341 | 2026 | 标题级 |
| 36 | The Confident Liar: Diagnosing Multi-Agent Debate with Log-Probabilities and LLM-as-Judge | 2606.10296 | 2026 | 标题级 |
| 37 | Self-Refine: Iterative Refinement with Self-Feedback | 2303.17651 | 2023 | 摘要已读 |
| 38 | CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing | 2305.11738 | 2023 | 摘要已读 |
| 39 | Chain-of-Verification Reduces Hallucination in Large Language Models (CoVe) | 2309.11495 | 2023 | 摘要已读 |
| 40 | Large Language Models Cannot Self-Correct Reasoning Yet | 2310.01798 | 2023 | 摘要已读 |
| 41 | Self-Consistency Improves Chain of Thought Reasoning in Language Models | 2203.11171 | 2022 | 摘要已读 |
| 42 | More Agents Is All You Need (Agent Forest) | 2402.05120 | 2024 | 摘要已读 |
| 43 | Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems | 2403.02419 | 2024 | 摘要已读 |
| 44 | Mixture-of-Agents Enhances Large Language Model Capabilities | 2406.04692 | 2024 | 摘要已读 |
| 45 | LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion | 2306.02561 | 2023 | 摘要已读 |
| 46 | Prediction-Powered Inference | 2301.09633 | 2023 | 摘要已读 |
| 47 | PPI++: Efficient Prediction-Powered Inference | 2311.01453 | 2023 | 摘要已读 |
| 48 | FAB-PPI: Frequentist, Assisted by Bayes, Prediction-Powered Inference | 2502.02363 | 2025 | 标题级 |
| 49 | Calibeating Prediction-Powered Inference | 2604.21260 | 2026 | 标题级 |
| 50 | Do More Predictions Improve Statistical Inference? Filtered Prediction-Powered Inference | 2602.10464 | 2026 | 标题级 |
| 51 | ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems | 2311.09476 | 2023 | 摘要已读（NAACL 2024，PPI 用） |
| 52 | Which Metrics Save the Most Human Annotation? Prediction-Powered Evaluation and Meta-Evaluation | 2608.26638 | 2026 | 摘要已读 |
| 53 | Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference | 2606.05308 | 2026 | 标题级 |
| 54 | Prediction-Powered Inference Across Many Tasks for AI Evaluation & Social Science Research | 2605.29249 | 2026 | 标题级 |
| 55 | Industrializing Prediction-Powered Inference: The GLIDE Library | 2605.31278 | 2026 | 标题级 |
| 56 | Prediction-Powered Smoothing and Validation for Disaggregated AI Evaluation | 2609.20758 | 2026 | 标题级 |
| 57 | Stratified Sampling for Model-Assisted Estimation with Surrogate Outcomes | 2602.12992 | 2026 | 标题级 |
| 58 | Human-Calibrated Automated Testing and Validation of Generative Language Models | 2411.16391 | 2024 | 标题级 |
| 59 | PRISM: Prompt Reliability via Iterative Simulation and Monitoring | 2605.15665 | 2026 | 标题级 |
| 60 | MM-Eval: A Multilingual Meta-Evaluation Benchmark for LLM-as-a-Judge and Reward Models | 2410.17578 | 2024 | 摘要已读 |
| 61 | Leveraging LLMs as Meta-Judges: A Multi-Agent Framework for Evaluating LLM Judgments | 2504.17087 | 2025 | 摘要已读 |
| 62 | Selective Annotation Makes Language Models Better Few-Shot Learners | 2209.01975 | 2022 | 摘要已读 |
| 63 | Active Learning Principles for In-Context Learning with Large Language Models | 2305.14264 | 2023 | 摘要已读 |
| 64 | ActiveDP: Bridging Active Learning and Data Programming | 2402.06056 | 2024 | 摘要已读 |
| 65 | Metric Match: A Subset Selection Approach to Evaluating LLM Judge Reliability | 2606.15029 | 2026 | 摘要已读 |
| 66 | Quantifying and Auditing LLM Evaluation via Positive-Unlabeled Learning | 2606.19057 | 2026 | 摘要已读 |
| 67 | LLMAuditor: A Framework for Auditing Large Language Models Using Human-in-the-Loop | 2402.09346 | 2024 | 摘要已读 |
| 68 | Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from LMs | 2305.14975 | 2023 | 摘要已读 |
| 69 | Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs | 2306.13063 | 2023 | 摘要已读 |
| 70 | Harnessing Diverse Perspectives: A Multi-Agent Framework for Enhanced Error Detection in Knowledge Graphs (MAKGED) | 2501.15791 | 2025 | 摘要已读 |
| 71 | A Graph-based Verification Framework for Fact-Checking | 2503.07282 | 2025 | 标题级 |
| 72 | Zero-resource Hallucination Detection for Text Generation via Graph-based Contextual Knowledge Triples Modeling | 2409.11283 | 2024 | 标题级 |
| 73 | Data Cleaning Using Large Language Models (Cocoon) | 2410.15547 | 2024 | 摘要已读 |
| 74 | Are Large Language Models Good Data Preprocessors? | 2502.16790 | 2025 | 标题级 |
| 75 | Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment | 2509.23564 | 2025 | 标题级 |
| 76 | AutoVerifier: An Agentic Automated Verification Framework Using Large Language Models | 2604.02617 | 2026 | 标题级 |
| 77 | iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models | 2409.03284 | 2024 | 标题级 |
| 78 | SMARTe: Slot-based Method for Accountable Relational Triple Extraction | 2504.12816 | 2025 | 标题级 |
| 79 | Scaling Knowledge Graph Construction through Synthetic Data Generation and Distillation | 2410.16597 | 2024 | 标题级 |
| 80 | Enhancing Knowledge Graph Construction Using Large Language Models | 2305.04676 | 2023 | 标题级 |
| 81 | GenRES: Rethinking Evaluation for Generative Relation Extraction in the Era of LLMs | 2402.10744 | 2024 | 摘要已读 |
| 82 | CORE: Comprehensive Ontological Relation Evaluation for Large Language Models | 2602.06446 | 2026 | 摘要已读 |
| 83 | Improving Recall of LLMs: A Model Collaboration Approach for Relational Triple Extraction | 2404.09593 | 2024 | 标题级 |
| 84 | Emergent Convergence in Multi-Agent LLM Annotation | 2512.00047 | 2025 | 摘要已读（EMNLP 2025） |
| 85 | CrowdAgent: Multi-Agent Managed Multi-Source Annotation System | 2509.14030 | 2025 | 摘要已读 |

**非 arXiv 的一手方法来源（规范引用）**

| 来源 | 说明 |
|---|---|
| Dawid, A. P., & Skene, A. M. (1979). Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm. *JRSS Series C*, 28(1), 20–28. | DS 原始论文（逐标注者混淆矩阵 + EM） |
| Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1), 37–46. | kappa |
| Fleiss, J. L. (1971). Measuring Nominal Scale Agreement Among Many Raters. *Psychological Bulletin*, 76(5), 378–382. | 多评定者 kappa |
| Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability. *Departmental Papers (ASC)*, University of Pennsylvania. | alpha |

记录：2026-09-19。检索受 arXiv API 限流（429）影响，采用退避重试；所有列出的 arXiv 条目
均已按 `id_list` 核到元数据。若后续出现"LLM judge 漂移基准"或"teacher 纠正 → student SFT
改善"的一手对照实验，应更新 §5/§6 与 Uncertainties。
