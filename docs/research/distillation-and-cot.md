# 蒸馏与 CoT — "no-think chatml 是否就不能蒸馏？" 一手调研

服务一个具体问题：本项目（kg-triplet-sft）用标准 SFT（no-think chatml + response-only
掩码）把 Qwen3 0.6B/1.7B/4B 训成 GraphRAG 式开放抽取器；teacher 是 API（qwen3-flash，
无 logits）。owner 问："保持 no-think chatml，是不是蒸馏就没有把 chain-of-thought
蒸进来？"本文只做一手文献调研，把"蒸馏传什么"和"是否蒸 CoT"拆开，逐条给证据；
凡无一手来源支撑的论断一律标注 "not found"，不臆测。

## Method（检索与核实方式）

- **检索通道**：arXiv API（`export.arxiv.org/api/query`，`search_query` + `id_list`），
  配合 arXiv abs 摘要页逐条复核；官方来源直接抓 arXiv HTML 全文与 Hugging Face
  model card。仅收一手论文 / 官方报告 / 官方 model card，不收博客或二手综述。
- **关键词**（按子主题迭代）：`chain-of-thought distillation`、`rationale distillation`、
  `distilling step by step`、`Orca explanation tuning`、`reasoning teacher`、
  `on-policy distillation`、`generalized knowledge distillation`、`knowledge
  distillation survey`、`Qwen3 thinking mode`、`distillation small language models`、
  `structured extraction rationale distillation`、`GoLLIE`、`InstructUIE`、`MimicSFT`、
  `relation extraction chain-of-thought`、`format restrictions reasoning JSON`。
- **核实方式**：每条拟引用来源都实际拉到 arXiv 元数据（title / arXiv ID / `published`
  / `comment` / `journal_ref`），venue **仅在元数据明确写出时标注**，否则只写
  "arXiv:ID (year)"。误命中即丢弃（例：`1905.12149` 实为 SATNet，与"理解蒸馏"无关，未采用）。
- **官方文档**：`Qwen3 Technical Report` 全文 HTML 已抓（`arxiv.org/html/2505.09388v1`，
  986 KB）并按 `distill` / `Off-policy` / `thinking budget` grep；`Qwen/Qwen3-0.6B`
  HF model card 已抓全文。
- **项目内证据**：ADR-0003（teacher thinking ON 的抽取盲评）、ADR-0004（teacher/student
  prompt 故意分离）、ADR-0008（thinking vs no-thinking judge 成本/质量对照）作为
  "本项目实测"单列，与文献证据区分。
- **局限**：没有检索到"开放自由文本关系 + strength、100–300 词 chunk、0.6–4B 学生、
  teacher think/no-think 对照"这一确切设定的一手实验；下述结论多为邻近任务
  （数学 / 常识推理、NER、经典与生成式 RE、KGC、text-to-SQL、JSON 结构化输出）的外推，
  迁移距离已在文中标注。

## Findings

### 1. "知识蒸馏"到底传什么——四种机制，没有一个内在地需要 CoT

- **(a) logit / soft-label 蒸馏**：Hinton 等让学生匹配 teacher 的 softmax 软标签
  （温度缩放）。知识载体是 **teacher 的完整输出分布**，需要 white-box logits。
  全文不涉及 CoT。[arXiv:1503.02531]
- **(b) 序列 / 黑盒输出蒸馏**：Kim & Rush 把 NMT 的 KD 做到序列级——学生直接学
  teacher 生成的 token 序列（word-level 与 sequence-level 两版）。teacher 只需产出
  **文本**，无需 logits。这是"API teacher 只能给文本"时可用的主流机制，本质上与
  CoT 无关。[arXiv:1606.07947] MiniLLM 明确把 KD 场景分成两类：white-box 分类模型，
  与"模仿 black-box 模型 API 输出"——后者正是本项目的情形。[arXiv:2306.08543]
- **(c) rationale / CoT 蒸馏**：Distilling step-by-step 把 LLM 的 rationale 作为
  **额外监督**训练小模型（多任务：同时学标签与理由），用更少数据超过 few-shot
  大模型。[arXiv:2305.02301] 同类：Teaching Small LMs to Reason（小模型在
  PaLM-540B 生成的 CoT 上微调）[arXiv:2212.08410]、Fine-tune-CoT
  [arXiv:2212.10071]、SCoTD（125M–1.3B）[arXiv:2306.14050]、Orca / Orca 2 的
  explanation traces [arXiv:2306.02707, 2311.11045]。**关键：这是"训练目标里放什么
  文本"的数据设计选择，不是蒸馏机制本身的组成部分。**
- **(d) on-policy / GKD**：GKD 让学生在**自生成序列**上接受 teacher 反馈（而非只在
  固定的 teacher 输出上训练），以缓解训练/推理分布错配，并允许选 f-divergence 等
  替代损失；论文在摘要、翻译、算术推理与 instruction-tuning 上验证。这里的
  "self-generated sequence = 学生的输出轨迹"，**输出是什么内容由任务决定**，
  机制不要求它是 CoT。[arXiv:2306.13649] MiniLLM 用 reverse-KL 的 on-policy
  目标，适用于 120M–13B。[arXiv:2306.08543]
- **分类学证据**：KD 综述把知识分为 response-based / feature-based / relation-based
  三类，并综述训练方案与算法——**reasoning trace 不是独立的知识类别**，它只是
  response-based 目标的一种内容选择。[arXiv:2006.05525]

> **结论（Q1）**：四种机制里，没有任何一种内在地要求蒸馏 CoT。(a) 传 logits，
> (b) 传输出文本，(d) 传"学生对自身输出分布与 teacher 的对齐"；只有 (c) 才是
> "专门蒸理由"。所以"no-think chatml"与"是否做蒸馏"是两个正交问题。

### 2. CoT / rationale 蒸馏：有用、中性还是有害？取决于什么

**正面证据（多在数学 / 常识 / 需要多步符号操作的任务）**
- Magister 等：T5 XXL 在 GSM8K 上由 8.11% → 21.99%（PaLM-540B 的 CoT 作监督）。
  [arXiv:2212.08410]
- Fine-tune-CoT：小模型获得显著推理能力，甚至超过 prompt 基线；**每个样本采样多条
  不同 rationale** 带来额外提升。[arXiv:2212.10071]
- SCoTD：125M–1.3B 也能从 CoT prompting 受益；**每实例采样多条链**至关重要。
  [arXiv:2306.14050] Socratic CoT 在 GSM8K / StrategyQA / SVAMP 上提升 >70%。
  [arXiv:2212.00193]
- Orca / Orca 2：explanation traces 使 13B 级模型在 BBH 等上远超同规模；Orca 2
  进一步让模型**自选策略**（step-by-step / recall-reason-generate / direct），
  而非强迫一律走 CoT。[arXiv:2306.02707, 2311.11045]

**中性 / 反证（何时不该蒸 reasoning）**
- **受控最直接的证据**：Scale or Reason 用同一个 teacher 只 toggle reasoning 模式，
  生成配对的 IFT 与 reasoning 输出，隔离"监督格式"为唯一变量，在 0.5B–14B 五个
  规模、18 个基准上比等 FLOPs：**IFT 在多数配置上位于或接近 Pareto 前沿；
  reasoning 只在 7B+ 的开放式任务上才上榜**；即便那里，混入 25–50% reasoning 数据
  的顺序课程就能拿到大部分收益，算力更省。[arXiv:2509.22193]
- Distilling CoT 的因子研究（4 teacher × 7 student × 7 数据集）：**SLM 对 reasoning
  粒度呈非单调关系**——较强的学生受益于更细推理，较弱的学生用更简单的 CoT 监督
  反而更好；**更强的 teacher 不必然产生更好的学生**（多样性与复杂度可能压过准确率）。
  [arXiv:2502.18001]
- **过度思考 / 性能退化**：D-CoT 报告从 LLM 蒸 CoT 常在 SLM 上诱发 "overthinking"，
  既掉点又浪费 token；用受控推理结构可同时降 token、涨点。[arXiv:2602.21786]
- **大模型才有涌现收益**：CoT prompting 的收益在足够大的模型上才出现；SCOTT 也指出
  CoT 增益只在足够大的 LM 上观察到，且生成的 rationale 未必与预测一致（faithfulness
  问题）。[arXiv:2201.11903, 2206.07682, 2305.01879]
- **任务域很关键**：对 >100 篇论文的元分析 + 14 模型 × 20 数据集实测显示，**CoT 主要
  在数学 / 逻辑 / 符号任务上有强增益，其他任务增益小得多**；作者主张按需使用以省成本。
  [arXiv:2409.12183] 另有六类"人类深思反而变差"的任务上，CoT 让 SOTA 模型显著掉点
  （o1-preview 最多 -36.3%）。[arXiv:2410.21333]
- **理由可能不可靠 / 不可信**：few-shot 解释只带来小到中等提升，且 LLM 的解释可能
  并不蕴含其预测、并不 grounded 于输入。[arXiv:2205.03401] CoT 解释可被偏置特征
  操纵，模型会"合理化"错误答案。[arXiv:2305.04388]
- **模仿学的是风格不是能力**：在 1.5B–13B 上模仿 ChatGPT，人类评分觉得接近，但
  在模仿数据未覆盖的任务上，能力差距几乎没被补上——模仿的是 style 而非 factuality。
  [arXiv:2305.15717]
- **容量差与更强 teacher**：学生-teacher 差距过大时学生退化，需要中间助教网络桥接；
  [arXiv:1902.03393] teacher 更强时学生-教师预测差异变大，精确 KL 匹配反而干扰训练。
  [arXiv:2205.10536] 这与 SLM 蒸 CoT 的非单调性一致。
- **长 CoT 的冗余 / 延迟 / 算力**：长推理轨迹含大量冗余，学生难学；截断、剪枝或
  budget 控制在数学任务上能以少量性能代价大幅降 token / 显存 / FLOPs。
  [arXiv:2512.21002, 2511.01470, 2505.18440]
- **显式 CoT 也非唯一推理范式**：Coconut 在连续隐空间做推理，在需要大量搜索的
  规划任务上超过 CoT，并取得更好的精度-效率权衡。[arXiv:2412.06769]
  （对本项目的含义：即便想要推理，也不必以"输出自然语言 CoT"为唯一形态。）

**与结构化输出格式的交互（对本项目最关键的一组）**
- **格式限制本身就是税**：让模型以 JSON / XML 等结构化格式作答，会显著降低推理能力，
  约束越严降得越多。[arXiv:2408.02442]
- **但这是容量问题，不是格式原罪**：Capacity, Not Format 发现结构化格式的代价强烈
  依赖模型余量——有 headroom 的模型（如 Sonnet）JSON 与 CoT 几乎无差，接近能力上限的
  模型严重退化（GPT-4o-mini 即使给足 token 也掉 28.0pp）；**"先自由推理、后格式化"
  （delayed structure）能挽回大部分精度**。[arXiv:2606.09410] The Format Tax 同样
  发现格式指令本身（在任何 decoder 约束之前）就造成大部分损失，**解耦推理与格式化**
  （先自由文本再二次格式化，或单次生成内开 extended thinking）能大量恢复精度。
  [arXiv:2604.03616]
- **JSON/schema 能力本身仍是瓶颈**：SchemaBench 发现当时最新 LLM 仍难以稳定产出
  合法 JSON，需用细粒度 schema validator 做 RL 才能明显改善。[arXiv:2502.18878]

### 3. 结构化抽取 / 关系抽取：rationale 蒸馏用不用？有用吗？

**先分清 white-box vs black-box**：logit KD（(a)）与 Qwen3 的 on-policy logit 对齐
都需要 **teacher logits**（white-box）。本项目 teacher 是 API，**没有 logits**，
所以只能用 (b) 黑盒文本蒸馏，或 (d) 的"学生 rollout + teacher 文本反馈"变体。
MiniLLM 与 GKD 都把"模仿 black-box API"单列，说明黑盒是成熟场景而非缺陷。
[arXiv:2306.08543, 2306.13649]

**抽取任务上"只用输出、不用 rationale"的黑盒蒸馏证据很强**
- UniversalNER：把 ChatGPT 定向蒸馏成小模型做开放 NER，43 个数据集上平均比
  Alpaca/Vicuna 高 30+ F1，**比 ChatGPT 本体还高 7–9 F1**，并大幅超过用监督 NER 的
  InstructUIE。目标是 typed spans，**没有 rationale 蒸馏**。[arXiv:2308.03279]
- Distill-SynthKG：LLM 合成 document–KG 对，把多步流程蒸成**单步**小模型；KG 质量
  超过所有 baseline（含 8× 大的模型），并改善下游检索与 QA。[arXiv:2410.16597]
- MetaIE：label-to-span 符号蒸馏，6 类 IE 任务的更好 few-shot 起点，output-only。
  [arXiv:2404.00457] InstructUIE：统一指令微调即可零样本超 GPT-3.5，无 rationale。
  [arXiv:2304.08085] GoLLIE：起作用的显式信号是 **annotation guidelines**（写进
  训练/提示的标注指南），而非自由文本 CoT；消融显示详细 guideline 是关键。
  [arXiv:2310.03668]
- **抽取上的 rationale 正例（但通常未与 output-only 严格隔离）**：
  - Revisiting RE in the era of LLMs：用 GPT-3 生成的 **CoT 风格解释**作监督微调
    Flan-T5，在标准 RE 上达到 SOTA；prompting GPT-3 本身也近 SOTA。任务=经典 RE。
    [arXiv:2305.05003]
  - CoT Prompt Distillation（MNER/MRE）：把 LLM 的 CoT 推理蒸进紧凑学生，并用
    conditional prompt distillation 让学生**在推理时只吃文本、不需要图片也不需要
    输出 CoT**。这正说明"teacher 有推理"与"student 输出推理"可以解耦。
    [arXiv:2306.14122]
  - Contextualization Distillation（KGC）：把紧凑三元组**扩写成上下文**再蒸馏，
    在多种 KGC 框架上稳定提升。[arXiv:2402.01729]
  - MimicSFT + R²GRPO（科学 IE）：**用结构化推理模板（不需要高质量 CoT 数据）**
    做 SFT，再用 relevance + rule 奖励做 GRPO，在关系抽取上超过 baseline 与专用
    监督模型。[arXiv:2505.22068]
  - Text-to-SQL 的结构化 CoT：在 text-to-SQL 上，**结构化** CoT 比非结构化 CoT
    绝对提升 8.1%，主要减少语法错误；另一工作也用中间 SQL 步骤作 rationale 蒸馏
    提升复杂查询执行准确率。[arXiv:2512.17053, 2502.06759]
- **抽取评测的 caveat**：GenRES 指出生成式 RE 用精确 P/R 会系统性误判，应转多维
  指标（topic similarity / uniqueness / granularity / factualness / completeness）。
  [arXiv:2402.10744]

> **结论（Q3）**：抽取/RE/KGC 上，**黑盒 output-only 蒸馏是证据最扎实、最主流**的
> 做法（UniversalNER / Distill-SynthKG / MetaIE / InstructUIE / GoLLIE）。rationale
> 或结构化中间步骤的蒸馏在 RE、多模态 RE、KGC、text-to-SQL 上有正例，但（i）多与
> output-only 混杂、未隔离；（ii）student 常是较大的 Flan-T5 / 领域模型。**未检索到
> 任何一手论文显示 rationale 蒸馏在开放关系抽取上优于 output-only 蒸馏，也未检索到
> 它明确有害**——本设定下的正负对照属于文献空白。

### 4. teacher 生成标签时要不要开 thinking？teacher 的 thinking 要不要进 student 目标？

这是两个必须分开的问题。

**(i) teacher 用 thinking 产出更好的最终标签**
- **项目内实测（最强相关）**：ADR-0003 记录，对 4 条 passage 的盲评显示 teacher
  **thinking ON** 在 description grounding 与数字召回上**实质更好、零编造**，而
  thinking OFF 出现一个错标 EVENT 与一段乱译，故 data generation 固定 thinking ON。
  [ADR-0003] 另一个近似任务是 referent 配对：ADR-0008 用 12 条已知答案探针 + 100 条
  争议集对照，**no-thinking 单票 / 硬化 / 严格 / agree-only 都达不到 thinking 质量**，
  且一个 hard over-merge 在所有 no-thinking prompt 下都存活；thinking 的代价约
  2.9–3.8× token、7–25× wall。[ADR-0008]
- **文献侧**：Scale or Reason 的受控实验表明，"teacher reasoning 是否更好"**取决于
  任务与规模**：在多数配置、等算力下，非推理（IFT）监督就在 Pareto 前沿；reasoning
  只在 7B+ 开放式任务上占优。[arXiv:2509.22193] 若目标是数学类推理，过程级
  （step-level）监督显著优于结果级监督，说明"监督的步骤质量"在需要推理时确实重要。
  [arXiv:2305.20050]
- DeepSeek-R1：大模型的推理模式可系统性用于指导小模型（R1-Distill 系列），即
  **thinking teacher 的推理能力可被转移**；但注意其 student 是被训练成**输出长 CoT**
  的。[arXiv:2501.12948] s1 也说明少量高质量推理轨迹即可显著提升（同样在数学）。
  [arXiv:2501.19393]

**(ii) teacher 的 thinking 文本要不要出现在 student 的 target 里**
- **这是一个独立选择，且可以不要**：Distilling System 2 into System 1 明确把
  System 2（推理/搜索）的输出 **compile 回 System 1**——学生**不生成中间推理
  token**，直接给更好的最终答案，推理时更省。这正对应"teacher 想、student 直答"。
  [arXiv:2407.06023] CoT Prompt Distillation 同样让学生推理时**不需要输出 CoT**。
  [arXiv:2306.14122]
- **若要学生输出 CoT**，则它通常在训练成"think 模式"，且证据显示对 SLM 要下调
  CoT 粒度、避免 overthinking、防冗余与格式税。[arXiv:2502.18001, 2602.21786,
  2408.02442, 2606.09410]
- **Qwen3 官方对"teacher thinking 是否进 student 目标"给出了直接的工程答案**：见下节。

> **结论（Q4）**：teacher 开 thinking 以提升**标签/判断质量**，在本项目内有正面实测，
> 文献侧则"看任务"。但"teacher 的 thinking 文本要不要进 student target"是另一个
> 决定，主流做法之一恰恰是**不进**（S2→S1、CoT prompt distillation、Qwen3 的
> /no_think 蒸馏支路）。

### 5. Qwen3 hybrid thinking 的官方说法，及 think / no-think 训练证据

**Qwen3 Technical Report（arXiv:2505.09388）与官方 model card（`Qwen/Qwen3-0.6B`）**
- 统一 thinking（复杂多步推理）与 non-thinking（快速上下文响应）于一个模型，支持
  按 query / chat template 动态切换，并有 **thinking budget** 机制在推理时权衡
  延迟与性能。[arXiv:2505.09388]
- **开关**：`enable_thinking=True/False` 是硬开关；`enable_thinking=False` 时**不生成
  任何 think 内容、也不含 `<think></think>` 块**。`enable_thinking=True` 时可用
  `/think`、`/no_think` 软开关，但输出**始终**被 `<think>...</think>` 包裹（内容可空）。
  [HF model card]
- **最佳实践**：thinking 模式建议 `T=0.6, TopP=0.95, TopK=20`，**不要 greedy**；
  non-thinking 建议 `T=0.7, TopP=0.8, TopK=20`。多轮历史中**不要放 thinking 内容**；
  基准评测时**标准化输出格式**（给了 JSON 示例）。[HF model card]
- **小模型怎么来的（直接回答 Q5）**：对 5 个 dense 小模型（0.6B/1.7B/4B/8B/14B）与
  一个 MoE（30B-A3B）用 **strong-to-weak distillation**，两阶段：
  1. **Off-policy**：把 teacher 在 **`/think` 与 `/no_think` 两种模式下**生成的输出
     一起用于 **response distillation**，先建立基本推理能力与**模式切换能力**；
  2. **On-policy**：用学生自生成序列做微调，并**对齐 teacher logits**（Qwen3-32B /
     235B-A22B）最小化 KL。
  报告称蒸馏在性能与训练效率上**显著优于 RL**（数学/代码上约 1/10 GPU hours），
  并称"直接蒸馏 teacher 输出 logits 到轻量学生，可在保持对推理过程细粒度控制的同时
  有效提升性能"；报告中同时强调蒸馏让学生 pass@64 提升、扩展探索空间。
  [arXiv:2505.09388 全文 grep]
- 语义上，`enable_thinking=False`（无 think 块）与项目说的 "no-think chatml" 一致。

> **结论（Q5）**：Qwen3 官方**同时**蒸馏 teacher 的 think 与 no-think 输出；no-think
> 输出本身就是被蒸馏的一支。所以"用 no-think 模板训 SFT"与"做蒸馏"不矛盾——它正是
> Qwen3 官方为小模型构建模式切换能力的配方的一半。Qwen3 的 on-policy 阶段用 teacher
> logits（white-box），本项目 API teacher 无法复现该阶段，只能做其 off-policy
> response distillation（黑盒）。

## Direct answer

**问题：保持 no-think chatml，distillation 是不是就没有蒸馏 chain-of-thought？**

**不是。蒸馏本身不蒸馏 CoT。** "蒸馏"描述的是**如何让学生学 teacher**（匹配
logits / 输出序列 / on-policy 对齐），而不是"训练目标里必须有什么文本"。四种机制
(a)–(d) 没有一个内在地要求 CoT（§1）。CoT/rationale 蒸馏（(c)）是**额外的、正交的
数据设计选择**：只有在你把 teacher 的 reasoning 文本也放进 student 的 target 时，
才算"蒸了 CoT"；这需要学生训练成 think/CoT 输出，与 no-think chatml 互斥——但它是
**可选项，不是蒸馏的前提**。

**本项目现在做的正是合法的黑盒蒸馏**：teacher 开 thinking 产出最终 GraphRAG JSON，
student 在 no-think chatml 下 response-only 学这份 JSON。这等价于
"teacher 用 System 2 想、student 只学 System 1 的答案"，有专门工作支持其有效性
（Distilling System 2 into System 1 [2407.06023]；CoT 蒸馏后学生推理时不输出 CoT
[2306.14122]），也与 Qwen3 官方对 0.6–14B 的 off-policy response distillation 同构
——**Qwen3 的 `/no_think` 输出正是被这样蒸馏的** [2505.09388]。

**对"结构化 JSON 抽取"这个任务，专门蒸 CoT 值不值？**
- 证据倾向**不必需、且有风险**：抽取/RE 上最扎实的是 **output-only 黑盒蒸馏**
  （UniversalNER 学生超 teacher 7–9 F1、Distill-SynthKG 超 8× 模型、MetaIE、GoLLIE）
  [2308.03279, 2410.16597, 2404.00457, 2310.03668]；rationale 在 RE/KGC 上的正例
  [2305.05003, 2402.01729] 未与 output-only 隔离，且 student 偏大。
- 负面/边界证据：CoT 增益主要在数学/逻辑而非一般抽取 [2409.12183]；SLM 蒸 CoT 易
  overthinking、掉点 [2602.21786]；弱学生更适合简单监督、更强 teacher 不必然更好
  [2502.18001]；JSON 格式本身对接近容量上限的小模型是重税 [2408.02442, 2606.09410]，
  reasoning token 还会叠加延迟 [2512.21002]。等算力下 IFT 多数配置在 Pareto 前沿
  [2509.22193]。
- 因此：**在本项目未见证据表明 rationale 进 target 会提升开放关系抽取**；若目标是
  需要多步推理的抽取（跨句、数值/时序），可试 **结构化推理 scaffold**（MimicSFT 式，
  不要求真 CoT）[2505.22068]，但应作为显式消融，而非默认。

**teacher 侧 thinking 值不值？** 分开看：
- **生成标签/判断时**：本项目内 ADR-0003（抽取盲评）与 ADR-0008（judge）都显示
  thinking 明显更优，建议 teacher **保持 thinking ON**；这与文献一致（需要推理时
  过程级/步骤级监督更好 [2305.20050]）。
- **是否把它写进 student target**：**默认不要**。黑盒学"最终 JSON"更便宜、更省
  推理、且与 no-think 部署一致；Qwen3 官方也是把 think 与 no_think 输出分别蒸馏，
  no_think 支路不要求学生复述推理 [2505.09388]。若将来想要学生输出推理，那是
  切换到 think 训练的另一条产品线，必须重估格式与延迟。

## 对本项目（kg-triplet-sft）的直接含义

| 选项 | 机制 | teacher 需求 | 与 no-think chatml | 抽取证据 |
|---|---|---|---|---|
| **现状：黑盒 response distillation** | (b) | 只需文本（think ON 产 JSON） | 完全兼容 | 强（UniversalNER/Distill-SynthKG/MetaIE） |
| rationale/CoT 蒸馏 | (c) | 需导出 rationale 文本 | 冲突（学生须学 think/CoT） | 抽取域正例不隔离；SLM 有 overthinking/格式风险 |
| logit / white-box 蒸馏 | (a)/(d-on-policy) | **需 teacher logits** | 兼容但不可行 | 本项目 API teacher 无 logits，**不可用** |
| 学生 rollout + teacher 文本反馈（黑盒 GKD） | (d) | 只需文本 | 兼容 | 抽取域未检索到一手实证 |

- "no-think chatml" 只约束**部署/训练模板**，不约束蒸馏机制；把它与 (b) 组合是最自然、
  证据最强的路径。
- 是否要"蒸 CoT"应作为**独立实验**（例如：在训练 target 里加一段 teacher rationale，
  或加 structured reasoning scaffold），并以本项目的 referent-level + consumer 指标
  评估，不要从"蒸馏"概念上默认它必须包含 CoT。

## Uncertainties / not found

- **未检索到**直接比较"开放自由文本关系 + strength 抽取"上 rationale 蒸馏 vs
  output-only 蒸馏的一手对照实验；§3 的结论由 NER / 经典 RE / KGC / text-to-SQL
  外推，迁移距离已在正文标注。
- **未检索到**"teacher-thinking 标签 vs no-think 标签"在中大规模上的系统对照；项目内
  ADR-0003 只有 4 条 passage、单个盲评，ADR-0008 的任务是 referent 配对而非抽取，
  均不足以当作强因果证据。
- **未检索到** "thinking-mode teacher 生成标注质量提升"的通用一手基准；最接近的是
  受控的监督格式对照 [2509.22193] 与过程监督 [2305.20050]，均非开放抽取。
- **未找到**"rationale 蒸馏在开放关系抽取上明确有害"的一手结果；有害证据来自通用
  推理 / SLM overthinking / 格式税，属邻近任务外推。
- 少数 2026 年预印本（D-CoT [2602.21786]、Capacity Not Format [2606.09410]、
  Format Tax [2604.03616]、Struct-SQL [2512.17053]）仅核到摘要/元数据层面，未逐条
  核对实验细节，引用时按较低证据强度看待。
- 所有 venue 仅在 arXiv 元数据明确写出时标注；其余以 arXiv ID + year 呈现。

## Sources

> 标注格式：标题 — venue（仅当 arXiv 元数据明确写出）— arXiv:ID — 年份。
> 未标 venue 者表示元数据未提供，不臆测。核实程度：元数据已核（默认）、全文已抓（注明）、官方文档。

**蒸馏机制 / 定义**
1. Distilling the Knowledge in a Neural Network — NIPS 2014 Deep Learning Workshop — arXiv:1503.02531 — 2015
2. Sequence-Level Knowledge Distillation — EMNLP 2016 — arXiv:1606.07947 — 2016
3. Knowledge Distillation: A Survey — International Journal of Computer Vision (IJCV) 2021 — arXiv:2006.05525 — 2020
4. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD) — ICLR 2024 — arXiv:2306.13649 — 2023
5. MiniLLM: On-Policy Distillation of Large Language Models — ICLR 2024 — arXiv:2306.08543 — 2023

**CoT / rationale 蒸馏（正反）**
6. Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes — Findings of ACL 2023 — arXiv:2305.02301 — 2023
7. Teaching Small Language Models to Reason — arXiv:2212.08410 — 2022
8. Large Language Models Are Reasoning Teachers (Fine-tune-CoT) — ACL 2023 — arXiv:2212.10071 — 2022
9. Symbolic Chain-of-Thought Distillation: Small Models Can Also "Think" Step-by-Step (SCoTD) — ACL 2023 — arXiv:2306.14050 — 2023
10. SCOTT: Self-Consistent Chain-of-Thought Distillation — ACL 2023 — arXiv:2305.01879 — 2023
11. Distilling Reasoning Capabilities into Smaller Language Models (Socratic CoT) — Findings of ACL 2023 — arXiv:2212.00193 — 2022
12. Orca: Progressive Learning from Complex Explanation Traces of GPT-4 — arXiv:2306.02707 — 2023
13. Orca 2: Teaching Small Language Models How to Reason — arXiv:2311.11045 — 2023
14. Distilling System 2 into System 1 — arXiv:2407.06023 — 2024
15. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models — arXiv:2201.11903 — 2022
16. Emergent Abilities of Large Language Models — TMLR 2022 — arXiv:2206.07682 — 2022
17. To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning — ICLR 2025 — arXiv:2409.12183 — 2024
18. Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Worse — arXiv:2410.21333 — 2024
19. The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning — NeurIPS 2022 — arXiv:2205.03401 — 2022
20. Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting — NeurIPS 2023 — arXiv:2305.04388 — 2023
21. The False Promise of Imitating Proprietary LLMs — arXiv:2305.15717 — 2023
22. Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning — ACL 2025 Findings — arXiv:2502.18001 — 2025
23. Scale or Reason? A Compute-Equivalent Analysis of Reasoning Distillation — arXiv:2509.22193 — 2025
24. D-COT: Disciplined Chain-of-Thought Learning for Efficient Reasoning in Small Language Models — arXiv:2602.21786 — 2026（摘要级）
25. Let's Verify Step by Step — arXiv:2305.20050 — 2023
26. Training Large Language Models to Reason in a Continuous Latent Space (Coconut) — COLM 2025 — arXiv:2412.06769 — 2024
27. s1: Simple test-time scaling — arXiv:2501.19393 — 2025
28. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — Nature 645:633-638 — arXiv:2501.12948 — 2025
29. Distilling the Essence: Efficient Reasoning Distillation via Sequence Truncation — arXiv:2512.21002 — 2025
30. BARD: budget-aware reasoning distillation — arXiv:2511.01470 — 2025
31. Efficient Long CoT Reasoning in Small Language Models — arXiv:2505.18440 — 2025

**容量差 / 更强 teacher**
32. Improved Knowledge Distillation via Teacher Assistant — AAAI 2020 — arXiv:1902.03393 — 2019
33. Knowledge Distillation from A Stronger Teacher (DIST) — NeurIPS 2022 — arXiv:2205.10536 — 2022

**结构化抽取 / IE / RE / KGC**
34. UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition — ICLR 2024 — arXiv:2308.03279 — 2023
35. InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction — arXiv:2304.08085 — 2023
36. GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction — ICLR 2024 — arXiv:2310.03668 — 2023
37. MetaIE: Distilling a Meta Model from LLM for All Kinds of Information Extraction Tasks — arXiv:2404.00457 — 2024
38. Scaling Knowledge Graph Construction through Synthetic Data Generation and Distillation (Distill-SynthKG) — arXiv:2410.16597 — 2024
39. Contextualization Distillation from Large Language Model for Knowledge Graph Completion — EACL 2024 Findings — arXiv:2402.01729 — 2024
40. Beyond path selection: Better LLMs for Scientific Information Extraction with MimicSFT and Relevance and Rule-induced (R²)GRPO — arXiv:2505.22068 — 2025
41. Revisiting Relation Extraction in the era of Large Language Models — ACL 2023 — arXiv:2305.05003 — 2023
42. Chain-of-Thought Prompt Distillation for Multimodal Named Entity Recognition and Multimodal Relation Extraction — arXiv:2306.14122 — 2023
43. GenRES: Rethinking Evaluation for Generative Relation Extraction in the Era of Large Language Models — arXiv:2402.10744 — 2024

**结构化输出 / JSON 与推理**
44. Learning to Generate Structured Output with Schema Reinforcement Learning — arXiv:2502.18878 — 2025
45. Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models — arXiv:2408.02442 — 2024
46. Capacity, Not Format: Rethinking Structured Reasoning Failures — arXiv:2606.09410 — 2026（摘要级）
47. The Format Tax — arXiv:2604.03616 — 2026（摘要级）
48. Rationalization Models for Text-to-SQL — ICLR 2025 Workshop on Reasoning and Planning for LLMs — arXiv:2502.06759 — 2025
49. Knowledge Distillation with Structured Chain-of-Thought for Text-to-SQL (Struct-SQL) — Canadian AI 2026 — arXiv:2512.17053 — 2025

**官方来源**
50. Qwen3 Technical Report — arXiv:2505.09388 — 2025（全文 HTML 已抓；§ distilling / off-policy / on-policy / thinking budget）
51. Qwen3-0.6B Model Card — Hugging Face `Qwen/Qwen3-0.6B` — 官方 model card（thinking/non-thinking 开关、`/think` `/no_think`、采样参数、no-think-in-history、标准化 JSON 输出）

**项目内证据（非文献）**
- `docs/adr/0003-graphrag-extraction.md` — teacher thinking ON 的抽取盲评（4 passages）
- `docs/adr/0004-deployment-interface-and-student-prompt.md` — teacher/student prompt 故意分离
- `docs/adr/0008-consumer-axes-and-nothink-cost.md` — thinking vs no-thinking judge 质量/成本对照
- `docs/2026-09-07-capacity-line-repair.md` — 4B masked 的 UPPERCASE schema 解析伪影（结构化输出风险）

记录：2026-09-19。检索窗口：arXiv 全库（截至 2026-09）。若后续出现"开放关系抽取
+ strength + 小模型上 think/no-think 蒸馏对照"的一手实验，应更新 §3 / §4 与
Direct answer。
