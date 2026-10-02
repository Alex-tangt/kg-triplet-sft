# 小模型开放 KG 关系抽取：评估方法、scaling vs 蒸馏 vs RL 的一手调研

本文服务一个具体抉择：kg-triplet-sft 已用 LoRA SFT 把 Qwen3 0.6B/1.7B/4B 训成
GraphRAG 式开放抽取器（实体 + 自由文本关系 + strength），消费方是 LightRAG。当前
teacher 约 0.92 edges/entity；student 仅 0.34（0.6B）→ 0.71（4B）；在 teacher 边里
端点被 referent 匹配上的那些，student 只复现 14%（0.6B）/ 35%（4B）；student 边能对上
teacher 边的精确率约 0.21–0.29。owner 在权衡三条路：(a) 继续放大模型、(b) 从更强模型
蒸馏（含 on-policy / GKD）、(c) 用本项目的 eval 当 reward 做 RL（如 GRPO）。本文只做
一手文献调研（arXiv / ACL / NeurIPS / ICML / ICLR），回答两个问题：(1) 结构化/KG 抽取
的评估除精确 F1 外还有哪些可操作指标；(2) 近期实证如何比较 scale、蒸馏、RL 对小型
抽取模型的效果与代价。所有引用均经 arXiv API 按标题/ID 核对；未核实者一律标注。

## Method

- **检索**：arXiv API（`export.arxiv.org/api/query`），按子主题迭代关键词，覆盖
  "LLM knowledge graph construction evaluation"、"GraphRAG evaluation"、"triple
  extraction LLM"、"relation extraction LLM benchmark"、"open information extraction
  LLM"、"knowledge graph hallucination evaluation"、"entailment relation verification"、
  "on-policy distillation"、"generalized knowledge distillation"、"distillation vs RL"、
  "GRPO verifiable reward"、"RLVR"、"reward hacking LLM"、"small language model
  distillation extraction" 等。API 有 429/502 限流，改为顺序请求 + 退避，每次取 top-10~15。
- **核验**：对入选论文再按 `id_list=` 拉一次元数据，逐条核对 exact title / arXiv ID /
  日期 / `comment`、`journal_ref` 里的 venue。**venue 只在元数据明确写出时才标注**，
  否则只写 "arXiv:ID (year)"，不凭记忆补 venue。凡标题与预期不符的 ID 直接丢弃
  （例：`1708.02596` 实为 model-based RL，与 "Deep Mutual Learning" 无关，故改用
  `1706.00384`；`2207.00593` 是物理论文，与 "Distilling from a Stronger Teacher" 无关，
  改用 `2205.10536`）。
- **范围**：只收一手论文（arXiv 预印本 / ACL Anthology / 会议论文），不收博客或二手综述；
  综述类论文只用于"领域地图"，具体方法结论回引其引用的一手工作（本文未逐条回引的，
  在正文标明是综述转述）。
- **局限**：文献几乎没有针对"开放自由文本关系 + strength、100–300 词 chunk、小模型
  0.6–4B"这一确切设定的对照实验；下述结论多为**邻近任务**（通用推理、NER、经典 RE、
  科学 IE、KGC）的外推，已在文中标注迁移距离。

## Findings

### 1. 结构化 / KG 抽取的评估：精确 F1 为何不够，替代指标有哪些

- **生成式关系抽取不能用精确匹配。** GenRES（`2402.10744`）指出，生成式 RE 常输出
  与人工参考不同但语义正确的多样关系，precision/recall 会系统性误判；提出五维评估：
  topic similarity、uniqueness、granularity、factualness、completeness，并用人工偏好
  验证其与人类排序一致，最后用它对 14 个 LLM 在 document/bag/sentence 三类 RE 数据上
  建立基准。论文还实证：人工参考关系本身可能不完整，且用固定关系/实体集 prompt LLM
  会诱发幻觉。
- **KG 构建需要把"忠实"与"合 schema"分开测。** Text2KGBench（`2308.02357`，ISWC 2023）
  用 7 个指标同时度量 fact extraction、ontology conformance（concepts / domain-range）
  和 hallucination；这正是本项目要区分"抽错关系"与"抽对但格式/本体不合"的模板。
- **幻觉 + 遗漏 + 图相似度。** Ghanem 等（`2502.05239`，KGSWC 2024）在 triple
  matching / graph matching 之外，加入 exact hallucination / omission 与 BERTScore 图
  相似度（阈值 95%），并发现微调虽降低幻觉与遗漏，却在 KELM-sub 上**泛化更差**——
  对"只在自家分布上变好"是直接警示。
- **GraphRAG 全链路评估。** GraphRAG-Bench（`2506.02404`）把评估拆到 graph
  construction、retrieval、answer generation，并在最终答案之外评**推理过程逻辑一致性**；
  这支持把"图构建质量"当成独立可评对象，而不是只测下游 QA。
- **语义 + 结构双视角。** D2A-HKG（`2510.11297`）在 schema-free 零样本设定下发现前沿
  LLM 能产出 document-faithful、幻觉有限的三元组，但不同构建阶段行为差异大——说明
  reference-based 的 "faithful" 已是前沿模型可达水位，质量瓶颈更可能在 recall/覆盖。
- **下游任务而非元数据指标。** KGrEaT（`2308.10537`，CIKM 2023）主张用分类/聚类/推荐等
  下游任务测 KG 的"有用性"，因为 correctness/completeness 指标看不到图是否真被用起来。
  这与本项目 `eval/lightrag_*.py` 的 consumer-impact e2e 同构，是对那条线的文献支撑。
- **图级结构幻觉。** Boudourides（`2603.01341`）提出 "structural hallucination"，用图相似
  度、中心性、引文完整性做网络化压力测试，报告 F1<0.05、幻觉率>93% 等。**注意：单作者
  预印本，结论仅作方向参考，不当作可靠效应量。**
- **经典 RE 基准的标注缺陷 = 支持语义/judge 评估的论据。** DocRED（`1906.06127`）、
  Re-DocRED（`2205.12696`，修正 false negative）、Re-TACRED（`2104.08398`，修正
  TACRED 缺陷）都表明精确匹配的 gold 本身不完整，直接拿 F1 会低估生成式系统。
- **鲁棒性维度。** "Preserving Knowledge Invariance"（`2305.13981`，EMNLP 2023）用
  knowledge-invariant clique 测同一语义的不同表述，发现既有模型最大掉 23.43 F1；
  抽取器在分布漂移下的稳定性是独立轴。
- **方向性（directionality）缺少专门基准。** 未找到专门评测关系方向/头尾顺序的一手
  论文。最接近的是 LA-RL（`2607.23420`）把 "reversed argument order" 明确列为 IE 自
  反思要区分的错误类型；传统 RE 数据默认方向由标签编码。**这是本项目可自行补的空白**
  （见 Uncertainties）。
- **LLM-as-judge 在 RE 上并不可靠。** 生物医学 RE 评测（`2506.00777`，ACL 2025）用 8 个
  LLM 当 judge 评 5 个 LLM、3 个数据集，发现 judge 准确率通常<50%，主因是模型输出的
  关系不遵循统一格式；强制结构化输出后 judge 平均提升约 15%。**结论：judge 可用，但
  必须先规范被测输出格式并做校准，不能裸用。**

### 2. 关系/图抽取的幻觉与忠实度评估（claim-level、entailment、judge 验证）

- **原子事实级。** FActScore（`2305.14251`，EMNLP 2023）把长文本拆成 atomic facts，问
  有可靠知识源支持的比例；自动化估计器对人工分误差<2%，并用它评估了 6,500 条生成
  （若人工需约 $26K）。这是"对每条生成关系逐条判支持"的直接范式。
- **NLI / 蕴含式一致性。** TRUE（`2204.04991`，NAACL 2022）在 11 个数据集上做 example-level
  meta-eval，推荐 NLI 与 question-generation-QA 两类方法；SummaC（`2111.09525`，TACL）
  用句级切分+聚合让 NLI 达到 74.4% balanced accuracy，并在 6 个数据集上验证；AlignScore
  （`2305.16739`，ACL 2023）用 4.7M 例、7 类任务统一 alignment 函数，355M 模型即可匹敌
  GPT-4 级评估器，覆盖 22 个评测集。
- **RAG 忠实度（无参考与有参考）。** RAGAS（`2309.15217`）提出 reference-free 的
  faithfulness / answer relevance / context relevance；ARES（`2311.09476`，NAACL 2024）
  自造合成数据微调轻量 judge，并用 prediction-powered inference（PPI）配合几百条人工
  标注做**纠偏与验证**，在 8 个任务与域偏移下仍有效。这两个是"如何验证 judge"的可操作
  模板。
- **KG 化的可解释幻觉检测。** GraphEval（`2407.10793`，KiL'24 @ KDD）把回答表示成 KG
  三元组，用 NLI 逐三元组判定，能定位幻觉位置且 balanced accuracy 优于裸 NLI。
- **人工审计式可验证性。** 生成式搜索审计（`2304.09848`，Findings of EMNLP 2023）定义
  citation recall/precision 并以人工评测：仅 51.5% 句子被引用完全支持、74.5% 引用真正
  支持其句子——说明"看似有据"与"真实有据"差距巨大。
- **judge 自身的偏差与验证。** LLM-as-a-judge（`2306.05685`，NeurIPS 2023 D&B）报告强
  judge 与人类偏好一致率>80%，同时明确列出 position / verbosity / self-enhancement 偏差
  及缓解手段。**对本项目"当 reward 用"的启示：judge 必须用人工小样、PPI 或双通道一致性
  来验证，且方向/位置偏置要显式控制。**

### 3. 训练信号对比：SFT / 数据 scaling vs 蒸馏 vs RL

**SFT + 数据/理由 scaling**
- Distilling step-by-step（`2305.02301`，Findings of ACL 2023）用 LLM rationale 作额外
  监督，比普通 finetuning/蒸馏用更少数据达到更好效果，770M T5 用 80% 数据超过 few-shot
  540B PaLM。
- Orca（`2306.02707`）与 Orca 2（`2311.11045`）证明"解释轨迹 + 教小模型自选策略"能让
  13B 级模型接近大得多模型（Orca 在 BBH 上比 Vicuna-13B 高 100%+，Orca 2 对标 5–10x
  参数模型）。
- 固定算力下 SFT 未必劣于蒸馏：`2404.19319`（NAACL 2024 Insights workshop）发现固定
  compute 下普通 KD≈从零预训练，但更强的 KD（TinyBERT/MiniLM）明显胜出；数据需重复时
  KD 收益更大。
- `2509.22193`（Scale or Reason）在 5 个规模（0.5–14B）、18 个基准上按等 FLOPs 比较：
  **IFT 在多数配置上位于 Pareto 前沿**；reasoning 监督只在 7B+ 的开放式任务上才上榜，
  且 25–50% reasoning 数据混合 IFT 即拿到大部分收益、算力更省。
- `2606.24747`（任务特定蒸馏 scaling law，金融域）显示 in-domain 随压缩可预测退化，
  监督格式是主因，CoT 监督能挽回被剪枝抹掉的通用知识。

**蒸馏（离线 targeted / 序列级 / on-policy）**
- Targeted distillation 在抽取上有直接效果：UniversalNER（`2308.03279`，ICLR 2024）把
  ChatGPT 蒸馏成小模型做开放 NER，在 43 个数据集上比 Alpaca/Vicuna 平均高 30+ F1，比
  ChatGPT 本体还高 7–9 F1，并大幅超过用监督 NER 的 InstructUIE。**这是"更强 teacher
  的监督确实抬高学生抽取质量、甚至超过 teacher"的最强一手证据（NER 而非关系）。**
- InstructUIE（`2304.08085`）用 32 个 IE 数据统一指令微调，零样本超过 GPT-3.5 与当时
  SoTA；GoLLIE（`2310.03668`，ICLR 2024）证明把 annotation guidelines 写进去后零样本
  IE 显著变好，详细 guideline 是关键。
- MetaIE（`2404.00457`）用"label-to-span"符号蒸馏从 LLM 造小 meta-model，作为 6 类 IE
  任务的更好 few-shot 起点。
- 图构建的 targeted 蒸馏：Distill-SynthKG（`2410.16597`）用 LLM 合成 document-KG 对，
  把多步流程蒸馏成单步小模型，**在 KG 质量上超过所有 baseline（含 8x 大的模型）**，
  并改善后续检索与 QA。Contextualization Distillation（`2402.01729`，EACL 2024 Findings）
  把三元组扩写成上下文再蒸馏，稳定提升 KGC。DistillER（`2602.05452`）在实体消解上发现
  **用 LLM teacher 的噪声标签做 SFT，稳定优于其他 KD 策略（含 RL）**。
- On-policy / GKD：GKD（`2306.13649`，ICLR 2024）用学生自生成序列 + teacher 反馈解决
  分布错配，可无缝结合 RLHF；MiniLLM（`2306.08543`，ICLR 2024）指出生成式 KD 该用
  reverse-KL，支持 120M–13B。OPD 综述（`2604.00626`）给出机制解释：**静态模仿的
  exposure bias 随序列长度约按平方增长，on-policy 把它降到约线性**，并把 OPD 形式化为
  f-divergence 最小化 →与 KL 约束 RL 同源。更新的 OPDVR（`2608.24696`）用 trajectory
  正确性 + ReLU gating 把 OPD 与 RLVR 合并（无新增超参），在 6 个推理基准上稳定超过
  纯 OPD。
- **OPD 的代价与坑**：SFD（`2605.30833`）发现学生前缀变长后 teacher 置信度/区分度下降，
  监督信号衰减；CaOPD（`2604.16830`）发现 OPD 提升准确率的同时系统性导致过度自信
  （校准 vs 能力解耦）。

**RL（PPO / DPO / GRPO / RLVR）**
- GRPO 出自 DeepSeekMath（`2402.03300`）；DeepSeek-R1（`2501.12948`，Nature 645:633-638,
  2025）展示纯 RL 可激励推理；Tulu 3（`2411.15124`）把 SFT/DPO/RLVR 组成开放配方。
- **但 RLVR 的增益边界有强烈反证**：`2504.13837`（NeurIPS 2025 Oral）用大 k 的 pass@k
  发现 RLVR 训练模型的推理能力并未超过 base 模型（大 k 时 base 更高），六个 RLVR 算法
  表现相近；**相比之下蒸馏能引入 teacher 的新推理模式、真正扩展能力**。即：RL 更像
  重加权采样，蒸馏才扩上限。
- **RL / GRPO 在抽取任务上的正面结果**：
  - MimicSFT + R²GRPO（`2505.22068`）在科学 IE 上先结构化模板 SFT、再用 relevance +
    rule 奖励做 GRPO，两者都提升"推理能力"，组合后在关系抽取上超过 baseline 与专用
    监督模型。
  - MR-UIE（`2509.09082`）用多视角推理 + RL 做通用 IE，跨域提升。
  - LA-RL（`2607.23420`）用任务诊断标签做 outcome-supervised GRPO，SciER 关系抽取
    平均 +6.83 F1、OOD RE ~+20 F1。
  - 自推导奖励 RE（`2510.06198`）：Qwen2.5-14B 在 One-shot NYT29 上 24.65% F1，加
    RL 中间奖励后 +23.46 点。
  - Schema RL（`2502.18878`）用细粒度 schema validator 当奖励，显著改善合法 JSON 生成
    与下游——对"输出必须是严格 JSON"的本项目直接相关。
  - 小模型奖励粒度（`2607.02869`）：Qwen2.5-0.5B + GRPO on GSM8K，**process-only
    63.73% 明显高于 outcome-only 53.75%**，且某些混合权重反而不如纯 outcome。说明
    小模型稀疏奖励下需要更细的过程信号。

### 4. 噪声 / 模型生成奖励下的 RL：reward hacking 与 GRPO 风险

- 概念界定：`2209.13085`（Skalse 等）形式化 "unhackable" 并刻画 reward hacking。
- 过优化 scaling law：`2210.10760`（Gao 等）在合成设定下测出 gold 分数随对 proxy
  reward 优化先升后降，且系数随 reward model 参数量平滑缩放，KL 惩罚可缓解。
- 奖励模型评估应与策略联动：`2505.12763`（ACL 2025）发现常见 RM benchmark 与优化后
  策略表现弱相关，建议用"过优化程度"而非静态准确率来评测 RM。
- 推理时也会 hack：`2506.19248`（NeurIPS 2025 Spotlight）证明 BoN 等一大类推理时机制
  必然出现先升后降的 hacking，提出 HedgeTune 对冲。
- **噪声奖励对 RLVR 是破坏性的（关键）**：`2603.16140` 修正了"RLVR 可从错误标注学习"
  的结论（原实验数据混入干净样本），严格去污后：训练在真错误标注上的模型在数学上比
  干净数据低 8–10%，在真实 Text2SQL 人工标注错误上低 5–12%；现有 RLVR 算法改进无法
  弥补。**直接推论：本项目的 teacher 边本身有噪声、eval judge 有偏差，把这些当
  reward 做 RL 之前必须先净化/验证奖励信号。**
- **judge 当 reward 的具体风险**：RE 上 LLM judge 裸用<50% 准确（`2506.00777`），
  与 `2209.13085`/`2210.10760` 叠加说明"用 flawed 自动 metric 当 reward"极易 hack。

### 5. 从有噪声的 teacher 蒸馏：更强 teacher 能否抬高学生上限

- **容量差会伤蒸馏**：Teacher Assistant（`1902.03393`，AAAI 2020）证明学生-teacher 差距
  过大时学生退化，中间助教网络可桥接——0.6B 学生直接模仿强 teacher 可能触发同一问题。
- **更强 teacher 不是自动更好**：DIST（`2205.10536`，NeurIPS 2022）发现更强 teacher 下
  学生-教师预测差异变大，精确 KL 匹配反而干扰训练，改用以相关系数捕捉类间关系才稳定。
- **teacher 标签噪声下的蒸馏**：`2109.10147` 是 NLU 上首个 KD + 噪声标签研究，提出缓解
  法但结论是"仍需更多研究"；TrueTeacher（`2305.11171`，EMNLP 2023）用 LLM 标注模型生成
  的摘要训练学生，**学生在 TRUE benchmark 上超过同容量 SoTA，也超过 LLM teacher 本体**，
  发布 1.4M 合成数据——说明蒸馏可超过 teacher（但任务是一致性判定，非关系抽取）。
  DistillER（`2602.05452`）同样发现 SFT on noisy LLM labels 是最稳的 KD 路径。
- **信心/课程门控**：GateKD（`2605.13136`，confidence-gated closed-loop distillation）、
  EGAD（`2605.01732`，entropy-guided adaptive distillation）、课程蒸馏（`2506.05695`）
  等按 teacher 置信度/熵动态加权或排序。**注意：这些是 arXiv 预印本，本文只核到标题/
  摘要层面，未逐一核对实验细节，引用时标为较低证据强度。**
- **上限证据（间接）**：UniversalNER（学生>teacher）、Distill-SynthKG（学生>8x 模型）、
  `2504.13837`（蒸馏引入 base 没有的新模式）三者共同支持"蒸馏能抬抽取上限"；但都
  不是"开放自由文本关系 + strength"的边复现实验。

## Decision-relevant comparison

> 说明：文献几乎不报绝对 GPU-hours，故成本列只给**相对量级**与可迁移的结构性成本；
> "对关系/边质量的作用"以**邻近任务**证据为主，迁移距离已在证据强度列标注。

| 方向 | 对关系/边质量的作用（证据） | 成本（相对） | 证据强度 | 主要风险 |
|---|---|---|---|---|
| **(a) 放大模型** | 最可靠、单调。项目自身容量线 0.34→0.71 edges/entity（0.6B→4B）；文献侧 `2606.24747`/`2509.22193` 说 in-domain 随规模/算力改善但会压缩。 | 高（更大 base 的训/推显存与时间）；数据不变 | **强**（项目内 + 通用 scaling law），但**非关系专属** | 边际收益递减；4B 已暴露 uppercase-schema 解析伪影（见 `docs/2026-09-07-capacity-line-repair.md`）；4090 单卡上限 |
| **(b) 离线/序列级蒸馏（更强 teacher → 学生 SFT）** | 抽取类任务中最直接有效：UniversalNER（NER，学生>teacher 7–9 F1）、Distill-SynthKG（KG，胜 8x 模型）、MetaIE / InstructUIE / GoLLIE。有噪声 teacher 标签下 SFT 仍稳（DistillER `2602.05452`）。 | 中（一次 teacher 生成 + 一次 SFT；本项目已有此流水线） | **强**（IE/KG 一手结果多），但**关系边复现**无直接实验 | 容量差（`1902.03393`）；更强 teacher 的 KL 失配（`2205.10536`）；teacher 噪声上限（`2109.10147`）；泛化变差（`2502.05239`） |
| **(b′) On-policy / GKD 蒸馏** | 针对暴露偏差，理论上正中学生"长 JSON 生成偏离 teacher 轨迹"的复现率问题：OPD 把 exposure bias 从 ~长度² 降到 ~线性（`2604.00626`）；OPDVR 在推理上超纯 OPD（`2608.24696`）。**未见其在 KG 抽取的实证。** | 中高（需学生 rollout + teacher 打分，4090 可做但吞吐下降） | **中**（机制清晰，抽取域未验证） | teacher 置信衰减 SFD（`2605.30833`）；校准崩坏/过度自信（`2604.16830`）；rollout 成本 |
| **(c) RL / GRPO（eval 当 reward）** | 抽取域有正面结果（R²GRPO `2505.22068`、LA-RL `2607.23420`、MR-UIE、自推导奖励、Schema RL），但通用 RLVR 被证基本不扩上限（`2504.13837`）；小模型需过程奖励（`2607.02869`）。 | 高（多次 rollout + 奖励计算/评判；单 4090 尤其贵）+ 奖励工程 | **中偏弱**（抽取域样本少、且 reward 质量决定成败） | **reward hacking**（`2209.13085`/`2210.10760`/`2506.19248`）；**噪声奖励破坏性**（`2603.16140`）；RE 上 LLM judge 裸用<50%（`2506.00777`）；可能只重加权采样、不扩上限 |

**对本项目的直接含义**

- 若目标是"复现 teacher 边"（当前 14%/35%）：这本质就是**从 teacher 蒸馏**，已在进行
  的 SFT 就是离线蒸馏；文献提示下一步的靶点是 **exposure bias / on-policy 纠正** 与
  **teacher 监督的信号格式（rationale/结构化）**，而非先上 RL。
- 若目标是"提高绝对边质量/覆盖"：证据顺序为 **数据与监督格式（rationale、guideline、
  去重/净化）> 更强 teacher 蒸馏 > 放大模型 > RL**；RL 收益最不确定且风险最高。
- **RL 的前置门槛**：先有一个**格式规范、经人工/PPI 校准**的 reward（`2506.00777`、
  `2311.09476`），且训练数据已去噪（`2603.16140`）。否则"eval 当 reward"极可能变成对
  judge 偏差的过优化。
- **评估侧**：无论走哪条路，报告不应只有精确 F1。建议对边采用 GenRES 式多维评估
  （`2402.10744`）+ 逐条关系 entailment/claim 支持（`2205.10536` 以外的 `2204.04991`/
  `2305.16739`）+ 图级结构指标（`2502.05239`/`2308.10537`/`2603.01341`），并显式补
  方向性/头尾一致性这一缺口。

## Uncertainties / not found

- **没有找到**直接比较 scale vs distillation vs RL 在**开放自由文本关系抽取 + strength**
  上的对照实验。最接近的是通用推理（`2509.22193`/`2504.13837`）与 NER/KGC（UniversalNER/
  Distill-SynthKG），外推需谨慎。
- **关系方向性/头尾顺序**无专门 benchmark；只有 LA-RL（`2607.23420`）把 "reversed
  argument order" 列为诊断类别。建议本项目自建方向一致性指标。
- **strength（0–10）校准**的评估方法未找到一手来源；`2606.24747` 只谈 CoT 监督与压缩，
  不涉及数值强度校准。
- **on-policy/GKD 蒸馏用于 KG 抽取**：未找到实证论文（OPD 文献集中在数学/代码/agent）。
- **GRPO 用于 0.6–4B 结构化抽取 + judge reward**：未找到；小模型 RLVR 研究基本在数学。
- **更强 teacher 是否抬高"学生边复现上限"**：只有 NER/KG 质量的间接证据，没有边级
  recall 的证据。
- **成本数字**：这些论文基本不报 GPU-hours，无法给出可信的绝对成本对照；表内为相对量级。
- 少数来源仅核到标题/摘要层面（`2605.13136`、`2605.01732`、`2506.05695`、`2602.19643`、
  `2605.09702`），已在正文标注为较低证据强度，未据其下结论。

## Sources

> 标注格式：标题 — venue（仅当 arXiv 元数据明确写出）— arXiv:ID — 年份。
> 未标 venue 者表示元数据未提供，不臆测。

**评估 / 图指标 / 抽取基准**
1. GenRES: Rethinking Evaluation for Generative Relation Extraction in the Era of Large Language Models — arXiv:2402.10744 — 2024
2. Text2KGBench: A Benchmark for Ontology-Driven Knowledge Graph Generation from Text — ISWC 2023 — arXiv:2308.02357
3. Enhancing Knowledge Graph Construction: Evaluating with Emphasis on Hallucination, Omission, and Graph Similarity Metrics — KGSWC 2024 — arXiv:2502.05239
4. GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation — arXiv:2506.02404 — 2025
5. Are Large Language Models Effective Knowledge Graph Constructors? — arXiv:2510.11297 — 2025
6. KGrEaT: A Framework to Evaluate Knowledge Graphs via Downstream Tasks — CIKM 2023 — arXiv:2308.10537
7. Structural Hallucination in Large Language Models: A Network-Based Evaluation of Knowledge Organization and Citation Integrity — arXiv:2603.01341 — 2026
8. DocRED: A Large-Scale Document-Level Relation Extraction Dataset — arXiv:1906.06127 — 2019
9. Revisiting DocRED — Addressing the False Negative Problem in Relation Extraction — arXiv:2205.12696 — 2022
10. Re-TACRED: Addressing Shortcomings of the TACRED Dataset — arXiv:2104.08398 — 2021
11. Preserving Knowledge Invariance: Rethinking Robustness Evaluation of Open Information Extraction — EMNLP 2023 — arXiv:2305.13981
12. LLM-empowered knowledge graph construction: A survey — arXiv:2510.20345 — 2025
13. Assessing LLMs Suitability for Knowledge Graph Completion — NESY 2024 — arXiv:2405.17249
14. Improving Automatic Evaluation of Large Language Models (LLMs) in Biomedical Relation Extraction via LLMs-as-the-Judge — ACL 2025 — arXiv:2506.00777

**幻觉 / 忠实度 / judge 验证**
15. FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation — EMNLP 2023 — arXiv:2305.14251
16. TRUE: Re-evaluating Factual Consistency Evaluation — NAACL 2022 — arXiv:2204.04991
17. SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization — TACL — arXiv:2111.09525
18. AlignScore: Evaluating Factual Consistency with a Unified Alignment Function — ACL 2023 — arXiv:2305.16739
19. Ragas: Automated Evaluation of Retrieval Augmented Generation — arXiv:2309.15217 — 2023
20. ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems — NAACL 2024 — arXiv:2311.09476
21. GraphEval: A Knowledge-Graph Based LLM Hallucination Evaluation Framework — KiL'24 @ KDD 2024 — arXiv:2407.10793
22. Evaluating Verifiability in Generative Search Engines — Findings of EMNLP 2023 — arXiv:2304.09848
23. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena — NeurIPS 2023 D&B — arXiv:2306.05685

**蒸馏 / SFT / 理由监督**
24. Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes — Findings of ACL 2023 — arXiv:2305.02301
25. Orca: Progressive Learning from Complex Explanation Traces of GPT-4 — arXiv:2306.02707 — 2023
26. Orca 2: Teaching Small Language Models How to Reason — arXiv:2311.11045 — 2023
27. UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition — ICLR 2024 — arXiv:2308.03279
28. InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction — arXiv:2304.08085 — 2023
29. GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction — ICLR 2024 — arXiv:2310.03668
30. MetaIE: Distilling a Meta Model from LLM for All Kinds of Information Extraction Tasks — arXiv:2404.00457 — 2024
31. Scaling Knowledge Graph Construction through Synthetic Data Generation and Distillation — arXiv:2410.16597 — 2024
32. Contextualization Distillation from Large Language Model for Knowledge Graph Completion — EACL 2024 Findings — arXiv:2402.01729
33. DistillER: Knowledge Distillation in Entity Resolution with Large Language Models — arXiv:2602.05452 — 2026
34. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD) — ICLR 2024 — arXiv:2306.13649
35. MiniLLM: On-Policy Distillation of Large Language Models — ICLR 2024 — arXiv:2306.08543
36. A Survey of On-Policy Distillation for Large Language Models — arXiv:2604.00626 — 2026
37. On-policy Distillation with Verifiable Reward (OPDVR) — arXiv:2608.24696 — 2026
38. Your Teacher Can't Help You Here: Combating Supervision Fidelity Decay in On-Policy Distillation — arXiv:2605.30833 — 2026
39. The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation — arXiv:2604.16830 — 2026
40. Knowledge Distillation from A Stronger Teacher (DIST) — NeurIPS 2022 — arXiv:2205.10536
41. Improved Knowledge Distillation via Teacher Assistant — AAAI 2020 — arXiv:1902.03393
42. Knowledge Distillation with Noisy Labels for Natural Language Understanding — arXiv:2109.10147 — 2021
43. TrueTeacher: Learning Factual Consistency Evaluation with Large Language Models — EMNLP 2023 — arXiv:2305.11171
44. Beyond Scaling Law: A Data-Efficient Distillation Framework for Reasoning — arXiv:2508.09883 — 2025
45. Scale or Reason? A Compute-Equivalent Analysis of Reasoning Distillation — arXiv:2509.22193 — 2025
46. Knowledge Distillation vs. Pretraining from Scratch under a Fixed (Computation) Budget — NAACL 2024 Insights (Negative Results) Workshop — arXiv:2404.19319
47. Scaling Laws for Task-Specific LLM Distillation — arXiv:2606.24747 — 2026
48. Deep Mutual Learning — arXiv:1706.00384 — 2017
49. GateKD: Confidence-Gated Closed-Loop Distillation for Robust Reasoning — arXiv:2605.13136 — 2026（仅核到摘要）
50. EGAD: Entropy-Guided Adaptive Distillation for Token-Level Knowledge Transfer — arXiv:2605.01732 — 2026（仅核到摘要）

**RL / GRPO / RLVR**
51. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models — arXiv:2402.03300 — 2024
52. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — Nature 645:633-638 — arXiv:2501.12948 — 2025
53. Tulu 3: Pushing Frontiers in Open Language Model Post-Training — arXiv:2411.15124 — 2024
54. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? — NeurIPS 2025 Oral — arXiv:2504.13837
55. Beyond path selection: Better LLMs for Scientific Information Extraction with MimicSFT and Relevance and Rule-induced (R²)GRPO — arXiv:2505.22068 — 2025
56. MR-UIE: Multi-Perspective Reasoning with Reinforcement Learning for Universal Information Extraction — arXiv:2509.09082 — 2025
57. LA-RL: Label-Aware Self-Reflection for Reinforcement Learning in Information Extraction — arXiv:2607.23420 — 2026
58. The Answer Lies Within: Self-Derived Rewards Enable Explainable Relation Extraction — arXiv:2510.06198 — 2025
59. Learning to Generate Structured Output with Schema Reinforcement Learning — arXiv:2502.18878 — 2025
60. Reward Granularity in RLVR: Comparing Process and Outcome Reward Structures for Mathematical Reasoning in Small Language Models — arXiv:2607.02869 — 2026

**奖励噪声 / reward hacking / RM 评估**
61. Defining and Characterizing Reward Hacking — arXiv:2209.13085 — 2022
62. Scaling Laws for Reward Model Overoptimization — arXiv:2210.10760 — 2022
63. Noisy Data is Destructive to Reinforcement Learning with Verifiable Rewards — arXiv:2603.16140 — 2026
64. Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization — ACL 2025 — arXiv:2505.12763
65. Inference-Time Reward Hacking in Large Language Models — NeurIPS 2025 Spotlight — arXiv:2506.19248

**GraphRAG / 消费方背景（评估动机）**
66. From Local to Global: A Graph RAG Approach to Query-Focused Summarization — arXiv:2404.16130 — 2024
67. LightRAG: Simple and Fast Retrieval-Augmented Generation — arXiv:2410.05779 — 2024（EMNLP 2025，见项目 ADR-0009）
68. How Significant Are the Real Performance Gains? An Unbiased Evaluation Framework for GraphRAG — arXiv:2506.06331 — 2025
69. RAG vs. GraphRAG: A Systematic Evaluation and Key Insights — arXiv:2502.11371 — 2025
70. GraphRAG under Fire — IEEE S&P 2026 — arXiv:2501.14050
71. AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora — arXiv:2505.23628 — 2025

记录：2026-09-19。检索窗口：arXiv 全库（截至 2026-09）。若后续定位到"开放关系抽取
上 scale vs distill vs RL"的专属对照实验，或关系方向性/strength 校准的一手基准，
应更新本文对应小节。
