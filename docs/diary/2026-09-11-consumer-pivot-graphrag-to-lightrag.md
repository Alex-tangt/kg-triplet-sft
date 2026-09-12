# 消费端 pivot：MS GraphRAG → LightRAG

日期：2026-09-11
分支：`exp/lightrag-e2e`

## 背景

Leg A（base 基线）收口后，下一步是 Leg B——"抽取质量是否真的传导到消费端"的
终端验证。原交接文档 `handoff-exp-B-graphrag-e2e.md` 的方案是：把 gold /
0.6B-masked / 4B-clean 三层抽取各灌成一份图，喂给 **MS GraphRAG** 的
Bring-Your-Own-Graph（BYOG）模式（自己提供 entities / relationships /
text_units parquet，跳过它自带的 LLM 抽取），再对比 Local / Global 问答质量。

在逐项敲定细节需求（grilling）时，消费端换成了 **LightRAG**。这篇记录为什么换、
换了之后方案长什么样。

## 为什么 pivot

1. **BYOG 的工程量与脆弱点都在 GraphRAG 的 parquet 工作流栈**：要凑
   `workflows: [create_communities, create_community_reports]`、text_unit 的
   id 对齐、Leiden 的 weight 映射，任何一处 schema 不对就是一次昂贵的调试循环。
   LightRAG 有一个**文档化的自定义 KG 写入入口** `insert_custom_kg`，直接吃
   chunks + entities + relationships，不用拼 parquet，也不用起它的 workflow。
2. **claim 大部分保留**。原本要验证的三个消费端性质里，实体描述检索（`local`）与
   跨文档关系检索（`global`）LightRAG 都有对应模式；但**社区/区域总结没有**——
   `lightrag-hku==1.5.7` 无 Leiden/社区/社区报告代码（源码审计 0 命中）。这是 pivot
   的**能力缺口**：换产品的同时砍掉了一项被验证能力。收尾复盘才发现，已写成
   claim 边界——要测社区总结得回 MS GraphRAG。
3. **多了一个天然的对照臂**。LightRAG 的 `naive` 模式只从原始 chunk 答题，完全
   不碰图，正好当"文本充分性 oracle"：图只在打赢 `naive` 的地方才算有增量价值。
   GraphRAG 侧没有这么干净的抽取无关臂。
4. 原来的"喂自己的图、跳过它的抽取器"这条理由完全保留——**stock 索引会用
   LightRAG 自己的 LLM 重新抽一遍，永远测不到我们替换掉的那一环**。BYOG/custom
   KG 是唯一能把变量锁死在"我们的抽取"上的路径。

## 读源码得到的硬事实（决定了构造器怎么写）

LightRAG 的 `insert_custom_kg` 是**直写器，不是成图器**——它归一化标识符后就落库，
**不做任何合并**。逐条核对源码后确认：

- **实体 last-wins**：同名实体只保留最后一条声明（`deduped_entities`）。
- **关系 last-wins**：按**无序端点对**去重（`sorted((src, tgt))`），保留最后一条，
  所以图里边是无向的。
- **自环会 raise**：`src_id == tgt_id`（归一化后）直接抛 `ValueError`。
- **悬挂端点自动补 `UNKNOWN` 节点**（`entity_type="UNKNOWN"`,
  `description="UNKNOWN"`），不会丢边。
- **`keywords` 必须显式给**：关系 VDB 的 content 是
  `f"{keywords}\t{src}\n{tgt}\n{desc}"`，源码直接下标 `relationship_data["keywords"]`，
  缺 key 就是 `KeyError`，没有默认值。
- **weight 有地板**：文档字符串写明"weight 下界 = 该关系的**不同真实证据 source 数**"。
  我们每条关系只给一个 `source_id`（一个规范段落），所以地板恒为 `1.0`；要保留
  teacher 的 0-10 strength 信号，就必须显式抬到 `1.0 + mean(strength)/10`。
- **source_id 是 chunk 别名**：先经 `chunk_to_source_map` 解析成 chunk hash，解析不到
  就是 `UNKNOWN`。

结论：**合并、去重、方向规范化、自环过滤都得我们在构造器里做完**，LightRAG 只负责
校验和写库。

## 定下来的方案（K1–K6）

- **消费端**：LightRAG，`insert_custom_kg`，零 LLM 建图。
- **数据**：不换数据集，用 fixed-200 里的一个 ~18 段多簇切片（space 8 +
  kurosawa 4 + afroasiatic 3 + andorra 3；egypt 附录、ankara 弃）。选簇标准是
  **具体共享实体**（df≤4、≥2 段），不是主题相邻。Apollo 神/登月两个子主题必须拆开，
  否则同名碰撞会凭空造边。官方语料（A Christmas Carol / UltraDomain）降级为可选
  Round-2。
- **三层**：gold（control）/ 0.6B-masked / 4B-clean；三层预测已存在，**零新增 GPU**。
- **构造规则**：实体按归一化 title 合并（描述 `" | "` 连接、type 取众数、
  source_id 取首现段落）；关系按无序端点对合并（留规范方向、描述连接、strength 聚合、
  `weight=1.0+mean(strength)/10`）；关系行显式 `keywords:""`；自环预丢；悬挂端点交给
  LightRAG 补 UNKNOWN 并计入 T0；三层 chunks 逐字节相同。
- **指标漏斗**：T0 图形成（离线、零成本、**硬闸**——链接实体存活率/fragmentation/
  conflation/UNKNOWN/空段落，平了就叫停）→ T1 检索层（`only_need_context`，免 judge，
  直接打召回）→ T2 答案层（严格 grounding + 蕴含）。四题型 A 交集/B 桥接/C 邻域聚合/
  D 全局主题，机械挖自 gold 图 + 双闸 + 预注册（~2× 候选落 ~10-15 题）。四种结局
  confirmed/partial/null/non-monotone，null 也是合法结果（T0 定位是"没进图"还是
  "进了没用"）。
- **角色**：KEYWORD `qwen-flash`、QUERY `qwen-max`、judge `qwen3.7-flash` thinking
  （与 QUERY 分离）、embedding 本地 `bge-m3`（1024 维，进程内 `EmbeddingFunc`）；
  reranker 本地 `bge-reranker-v2-m3`，**有/无双臂 × 全 4 模式**。QueryParam 固定不调参。
- **刹车**：建图 LLM=0；query+judge 累计 >3M token 或 >¥30 停并报告。

## 明确否决

- 不把 2575 训练数据并进主实验：与 fixed-200 切片 ~93% 重叠，会稀释层间差异，
  且让学生"背过的"训练 chunk 污染读数。
- 不新增任何 GPU 推理。
- 不拿"多跳成功率"当 headline：LightRAG 没有跳算子，这个数字是 prompt 措辞的
  产物，不是消费端性质；B 桥接题型才是它想表达的跨段可达性。

## 诚实的局限

~18 段、~10-15 题的 case study，没有统计功效，只出方向性读数。`keywords` 为空意味着
`global` 只能靠描述文本够到关系。学生模型 garble 标题既是传导机制、也是测量不确定度
来源——这正是 T0 必须当闸门而不是脚注的原因。

## 文档影响

- ADR-0003 / 0004：只换消费端产品名（"MS GraphRAG"→"LightRAG"），决策正文不动；
  通用词 "GraphRAG-style extraction" 保留（teacher 仍是官方 MS GraphRAG 抽取 prompt）。
- ADR-0009：**Leg B 重写为 LightRAG custom KG**，Leg A（base 基线）原样保留。
- ADR-0008：reachability 轴的一句消费端描述同步（LightRAG 同样用实体描述做向量路由）。
- `CONTEXT.md` / `AGENTS.md`：`consumer-impact e2e` 词条与数据生成 pivot 行同步。
- 不新开 ADR-0010——这是对既有消费端决策的就地修订，不是新决策。

## 结果（2026-09-11 收口）

跑完了：22 段 / 5 簇、16 道预注册自然题、三层图、T1（384 query）+ T2（192 生成，
2.24M token，未触刹车）。

- **T0 强区分**：0.6B 图碎成 381 个连通块（最大 26、351 个孤立点），4B 最大块 72，
  gold 136。抽取质量确实传到了图结构。
- **T1/T2 饱和**：`naive`（纯文本、不碰图）召回就已是 1.000，图模式 0.85–1.0 无层间
  次序；T2 三层正确率 0.94–1.0，失败集中在题目（andorra 机构链接题处处 partial），
  不在层。**三层在消费端不可区分。**

结论：(a) 可行性坐实；(b) 传导是**结构层面成立、消费端层面在本 case study 为 null**
——22 段 + `top_k=60` 下原文已经带着答案，图不是承重结构。要做出区分度，语料得大到
检索有选择性。这是诚实的后续方向，不是失败。详见 `outputs/lightrag_e2e/RESULT.md`。

顺带两条工程教训（都踩过）：Windows 控制台 GBK 会吞中文日志（设 `PYTHONIOENCODING`）；
LightRAG 的 worker 池是非守护线程，`asyncio.run` 收尾会挂——用 `os._exit` 硬退。
