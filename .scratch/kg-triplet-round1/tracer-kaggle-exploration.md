# Tracer SFT on Kaggle — 探索简报(2026-09-02/03)

Issue 06 的 Kaggle 训练踩坑与决策全过程。目标读者:项目 owner(中文)。
按时间顺序记录**每轮尝试 → 确切错误 → 根因 → 决策**。

---

## 0. 上下文:为什么走了这条路

- 你原计划手工在 Kaggle notebook UI 里跑,受困于"路径/版本/yaml"反复出错(每次试错排队 30-60 分钟)。
- 你要求:配置 MCP,让 AI 全权驱动训练、拉回结果。
- 本机无 GPU(训练只能在 Kaggle),本机 `huggingface.co` 不通(本地拉不了模型,但 Kaggle 上能通)。

## 1. 通道调研:MCP vs CLI(决策点 A)

**结论:Kaggle 官方 MCP 服务器存在且读操作可用,但写操作(token 鉴权)对你不可用 → 实际用官方 kaggle CLI 做写操作。**

| 通道 | 读 | 写 | 结论 |
|---|---|---|---|
| 官方 MCP `https://www.kaggle.com/mcp` | ✅ `authorize`/`get_accelerator_quota`/搜索都通 | ❌ `update_dataset_metadata`/`upload_dataset_file`/`save_notebook` 全部返回 `Unauthenticated` | 注册到 opencode.jsonc 备用,但**写不可靠** |
| 官方 `kaggle` CLI(pip 包) | ✅ | ✅ `kernels push/pull/output/status`,同 KGAT token 全通 | **最终采用的写通道** |
| `kagglehub` | ✅ | ✅ dataset 上传 | 传数据可用 |

要点:
- 鉴权用**单 token**(`KGAT_...`,存环境变量 `KAGGLE_API_TOKEN`),不是旧式 `kaggle.json` 两段式。
- 真实 Kaggle 用户名是 **`idalextan`**(不是 `alextan`——那是另一个老账号 user_id 477848;你是 user_id 28715211)。用错 owner slug 时私有 API 全报 Unauthenticated。
- 已把 `kaggle` remote MCP 注册进 `C:\Users\Tan\.config\opencode\opencode.jsonc`(`{env:KAGGLE_API_TOKEN}`),**重启 opencode 后生效**。

## 2. 复用现有资产

你 9/1 手搓时已经上传过数据,全部字节数与本地一致,直接复用:

- `idalextan/trace-dataset` → `alpaca_train.jsonl`(95)/ `alpaca_val.jsonl`(5)
- `idalextan/trace-config` → yaml + dataset_info
- 旧 kernel `tracer-0901` = **ERROR**,pull 下来看是手搓产物:GPU/Internet **根本没开**(`accelerator:none`),路径用错一层,`lora_target` 是旧字段名,还有乱码调试 cell。

> 你之前不是"跑不过",是 notebook 元数据连 GPU/网都没开。

## 3. 探针 kernel 证实挂载路径(结论可复用)

用一个 CPU 探针 kernel 实测(不做任何假设):

- 模型挂载:**`/kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1/`**(Kaggle Models 官方路径,含 config.json+model.safetensors+tokenizer 全套)
- 数据挂载:**`/kaggle/input/datasets/idalextan/trace-dataset/alpaca_train.jsonl`**(Kaggle dataset 新式路径带 `datasets/`)
- 镜像自带 torch;kernel 里 pip 装依赖即可,Internet 需在 metadata 里 `enable_internet: true`。
- 你的旧 notebook 里写的模型路径其实是对的,dataset 路径也是对的——**路径不是你的主问题**。

## 4. 脚本 kernel 的硬约束(踩过的坑)

Kaggle **script** kernel 只把 `code_file` 单个文件打包成 `/kaggle/src/script.py`,
**同目录其它 `.py` 不会带上**(第一次验证版直接 `ModuleNotFoundError`)。

→ kernel 必须是**单文件自包含**。本仓库做法:`finetune/kaggle/kernel/tracer_kernel.py`(源)+
`gen_inputs.py`(把 10 个 eval passage + STUDENT_PROMPT 生成 `INPUTS = {...}`)+
`merge_kernel.py` + `make_variants.py`(产出 full / smoke / verify 三个单文件变体)。
产物:`finetune/kaggle/kernel/kg_tracer_{full,smoke,verify}.py`。

## 5. llamafactory 失败全记录(决策点 B — 你问的重点)

### 5.1 失败的真正形态

**不是"yaml 错/路径错/参数错"。llamafactory-cli 在 Kaggle 镜像上静默死在 Trainer 初始化中途,exit 0,无 traceback、无 loss、无 Running training 输出。** 逐层排查:

| 版本 | 症状 | 停在何处 |
|---|---|---|
| llamafactory 0.9.5 + 镜像自带 transformers 5.0 | 模型加载完 → `max_steps is given` 日志后无声退出 | trainer.py:675(Trainer.__init__ 中) |
| llamafactory 0.9.5 + 降级 transformers 4.57.6 | 同上,多走一步 | trainer.py:749(`Using auto half precision backend`) |

- 数据加载 OK(95 条)、模型加载 OK(`all params: 751,632,384`)、Trainer 构造 OK——**之后 train() 从未真正跑**,进程就 0 退出。
- 用 subprocess 包裹且 stdout 没落盘时,一度被误判为"training finished"(那只是 subprocess 返回)。
- 后来把训练输出**重定向到 `train.log` 文件**才看到真实停顿点。

### 5.2 归因(我的判断)

Kaggle 镜像自带 accelerate 1.13 / transformers 5 系列,llamafactory 的 launcher/加速器
初始化在该环境下不兼容,**静默 `os._exit`/空转,不抛异常**。不同 transformers 版本只是换了
一个停点。这不是配置能救的,属于 llamafactory↔Kaggle 镜像组合的兼容黑盒。

### 5.3 附带确认的 llamafactory 0.9.5 字段事实(避免再错)

- LoRA 模块字段叫 **`lora_target`**(不是 `target_modules`,那是旧版/其它库名),默认值已是 `"all"`。
- Qwen3 关闭思考用模板 **`template: qwen3_nothink`**,且数据参数 `enable_thinking: false`。
- `cutoff_len`、`dataset_dir` 都是合法参数。
- 多卡 T4 kernel 会被 llamafactory 默认当 2 卡起 DDP(`nproc_per_node = get_device_count`)→ 需 `NPROC_PER_NODE=1` + `CUDA_VISIBLE_DEVICES=0`。
- yaml 不要用 f-string/textwrap 手拼(缩进变量会错位致 `yaml ParserError`);用 dict + `yaml.safe_dump`。

### 5.4 决策

不再与 llamafactory 黑盒搏斗。**改用手写 `transformers.Trainer` + `peft` LoRA**:
代码全在我们控制内、错误直接可见。这也是当时给你的三选项里你选的那个。

## 6. 手写 Trainer 的调试(3 个内存/梯度坑)

`finetune/kaggle/kernel/tracer_kernel.py` 手写管线:
tokenizer+model(从挂载加载)→ alpaca 用 qwen3 chat template(thinking OFF)编码,
user 部分 mask → peft LoRA(r/α 32,全线性层)→ `Trainer`。

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| 1 | `CUDA OOM`,bs2 时 13.8GB 用满 | cutoff 6144 + bs2 激活超 T4 16GB | bs1 + grad-accum 8(等效全局 batch 8 不变) |
| 2 | backward `does not require grad` | 冻结主干 + gradient_checkpointing 需要 embedding 输入梯度钩子 | `get_peft_model` 后调 `model.enable_input_require_grads()` |
| 3 | FULL 95 条**第一步**仍 OOM,但 smoke 8 条过 | 分配器**碎片化**:`allocated 8.6GB + reserved 5.7GB`(无法服务的保留块),并非真不够;95 条里随机先碰到长样本 | `PYTORCH_ALLOC_CONF=expandable_segments:True`(报错自身建议) |

smoke(6 步)在 expandable_segments 之前已经跑通并**真实产出 adapter**(`adapter_config.json`+`adapter_model.safetensors`),
证明手写管线机制成立。FULL(5 epochs,95 条)已用 expandable_segments 重新提交并 RUNNING。

## 7. "100 条为什么跑这么久/总 OOM"——数据×硬件本质冲突

- **标签极长**:GraphRAG 格式的 `{entities, relationships}` JSON,实测单条 char 中位数 ~13.9k、p90 ~20k,
  估 token 中位 ~3.5-4k、p90 ~5k、上限 ~7k。`cutoff_len: 6144` 是**必需的**,不是保守。
- attention 显存 ~ 序列长² ,6144 在 T4 上 **batch=1 也要 ~14GB**,只能 bs1;
  bs1 每步只吃 1 样本,时间主要花在"每步前向+反向跑一个 4-6k token 序列",与条数关系小。
- 由此 95 条×5 epochs ≈ 60 步 ≈ 60-90 分钟;Full 全量(3349 条)在此硬件/长度下会是**很大一笔时间**,
  这个事实应进入 go/no-go 的权衡(例如是否值得为长标签另想办法,见后)。

## 8. 当前状态 & 产物地图

| Kaggle kernel | 状态 | 内容 |
|---|---|---|
| `idalextan/kg-tracer-probe1` | COMPLETE | 挂载路径证实 |
| `idalextan/kg-tracer-verify` | COMPLETE | yaml schema 预检 |
| `idalextan/kg-tracer-smoke` | COMPLETE | 6 步真训练,adapter 产出 |
| `idalextan/kg-tracer-sft` | RUNNING(v4) | FULL 5 epochs + 推理,当前跑中 |

本地文件(都在 git 工作区,未提交):
- `finetune/kaggle/kernel/{tracer_kernel.py,inputs.py,kg_tracer_{single,full,verify,smoke}.py}`
- `finetune/kaggle/{gen_inputs.py,merge_kernel.py,make_variants.py}`
- `finetune/kaggle/kernel_{full,verify,smoke}/`(各含 metadata + 单文件变体,供 `kaggle kernels push`)

## 9. 后续建议(gate 决策时的输入)

- 等 FULL v4 结果:若 loss 正常下降 + `predictions.jsonl` 出来 → 本地 `eval/tracer_eval.py` → 盲审 → go/no-go。
- 若 v4 仍 OOM(长样本单序列极限),备选:把 `cutoff_len` 降至 5120(牺牲最长样本尾部,README 原也提过这个 OOM 退路)。
- 若决定走向 full 3349:serious 权衡 `6144×bs1×T4` 的时长预算;值得重新审视
  **是否需要 5 epochs + 全序列监督**,或未来考虑 flash-attn(需重编)/分块监督。
- 代码风格警告已内置:`enable_input_require_grads`、`expandable_segments`、`save_strategy: epoch`
  (避免 save_steps 超过总步数导致从不落盘)、重定向训练日志到文件以便失败时看全量。
