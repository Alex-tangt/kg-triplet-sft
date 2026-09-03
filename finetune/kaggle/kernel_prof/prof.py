#!/usr/bin/env python3
"""A/B the compute dtype on T4: bf16 vs fp16 GEMM path + wall time.

bf16 profiler v2 showed the top kernels are FP32-accumulating bf16 GEMMs
(`magma_sgemmEx_kernel<float, __nv_bfloat16, ...>`, `volta_sgemm_*`) — i.e.
bf16 on Turing falls back off the tensor cores (no native bf16 support) and is
slow. Hypothesis: fp16 (natively supported at 65 TFLOPS on T4) uses the tensor
core path and is several times faster.

This kernel measures one optimizer step each under bf16 and fp16 with the SAME
8 real rows, LoRA r/α 32, gradient checkpointing (required at 6144), and lists
the top CUDA kernels per dtype so we can see the GEMM name change. fp16 training
loss-scales via AMP? No — we use plain fp16 autocast so we can read the raw
matmul dtype in the profiler; if the kernel list shows fp16 tensor-core GEMMs
and wall time drops, that is the answer.

Run: python /kaggle/src/script.py   (metadata in kernel_prof/)
"""

import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

WORK = "/kaggle/working"
MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1"
DS_DIR = "/kaggle/input/datasets/idalextan"
TRAIN_SRC = os.path.join(DS_DIR, "trace-dataset/alpaca_train.jsonl")
CUTOFF = 6144
N_ROWS = 8
GRAD_ACCUM = 8


def log(msg: str) -> None:
    print(msg, flush=True)


def check_env() -> None:
    import torch
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    log(f"gpu: {torch.cuda.get_device_name(0)} sm={torch.cuda.get_device_capability(0)}")
    log(f"cuda supports bf16 matmul: {torch.cuda.is_bf16_supported()}")
    for p in (MODEL_PATH, TRAIN_SRC):
        ok = os.path.exists(p)
        log(f"{'OK ' if ok else 'MISSING'} {p}")
        if not ok:
            raise SystemExit(f"missing {p}")


def install() -> None:
    import importlib.metadata as md
    import subprocess
    for p in ("transformers", "peft"):
        try:
            log(f"pre {p}: {md.version(p)}")
        except Exception:
            pass
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "transformers>=4.49.0,<5.0.0", "peft>=0.14.0,<=0.18.1"],
        stdout=subprocess.DEVNULL,
    )
    for p in ("transformers", "peft"):
        try:
            log(f"post {p}: {md.version(p)}")
        except Exception:
            pass


def load_rows(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def build_one(tokenizer, r):
    user = f"{r['instruction']}\n{r['input']}"
    msgs = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": r["output"]},
    ]
    full = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False,
        enable_thinking=False, return_dict=True,
    )
    ids = full["input_ids"][:CUTOFF]
    user_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )
    labels = ([-100] * len(user_ids) + ids[len(user_ids):])[:CUTOFF]
    labels = labels + [-100] * (len(ids) - len(labels))
    return {"input_ids": ids, "labels": labels}


def bench_dtype(dtype: str, enc: list) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForSeq2Seq
    from datasets import Dataset

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    log(f"\n===== dtype = {dtype} ({torch_dtype}) =====")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch_dtype, trust_remote_code=True
    )
    model.config.use_cache = False
    log(f"attn_implementation = {getattr(model.config, '_attn_implementation', 'UNSET')}")
    lora_cfg = LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = model.to("cuda:0")

    ds = Dataset.from_list(enc)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-5
    )
    model.train()

    # warm up one micro-batch so the first real measurement excludes lazy init
    with torch.no_grad():
        wb = collator([ds[0]])
        model(input_ids=wb["input_ids"].to("cuda:0"))

    t_fetch = t_fwd = t_bwd = t_opt = 0.0
    grad_accum = 0
    for step in range(1):
        dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collator)
        for batch in dl:
            t0 = time.time()
            input_ids = batch["input_ids"].to("cuda:0")
            labels = batch["labels"].to("cuda:0")
            t_fetch += time.time() - t0
            t0 = time.time()
            out = model(input_ids=input_ids, labels=labels)
            (out.loss / GRAD_ACCUM).backward()
            t_fwd += time.time() - t0
            # NOTE: split forward/backward inside one timed region for simplicity
            grad_accum += 1
            if grad_accum == GRAD_ACCUM:
                t0 = time.time()
                optim.step()
                optim.zero_grad()
                t_opt += time.time() - t0
                grad_accum = 0

    per_step = (t_fetch + t_fwd + t_bwd + t_opt) / 1
    log(f"dtype={dtype}: one optimizer step = {per_step:.2f}s "
        f"(fwd+bwd {t_fwd:.2f}s, opt {t_opt:.2f}s)")
    log(f"dtype={dtype}: theoretical tensor-core TFLOPS util check below via kernels")

    # profiler for kernel dtype names (one step, best-effort)
    try:
        dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collator)
        grad_accum = 0
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for batch in dl:
                input_ids = batch["input_ids"].to("cuda:0")
                labels = batch["labels"].to("cuda:0")
                out = model(input_ids=input_ids, labels=labels)
                (out.loss / GRAD_ACCUM).backward()
                grad_accum += 1
                if grad_accum == GRAD_ACCUM:
                    optim.step()
                    optim.zero_grad()
                    break
        events = sorted(
            prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True
        )[:8]
        for e in events:
            log(f"  {e.self_device_time_total/1e3:9.1f}ms  {e.key[:100]}")
    except Exception as exc:  # noqa: BLE001
        log(f"profiler section failed for {dtype}: {exc}")

    del model, optim, ds
    torch.cuda.empty_cache()


def main() -> int:
    check_env()
    install()
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    rows = load_rows(TRAIN_SRC, N_ROWS)
    enc = [build_one(tokenizer, r) for r in rows]
    for e in enc:
        real = sum(1 for x in e["labels"] if x != -100)
        log(f"seq_len={len(e['input_ids'])} label_tokens={real}")

    bench_dtype("bf16", enc)
    bench_dtype("fp16", enc)
    log("\n=== A/B DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
