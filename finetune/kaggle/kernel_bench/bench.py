#!/usr/bin/env python3
"""A/B throughput bench: bf16 vs fp16 (and single vs dual GPU if available).

Measures per-step wall time and peak VRAM on a fixed smoke dataset (first N
train rows), one config at a time, prints a comparison table. No adapter save,
no inference — pure training-throughput probe.

Purpose (issue 06 / ADR-0005): decide the Round-1 canonical training config.
T4 is Turing (no bf16 tensor cores) so fp16+AMP may be the native fast path;
Kaggle T4 kernels may expose 2 GPUs. Both are hypotheses to measure.

Every config is a fresh Trainer on a fresh model so results don't leak across
runs. Uses a short fixed number of steps; ~5-8 min per config on a T4.
"""

import json
import os
import sys
import time

WORK = "/kaggle/working"
MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1"
DS_DIR = "/kaggle/input/datasets/idalextan"
TRAIN_SRC = os.path.join(DS_DIR, "trace-dataset/alpaca_train.jsonl")
CUTOFF = 6144
N_ROWS = 8          # smoke rows; short-ish but real 6144-capable
BENCH_STEPS = 6
GRAD_ACCUM = 8
# leave CUDA_VISIBLE_DEVICES unset so the bench sees the real device count


def log(msg: str) -> None:
    print(msg, flush=True)


def check_env() -> None:
    log("=== env ===")
    import torch

    log(f"torch: {torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu_count={torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            log(f"  gpu[{i}] {p.name} total_mem={p.total_memory/1e9:.1f}GB")
    for p in (MODEL_PATH, TRAIN_SRC):
        ok = os.path.exists(p)
        log(f"{'OK ' if ok else 'MISSING'} {p}")
        if not ok:
            raise SystemExit(f"required path missing: {p}")


def install() -> None:
    import importlib.metadata as md

    def ver(pkg: str) -> str:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "MISSING"

    for p in ("transformers", "peft", "accelerate", "datasets"):
        log(f"pre {p}: {ver(p)}")
    subprocess = __import__("subprocess")
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install", "--quiet",
            "transformers>=4.49.0,<5.0.0",
            "peft>=0.14.0,<=0.18.1",
            "accelerate>=1.0.0",
            "datasets>=2.16.0,<5.0.0",
        ]
    )
    for p in ("transformers", "peft", "accelerate", "datasets"):
        log(f"post {p}: {ver(p)}")


def _rows(path: str, n: int) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def bench_one(name: str, use_fp16: bool, use_bf16: bool, device: str) -> dict:
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    )
    from transformers import DataCollatorForSeq2Seq
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model

    log(f"\n=== bench {name} (fp16={use_fp16} bf16={use_bf16} device={device}) ===")
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.config.use_cache = False
    lora_cfg = LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    rows = _rows(TRAIN_SRC, N_ROWS)
    enc = []
    for r in rows:
        user = f"{r['instruction']}\n{r['input']}"
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": r["output"]},
        ]
        full = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            enable_thinking=False, return_dict=True,
        )
        ids = full["input_ids"][:CUTOFF]
        user_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        labels = ([-100] * len(user_text) + ids[len(user_text):])[:CUTOFF]
        labels = labels + [-100] * (len(ids) - len(labels))
        enc.append({"input_ids": ids, "labels": labels})
    ds = Dataset.from_list(enc)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    out_dir = os.path.join(WORK, f"bench_{name.replace('/', '_')}")
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=5.0e-5,
        num_train_epochs=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=2,
        max_steps=BENCH_STEPS,
        save_strategy="no",
        report_to=[],
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    per_step = wall / BENCH_STEPS
    log(f"RESULT {name}: {BENCH_STEPS} steps in {wall:.1f}s "
        f"= {per_step:.2f}s/step | peak_vram={peak:.2f}GB | "
        f"steps/s={BENCH_STEPS/wall:.4f}")
    return {"name": name, "wall_s": wall, "per_step_s": per_step,
            "peak_gb": peak, "steps_s": BENCH_STEPS / wall}


def main() -> int:
    import torch
    check_env()
    install()
    import importlib.metadata as md

    results = []
    results.append(bench_one("bf16_single", use_fp16=False, use_bf16=True, device="cuda:0"))
    results.append(bench_one("fp16_single", use_fp16=True, use_bf16=False, device="cuda:0"))

    if torch.cuda.device_count() >= 2:
        log(f"\nsecond GPU detected: {torch.cuda.get_device_name(1)}")
        results.append({"name": "dual_GPU_available", "note": "see metadata T4x2"})

    log("\n=== SUMMARY ===")
    for r in results:
        log(json.dumps(r))
    out = os.path.join(WORK, "bench_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {out}")
    return 0


sys.exit(main())
