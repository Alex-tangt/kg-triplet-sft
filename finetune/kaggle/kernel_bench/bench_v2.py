#!/usr/bin/env python3
"""A/B bench v2: dual-GPU DDP throughput on the two T4s.

bench v1 (single GPU, bf16 vs fp16) showed: fp16 == bf16 (~81s/step), peak VRAM
only 8.86/16GB, and Kaggle exposes 2x Tesla T4. This kernel measures the real
dual-card lever via torch DDP inside the single-file script kernel:

  configs (each max_steps=6, effective global batch 8):
    ddp_bs1x2  per_device=1 world=2 accum=4  ->  8 samples/opt-step
    ddp_bs2x2  per_device=2 world=2 accum=2  ->  8 samples/opt-step

Each config is one mp.spawn over world=2 (nccl on the same node). Rank 0
collects per-step wall time + peak VRAM per device.

Purpose: does dual-GPU get full run (2575x5x6144) inside the ~20h remaining
quota? If ddp_bs2x2 runs ~40-50s/opt-step and fits VRAM, full ~= 1610 steps...
steps drop to 1610 (same samples) but each step covers 2 GPUs -> estimate
helps decide Round-1 config.
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
N_ROWS = 16          # enough rows that 2 ranks x accum don't exhaust 1 epoch
BENCH_STEPS = 6


def log(msg: str) -> None:
    print(msg, flush=True)


def check_env() -> None:
    log("=== env ===")
    import torch

    log(f"torch: {torch.__version__} cuda={torch.cuda.is_available()}")
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
    import subprocess

    def ver(pkg: str) -> str:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "MISSING"

    for p in ("transformers", "peft", "accelerate", "datasets"):
        log(f"pre {p}: {ver(p)}")
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


def _build_ds(rows: list, tokenizer) -> "Dataset":
    from datasets import Dataset

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
    return Dataset.from_list(enc)


def ddp_train(rank: int, world: int, per_device_bs: int, accum: int,
              out_json: str) -> None:
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
        )
        from transformers import DataCollatorForSeq2Seq
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(f"cuda:{rank}")
        model.config.use_cache = False
        lora_cfg = LoraConfig(
            r=32, lora_alpha=32, lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.enable_input_require_grads()

        if rank == 0:
            model.print_trainable_parameters()
        dist.barrier()

        rows = _rows(TRAIN_SRC, N_ROWS)
        ds = _build_ds(rows, tokenizer)
        collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

        out_dir = os.path.join(WORK, f"ddp_bs{per_device_bs}x{world}_r{rank}")
        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=per_device_bs,
            gradient_accumulation_steps=accum,
            learning_rate=5.0e-5,
            num_train_epochs=2.0,   # enough for max_steps across the sampler
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            bf16=True,
            logging_steps=2,
            max_steps=BENCH_STEPS,
            save_strategy="no",
            report_to=[],
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            optim="adamw_torch",
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
            local_rank=rank,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=collator,
            tokenizer=tokenizer,
        )
        if rank == 0:
            log(f"rank0 starting train bs{per_device_bs} accum{accum}")
        t0 = time.time()
        trainer.train()
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        if rank == 0:
            log(f"RESULT ddp_bs{per_device_bs}x{world}: {BENCH_STEPS} steps "
                f"{wall:.1f}s = {wall/BENCH_STEPS:.2f}s/step peak_r0={peak:.2f}GB")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({"wall": wall, "per_step": wall / BENCH_STEPS,
                           "peak_r0": peak, "exitcode": 0}, f)
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            log(f"rank0 EXCEPTION: {e}")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({"error": str(e), "exitcode": 1}, f)
    finally:
        dist.destroy_process_group()


def bench(name: str, per_device_bs: int, accum: int) -> dict:
    import multiprocessing as mp

    out_json = os.path.join(WORK, f"{name}_result.json")
    if os.path.exists(out_json):
        os.remove(out_json)
    log(f"\n=== bench {name}: world=2 per_device_bs={per_device_bs} accum={accum} ===")
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=ddp_train,
                      args=(i, 2, per_device_bs, accum, out_json))
          for i in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    codes = [p.exitcode for p in ps]
    if not os.path.exists(out_json):
        log(f"{name} FAILED no result exitcodes={codes}")
        return {"name": name, "error": f"no result exitcodes={codes}"}
    r = json.load(open(out_json, encoding="utf-8"))
    r["name"] = name
    if r.get("exitcode") != 0:
        r["exitcodes"] = codes
    log(f"{name} done: {json.dumps(r)}")
    return r


def main() -> int:
    check_env()
    install()
    import torch

    if torch.cuda.device_count() < 2:
        log("need 2 GPUs; abort")
        return 1

    results = []
    results.append(bench("ddp_bs1x2", per_device_bs=1, accum=4))
    results.append(bench("ddp_bs2x2", per_device_bs=2, accum=2))

    log("\n=== SUMMARY ===")
    for r in results:
        log(json.dumps(r))
    with open(os.path.join(WORK, "bench_v2_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log("wrote bench_v2_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
