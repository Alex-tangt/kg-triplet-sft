#!/usr/bin/env python3
"""Verify LLaMA-Factory runs on Kaggle T4 with its OWN resolved dependency matrix.

Background: earlier llamafactory attempts failed (silent exit 0 / hang) because
the stack was force-installed ON TOP of Kaggle's preinstalls (accelerate 1.13.0
etc.) WITHOUT satisfying llamafactory's declared constraints. Research (issue
06 + hiyouga/LlamaFactory) shows the current matrix pins accelerate<=1.11.0,
peft 0.18.0-0.18.1, transformers in [4.55, 5.8]. The clean route is what the
community does on Kaggle: git clone + `pip install -e ".[torch,metrics]"` so pip
resolves a compatible set, then run the CLI.

This kernel does exactly that at smoke scale (8 rows, 6 steps) and prints the
resolved versions + the training loss to prove the framework works. If it does,
LLaMA-Factory becomes the Round-1 carrier and the hand-written Trainer is retired.

Run: python /kaggle/src/script.py   (metadata in kernel_lfcheck/)
"""

import json
import os
import shutil
import subprocess
import sys

# Do NOT pin CUDA_VISIBLE_DEVICES to a single card: llamafactory's launcher
# branches on get_device_count()>1 (Kaggle exposes 2x T4) and then runs
# torchrun DDP; forcing 1 card while Trainer's device_map still sees 2 GPUs is
# what silently exits mid-Trainer-init (trainer.py:906 "already on multiple
# devices"). Let llamafactory take its native dual-GPU path.
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
# llamafactory version checker is noise; also avoid the Kaggle preinstall clash
os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

WORK = "/kaggle/working"
MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1"
DS_DIR = "/kaggle/input/datasets/idalextan"
TRAIN_SRC = os.path.join(DS_DIR, "trace-dataset/alpaca_train.jsonl")
LF_DIR = os.path.join(WORK, "LLaMA-Factory")
LF_DATA = os.path.join(LF_DIR, "data")
YAML = os.path.join(WORK, "lf_smoke.yaml")


def log(msg: str) -> None:
    print(msg, flush=True)


def check_env() -> None:
    log("=== 1. environment ===")
    import torch
    log(f"python: {sys.version.split()[0]}")
    log(f"torch: {torch.__version__} cuda={torch.cuda.is_available()}")
    for p in (MODEL_PATH, TRAIN_SRC):
        ok = os.path.exists(p)
        log(f"{'OK ' if ok else 'MISSING'} {p}")
        if not ok:
            raise SystemExit(f"required path missing: {p}")


def setup() -> None:
    log("=== 2. clone + pip install -e (community Kaggle route) ===")
    subprocess.check_call(
        ["git", "clone", "--depth", "1",
         "https://github.com/hiyouga/LLaMA-Factory.git", LF_DIR],
        stdout=subprocess.DEVNULL,
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "-e", ".[torch,metrics]"],
        cwd=LF_DIR,
        stdout=subprocess.DEVNULL,
    )
    # llamafactory's clean install resolves transformers 5.0.0, which silently
    # exits mid-Trainer-init (trainer.py:675) in this image. Its CI pins 4.57.1
    # (4.57.0 is excluded) — pin that, keep accelerate at the resolved 1.11.0.
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "transformers==4.57.1"],
        stdout=subprocess.DEVNULL,
    )
    import importlib.metadata as md

    def ver(pkg: str) -> str:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "MISSING"

    for p in ("llamafactory", "transformers", "peft", "accelerate", "datasets", "torch"):
        log(f"  resolved {p}: {ver(p)}")
    # llamafactory pins `lora_target`; log the version to confirm freshness
    log(f"  llamafactory loc: {md.distribution('llamafactory').locate_file('')}")

    # stage data: copy alpaca + register in llamafactory's data/
    shutil.copy(TRAIN_SRC, os.path.join(LF_DATA, "alpaca_train.jsonl"))
    ds_cfg = {
        "lf_smoke": {
            "file_name": "alpaca_train.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
            "formatting": "alpaca",
        }
    }
    with open(os.path.join(LF_DATA, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(ds_cfg, f)


def write_yaml() -> None:
    log("=== 3. smoke yaml (8 rows via max_samples, 6 steps) ===")
    cfg = {
        "model_name_or_path": MODEL_PATH,
        "template": "qwen3_nothink",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "dataset": "lf_smoke",
        "cutoff_len": 6144,
        "max_samples": 16,
        "per_device_train_batch_size": 1,
        # 2 GPUs x bs1 x accum4 = effective batch 8 (matches the single-GPU
        # bs1 x accum8 that fits the memory; same global batch on both cards)
        "gradient_accumulation_steps": 4,
        "learning_rate": 5.0e-5,
        "num_train_epochs": 5.0,
        "max_steps": 6,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "lora_rank": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target": "all",
        "logging_steps": 2,
        "save_strategy": "no",
        "report_to": "none",
        "output_dir": os.path.join(WORK, "saves/lf-smoke"),
        "overwrite_output_dir": True,
    }
    import yaml
    with open(YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log(f"wrote {YAML}")
    log(yaml.safe_dump(cfg, sort_keys=False)[:400])


def run_train() -> None:
    log("=== 4. llamafactory-cli train (log to file; inherit nothing via PIPE) ===")
    train_log = os.path.join(WORK, "train.log")
    # PR #10584 lesson: never leave stdout/stderr on an unread PIPE; redirect
    # to a file inside the child.
    with open(train_log, "wb") as f:
        r = subprocess.run(
            [sys.executable, "-u", "-m", "llamafactory.cli", "train", YAML],
            cwd=LF_DIR, stdout=f, stderr=subprocess.STDOUT,
        )
    log(f"cli exit code: {r.returncode}")
    tail = open(train_log, encoding="utf-8", errors="replace").read()
    print(tail[-2500:])
    if r.returncode != 0:
        raise SystemExit(f"llamafactory-cli train failed rc={r.returncode}")


def main() -> int:
    check_env()
    setup()
    write_yaml()
    run_train()
    log("=== LF SMOKE DONE: llamafactory trains on Kaggle T4 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
