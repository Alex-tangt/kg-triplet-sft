#!/usr/bin/env python3
"""Tracer SFT for Qwen3-0.6B on the GraphRAG-format 95/5 alpaca set.

Runs on a Kaggle T4. Uses a hand-written transformers.Trainer + peft LoRA
pipeline (NO llama-factory launcher — that silently exits 0 mid-Trainer-init
in the Kaggle image, both under transformers 5.0 and 4.57).

Pipeline:
  1. verify env (torch/GPU/model/dataset mounts).
  2. pip install the pinned stack (transformers 4.x, peft, accelerate).
  3. load tokenizer + Qwen3-0.6B base from the Kaggle Models mount.
  4. build a torch Dataset from alpaca jsonl: apply the qwen3 chat template
     with enable_thinking=False, mask the user turn, keep the JSON answer as
     the label. cutoff_len 6144.
  5. LoRA via peft (rank/alpha 32, dropout 0.1, all linear modules).
  6. Trainer: lr 5e-5, cosine, warmup 0.1, bf16, bs2 x grad-accum 4,
     save_strategy epoch, 5 epochs (smoke: max_steps 6).
  7. merge adapter, infer on the 10 embedded tracer-eval passages single-shot
     thinking OFF, write predictions.jsonl.

Every path is absolute and checked. No interactive steps.
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# T4 16GB + 6144-token sequences fragment the allocator badly (reserved-but-
# unallocated blocks can't serve a new request). expandable_segments grows
# segments on demand instead of pre-reserving huge pools.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

WORK = "/kaggle/working"
SAVE = os.path.join(WORK, "saves")
os.makedirs(SAVE, exist_ok=True)

MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1"
DS_DIR = "/kaggle/input/datasets/idalextan"
TRAIN_SRC = os.path.join(DS_DIR, "trace-dataset/alpaca_train.jsonl")
VAL_SRC = os.path.join(DS_DIR, "trace-dataset/alpaca_val.jsonl")
ADAPTER_DIR = os.path.join(SAVE, "qwen3-0.6b-graphrag-tracer")
CUTOFF = 6144

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inputs import INPUTS  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def check_env() -> None:
    log("=== 1. environment ===")
    log(f"python: {sys.version.split()[0]}")
    import torch

    log(f"torch: {torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu: {torch.cuda.get_device_name(0)}  count={torch.cuda.device_count()}")
    for p in (MODEL_PATH, TRAIN_SRC, VAL_SRC):
        ok = os.path.exists(p)
        log(f"{'OK ' if ok else 'MISSING'} {p}")
        if not ok:
            raise SystemExit(f"required path missing: {p}")


def install() -> None:
    log("=== 2. install stack ===")
    import importlib.metadata as md

    def ver(pkg: str) -> str:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "MISSING"

    for p in ("torch", "transformers", "peft", "accelerate", "datasets"):
        log(f"pre-install {p}: {ver(p)}")

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
        log(f"post-install {p}: {ver(p)}")


def build_dataset(tokenizer, path: str, smoke: bool) -> tuple:
    """Return (dataset, num_trainable_examples). Encodes each alpaca row via the
    qwen3 chat template with thinking OFF; user turn ids become -100 in labels.
    """
    from datasets import Dataset

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if smoke:
        rows = rows[:8]

    enc = []
    skipped = 0
    for r in rows:
        user = f"{r['instruction']}\n{r['input']}"
        answer = r["output"]
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]
        full = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            enable_thinking=False, return_dict=True,
        )
        ids = full["input_ids"]
        if len(ids) > CUTOFF:
            ids = ids[:CUTOFF]
        # find where the assistant turn begins: tokenize the user-only prefix
        user_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        n_user = len(user_text)
        labels = [-100] * n_user + ids[n_user:]
        # labels may exceed CUTOFF after truncation; align lengths
        labels = labels[:CUTOFF]
        labels = labels + [-100] * (len(ids) - len(labels))
        if all(x == -100 for x in labels):
            skipped += 1
            continue
        enc.append({"input_ids": ids, "labels": labels})

    log(f"built dataset from {len(rows)} rows (skipped {skipped} all-masked) "
        f"from {path}")
    return Dataset.from_list(enc), len(enc)


def train(smoke: bool) -> None:
    log("=== 3. train (hand-written Trainer + peft) ===")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForSeq2Seq

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    # gradient checkpointing over a frozen base needs input grad hooks on the
    # embeddings, else backward dies with "does not require grad"
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_ds, n = build_dataset(tokenizer, TRAIN_SRC, smoke)
    if n == 0:
        raise SystemExit("no trainable examples after masking")
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    out_dir = ADAPTER_DIR if not smoke else ADAPTER_DIR + "-smoke"
    epochs = 1.0 if smoke else 5.0
    args = TrainingArguments(
        output_dir=out_dir,
        # cutoff 6144 x bf16 on a 16GB T4: bs2 blew up 13.8GB on activations.
        # bs1 + grad-accum 8 keeps the effective global batch (8) while fitting.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5.0e-5,
        num_train_epochs=epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch" if not smoke else "steps",
        save_steps=3 if smoke else 500,
        save_total_limit=3,
        report_to=[],
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_steps=6 if smoke else -1,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    log("trainer built — starting train()")
    trainer.train()
    log("train() returned")
    trainer.save_model(out_dir)
    log(f"saved adapter -> {out_dir}")
    log(f"adapter_config.json exists: "
        f"{os.path.exists(os.path.join(out_dir, 'adapter_config.json'))}")


def _parse_answer(raw: str) -> dict:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in completion: {raw[:200]!r}")
    return json.loads(raw[start : end + 1])


def infer() -> None:
    log("=== 4. inference on 10 eval passages ===")
    import glob
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    candidates = glob.glob(os.path.join(ADAPTER_DIR, "**", "adapter_config.json"), recursive=True)
    candidates = [c for c in candidates if "-smoke" not in c]
    log(f"adapter candidates: {candidates}")
    if not candidates:
        raise SystemExit(f"no adapter_config.json under {ADAPTER_DIR}")
    adapter_dir = os.path.dirname(candidates[0])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.eval()

    prompt = INPUTS["student_prompt"]
    rows = []
    for i, p in enumerate(INPUTS["passages"], 1):
        content = f"{prompt}\n{p['text']}"
        msgs = [{"role": "user", "content": content}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=4096, do_sample=False,
                temperature=1.0, top_p=1.0,
            )
        out_ids = gen[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(out_ids, skip_special_tokens=True)
        try:
            parsed = _parse_answer(raw)
        except Exception as e:  # noqa: BLE001
            log(f"[{i}/10] {p['id']}: PARSE FAIL ({e})")
            parsed = {"entities": [], "relationships": []}
        rows.append({"id": p["id"], "entities": parsed.get("entities", []),
                     "relationships": parsed.get("relationships", [])})
        log(f"[{i}/10] {p['id']}: entities={len(rows[-1]['entities'])} "
            f"rels={len(rows[-1]['relationships'])}")

    pred_path = os.path.join(WORK, "predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"wrote {pred_path} ({len(rows)} rows)")


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny train (6 steps) then exit")
    args = parser.parse_args(argv)

    check_env()
    install()
    train(smoke=args.smoke)
    if args.smoke:
        log("=== SMOKE DONE (training ok) ===")
        return 0
    infer()
    log("=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
