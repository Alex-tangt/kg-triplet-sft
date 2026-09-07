"""Round-1 capacity-curve training (chat protocol, Unsloth, single 4090).

Format/protocol decision (see issue 08 Finding 2026-09-06):
training texts are rendered through the base tokenizer's chat template with
thinking disabled (``enable_thinking=False``), i.e. the same qwen3 no-think
protocol the eval harness consumes. A plain-concat training line is known to be
suppressed under chat eval and is deliberately not reproduced here.

Uniform run behaviour (identical across every curve member):
r=32 alpha=32, bf16, cutoff 6144, bs 2 x ga 8, lr 5e-5, cosine + warmup 0.03,
5 epochs, seed 42, adamw_8bit, packing=True, eval=OFF, save_every 50.
The ONLY allowed difference between runs is --model / --out.

Integrity artifacts written under <out>/:
  run_config.json  - pure-file config card (data sha256, tokenizer markers, hyperparams)
  format_check.json- rendered sample + chat-marker assertions (from the loaded tokenizer)
  receipt.json     - post-train evidence (global_step, final loss, checkpoints present)

Usage:
  python train_unsloth.py --write-card --model M --train-jsonl D --out O   # card only
  python train_unsloth.py --diff-cards C1 C2 C3 ...                        # fail if runs differ beyond --model/--out
  python train_unsloth.py --model M --train-jsonl D --out O [--merge]      # real run
  python train_unsloth.py ... --limit 50                                   # debug cap
"""
import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime

SEQ = 6144
LORA_R = 32
LORA_A = 32
LORA_DROPOUT = 0.05
GA = 8
LR = 5e-5
EPOCHS = 5
SEED = 42
FORMAT = "chat_qwen3_nothink"

DIFF_ALLOWED = {"model", "model_dir", "out", "timestamp"}
CARD_KEYS = ["model", "model_dir", "tokenizer_eos", "chat_template_in_cfg",
             "chat_template_jinja", "im_tokens", "train_jsonl", "data_sha256",
             "rows", "bs", "ga", "eff_bs", "epochs", "seq", "lr", "r", "alpha",
             "dropout", "precision", "scheduler", "warmup_ratio", "seed",
             "optim", "weight_decay", "packing", "format", "eval", "save_every",
             "merge", "limit", "out", "timestamp"]


def file_sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for i, chunk in enumerate(iter(lambda: f.read(1 << 20), b"")):
            if limit and i >= limit:
                break
            h.update(chunk)
    return h.hexdigest()


def count_rows(path, limit):
    n = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            n += 1
            if limit and n >= limit:
                break
    return n


def read_rows(path, limit):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def model_dir_info(model):
    d = {}
    tc = {}
    try:
        with open(os.path.join(model, "tokenizer_config.json"), encoding="utf-8") as f:
            tc = json.load(f)
    except OSError:
        pass
    tk = {}
    try:
        with open(os.path.join(model, "tokenizer.json"), encoding="utf-8") as f:
            tk = json.load(f)
    except OSError:
        pass
    d["tokenizer_eos"] = tc.get("eos_token")
    d["chat_template_in_cfg"] = bool(tc.get("chat_template"))
    d["chat_template_jinja"] = os.path.exists(os.path.join(model, "chat_template.jinja"))
    added = [a.get("content", "") for a in tk.get("added_tokens", [])]
    d["im_tokens"] = sum(1 for a in added if a.startswith("<|im_"))
    return d


def make_card(model, train_jsonl, out, bs, epochs, ga, limit, merge, save_every):
    info = model_dir_info(model)
    card = {
        "model": model,
        "model_dir": os.path.basename(model.rstrip("/")),
        "tokenizer_eos": info["tokenizer_eos"],
        "chat_template_in_cfg": info["chat_template_in_cfg"],
        "chat_template_jinja": info["chat_template_jinja"],
        "im_tokens": info["im_tokens"],
        "train_jsonl": train_jsonl,
        "data_sha256": file_sha256(train_jsonl),
        "rows": count_rows(train_jsonl, limit),
        "bs": bs, "ga": ga, "eff_bs": bs * ga, "epochs": epochs, "seq": SEQ,
        "lr": LR, "r": LORA_R, "alpha": LORA_A, "dropout": LORA_DROPOUT,
        "precision": "bf16", "scheduler": "cosine", "warmup_ratio": 0.03,
        "seed": SEED, "optim": "adamw_8bit", "weight_decay": 0.0,
        "packing": True, "format": FORMAT, "eval": "off",
        "save_every": save_every, "merge": bool(merge), "limit": limit or 0,
        "out": out,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return card


def render_chat(tokenizer, row):
    instruction = row["instruction"]
    extra = row.get("input") or ""
    user = instruction + ("\n" + extra if extra else "")
    conv = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["output"]},
    ]
    try:
        text = tokenizer.apply_chat_template(
            conv, add_generation_prompt=False, tokenize=False,
            enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(
            conv, add_generation_prompt=False, tokenize=False)
    if "<think>" in text:
        raise RuntimeError("chat template leaked thinking tokens into a training row")
    stripped = text.rstrip()
    if not stripped.endswith("<|im_end|>"):
        raise RuntimeError("chat template did not close the assistant turn with <|im_end|>")
    return text


def run_diff(card_paths):
    cards = []
    for p in card_paths:
        with open(p, encoding="utf-8") as f:
            cards.append(json.load(f))
    keys = set()
    for c in cards:
        keys.update(c.keys())
    bad = {}
    for k in sorted(keys):
        if k in DIFF_ALLOWED:
            continue
        vals = [c.get(k) for c in cards]
        if any(v != vals[0] for v in vals):
            bad[k] = vals
    if bad:
        print("DIFF-CARDS FAIL extra differences:", json.dumps(bad, ensure_ascii=False, indent=2))
        return 1
    print("DIFF-CARDS PASS only allowed keys differ:",
          {k: [c.get(k) for c in cards] for k in sorted(DIFF_ALLOWED)})
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--train-jsonl", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--ga", type=int, default=GA)
    ap.add_argument("--limit", type=int, default=0, help="debug cap on rows")
    ap.add_argument("--merge", action="store_true",
                    help="also save a merged bf16 full model under <out>/merged")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--write-card", action="store_true")
    ap.add_argument("--diff-cards", nargs="+", default=None)
    args = ap.parse_args()

    if args.diff_cards:
        sys.exit(run_diff(args.diff_cards))

    if not (args.model and args.train_jsonl and args.out):
        ap.error("--model/--train-jsonl/--out are required unless --diff-cards is used")
    os.makedirs(args.out, exist_ok=True)

    card = make_card(args.model, args.train_jsonl, args.out,
                     args.bs, args.epochs, args.ga, args.limit,
                     args.merge, args.save_every)
    card_path = os.path.join(args.out, "run_config.json")
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    rows = count_rows(args.train_jsonl, args.limit)
    eff_bs = args.bs * args.ga
    per_epoch = math.ceil(rows / eff_bs)
    print(f"PLAN model={args.model} rows={rows} bs={args.bs} ga={args.ga} "
          f"eff_bs={eff_bs} epochs={args.epochs} seq={SEQ} lr={LR} r={LORA_R} "
          f"format={FORMAT} eval=off save_every={args.save_every}", flush=True)
    if args.write_card:
        print("CARD_WRITTEN", card_path, flush=True)
        return

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=SEQ,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    tokenizer_eos = getattr(tokenizer, "eos_token", None)
    has_tpl = getattr(tokenizer, "chat_template", None) is not None
    if tokenizer_eos != "<|im_end|>":
        raise RuntimeError(f"unexpected eos_token {tokenizer_eos!r}; expected <|im_end|> (chat release)")
    if not has_tpl:
        raise RuntimeError("loaded tokenizer has no chat_template; refusing plain-concat fallback")

    data_rows = read_rows(args.train_jsonl, args.limit)
    sample = render_chat(tokenizer, data_rows[0])
    format_check = {
        "ok": True, "format": FORMAT, "eos_token": tokenizer_eos,
        "sample_head": sample[:160],
    }
    with open(os.path.join(args.out, "format_check.json"), "w", encoding="utf-8") as f:
        json.dump(format_check, f, ensure_ascii=False, indent=2)
    print("FORMAT_OK", flush=True)

    texts = [render_chat(tokenizer, r) for r in data_rows]
    train_ds = Dataset.from_list([{"text": t} for t in texts])

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R, lora_alpha=LORA_A, lora_dropout=LORA_DROPOUT, bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED, use_rslora=False,
    )

    training_args = UnslothTrainingArguments(
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        num_train_epochs=args.epochs,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=1,
        output_dir=args.out,
        report_to="none",
        optim="adamw_8bit",
        weight_decay=0.0,
        seed=SEED,
        save_strategy="steps",
        save_steps=args.save_every,
        save_only_model=True,
        save_total_limit=None,
    )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=training_args,
        packing=True,
        dataset_text_field="text",
        max_seq_length=SEQ,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.out, "adapter"))

    if args.merge:
        base2, _ = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=SEQ,
            dtype=torch.bfloat16, load_in_4bit=False)
        m = PeftModel.from_pretrained(base2, os.path.join(args.out, "adapter"))
        m = m.merge_and_unload()
        m.save_pretrained(os.path.join(args.out, "merged"), safe_serialization=True)
        tokenizer.save_pretrained(os.path.join(args.out, "merged"))

    state = {}
    try:
        with open(os.path.join(args.out, "trainer_state.json"), encoding="utf-8") as f:
            state = json.load(f)
    except OSError:
        pass
    history = state.get("log_history", [])
    final = {}
    for e in history:
        if "train_loss" in e:
            final = e
    checkpoints = sorted(
        int(p.split("-")[1]) for p in os.listdir(args.out)
        if p.startswith("checkpoint-") and p.split("-")[1].isdigit())
    receipt = {
        "model": args.model, "out": args.out, "card": card_path,
        "global_step": state.get("global_step"), "epoch": state.get("epoch"),
        "final_train_loss": final.get("train_loss"),
        "num_checkpoints": len(checkpoints), "checkpoints": checkpoints,
        "adapter_saved": os.path.isdir(os.path.join(args.out, "adapter")),
        "merged_saved": os.path.isdir(os.path.join(args.out, "merged")),
    }
    with open(os.path.join(args.out, "receipt.json"), "w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    print("RECEIPT", json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
