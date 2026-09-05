"""Round-1 training for the bf16 capacity-curve line (Unsloth, single 4090).

Canonical hyperparameters (fixed across every curve/ablation member):
r=32 alpha=32, bf16, cutoff 6144, ga 8, lr 5e-5, cosine+warmup 0.03, 5 epochs.
Config differs between runs ONLY in --model / --train-jsonl / --out.

Usage examples:
  python train_unsloth.py --model /model/.../Qwen3-0.6B \
      --train-jsonl /root/kg/data/alpaca_full_train.jsonl --out /root/kg/out/0.6b
  python train_unsloth.py --model ... --train-jsonl subset500.jsonl \
      --epochs 5 --out /root/kg/out/abla500
Dry-run (no GPU, prints resolved plan): add --dry-run
"""
import argparse
import json
import math
import os

SEQ = 6144
LORA_R = 32
LORA_A = 32
GA = 8
LR = 5e-5
EPOCHS = 5


def build_text(row):
    instruction = row["instruction"]
    extra = row.get("input")
    user = instruction + ("\n" + extra if extra else "")
    return user + "\n\n" + row["output"]


def load_texts(jsonl_path, limit):
    texts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            texts.append(build_text(json.loads(line)))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--ga", type=int, default=GA)
    ap.add_argument("--limit", type=int, default=0, help="debug cap on rows")
    ap.add_argument("--merge", action="store_true",
                    help="also save a merged bf16 full model under <out>/merged")
    ap.add_argument("--save-every", type=int, default=50,
                    help="checkpoint + eval cadence (optimizer steps); "
                         "smaller = less wall time lost on a crash")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    n_rows = 0
    with open(args.train_jsonl, encoding="utf-8") as f:
        for _ in f:
            n_rows += 1
    if args.limit:
        n_rows = min(n_rows, args.limit)
    eff_bs = args.bs * args.ga
    per_epoch = math.ceil(n_rows / eff_bs)
    total_steps = per_epoch * args.epochs
    print(f"PLAN model={args.model} rows={n_rows} bs={args.bs} ga={args.ga} "
          f"eff_bs={eff_bs} per_epoch_steps={per_epoch} epochs={args.epochs} "
          f"total_steps={total_steps} seq={SEQ} lr={LR} r={LORA_R}", flush=True)
    if args.dry_run:
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
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R, lora_alpha=LORA_A, lora_dropout=0.05, bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42, use_rslora=False,
    )

    train_texts = load_texts(args.train_jsonl, args.limit)
    train_ds = Dataset.from_list([{"text": t} for t in train_texts])

    save_steps = args.save_every
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
        seed=42,
        save_strategy="steps",
        save_steps=save_steps,
        save_only_model=True,
        save_total_limit=None,
    )

    eval_ds = None
    if args.val_jsonl:
        eval_texts = load_texts(args.val_jsonl, args.limit)
        eval_ds = Dataset.from_list([{"text": t} for t in eval_texts])
        training_args.eval_strategy = "steps"
        training_args.eval_steps = save_steps
        training_args.per_device_eval_batch_size = 1

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
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


if __name__ == "__main__":
    main()
