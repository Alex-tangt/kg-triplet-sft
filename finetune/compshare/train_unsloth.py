"""Round-1 capacity-curve training (canonical masked recipe, Unsloth, 4090).

Root-cause fix (recipe-audit 2026-09-07, issue 08): the previous capacity runs
trained on a hand-stitched raw ``text`` field, which Unsloth treats as
continued-pretraining -- causal LM loss over EVERY token including the long
user prompt. The reference fp16 line (LLaMA-Factory SFT) masks the prompt and
supervises only the assistant response; ~85-90% of our loss budget was spent
predicting prompt continuation, which is why wrapper changes (plain/chat) never
moved the referent score (~0.44 vs reference ~0.53).

This canonical recipe mirrors the reference on every learning-relevant axis:
  - text rendered through a NO-THINK chatml template (the Qwen3 default template
    injects an empty <think>\\n\\n</think> preamble the reference does not train
    with) via messages -> apply_chat_template (canonical Unsloth flow)
  - response-only loss masking via unsloth.chat_templates.train_on_responses_only
  - eff batch 8 (bs 2 x ga 4), warmup 0.1, dropout 0.1, adamw_torch, bf16,
    lr 5e-5 cosine, 5 epochs, seed 42, cutoff 6144
  - NON-packed (one example per sequence): unsloth ignores packing while
    UNSLOTH_RETURN_LOGITS=1, and non-packed also mirrors the reference
    LLaMA-Factory SFT step semantics one-to-one
  - UNSLOTH_RETURN_LOGITS=1 (community workaround for fused-loss sparse-mask
    silent zero-gradient, unsloth#5230)

Integrity artifacts under <out>/: run_config.json, format_check.json
(masked-sample evidence), receipt.json.

Usage:
  python train_unsloth.py --write-card --model M --train-jsonl D --out O
  python train_unsloth.py --diff-cards C1 C2 C3 ...
  python train_unsloth.py --model M --train-jsonl D --out O [--merge]
  ... --no-mask                       # legacy full-seq variant
  ... --packing                       # re-enable packing (ignored while RETURN_LOGITS=1)
  ... --limit 50                         # smoke cap
"""
import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

SEQ = 6144
LORA_R = 32
LORA_A = 32
LORA_DROPOUT = 0.1
GA = 4
LR = 5e-5
EPOCHS = 5
SEED = 42
WARMUP_RATIO = 0.1
OPTIM = "adamw_torch"
PRECISION = "bf16"
FORMAT = "chat_qwen3_nothink_masked"

DIFF_ALLOWED = {"model", "model_dir", "out", "timestamp"}
CARD_KEYS = ["model", "model_dir", "tokenizer_eos", "chat_template_in_cfg",
             "chat_template_jinja", "im_tokens", "train_jsonl", "data_sha256",
             "rows", "bs", "ga", "eff_bs", "epochs", "seq", "lr", "r", "alpha",
             "dropout", "precision", "scheduler", "warmup_ratio", "seed",
             "optim", "weight_decay", "packing", "mask", "format", "eval",
             "save_every", "merge", "limit", "out", "timestamp"]

# Qwen3 chatml WITHOUT the think preamble (reference/llamafactory qwen3_nothink
# layout). Plain chatml: each message is <|im_start|><role>\\n<content><|im_end|>\\n.
NO_THINK_CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
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


def make_card(model, train_jsonl, out, bs, epochs, ga, limit, merge, save_every,
              packing, mask, optim):
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
        "precision": PRECISION, "scheduler": "cosine",
        "warmup_ratio": WARMUP_RATIO,
        "seed": SEED, "optim": optim, "weight_decay": 0.0,
        "packing": bool(packing), "mask": "response_only" if mask else "none",
        "format": FORMAT, "eval": "off",
        "save_every": save_every, "merge": bool(merge), "limit": limit or 0,
        "out": out,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return card


def render_chat(tokenizer, row):
    """Render one row through the no-think chatml template (canonical path).

    The base Qwen3 tokenizer's default chat_template injects an empty
    <think>\\n\\n</think> block before the assistant turn; callers must install
    NO_THINK_CHATML (see main) before rendering so the text byte-matches the
    reference no-think layout.
    """
    instruction = row["instruction"]
    extra = row.get("input") or ""
    user = instruction + ("\n" + extra if extra else "")
    conv = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["output"]},
    ]
    text = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=False)
    if "<|im_start|>user" not in text or "<|im_start|>assistant" not in text:
        raise RuntimeError("no-think chatml did not wrap the user/assistant turns")
    if not text.rstrip().endswith("<|im_end|>"):
        raise RuntimeError("no-think chatml did not close the assistant turn with <|im_end|>")
    if "<think>" in text:
        raise RuntimeError("no-think chatml leaked a thinking preamble")
    return text


def verify_mask(dataset, tokenizer, n=3):
    """Evidence that train_on_responses_only masking is real, not all -100."""
    out = {"checked": 0, "all_masked": 0, "samples": []}
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        keys = set(item.keys())
        if "labels" not in keys:
            out["samples"].append({"idx": i, "keys": sorted(keys),
                                   "note": "no labels column yet"})
            continue
        labels = item["labels"]
        active = sum(1 for v in labels if v != -100)
        head = labels[:200]
        masked = all(v == -100 for v in head)
        out["checked"] += 1
        if active == 0:
            out["all_masked"] += 1
        out["samples"].append({
            "idx": i, "len": len(labels), "active_labels": active,
            "head_all_masked": masked,
            "decoded_tail": tokenizer.decode(
                [v if v != -100 else 0 for v in labels][-120:])[-80:],
        })
    return out


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
    ap.add_argument("--packing", dest="packing", action="store_true", default=False,
                    help="pack multiple examples per sequence (NOTE: unsloth ignores "
                         "packing while UNSLOTH_RETURN_LOGITS=1; default run is non-packed)")
    ap.add_argument("--mask", dest="mask", action="store_true", default=True)
    ap.add_argument("--no-mask", dest="mask", action="store_false",
                    help="legacy full-sequence LM loss (for A/B only)")
    ap.add_argument("--optim", default=OPTIM)
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
                     args.merge, args.save_every, args.packing, args.mask,
                     args.optim)
    card_path = os.path.join(args.out, "run_config.json")
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    rows = count_rows(args.train_jsonl, args.limit)
    eff_bs = args.bs * args.ga
    per_epoch = math.ceil(rows / eff_bs)
    print(f"PLAN model={args.model} rows={rows} bs={args.bs} ga={args.ga} "
          f"eff_bs={eff_bs} epochs={args.epochs} seq={SEQ} lr={LR} r={LORA_R} "
          f"warmup={WARMUP_RATIO} dropout={LORA_DROPOUT} optim={args.optim} "
          f"packing={args.packing} mask={'response_only' if args.mask else 'none'} "
          f"format={FORMAT} eval=off save_every={args.save_every}", flush=True)
    if args.write_card:
        print("CARD_WRITTEN", card_path, flush=True)
        return

    import unsloth  # noqa: F401  (must precede transformers/peft imports)
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
    import torch
    from datasets import Dataset
    from peft import PeftModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=SEQ,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    tokenizer_eos = getattr(tokenizer, "eos_token", None)
    if tokenizer_eos != "<|im_end|>":
        raise RuntimeError(f"unexpected eos_token {tokenizer_eos!r}; expected <|im_end|> (chat release)")
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError("loaded tokenizer has no chat_template")

    # no-think chatml for the whole training render; keep the original to
    # restore before saving artifacts (so merged/adapter templates stay stock).
    orig_template = tokenizer.chat_template
    tokenizer.chat_template = NO_THINK_CHATML

    data_rows = read_rows(args.train_jsonl, args.limit)
    sample = render_chat(tokenizer, data_rows[0])
    format_check = {
        "ok": True, "format": FORMAT, "eos_token": tokenizer_eos,
        "template_override": True, "sample_head": sample[:160],
    }
    texts = [render_chat(tokenizer, r) for r in data_rows]
    train_ds = Dataset.from_list([{"text": t} for t in texts])

    tokenizer.chat_template = orig_template

    with open(os.path.join(args.out, "format_check.json"), "w", encoding="utf-8") as f:
        json.dump(format_check, f, ensure_ascii=False, indent=2)
    print("FORMAT_OK", flush=True)

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
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        logging_steps=1,
        output_dir=args.out,
        report_to="none",
        optim=args.optim,
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
        packing=args.packing,
        dataset_text_field="text",
        max_seq_length=SEQ,
    )

    if args.mask:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        check = verify_mask(trainer.train_dataset, tokenizer)
        mask_path = os.path.join(args.out, "mask_check.json")
        with open(mask_path, "w", encoding="utf-8") as f:
            json.dump(check, f, ensure_ascii=False, indent=2)
        if check["all_masked"]:
            raise RuntimeError("masking produced all--100 labels; wrong markers")
        print("MASK_OK", json.dumps(check, ensure_ascii=False), flush=True)

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
    state_path = os.path.join(args.out, "trainer_state.json")
    if not os.path.exists(state_path):
        cks = sorted(p for p in os.listdir(args.out)
                     if p.startswith("checkpoint-") and p.split("-")[1].isdigit())
        if cks:
            state_path = os.path.join(args.out, cks[-1], "trainer_state.json")
    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    except OSError:
        pass
    history = state.get("log_history", [])
    losses = [e for e in history if "loss" in e]
    final_step_loss = losses[-1].get("loss") if losses else None
    checkpoints = sorted(
        int(p.split("-")[1]) for p in os.listdir(args.out)
        if p.startswith("checkpoint-") and p.split("-")[1].isdigit())
    receipt = {
        "model": args.model, "out": args.out, "card": card_path,
        "global_step": state.get("global_step"), "epoch": state.get("epoch"),
        "final_step_loss": final_step_loss,
        "num_checkpoints": len(checkpoints), "checkpoints": checkpoints,
        "adapter_saved": os.path.isdir(os.path.join(args.out, "adapter")),
        "merged_saved": os.path.isdir(os.path.join(args.out, "merged")),
    }
    with open(os.path.join(args.out, "receipt.json"), "w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    print("RECEIPT", json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
