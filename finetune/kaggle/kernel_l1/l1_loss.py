"""L1 — teacher-forced, per-field train/val loss of a Qwen3 adapter (Kaggle T4x2).

What it measures
----------------
Given the exact training target (Alpaca instruction/input/output rendered through
the no-think chatml template) this runs a FORWARD PASS only (no generation) and
computes the cross-entropy the model assigns to the *response* tokens, split by
JSON field (entity title/type/description, relationship source/target/
description/strength, plus structural tokens). It writes per-row sums so train
vs val, entity vs relationship, and position profiles can be read offline.

It does NOT generate, does not train, and does not judge correctness.

Faithfulness to the canonical recipe (train_unsloth.py)
-------------------------------------------------------
- same render: ``instruction + "\\n" + input`` -> user, ``output`` -> assistant
- same no-think chatml text (rendered by hand, byte-identical to the template
  in train_unsloth.NO_THINK_CHATML with add_generation_prompt=False)
- response-only: loss taken only on tokens after ``<|im_start|>assistant\\n``
- same cutoff: 6144, truncation, no special tokens

Tokenization uses the `tokenizers` library directly on the adapter's
tokenizer.json, so it does not depend on the installed transformers version.

Robustness: fp16 is FORCED (T4 has no native bf16); the loaded dtype is logged
and re-cast if the transformers arg name is ignored. sdpa attention + inference
mode. Results are appended per row (survive a crash). LIMIT caps rows for smoke.

Two models, one GPU each (base on rank 1, adapter on rank 0) via spawn workers.

Smoke:   LIMIT = 20  -> set below, push, read logs, then set None for the full run.
Local:   python l1_loss.py --dry-run --n 5 --adapter <dir> --data <train.jsonl>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

WORK = "/kaggle/working"
SEQ = 6144
LIMIT = None  # per split; None = full run
ASSISTANT_MARKER = "<|im_start|>assistant\n"

MODEL_SLUG = "qwen-lm/qwen-3/transformers/0.6b/1"
MODEL_PATH = f"/kaggle/input/models/{MODEL_SLUG}"
DATA_SLUG = "trace-full-dataset"
ADAPTER_SLUGS = ("l1-masked-adapter", "kg-cap06b-masked-adapter")

FIELDS = (
    "e_title", "e_type", "e_desc",
    "r_source", "r_target", "r_desc", "r_strength",
    "structural",
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- rendering

def render_chat(row: dict) -> str:
    """Byte-identical to train_unsloth NO_THINK_CHATML, add_generation_prompt=False."""
    instruction = row["instruction"]
    extra = row.get("input") or ""
    user = instruction + ("\n" + extra if extra else "")
    conv = (("user", user), ("assistant", row["output"]))
    return "".join(f"<|im_start|>{role}\n{content}<|im_end|>\n" for role, content in conv)


def load_tokenizer(adapter_dir: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(adapter_dir, "tokenizer.json"))


def encode_tokens(tok, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    enc = tok.encode(text, add_special_tokens=False)
    return enc.ids[:SEQ], list(enc.offsets[:SEQ])


# ---------------------------------------------------------------- field spans

_STR = r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"'


def field_value_spans(out: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    rel_i = out.find('"relationships"')
    ent_region = out[:rel_i] if rel_i >= 0 else out
    for key, fld in (("title", "e_title"), ("type", "e_type"), ("description", "e_desc")):
        for m in re.finditer(_STR % key, ent_region):
            spans.append((fld, m.start(1), m.end(1)))
    if rel_i >= 0:
        rel = out[rel_i:]
        for key, fld in (("source", "r_source"), ("target", "r_target"), ("description", "r_desc")):
            for m in re.finditer(_STR % key, rel):
                spans.append((fld, m.start(1), m.end(1)))
        for m in re.finditer(r'"strength"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?|null)', rel):
            spans.append(("r_strength", m.start(1), m.end(1)))
    return spans


def label_for_token(off_start: int, off_end: int, spans: list[tuple[str, int, int]]) -> str:
    mid = (off_start + off_end) / 2.0
    for fld, s, e in spans:
        if s <= mid <= e:
            return fld
    return "structural"


def _response_bounds(text: str, out: str) -> tuple[int, list[tuple[str, int, int]]]:
    resp_char = text.index(ASSISTANT_MARKER) + len(ASSISTANT_MARKER)
    if text[resp_char:resp_char + len(out)] != out:
        raise RuntimeError("assistant content is not verbatim after the marker")
    spans = [(f, resp_char + s, resp_char + e) for f, s, e in field_value_spans(out)]
    return resp_char, spans


def field_token_counts(tok, row: dict) -> dict:
    text = render_chat(row)
    resp_char, spans = _response_bounds(text, row["output"])
    ids, offs = encode_tokens(tok, text)
    counts = {f: 0 for f in FIELDS}
    n_resp = 0
    for (s, e) in offs:
        if s < resp_char:
            continue
        n_resp += 1
        counts[label_for_token(s, e, spans)] += 1
    counts["_response_tokens"] = n_resp
    counts["_total_tokens"] = len(ids)
    counts["_truncated"] = len(ids) >= SEQ
    return counts


# ---------------------------------------------------------------- data paths

def _candidates(rel: str) -> list[str]:
    return [
        f"/kaggle/input/{rel}",
        f"/kaggle/input/datasets/idalextan/{rel}",
        f"/kaggle/input/datasets/{rel}",
    ]


def resolve_data_file(name: str) -> str:
    for cand in _candidates(f"{DATA_SLUG}/{name}"):
        if os.path.exists(cand):
            return cand
    return f"/kaggle/input/{DATA_SLUG}/{name}"


def resolve_adapter_dir() -> str:
    for slug in ADAPTER_SLUGS:
        for cand in _candidates(slug):
            if os.path.exists(os.path.join(cand, "adapter_config.json")):
                return cand
    return f"/kaggle/input/{ADAPTER_SLUGS[0]}"


# ---------------------------------------------------------------- model load

def _load_base(torch):
    from transformers import AutoModelForCausalLM
    common = dict(device_map="cuda", trust_remote_code=True)
    last = None
    for extra in (dict(dtype=torch.float16, attn_implementation="sdpa"),
                  dict(dtype=torch.float16),
                  dict(torch_dtype=torch.float16, attn_implementation="sdpa"),
                  dict(torch_dtype=torch.float16),
                  dict()):
        try:
            return AutoModelForCausalLM.from_pretrained(MODEL_PATH, **{**common, **extra})
        except TypeError as e:  # unknown kwarg for this transformers version
            last = e
            continue
    raise last


# ---------------------------------------------------------------- forward pass

def _run_worker(kind: str, rank: int, out_path: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch
    import torch.nn.functional as F

    adapter = resolve_adapter_dir()
    log(f"[{kind}] adapter/tokenizer dir={adapter}")
    tok = load_tokenizer(adapter)
    log(f"[{kind}] model_path_exists={os.path.exists(MODEL_PATH)} loading on rank {rank}")
    model = _load_base(torch)
    if kind == "masked":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    if next(model.parameters()).dtype != torch.float16:
        model = model.half()
    model.eval()
    log(f"[{kind}] cuda_count={torch.cuda.device_count()} "
        f"dev={torch.cuda.get_device_name(0)} model_dtype={next(model.parameters()).dtype} "
        f"limit={LIMIT}")
    dev = torch.device("cuda")
    n_bins = 10

    fout = open(out_path, "w", encoding="utf-8")
    for split, fname in (("train", "alpaca_full_train.jsonl"), ("val", "alpaca_full_val.jsonl")):
        path = resolve_data_file(fname)
        log(f"[{kind}] {split} from {path}")
        n = 0
        idx = 0
        t0 = time.time()
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if LIMIT and idx >= LIMIT:
                    break
                row = json.loads(line)
                text = render_chat(row)
                resp_char, spans = _response_bounds(text, row["output"])
                ids_list, offs = encode_tokens(tok, text)
                resp_pos = [i for i, (s, _) in enumerate(offs) if s >= resp_char]
                if not resp_pos or resp_pos[0] == 0:
                    idx += 1
                    continue
                ids = torch.tensor([ids_list], device=dev)
                with torch.inference_mode():
                    logits = model(ids).logits[0]
                    pos = torch.tensor(resp_pos, device=dev)
                    chunks = []
                    for b in range(0, len(resp_pos), 256):
                        pp = pos[b:b + 256]
                        chunks.append(F.cross_entropy(logits[pp - 1].float(), ids[0, pp], reduction="none"))
                    losses = torch.cat(chunks).tolist()
                    del logits
                fields = {f: [0.0, 0] for f in FIELDS}
                bins = {f: [[0.0, 0] for _ in range(n_bins)] for f in FIELDS}
                R = len(resp_pos)
                tot = 0.0
                for k, tok_i in enumerate(resp_pos):
                    l = losses[k]
                    fld = label_for_token(offs[tok_i][0], offs[tok_i][1], spans)
                    fields[fld][0] += l
                    fields[fld][1] += 1
                    b = min(int(n_bins * k / R), n_bins - 1)
                    bins[fld][b][0] += l
                    bins[fld][b][1] += 1
                    tot += l
                rec = {
                    "id": f"{split}-{idx}", "model": kind, "split": split,
                    "n_resp": R, "total_sum": tot,
                    "fields": {f: {"sum": v[0], "n": v[1]} for f, v in fields.items()},
                    "pos_bins": bins,
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n += 1
                idx += 1
                if n % 10 == 0:
                    el = time.time() - t0
                    log(f"[{kind}] {split} {n} rows {el:.1f}s ({el / n:.2f}s/row)")
        log(f"[{kind}] {split} done: {n} rows in {time.time() - t0:.1f}s")
    fout.close()
    log(f"[{kind}] wrote {out_path}")


def _spawn(kind: str, rank: int, out_path: str):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_run_worker, args=(kind, rank, out_path))
    p.start()
    return p


def _prepare_env() -> None:
    """PEFT hard-imports torchao; Kaggle ships torchao 0.10.0 which it rejects.
    Uninstall it (mirrors finetune/kaggle/kernel_cap06/kg_tracer_cap06.py)."""
    import subprocess
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
            timeout=120, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log("torchao uninstalled (or absent)")
    except Exception as e:  # noqa: BLE001
        log(f"torchao uninstall failed (continuing): {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="outputs/graphrag_full/alpaca_full_train.jsonl")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.worker:
        _run_worker(args.worker, args.rank, args.out or f"{WORK}/l1_{args.worker}.jsonl")
        return 0

    if args.dry_run:
        tok = load_tokenizer(args.adapter)
        with open(args.data, encoding="utf-8") as f:
            rows = [json.loads(next(f)) for _ in range(args.n)]
        for i, r in enumerate(rows):
            print(i, json.dumps(field_token_counts(tok, r)))
        return 0

    os.makedirs(WORK, exist_ok=True)
    import torch
    log(f"start: cuda_count={torch.cuda.device_count()} limit={LIMIT}")
    _prepare_env()
    procs = [
        _spawn("base", 1, f"{WORK}/l1_base.jsonl"),
        _spawn("masked", 0, f"{WORK}/l1_masked.jsonl"),
    ]
    for p in procs:
        p.join()
    log("all workers done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
