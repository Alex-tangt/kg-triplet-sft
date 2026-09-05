#!/usr/bin/env python3
"""Batch-decode A/B on the full-trained adapter (T4).

Decode is GEMV-bound (~16-20 tok/s at bs=1; profiler shows cublas
`gemv2T_kernel` / `gemvx::kernel`, memory-bandwidth limited, NOT the tensor-core
GEMM path). Hypothesis: batching N passages turns the per-layer weight
application into a GEMM (matrix x matrix) which on T4 fp16 hits the tensor
cores / much better bandwidth utilisation, so N parallel sequences decode at
~N x the single-stream rate.

This kernel measures decode tok/s at batch 1, 4, 8 on identical short
passages (padded), same model/adapter, to quantify the real speedup before
committing the inference kernel to batch decoding.

Run: python /kaggle/src/script.py   (metadata in kernel_benchbatch/)
"""

import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

WORK = "/kaggle/working"
MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3/transformers/0.6b/1"
ADAPTER_DIR = "/kaggle/input/notebooks/idalextan/kg-tracer-lffull/saves/qwen3-0.6b-graphrag-full-lf"


def log(msg):
    print(msg, flush=True)


def check_env():
    import torch
    log(f"torch {torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    for p in (MODEL_PATH, ADAPTER_DIR):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    log("env ok")


def install():
    import importlib.metadata as md
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "transformers>=4.49.0,<5.0.0", "peft>=0.14.0,<=0.18.1"],
        stdout=subprocess.DEVNULL,
    )


PROMPT = (
    "-GOAL-\nGiven a text document, identify all entities and relationships.\n"
    "-OUTPUT-\nReturn a single JSON object with two keys: "
    '{"entities": [...], "relationships": [...]}. '
    'If nothing can be extracted, return {"entities": [], "relationships": []}.'
)
PASSAGE = (
    "Arne Kaijser (born 1950) is a professor emeritus of history of technology "
    "at the KTH Royal Institute of Technology in Stockholm, and a former "
    "president of the Society for the History of Technology. Kaijser is a "
    "member of the Royal Swedish Academy of Engineering Sciences since 2007 "
    "and also a member of the editorial board of Journal of Urban Technology "
    "and Centaurus. Kaijser has published two books in Swedish and has "
    "co-edited several anthologies."
)


def make_prompts(tokenizer, n):
    msgs = [{"role": "user", "content": f"{PROMPT}\n{PASSAGE}"}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return [text] * n


def bench(model, tokenizer, bs, n_gen=200):
    import torch
    texts = make_prompts(tokenizer, bs)
    enc = tokenizer(texts, return_tensors="pt", padding=True)
    prefix = '{"entities": ['
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    prefix_t = torch.tensor([prefix_ids] * bs, dtype=torch.long, device="cuda")
    gen_input = torch.cat([enc["input_ids"].to("cuda"), prefix_t], dim=1)
    attn = enc["attention_mask"].to("cuda")
    attn = torch.cat([attn, torch.ones(bs, prefix_t.shape[1], dtype=attn.dtype,
                                       device="cuda")], dim=1)
    n_prompt = gen_input.shape[1]

    torch.cuda.synchronize()
    t0 = time.time()
    gen = model.generate(
        input_ids=gen_input, attention_mask=attn, max_new_tokens=n_gen,
        do_sample=True, temperature=0.7, top_p=0.95, repetition_penalty=1.15,
    )
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_new = gen.shape[1] - n_prompt
    per_seq = n_new / bs
    rate_total = n_new * bs / dt          # aggregate tokens/sec across batch
    rate_per = per_seq / dt               # wall time per sequence
    log(f"bs={bs}: gen {dt:.1f}s, {n_new} new/seq -> "
        f"{rate_total:.0f} agg tok/s, {rate_per:.0f} tok/s per-seq wall")
    return rate_per


def main():
    check_env()
    install()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True
    ).to("cuda")
    model = PeftModel.from_pretrained(
        model, ADAPTER_DIR, torch_dtype=torch.float16
    ).merge_and_unload()
    model.eval()

    log("warmup bs=1 ...")
    bench(model, tokenizer, 1, n_gen=30)
    torch.cuda.empty_cache()

    for bs in (1, 4, 8):
        rate = bench(model, tokenizer, bs, n_gen=200)
        log(f"-> bs={bs} per-seq wall rate = {rate:.0f} tok/s")
        torch.cuda.empty_cache()

    log("=== BATCH BENCH DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
