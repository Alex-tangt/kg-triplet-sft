#!/usr/bin/env python3
"""Profile inference kernels on the full-trained adapter (T4).

Training profiler showed bf16 falls back to FP32 GEMMs on T4 (magma_sgemmEx)
while fp16 uses tensor cores (turing_fp16_s1688gemm). Inference on the SAME
model plateaus at ~20 tok/s in fp16 regardless of prompt length or dtype, which
looks wrong for a 0.6B model (expect ~100+ tok/s on T4). This kernel profiles
real generation to answer two questions:

1. What GEMM kernels actually run during fp16 decode — tensor-core
   `turing_fp16_s1688gemm` or a fallback path?
2. What is the attention implementation (eager / sdpa / flash) and how much
   time does non-matmul overhead (kernel launch, python sampling loop) take?

Run: python /kaggle/src/script.py   (metadata in kernel_inferprof/)
"""

import json
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
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    log(f"gpu: {torch.cuda.get_device_name(0)} sm={torch.cuda.get_device_capability(0)}")
    for p in (MODEL_PATH, ADAPTER_DIR):
        ok = os.path.exists(p)
        log(f"{'OK ' if ok else 'MISSING'} {p}")
        if not ok:
            raise SystemExit(f"missing {p}")


def install():
    import importlib.metadata as md
    import subprocess
    for p in ("torchao", "transformers", "peft", "accelerate"):
        try:
            log(f"pre {p}: {md.version(p)}")
        except Exception:
            pass
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "transformers>=4.49.0,<5.0.0", "peft>=0.14.0,<=0.18.1"],
        stdout=subprocess.DEVNULL,
    )
    for p in ("transformers", "peft"):
        log(f"post {p}: {md.version(p)}")


PROMPT = (
    "-GOAL-\nGiven a text document, identify all entities and relationships.\n"
    "-OUTPUT-\nReturn a single JSON object with exactly two keys: "
    '{"entities": [{"title", "type", "description"}], '
    '{"source", "target", "description", "strength"}}. '
    'If nothing can be extracted, return {"entities": [], "relationships": []}.'
)
PASSAGE = (
    "Arne Kaijser (born 1950) is a professor emeritus of history of technology "
    "at the KTH Royal Institute of Technology in Stockholm, and a former "
    "president of the Society for the History of Technology. Kaijser is a "
    "member of the Royal Swedish Academy of Engineering Sciences since 2007 "
    "and also a member of the editorial board of Journal of Urban Technology "
    "and Centaurus."
)


def run_decode(model, tokenizer, n_gen=300):
    import torch
    msgs = [{"role": "user", "content": f"{PROMPT}\n{PASSAGE}"}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    schema_prefix = '{"entities": ['
    prefix_ids = tokenizer.encode(schema_prefix, add_special_tokens=False)
    prefix_t = torch.tensor([prefix_ids], dtype=torch.long, device="cuda")
    gen_input = torch.cat([inputs["input_ids"], prefix_t], dim=1)
    n_prompt = gen_input.shape[1]

    t0 = time.time()
    gen = model.generate(
        input_ids=gen_input, max_new_tokens=n_gen, do_sample=True,
        temperature=0.7, top_p=0.95, repetition_penalty=1.15,
    )
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_new = gen.shape[1] - n_prompt
    log(f"decode {n_new} tok in {dt:.1f}s = {n_new/dt:.1f} tok/s "
        f"(prompt {n_prompt} tok)")
    return gen


def main():
    check_env()
    install()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from torch.profiler import profile, ProfilerActivity

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    log("\n===== FP16 load =====")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True
    ).to("cuda")
    model = PeftModel.from_pretrained(
        model, ADAPTER_DIR, torch_dtype=torch.float16
    ).merge_and_unload()
    model.eval()
    log(f"attn_implementation = {getattr(model.config, '_attn_implementation', 'UNSET')}")
    log(f"model dtype = {next(model.parameters()).dtype}")

    # warmup
    run_decode(model, tokenizer, n_gen=30)
    torch.cuda.empty_cache()

    # profiled decode: capture kernel names + timing breakdown
    from torch.profiler import profile as prof, ProfilerActivity
    with prof(activities=[ProfilerActivity.CUDA]) as p:
        run_decode(model, tokenizer, n_gen=120)

    log("\n--- top CUDA kernels by device time (decode) ---")
    events = sorted(
        p.key_averages(), key=lambda e: e.self_device_time_total, reverse=True
    )[:15]
    for e in events:
        log(f"  {e.self_device_time_total/1e3:9.1f}ms  {e.key[:110]}")

    # count of kernel launches per token region
    total_launches = len(p.key_averages())
    log(f"total distinct kernel keys: {total_launches}")

    # naive (non-profiled) full decode rate already printed by run_decode

    # second: compare fp16 vs bf16 load wall rate only (no profiler overhead)
    log("\n===== BF16 load (compare) =====")
    del model
    torch.cuda.empty_cache()
    model2 = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    model2 = PeftModel.from_pretrained(
        model2, ADAPTER_DIR, torch_dtype=torch.bfloat16
    ).merge_and_unload()
    model2.eval()
    run_decode(model2, tokenizer, n_gen=200)
    log("\n=== INFER PROF DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
