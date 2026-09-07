#!/usr/bin/env python3
"""Inference on a CAPACITY-CURVE LoRA adapter (bf16 Unsloth line, ADR-0006).

Loads the base Qwen3 (Kaggle model slug), merges a capacity adapter mounted
from the `idalextan/cap-upload` dataset (qwen3-0.6b / qwen3-1.7b / qwen3-4b),
and runs single-shot extraction on the 200-passage referent sample
(sample_200_passages.jsonl) with the STUDENT_PROMPT (thinking OFF).

PROTOCOL selects the prompt wrapper (baked per model by make_capinfer.py):
  - "chat":  Qwen3 chat template (reproduction-line protocol, self-consistent
             with the chat-trained reference baseline);
  - "plain": training-faithful format for the capacity line
             (STUDENT_PROMPT\\n<passage>\\n\\n, no chat markers — matches
             train_unsloth.py build_text; see issue 08 Finding 2026-09-06).
Both keep the schema-prefix injection + tolerant parser + length-sorted batched
decode + one worker per GPU, so the two differ only in the wrapper.

Writes predictions.jsonl + raw_<id>.txt + summary.csv under /kaggle/working.

Output rows match the deploy extraction format consumed by eval/referent_*:
  {"id", "entities": [{title,type,description}], "relationships":
   [{source,target,description,strength}]}

Usage on Kaggle: python /kaggle/src/script.py  (metadata per model dir)
"""

import json
import os
import re
import sys

# Device selection is per-worker inside _infer_worker; the parent must NOT pin
# CUDA_VISIBLE_DEVICES (see main()).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ---- embedded eval inputs (STUDENT_PROMPT only; injected by merge) ----
# INPUTS_HERE

WORK = "/kaggle/working"
# Placeholders are filled per-model by make_capinfer.py.
MODEL_SLUG = "%%MODEL_SLUG%%"          # e.g. qwen-lm/qwen-3/transformers/0.6b/1
ADAPTER_NAME = "%%ADAPTER_NAME%%"      # e.g. qwen3-0.6b
MODEL_PATH = f"/kaggle/input/models/{MODEL_SLUG}"
ADAPTER_DIR = f"/kaggle/input/cap-upload/{ADAPTER_NAME}"
# Smoke cap: make_capinfer.py bakes --limit into LIMIT (kernel env is not
# settable at push time, so per-run limits are baked at generation).
LIMIT = None
# Batch size for length-sorted decode (measured 22/85/164 tok/s at bs 1/4/8 on
# 0.6B). Per-model override baked by the generator.
BATCH_SIZE = 4
# Prompt wrapper: "chat" applies the Qwen3 chat template (reproduction-line
# protocol); "plain" uses the capacity training format verbatim
# (`STUDENT_PROMPT\n<passage>\n\n`, no chat markers — train_unsloth build_text).
# Schema-prefix injection and sampling are identical in both, so the two differ
# only in the wrapper. See issue 08 Finding (2026-09-06).
PROTOCOL = "%%PROTOCOL%%"
# Stop decoding as soon as the top-level JSON object closes (bracket balance
# returns to zero after the forced `{"entities": [` prefix) instead of decoding
# to max_new_tokens. Plain prompts have no chat stop marker and would otherwise
# generate ~8k tokens per passage (~5.6 h for 200); early stop cuts wall time
# to ~chat levels. Backstop MAX_NEW_TOKENS still applies if JSON never closes.
EARLY_STOP = False
MAX_NEW_TOKENS = 8192


def log(msg: str) -> None:
    print(msg, flush=True)


def _tree(roots=("/kaggle/input",), depth=4) -> list[str]:
    lines = []
    for root in roots:
        if not os.path.isdir(root):
            lines.append(f"MISSING {root}")
            continue
        base = root.count(os.sep)
        for cur, dirs, files in os.walk(root):
            if cur.count(os.sep) - base > depth:
                dirs[:] = []
                continue
            lines.append(f"DIR  {cur}")
            for f in sorted(files)[:8]:
                lines.append(f"  {f}")
    return lines


# Dataset-source mount layouts observed on Kaggle (slug-root and the
# /datasets/<owner>/<slug> forms); model sources mount under /kaggle/input/models.
def _candidates(rel: str) -> list[str]:
    return [
        f"/kaggle/input/{rel}",
        f"/kaggle/input/datasets/idalextan/{rel}",
        f"/kaggle/input/datasets/{rel}",
    ]


def resolve_adapter_dir() -> str:
    """Return the adapter dir the dataset actually mounted at.

    Spawned workers re-import this module and re-evaluate module constants, so
    path resolution must be a pure function both the parent (check_env) and
    each worker call — never a mutated module global.
    """
    for cand in _candidates(f"cap-upload/{ADAPTER_NAME}"):
        if os.path.exists(os.path.join(cand, "adapter_config.json")):
            return cand
    return ADAPTER_DIR


def check_env() -> None:
    log("=== env ===")
    import torch

    log(f"torch: {torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu: {torch.cuda.get_device_name(0)}  count={torch.cuda.device_count()}")
    adapter = resolve_adapter_dir()
    log(f"MODEL_PATH={MODEL_PATH}\nADAPTER_DIR={adapter}")
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"required base model missing: {MODEL_PATH}")
    if not os.path.exists(os.path.join(adapter, "adapter_config.json")):
        log("--- /kaggle/input tree ---\n" + "\n".join(_tree()))
        raise SystemExit(f"no adapter_config.json under {adapter}")


def _array_span(text: str, key: str, stop_key: str | None = None) -> str | None:
    """Return the JSON array value for `key` (with surrounding brackets).

    Respects string/nesting depth. If the array never closes, stop at the
    next `stop_key` (so an unterminated entities array does not swallow the
    relationships that follow) and return the unclosed prefix — the caller
    repairs it by appending ']'. Returns None if the key or its array is
    absent.
    """
    i = text.find('"' + key + '"')
    if i < 0:
        return None
    j = text.find("[", i)
    if j < 0:
        return None
    limit = len(text)
    if stop_key:
        k = text.find('"' + stop_key + '"', i + len(key))
        if k > j:
            limit = k
    depth = 0
    in_str = False
    esc = False
    for idx in range(j, min(limit + 1, len(text))):
        c = text[idx]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[j : idx + 1]
    return text[j:limit]


def _fix_array(seg: str | None, item_key: str) -> list[dict]:
    """Best-effort parse of a possibly-broken array into a list of dicts.

    Tries, in order: strict parse, append ']', prepend '[', per-object salvage
    (each non-nested {...} parsed independently with light char repairs).
    Only dicts carrying `item_key` (e.g. "title" or "source") are kept.
    """
    if not seg:
        return []
    cands = [seg]
    if not seg.endswith("]"):
        cands.append(seg + "]")
    if not seg.startswith("["):
        cands.append("[" + seg)
    cands += [seg.replace(")", "}"), seg.replace("),", "},")]
    for c in cands:
        try:
            v = json.loads(c)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, list):
            return [d for d in v if isinstance(d, dict) and d.get(item_key)]
    # per-object salvage
    out = []
    for o in re.findall(r"\{[^{}]*\}", seg):
        try:
            d = json.loads(o.replace(")", "}"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and d.get(item_key):
            out.append(d)
    return out


def _parse_answer(raw: str) -> tuple[dict, str]:
    """Parse a completion into {entities, relationships} with repair.

    Returns (parsed, status) where status is "ok" (strict JSON parse) or
    "repaired:N" (recovered N of {entities,relationships} arrays by salvage).
    Never raises; on total failure returns empty lists with status "failed".
    """
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return {"entities": obj.get("entities", []),
                    "relationships": obj.get("relationships", [])}, "ok"
        except Exception:  # noqa: BLE001
            pass
    entities = _fix_array(_array_span(raw, "entities", stop_key="rel"), "title")
    # the model is unstable about the relationship key name: it has emitted
    # "relationships", the misspelt "relationship", and "relations". Try the
    # candidates in order (the standard one first), bounded by the entities
    # array on the left.
    relationships = []
    for key in ("relationships", "relations", "relationship"):
        span = _array_span(raw, key)
        if span:
            relationships = _fix_array(span, "source")
            if relationships:
                break
    if not relationships:
        # degenerate emission seen in the wild:
        # {"relationship": ["ENGLAND", "RUDDER"], "strength": 9} — a two-element
        # array standing in for source/target. Convert to the canonical form.
        rels = []
        for m in re.finditer(
            r'\{\s*"relationship"\s*:\s*\[("[^"]*")\s*,\s*("[^"]*")\]\s*(?:,\s*"strength"\s*:\s*([0-9.]+))?\s*\}',
            raw,
        ):
            src, tgt, strength = m.groups()
            d = {"source": json.loads(src), "target": json.loads(tgt)}
            if strength:
                d["strength"] = float(strength)
            rels.append(d)
        relationships = rels
    if entities or relationships:
        return {"entities": entities, "relationships": relationships}, "repaired"
    return {"entities": [], "relationships": []}, "failed"


def _truncate_at_json_close(raw: str) -> str:
    """Cut `raw` right after the top-level JSON object closes.

    Scans char-by-char tracking bracket depth, string state and escapes. The
    forced schema prefix `{"entities": [` is always the start of `raw`, so a
    return to depth 0 is the balanced end of the whole object; everything after
    it (runaway plain-mode generation) is dropped. If the object never closes,
    returns `raw` unchanged (the tolerant parser still salvages it).
    """
    depth = 0
    in_str = False
    esc = False
    for i, c in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth <= 0:
                return raw[: i + 1]
    return raw


def install() -> None:
    """Neutralise torchao only; no transformers/peft downgrade.

    torchao 0.10.0 preinstalled in the Kaggle image makes peft's
    `PeftModel.from_pretrained` hard-fail before reading the adapter.
    Uninstalling it is enough (inference does not use torchao).

    Earlier versions downgraded to the v4-validated matrix (transformers
    4.57.x / peft 0.18.1) but the Kaggle pip backend repeatedly hung mid-
    install on those runs. Inference does not need the training matrix, so we
    now try the preinstalled transformers 5.0 / peft 0.19.1 directly and only
    uninstall torchao (with a hard timeout so a hung backend fails fast).
    """
    import importlib.metadata as md
    import subprocess

    log("=== install stack ===")

    def ver(pkg: str) -> str:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "MISSING"

    for p in ("torchao", "transformers", "peft"):
        log(f"pre {p}: {ver(p)}")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
            timeout=60, check=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log("uninstalled torchao")
    except Exception as e:  # noqa: BLE001
        log(f"torchao uninstall failed (continuing): {e}")
    # No transformers/peft downgrade: the Kaggle pip backend has repeatedly
    # hung mid-install, and inference does not actually need the training
    # matrix — try the preinstalled transformers 5.0 / peft 0.19.1 directly.
    for p in ("torchao", "transformers", "peft"):
        log(f"post {p}: {ver(p)}")


def _infer_worker(rank: int, nprocs: int, passages: list[dict]) -> list[dict]:
    """Run batch inference for `passages` on GPU `rank`. Top-level for spawn.

    Each worker process pins its own CUDA_VISIBLE_DEVICES so two Kaggle T4s are
    used (each loads its own fp16 copy of model+adapter and decodes a half of
    the passages). Writes raw_<id>.txt into WORK as it goes; returns the parsed
    rows so the parent can merge them into predictions.jsonl.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import StoppingCriteria
    from peft import PeftModel

    if EARLY_STOP:
        class _JsonClose(StoppingCriteria):
            """Stop each row as soon as its JSON object closes.

            Generation starts inside the forced schema prefix `{"entities": [`
            (depth 2). Each call decodes only the newly generated token per row
            and updates bracket/string/escape state; a row stops when depth
            returns to 0 (its object is balanced). Returns a per-batch bool
            tensor so finished rows stop decoding while others continue.
            """

            def __init__(self, tokenizer, n):
                super().__init__()
                self.tok = tokenizer
                self.depth = [2] * n
                self.in_str = [False] * n
                self.esc = [False] * n
                self.closed = [False] * n
                self.prev = None

            def __call__(self, input_ids, scores):
                cur = input_ids.shape[1]
                start = cur - 1 if self.prev is None else self.prev
                self.prev = cur
                for i in range(input_ids.shape[0]):
                    if self.closed[i]:
                        continue
                    text = self.tok.decode(
                        input_ids[i, start:cur].tolist(), skip_special_tokens=True)
                    d, s, e = self.depth[i], self.in_str[i], self.esc[i]
                    for c in text:
                        if s:
                            if e:
                                e = False
                            elif c == "\\":
                                e = True
                            elif c == '"':
                                s = False
                            continue
                        if c == '"':
                            s = True
                        elif c in "[{":
                            d += 1
                        elif c in "]}":
                            d -= 1
                            if d <= 0:
                                self.closed[i] = True
                                d = 0
                                break
                    self.depth[i], self.in_str[i], self.esc[i] = d, s, e
                return torch.tensor(self.closed, dtype=torch.bool,
                                    device=input_ids.device)

    adapter_dir = resolve_adapter_dir()
    log(f"[gpu{rank}] worker on {torch.cuda.get_device_name(0)}; "
        f"{len(passages)} passages; adapter={adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    log(f"[gpu{rank}] tokenizer loaded")
    # FP16 load (bf16 also works — decode is GEMV-bound so dtype does not matter
    # at bs=1 — but fp16 is the training-validated path).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True,
    )
    log(f"[gpu{rank}] base model loaded")
    # Adapter weights are fp32 (Unsloth saves) while base is fp16; torch_dtype
    # casts them on merge. PeftModel.load accepts unknown peft-0.20 config keys.
    model = PeftModel.from_pretrained(
        model, adapter_dir, torch_dtype=torch.float16
    ).merge_and_unload()
    model.eval()
    log(f"[gpu{rank}] adapter merged, model ready")

    prompt = INPUTS["student_prompt"]
    rows: list[dict] = []
    # Batch decode: bs=1 decode is GEMV-bound (~20 tok/s on T4; cublas gemv2T/
    # gemvx kernels, memory-bandwidth limited, fp16 vs bf16 makes no diff).
    # Batching N sequences turns weight application into a GEMM: measured
    # 22 -> 85 -> 164 aggregate tok/s at bs 1/4/8. Passages differ in length,
    # so sort by length and pack consecutive ones to keep padding waste small.
    sorted_ps = sorted(enumerate(passages), key=lambda ep: len(ep[1]["text"]))
    schema_prefix = '{"entities": ['
    prefix_ids = tokenizer.encode(schema_prefix, add_special_tokens=False)
    batch_size = int(os.environ.get("BATCH_SIZE", str(BATCH_SIZE)))

    def run_batch(batch: list[tuple[int, dict]]) -> None:
        texts = []
        for _, p in batch:
            content = f"{prompt}\n{p['text']}"
            if PROTOCOL == "plain":
                # training-faithful: no chat markers, blank line before output
                texts.append(content + "\n\n")
            else:
                texts.append(tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False))
        enc = tokenizer(texts, return_tensors="pt", padding=True)
        prefix_t = torch.tensor([prefix_ids] * len(batch), dtype=torch.long,
                                device="cuda")
        gen_input = torch.cat([enc["input_ids"].to("cuda"), prefix_t], dim=1)
        attn = torch.cat([enc["attention_mask"].to("cuda"),
                          torch.ones(len(batch), prefix_t.shape[1],
                                     dtype=torch.long, device="cuda")], dim=1)
        # True (unpadded) prompt length per row — the generated tail starts at
        # mask.sum(). Slice from prompt_end = mask.sum() - prefix_len so the
        # schema prefix stays in the decoded raw (parser expects it).
        prompt_ends = (attn.sum(dim=1) - len(prefix_ids)).tolist()
        kwargs = {}
        if EARLY_STOP:
            kwargs["stopping_criteria"] = [_JsonClose(tokenizer, len(batch))]
        with torch.no_grad():
            gen = model.generate(
                input_ids=gen_input, attention_mask=attn,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=0.7, top_p=0.95, repetition_penalty=1.15,
                **kwargs,
            )
        for bi, (_orig_i, p) in enumerate(batch):
            out_ids = gen[bi][int(prompt_ends[bi]):]
            raw = tokenizer.decode(out_ids, skip_special_tokens=True)
            if EARLY_STOP:
                raw = _truncate_at_json_close(raw)
            with open(os.path.join(WORK, f"raw_{p['id']}.txt"), "w",
                      encoding="utf-8") as f:
                f.write(raw)
            try:
                parsed, status = _parse_answer(raw)
            except Exception as e:  # noqa: BLE001
                status = f"except:{e}"
                parsed = {"entities": [], "relationships": []}
            if status != "ok":
                log(f"[gpu{rank}][{p['id']}]: PARSE {status} len={len(raw)}")
            rows.append({"id": p["id"],
                         "entities": parsed.get("entities", []),
                         "relationships": parsed.get("relationships", [])})
            log(f"[gpu{rank}][{p['id']}]: entities={len(rows[-1]['entities'])} "
                f"rels={len(rows[-1]['relationships'])} raw_len={len(raw)}")

    for start in range(0, len(sorted_ps), batch_size):
        run_batch(sorted_ps[start:start + batch_size])
    log(f"[gpu{rank}] worker done, {len(rows)} rows")
    return rows


def _load_passages() -> list[dict]:
    """Load the 200-passage sample from the mounted dataset.

    Sources, in priority order: PASSAGES_FILE env (explicit override), then the
    cap-upload / 200-passages datasets under any of the path layouts Kaggle has
    used (slug-root vs /datasets/<owner>/<slug>). Each line must be
    {"id": ..., "text": ...}. Hard-fails (no silent embedded fallback) because a
    wrong passage set would invalidate the eval.
    """
    cands = [os.environ.get("PASSAGES_FILE")] + _candidates(
        "cap-upload/sample_200_passages.jsonl") + _candidates(
        "200-passages/sample_200_passages.jsonl")
    for pf in cands:
        if pf and os.path.exists(pf):
            log(f"reading passages from {pf}")
            out = []
            with open(pf, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        out.append({"id": r["id"], "text": r["text"]})
            return out
    raise SystemExit(
        "no sample_200_passages.jsonl under /kaggle/input; tree:\n"
        + "\n".join(_tree()))


def infer(limit: int | None = None) -> None:
    passages = _load_passages()
    if limit:
        passages = passages[:limit]
    log(f"=== inference on {len(passages)} eval passages (max_new_tokens=8192) ===")
    import torch

    n_gpu = torch.cuda.device_count()
    log(f"CUDA devices available: {n_gpu}")
    # Spawn one worker per GPU (each loads its own model copy + decode half the
    # passages -> ~2x wall throughput on two T4s). Single-GPU falls back to one
    # worker (rank 0).
    nprocs = n_gpu if n_gpu >= 2 else 1
    if nprocs == 1:
        rows = _infer_worker(0, 1, passages)
    else:
        from torch import multiprocessing as mp
        chunks = [passages[i::nprocs] for i in range(nprocs)]
        ctx = mp.get_context("spawn")
        procs = []
        results: dict[int, list[dict]] = {}
        q = ctx.Queue()
        for rank in range(nprocs):
            p = ctx.Process(target=_worker_send, args=(rank, chunks[rank], q, nprocs))
            p.start()
            procs.append(p)
        for _ in range(nprocs):
            rank, part = q.get()
            results[rank] = part
        for p in procs:
            p.join(timeout=3600)
        rows = []
        for rank in range(nprocs):
            rows.extend(results.get(rank, []))

    pred_path = os.path.join(WORK, "predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"wrote {pred_path} ({len(rows)} rows)")
    with open(os.path.join(WORK, "summary.csv"), "w", encoding="utf-8") as f:
        f.write("id,entities,relationships\n")
        for r in rows:
            f.write(f"{r['id']},{len(r['entities'])},{len(r['relationships'])}\n")


def _worker_send(rank: int, passages: list[dict], q, nprocs: int) -> None:
    """spawn target: run worker and ship its rows back on the queue."""
    try:
        q.put((rank, _infer_worker(rank, nprocs, passages)))
    except Exception as e:  # noqa: BLE001
        log(f"[gpu{rank}] worker failed: {e}")
        q.put((rank, []))


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="infer only the first N passages (smoke check); "
                             "defaults to baked LIMIT then env TRACE_LIMIT")
    args = parser.parse_args(argv)
    check_env()
    install()
    n = args.limit if args.limit is not None else LIMIT
    if n is None and os.environ.get("TRACE_LIMIT"):
        n = int(os.environ["TRACE_LIMIT"])
    log(f"=== inference {'(limit=%d)' % n if n else '(all passages)'} ===")
    infer(limit=n)
    log("=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
