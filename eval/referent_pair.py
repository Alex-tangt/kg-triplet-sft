"""Referent entity pairing between the teacher gold and a student's extraction.

Two modes:
  --mode title : offline greedy 1-1 pairing by normalized-title containment
                 (the deterministic baseline; zero API cost).
  --mode llm   : per-passage DashScope (qwen3.7-flash) call that reads the two
                 numbered entity lists and emits the matching pairs as JSON.

Either way the output is one JSONL row per passage:
  {"id": ..., "pairs": [[teacher_idx, student_idx], ...], "status": "ok"}
Pair indices are into the *original* entity lists of the input files. A student
entity may appear in several pairs (the student merged several gold referents);
a teacher entity appears at most once per pair list.

Usage:
  python eval/referent_pair.py --mode title
  python eval/referent_pair.py --mode llm --limit 5        # probe
  python eval/referent_pair.py --mode llm                  # full 699 (resumable)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.io import read_jsonl  # noqa: E402
from dataset import teacher as teacher_api  # noqa: E402
from dataset.teacher import http_transport  # noqa: E402

from eval.referent_common import (  # noqa: E402
    DEFAULT_STUDENT,
    DEFAULT_TEACHER,
    OUT_DIR,
    normalize,
    write_jsonl,
)

MODEL = "qwen3.7-flash"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
IDENTITY_RUBRIC = (
    "ENTITY IDENTITY: two records refer to the SAME real-world object/individual/concept within "
    "this passage only if — allowing surface differences (spelling mistakes, case, plural/singular, "
    "abbreviations, word order, added or dropped qualifiers) — the titles/descriptions denote ONE "
    "single referent that the passage talks about, and both records describe that same referent.\n"
    "NOT SAME when either record denotes a genuinely different referent, including: two distinct "
    "objects/individuals/concepts even in the same topic; a generic category vs its specific "
    "instance/hyponym; a property/quantity/unit/part/effect vs the object that carries it or its "
    "whole; two different members of one class; or when the two merely appear together or share "
    "vocabulary. If the evidence is ambiguous, answer NOT SAME."
)

SYSTEM_PROMPT = (
    "You resolve entities between two extractions of the same passage (list A = reference "
    "\"teacher\", list B = another model's extraction). For each list-A entity output which "
    "list-B entity (index) denotes the SAME referent.\n"
    "RULES:\n"
    "- A list-A entity stays UNMATCHED unless the identity rule below firmly identifies a "
    "list-B counterpart; never force a match.\n"
    "- A list-B entity may cover several list-A entities ONLY when it is an explicit combined "
    "node (its title/description merge several distinct list-A referents); otherwise keep 1-to-1.\n"
    "- The passage is evidence for DISTINGUISHING referents, never a reason to merge.\n"
    + IDENTITY_RUBRIC + "\n"
    "Reply with ONLY a JSON object, no prose:\n"
    '{"pairs": [[a, b], ...]}   (a = list-A index, b = list-B index; empty array allowed)'
)

# Cost-reduction experiment variants (ADR-0007). "hard" adds over-merge hardening
# + worked exemplars; "strict" pins every ambiguous case to NOT SAME. The two votes
# of the agree-only scheme are meant to differ in exactly this way.
_EXAMPLES = (
    "WORKED EXAMPLES (study these; they define 'same referent'):\n"
    "1. teacher [0] PERSON 'Gene Tierney' - American actress, star of the 1944 film Laura.\n"
    "   student [4] PERSON 'Geren Tierney' - actress, played in the movie Laura.\n"
    "   -> same real-world person despite the misspelled title: pairs [[0, 4]].\n"
    "2. teacher [2] ORGANIZATION 'Ferrari' - Italian luxury sports car manufacturer.\n"
    "   student [7] PERSON 'Enzo Ferrari' - the founder of the car maker Ferrari.\n"
    "   -> the person and the company merely co-occur; NOT SAME: pairs [].\n"
    "3. teacher [1] CONCEPT 'Deep learning' - a branch of machine learning using many layers.\n"
    "   student [3] CONCEPT 'Convolutional neural networks' - a specific model family within it.\n"
    "   -> a general category and its instance are DIFFERENT referents: pairs [].\n"
    "4. teacher [5] ORGANIZATION 'International Business Machines Corporation'.\n"
    "   student [9] ORGANIZATION 'IBM'.\n"
    "   -> abbreviation/word-order differences, same referent: pairs [[5, 9]].\n"
)
_HARD_EXTRA = (
    "HARDENING:\n"
    "- When two candidate titles differ but the DESCRIPTIONS point at the same individual/"
    "object in the passage, they are the same referent (typos, case, plural, abbreviations, "
    "word order). Prefer SAME here.\n"
    "- But NEVER merge because entities share a topic, appear together, share one vocabulary "
    "word, or are a type/specimen of another. A person is not their company, product, work, "
    "role, or family name; a system is not its component or behavior.\n"
    "- Different members of the same class (two people, two laws, two years) are never the "
    "same referent.\n"
)
_STRICT_PIN = (
    "FINAL RULE: if you cannot name, from the passage text, concrete evidence that the two "
    "records are the SAME real-world referent, output NOT SAME (omit the pair). An empty "
    "array is a valid and preferred answer when in doubt.\n"
)


def system_prompt(variant: str) -> str:
    base = SYSTEM_PROMPT
    if variant == "hard":
        return SYSTEM_PROMPT + "\n" + _HARD_EXTRA + "\n" + _EXAMPLES
    if variant == "strict":
        return SYSTEM_PROMPT + "\n" + _STRICT_PIN
    return base


def _api_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(".").resolve() / ".env")
    except ImportError:
        pass
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("DASHSCOPE_API_KEY not set (see .env.example)")
    return key


def _re_sub(s: str) -> str:
    import re

    return re.sub(r"\s+", " ", s)


def exact_pairs(teacher_entities: list[dict], student_entities: list[dict]) -> tuple[list[list[int]], set[int], set[int]]:
    """Tier 1: 1-1 pairs on normalized-title equality. Returns (pairs, unused teacher idx, unused student idx)."""
    tn = {normalize(e.get("title")): i for i, e in enumerate(teacher_entities) if normalize(e.get("title"))}
    sn = {normalize(e.get("title")): i for i, e in enumerate(student_entities) if normalize(e.get("title"))}
    pairs: list[list[int]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for t, ti in tn.items():
        si = sn.get(t)
        if si is not None:
            pairs.append([ti, si])
            used_a.add(ti)
            used_b.add(si)
    all_a = set(range(len(teacher_entities)))
    all_b = set(range(len(student_entities)))
    return pairs, all_a - used_a, all_b - used_b


def _entry_line(i: int, e: dict, desc_len: int) -> str:
    title = (e.get("title") or "").strip()
    etype = (e.get("type") or "").strip()
    desc = _re_sub((e.get("description") or "").strip())
    return f"[{i}] {etype} '{title}' — {desc[:desc_len]}"


def build_subset_content(passage: str | None, subset_a: list[int], teacher: dict, subset_b: list[int], student: dict, desc_len: int = 160) -> str:
    la = "\n".join(_entry_line(i, teacher["entities"][i], desc_len) for i in sorted(subset_a))
    lb = "\n".join(_entry_line(i, student["entities"][i], desc_len) for i in sorted(subset_b))
    head = f"PASSAGE:\n{passage[:1200]}\n\n" if passage else ""
    return (
        head
        + "These are the STILL-UNMATCHED entities (indices refer to the original lists).\n"
        + "LIST A (teacher, unmatched):\n" + (la or "(empty)")
        + "\n\nLIST B (student, unmatched):\n" + (lb or "(empty)")
        + "\n\nReturn the JSON object of matching pairs."
    )


def _parse_pairs(raw: str, n_a: int, n_b: int) -> list[list[int]] | None:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    pairs = obj.get("pairs")
    if not isinstance(pairs, list):
        return None
    out: list[list[int]] = []
    for p in pairs:
        if not isinstance(p, list) or len(p) != 2:
            return None
        a, b = p
        if a is None or b is None:
            continue  # the model marks "no counterpart" as null; skip, not an error
        if not isinstance(a, int) or not isinstance(b, int):
            return None
        if not (0 <= a < n_a and 0 <= b < n_b):
            continue  # hallucinated out-of-range index: drop the pair, keep the rest
        out.append([a, b])
    return out


def title_pair_entities(teacher_entities: list[dict], student_entities: list[dict]) -> list[list[int]]:
    """Greedy 1-1 pairing: exact normalized title wins, then containment."""
    ta = {normalize(e.get("title")): i for i, e in enumerate(teacher_entities) if normalize(e.get("title"))}
    sb = {normalize(e.get("title")): i for i, e in enumerate(student_entities) if normalize(e.get("title"))}
    cand = []
    for t, ti in ta.items():
        for s, si in sb.items():
            if t == s:
                cand.append((2, ti, si))
            elif t in s or s in t:
                ratio = min(len(t), len(s)) / max(len(t), len(s))
                cand.append((1 + ratio, ti, si))
    cand.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[list[int]] = []
    for _, ti, si in cand:
        if ti in used_a or si in used_b:
            continue
        used_a.add(ti)
        used_b.add(si)
        pairs.append([ti, si])
    return pairs


def run_title(teacher_path: Path, student_path: Path, out: Path, limit: int | None, only_ids: list[str] | None) -> None:
    teacher = {r["id"]: r for r in read_jsonl(teacher_path)}
    student = {r["id"]: r for r in read_jsonl(student_path)}
    ids = sorted(teacher)
    if only_ids:
        ids = [i for i in ids if i in set(only_ids)]
    if limit is not None:
        ids = ids[:limit]
    rows = []
    for pid in ids:
        st = student.get(pid)
        if not st:
            rows.append({"id": pid, "status": "failed", "reason": "no student row"})
            continue
        pairs = title_pair_entities(teacher[pid]["entities"], st.get("entities", []))
        rows.append({"id": pid, "status": "ok", "pairs": pairs})
    write_jsonl(out, rows)
    print(f"title pairing: {len(rows)} passages -> {out}")


def run_llm(teacher_path: Path, student_path: Path, out: Path, workers: int,
            limit: int | None, only_ids: list[str] | None, model: str, thinking: bool,
            no_passage: bool, desc_len: int, variant: str = "default") -> None:
    teacher = {r["id"]: r for r in read_jsonl(teacher_path)}
    student = {r["id"]: r for r in read_jsonl(student_path)}
    ids = sorted(teacher)
    if only_ids:
        ids = [i for i in ids if i in set(only_ids)]
    if limit is not None:
        ids = ids[:limit]

    existing = {r["id"]: r for r in read_jsonl(out)} if out.exists() else {}
    remaining = [i for i in ids if i not in existing or existing[i].get("status") != "ok"]
    print(f"[llm pairing] {len(ids)} requested, {len(remaining)} to run ({len(ids) - len(remaining)} done)")

    if not remaining:
        return
    transport = http_transport(
        model=model, base_url=BASE_URL, api_key=_api_key(),
        temperature=0.0, max_tokens=2000, timeout=120,
        enable_thinking=thinking,  # thinking off: ~5-10x faster; pair resolution is a judgement call it handles fine
    )
    lock = threading.Lock()
    done = 0
    start = time.time()
    sys_content = system_prompt(variant)

    def work(pid: str) -> None:
        nonlocal done
        rec = {"id": pid}
        t_entities = teacher[pid]["entities"]
        s_entities = (student.get(pid, {}) or {}).get("entities", [])
        exact, unused_t, unused_s = exact_pairs(t_entities, s_entities)
        base_pairs = list(exact)
        if unused_t and unused_s:
            for attempt in range(4):
                try:
                    passage = None if no_passage else teacher[pid]["text"]
                    raw = transport([
                        {"role": "system", "content": sys_content},
                        {"role": "user", "content": build_subset_content(passage, unused_t, teacher[pid], unused_s, student[pid], desc_len)},
                    ])
                    usage = teacher_api.LAST_USAGE
                    pairs = _parse_pairs(raw, len(t_entities), len(s_entities))
                    if pairs is None:
                        raise ValueError("unparseable pair JSON")
                    # only accept pairs within the unmatched sets (reject duplicates/merge spillover)
                    extra = [p for p in pairs if p[0] in unused_t and p[1] in unused_s]
                    base_pairs = exact + extra
                    rec.update({"status": "ok", "pairs": base_pairs, "attempts": attempt + 1})
                    if usage:
                        rec["tokens"] = {
                            k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                        }
                    break
                except Exception as exc:  # Retryable/NonRetryable + parse errors
                    if attempt == 3:
                        rec.update({"status": "failed", "reason": str(exc)[:300], "attempts": 4})
        else:
            rec.update({"status": "ok", "pairs": base_pairs, "attempts": 0, "note": "exact-only"})
        with lock:
            done += 1
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n = len(rec["pairs"]) if rec["status"] == "ok" else "FAIL"
            print(f"[{done}/{len(remaining)}] {pid} pairs={n} attempts={rec.get('attempts')} "
                  f"elapsed={time.time() - start:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, remaining))
    print(f"llm pairing done: {done} attempts wrote to {out} (elapsed {time.time() - start:.0f}s)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    p.add_argument("--mode", choices=("title", "llm"), required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--thinking", action="store_true", help="enable qwen reasoning (slower; default off)")
    p.add_argument("--no-passage", action="store_true", help="omit the passage from the LLM call (ablation)")
    p.add_argument("--desc-len", type=int, default=160, help="description chars per entity (0 = omit descriptions)")
    p.add_argument("--variant", choices=("default", "hard", "strict"), default="default",
                   help="prompt variant: default | hard (rules+exemplars) | strict (ambiguous => NOT SAME)")
    args = p.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or OUT_DIR / f"pairings_{args.mode}.jsonl"
    if args.mode == "title":
        run_title(args.teacher, args.student, out, args.limit, args.ids)
    else:
        run_llm(args.teacher, args.student, out, args.workers, args.limit, args.ids, args.model, args.thinking,
                args.no_passage, args.desc_len, args.variant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
