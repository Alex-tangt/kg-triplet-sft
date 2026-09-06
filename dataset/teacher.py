"""Stage 4 — teacher labeling tracer (pilot).

Runs the DashScope teacher (qwen3-flash family) over a stratified sample of the
corpus to prove the labeling machinery before the full run pays for it: prompt,
parallel client with retry/backoff, per-run budget hard stop, single-call
context < 32k, schema validation, and semantic (weight-anchor + grounding)
checks. Incremental by design — ``--limit 1`` then ``10`` then ``100`` reuse the
same seeded sample order, and an interrupted run resumes without re-calling
already-written passages.

Writes ``<data-dir>/03_labels_pilot.jsonl``; the human-review report is built
from it by ``dataset/report.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg_contract import extract_json_array, render_teacher_prompt, validate_triplet_list  # noqa: E402
from kg_contract.prompts import TEACHER_SECTIONS_PROSE  # noqa: E402

from dataset.io import read_jsonl  # noqa: E402
from dataset.semantic import check_semantics, violation_counts  # noqa: E402

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-flash"
MAX_CONTEXT_TOKENS = 32_000
CHAR_TOKENS = 4  # conservative English estimate: ~4 chars per token

WEIGHT_RULE = "weight"
GROUNDING_RULE = "grounding"


class RetryableError(Exception):
    """Transient failure: network, timeout, HTTP 5xx or 429. Safe to retry."""


class NonRetryableError(Exception):
    """Permanent failure: HTTP 4xx (except 429) or a guard violation."""


class BudgetExhausted(Exception):
    """Per-run call budget consumed; labeling must stop."""


@dataclass
class Budget:
    """Thread-safe per-run call counter. Every transport attempt consumes one."""

    cap: int
    _used: int = field(default=0)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def try_consume(self) -> bool:
        with self._lock:
            if self._used >= self.cap:
                return False
            self._used += 1
            return True

    @property
    def used(self) -> int:
        with self._lock:
            return self._used


def build_request_body(model, messages, temperature, max_tokens, enable_thinking=True) -> bytes:
    """The OpenAI-compatible request payload.

    ``enable_thinking`` defaults to on — qwen3's reasoning mode is the faithful
    behaviour for the teacher task and measurably improves rubric adherence; it
    costs latency, which parallelism absorbs. qwen3-flash accepted the field in a
    live probe; on a model that rejects unknown fields this is the flag to flip.
    """
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
    }
    return json.dumps(body).encode()


LAST_USAGE: dict | None = None


def http_transport(*, model, base_url, api_key, temperature, max_tokens, timeout, enable_thinking=True):
    """Real DashScope (OpenAI-compatible) transport. 5xx/429/network raise
    :class:`RetryableError`; other 4xx raise :class:`NonRetryableError`.

    Each successful call also records the response ``usage`` dict on the module
    global ``LAST_USAGE`` (consumers read-and-reset it after each call)."""
    def call(messages) -> str:
        body = build_request_body(model, messages, temperature, max_tokens, enable_thinking)
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                raise RetryableError(f"HTTP {e.code}: {e.reason}") from e
            raise NonRetryableError(f"HTTP {e.code}: {e.reason}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise RetryableError(str(e)) from e
        global LAST_USAGE
        LAST_USAGE = data.get("usage")
        return data["choices"][0]["message"]["content"]

    return call


def estimate_tokens(*texts) -> int:
    return sum(len(t) for t in texts) // CHAR_TOKENS


def _default_extractor(raw: str) -> tuple[list | None, list[str]]:
    """Standard extractor: tolerant JSON-array parse + full contract validation.

    Returns (triplets, []) on success or (None, error messages) otherwise.
    Probe modes (open relation set, W1 two-stage output) inject their own.
    """
    triplets = extract_json_array(raw)
    if triplets is None:
        return None, ["output is not a JSON array"]
    valid, errors = validate_triplet_list(triplets)
    if not valid:
        return None, ["; ".join(f"{e.path}: {e.reason}" for e in errors)]
    return triplets, []


class TeacherClient:
    """Labels one passage: retry/backoff + schema re-request, budget and context guards."""

    def __init__(
        self,
        transport,
        *,
        system_prompt: str,
        extractor=_default_extractor,
        network_retries: int = 3,
        schema_retries: int = 2,
        backoff_base: float = 1.0,
        max_tokens: int = 2000,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
        budget: Budget | None = None,
    ):
        self.transport = transport
        self.system_prompt = system_prompt
        self.extractor = extractor
        self.network_retries = network_retries
        self.schema_retries = schema_retries
        self.backoff_base = backoff_base
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens
        self.budget = budget

    def _guard_context(self, messages) -> None:
        chars = sum(len(m["content"]) for m in messages)
        if estimate_tokens(*[m["content"] for m in messages]) + self.max_tokens > self.max_context_tokens:
            raise NonRetryableError(
                f"estimated context exceeds {self.max_context_tokens} tokens ({chars} chars)"
            )

    def _call(self, messages) -> str:
        self._guard_context(messages)
        for attempt in range(self.network_retries + 1):
            if self.budget is not None and not self.budget.try_consume():
                raise BudgetExhausted()
            try:
                return self.transport(messages)
            except RetryableError as e:
                if attempt >= self.network_retries:
                    raise
                time.sleep(self.backoff_base * (2 ** attempt))
        raise AssertionError("unreachable")

    def label(self, text: str) -> dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        attempts = 0
        last_raw = None
        schema_errors: list[str] = []
        for _ in range(self.schema_retries + 1):
            try:
                last_raw = self._call(messages)
            except (RetryableError, NonRetryableError) as e:
                return {
                    "status": "failed",
                    "failure_reason": "network" if isinstance(e, RetryableError) else "http",
                    "attempts": attempts or 1,
                    "teacher_raw": None,
                    "triplets": [],
                    "hard_negative": False,
                    "schema_errors": [str(e)],
                    "semantic_violations": [],
                }
            attempts += 1
            triplets, extract_errors = self.extractor(last_raw)
            if triplets is None:
                schema_errors = [f"attempt {attempts}: " + "; ".join(extract_errors)]
                continue
            return {
                "status": "ok",
                "failure_reason": None,
                "attempts": attempts,
                "teacher_raw": last_raw,
                "triplets": triplets,
                "hard_negative": len(triplets) == 0,
                "schema_errors": None,
                "semantic_violations": [v.__dict__ for v in check_semantics(triplets, text)],
            }
        return {
            "status": "failed",
            "failure_reason": "schema",
            "attempts": attempts,
            "teacher_raw": last_raw,
            "triplets": [],
            "hard_negative": False,
            "schema_errors": schema_errors,
            "semantic_violations": [],
        }


def length_strata(pool: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Draw ``k`` passages spread across three word-count terciles."""
    counts = sorted(r["word_count"] for r in pool)
    lo = counts[len(counts) // 3]
    hi = counts[2 * len(counts) // 3]
    buckets: dict[str, list[dict]] = {"lo": [], "mid": [], "hi": []}
    for r in pool:
        bucket = "lo" if r["word_count"] <= lo else "mid" if r["word_count"] <= hi else "hi"
        buckets[bucket].append(r)
    drawn: list[dict] = []
    per = k // 3
    remainder = k % 3
    for i, name in enumerate(("lo", "mid", "hi")):
        want = per + (1 if i < remainder else 0)
        rng.shuffle(buckets[name])
        drawn.extend(buckets[name][:want])
    return drawn


def sample_passages(rows: list[dict], n: int = 100, seed: int = 0) -> list[dict]:
    """Stratified sample: 60/40 source split, length terciles within each, seeded.

    Returns a deterministically shuffled list; the first ``limit`` entries are
    the incremental (1 → 10 → 100) batches.
    """
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    selected: list[dict] = []
    quotas = (("wikipedia", round(n * 0.6)), ("arxiv", n - round(n * 0.6)))
    for source, k in quotas:
        pool = by_source.get(source, [])
        if pool:
            selected.extend(length_strata(pool, min(k, len(pool)), rng))
    rng.shuffle(selected)
    return selected


def label_tasks(
    client: TeacherClient,
    tasks: list[dict],
    workers: int,
    out: Path | None = None,
    progress=None,
) -> tuple[list[dict], bool]:
    """Label passages with ``workers`` parallel threads.

    Returns (completed records in task order, stopped_on_budget). Each record is
    appended to ``out`` (if given) as soon as it finishes, so a killed or
    budget-stopped run persists partial progress and resumes without re-calling
    completed passages. ``progress`` (if given) is invoked per completed record.
    """
    records: dict[str, dict] = {}
    task_ids = [r["id"] for r in tasks]
    by_id = {r["id"]: r for r in tasks}
    index = threading.Lock()
    write_lock = threading.Lock()
    cursor = 0
    stop = threading.Event()

    def finish(rec: dict) -> None:
        if out is not None:
            with write_lock:
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if progress is not None:
            progress(rec)

    def worker() -> None:
        nonlocal cursor
        while not stop.is_set():
            with index:
                if cursor >= len(task_ids):
                    return
                pid = task_ids[cursor]
                cursor += 1
            passage = by_id[pid]
            try:
                rec = client.label(passage["text"])
            except BudgetExhausted:
                stop.set()
                return
            rec = dict(rec)
            rec.update(
                {
                    "id": pid,
                    "source": passage["source"],
                    "title": passage["title"],
                    "word_count": passage["word_count"],
                    "text": passage["text"],
                }
            )
            with index:
                records[pid] = rec
            finish(rec)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker) for _ in range(workers)]
        for f in futures:
            f.result()

    return [records[r["id"]] for r in tasks if r["id"] in records], stop.is_set()


def prepare_resume(out: Path, retry_failed: bool) -> tuple[list[dict], set[str]]:
    """Return (existing rows, ids already labeled).

    With ``retry_failed`` the failed rows are removed from the output file so a
    prompt iteration can re-label them; the file keeps only ``ok`` records.
    """
    existing = read_jsonl(out) if out.exists() else []
    if retry_failed:
        keep = [r for r in existing if r["status"] == "ok"]
        if len(keep) != len(existing):
            with open(out, "w", encoding="utf-8") as f:
                for r in keep:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        existing = keep
    return existing, {r["id"] for r in existing}


def run(args) -> dict:
    data_dir = Path(args.data_dir)
    corpus = read_jsonl(Path(args.corpus) if args.corpus else data_dir / "corpus.jsonl")
    out = Path(args.out) if args.out else data_dir / "03_labels_pilot.jsonl"

    sampled = sample_passages(corpus, n=args.sample_size, seed=args.seed)[: args.limit]
    existing, existing_ids = prepare_resume(out, args.retry_failed)
    tasks = [r for r in sampled if r["id"] not in existing_ids]
    skipped_existing = len(sampled) - len(tasks)

    if args.dry_run:
        prompt = render_teacher_prompt(TEACHER_SECTIONS_PROSE)
        return {
            "dry_run": True,
            "prompt": prompt,
            "prompt_tokens_est": estimate_tokens(prompt),
            "sample": [{"id": r["id"], "source": r["source"], "word_count": r["word_count"]} for r in sampled],
            "tasks_new": len(tasks),
            "already_labeled": skipped_existing,
        }

    system_prompt = render_teacher_prompt(TEACHER_SECTIONS_PROSE)
    budget = Budget(args.budget)
    if args.api_key is None:
        raise SystemExit("DASHSCOPE_API_KEY is required (set it in .env)")
    transport = http_transport(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        enable_thinking=not args.no_thinking,
    )
    client = TeacherClient(transport, system_prompt=system_prompt, budget=budget)

    started = time.time()
    done = {"n": 0}

    def progress(rec: dict) -> None:
        done["n"] += 1
        status = rec["status"]
        extra = f"{len(rec['triplets'])} triplets" if status == "ok" else f"failed: {rec['failure_reason']}"
        print(
            f"[{done['n']}/{len(tasks)}] {rec['id']} {status} {extra}  "
            f"budget {budget.used}/{args.budget}  elapsed {time.time() - started:.0f}s",
            flush=True,
        )

    if tasks:
        print(
            f"labeling {len(tasks)} passages (workers={args.workers}, budget={args.budget}, "
            f"model={args.model}, thinking={'off' if args.no_thinking else 'on'})",
            flush=True,
        )
    rows, stopped_on_budget = label_tasks(client, tasks, args.workers, out=out, progress=progress)
    out.parent.mkdir(parents=True, exist_ok=True)

    return {
        "pilot": {
            "drawn": len(sampled),
            "already_labeled": skipped_existing,
            "labeled_new": len(rows),
            "budget_used": budget.used,
            "budget_cap": args.budget,
            "stopped_on_budget": stopped_on_budget,
            "status_counts": _count(rows, lambda r: r["status"]),
            "failure_reasons": _count(rows, lambda r: r["failure_reason"]),
            "hard_negatives": sum(1 for r in rows if r["hard_negative"]),
            "hard_negative_rate": round(sum(1 for r in rows if r["hard_negative"]) / len(rows), 3) if rows else 0,
            "violations": violation_counts([v for r in rows for v in r["semantic_violations"]]),
        }
    }


def _count(rows: list[dict], key) -> dict:
    counts: dict = {}
    for r in rows:
        counts[key(r)] = counts.get(key(r), 0) + 1
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--corpus", default=None, help="path to corpus.jsonl (default <data-dir>/corpus.jsonl)")
    parser.add_argument("--out", default=None, help="output jsonl (default <data-dir>/03_labels_pilot.jsonl)")
    parser.add_argument("--limit", type=int, default=100, help="max new passages to label this run")
    parser.add_argument("--sample-size", type=int, default=100, help="size of the stratified draw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--budget", type=int, default=150, help="total calls (incl. retries) cap per run")
    parser.add_argument("--dry-run", action="store_true", help="print prompt + sample order, make no calls")
    parser.add_argument("--no-thinking", action="store_true", help="disable qwen reasoning mode (faster, weaker rubric adherence)")
    parser.add_argument("--retry-failed", action="store_true", help="drop failed records from the output and re-label them (prompt iteration)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None, help="defaults to DASHSCOPE_API_KEY from .env")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096, help="output budget; room for reasoning trace + answer")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass
    if args.api_key is None:
        args.api_key = os.environ.get("DASHSCOPE_API_KEY")

    print(json.dumps(run(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
