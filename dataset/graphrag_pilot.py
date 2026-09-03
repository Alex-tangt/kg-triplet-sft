"""Phase-0 pilot: Microsoft GraphRAG official extraction prompt as the teacher.

Runs the official GRAPH_EXTRACTION_PROMPT (+ continuation loop) via
qwen3-flash on 10 passages, alongside the current 20-set teacher, so a blind
judge can compare the two on faithfulness / completeness (incl. numeric facts)
/ noise. Decides whether data generation should switch to GraphRAG-style output
(open relation descriptions + strength) for the MS GraphRAG consumer.

Usage:
    python dataset/graphrag_pilot.py                 # run both teachers
    python dataset/graphrag_pilot.py --teacher graphrag
    python dataset/graphrag_pilot.py --teacher twenty

Writes outputs/graphrag_pilot/graphrag.jsonl and twenty.jsonl, resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg_contract import render_teacher_prompt  # noqa: E402
from kg_contract.graphrag_prompts import CONTINUE_PROMPT, GRAPH_EXTRACTION_PROMPT, LOOP_PROMPT  # noqa: E402
from kg_contract.prompts import TEACHER_SECTIONS_PROSE  # noqa: E402

from dataset.io import read_jsonl  # noqa: E402
from dataset.graphrag import dedup_graphrag, parse_graphrag_output  # noqa: E402
from dataset.teacher import (  # noqa: E402
    Budget,
    BudgetExhausted,
    RetryableError,
    TeacherClient,
    http_transport,
)

OUT_DIR = Path("outputs/graphrag_pilot")

PASSAGE_IDS = [
    "wikipedia-02909",  # easy gold (person bio)
    "arxiv-00944",      # medium gold (physics)
    "wikipedia-01557",  # hard gold (ecology)
    "wikipedia-02266",  # held-out (corporate history, years)
    "wikipedia-01974",  # astronomy (diameter, distances)
    "wikipedia-01420",  # population stats
    "wikipedia-01252",  # seas list (areas)
    "wikipedia-00083",  # stadiums (dates)
    "wikipedia-01421",  # languages (speaker counts)
    "arxiv-00634",      # physics narrative (mode conversion)
]

ENTITY_TYPES = "PERSON, ORGANIZATION, GEO, EVENT, CONCEPT"
MODEL = "qwen3.7-flash"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_ROUNDS = 2
MAX_TOKENS = 4096


def _load_api_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(".").resolve() / ".env")
    except ImportError:
        pass
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("DASHSCOPE_API_KEY not set (see .env.example)")
    return key


def _call(transport, budget, messages, network_retries=3, backoff_base=1.0) -> str:
    for attempt in range(network_retries + 1):
        if not budget.try_consume():
            raise BudgetExhausted()
        try:
            return transport(messages)
        except RetryableError as e:
            if attempt >= network_retries:
                raise
            time.sleep(backoff_base * (2**attempt))
    raise AssertionError("unreachable")


def run_graphrag(transport, budget, text: str, entity_types: str, max_rounds: int) -> dict:
    """GraphRAG extraction with the completeness loop; returns record dict.

    All rounds' raw records are collected, then deduplicated and grounding-
    filtered once (``dedup_graphrag``) so the continuation loop cannot inflate
    the graph with near-identical variants.
    """
    prompt = GRAPH_EXTRACTION_PROMPT.format(entity_types=entity_types, input_text=text)
    messages = [{"role": "user", "content": prompt}]
    raws: list[str] = []
    calls = 0
    raw_entities: list[dict] = []
    raw_rels: list[dict] = []

    def absorb(raw: str) -> None:
        parsed = parse_graphrag_output(raw)
        if parsed:
            raw_entities.extend(parsed["entities"])
            raw_rels.extend(parsed["relationships"])

    raw = _call(transport, budget, messages)
    calls += 1
    raws.append(raw)
    absorb(raw)

    for _ in range(max_rounds):
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": LOOP_PROMPT})
        answer = _call(transport, budget, messages)
        calls += 1
        raws.append(answer)
        if not answer.strip().upper().startswith("Y"):
            break
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": CONTINUE_PROMPT})
        raw = _call(transport, budget, messages)
        calls += 1
        raws.append(raw)
        absorb(raw)

    raw_ent_n = len(raw_entities)
    raw_rel_n = len(raw_rels)
    out = dedup_graphrag({"entities": raw_entities, "relationships": raw_rels}, text)
    return {
        "calls": calls,
        "raw_rounds": raws,
        "raw_entity_count": raw_ent_n,
        "raw_relationship_count": raw_rel_n,
        "dropped": raw_ent_n - len(out["entities"]),
        **out,
    }


def run_twenty(client: TeacherClient, text: str) -> dict:
    rec = client.label(text)
    return {
        "status": rec["status"],
        "schema_errors": rec["schema_errors"],
        "triplets": rec["triplets"],
        "semantic_violations": rec["semantic_violations"],
    }


def _write(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["id"] for r in read_jsonl(path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=("graphrag", "twenty"), default=None)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ids", nargs="*", default=PASSAGE_IDS)
    parser.add_argument("--entity-types", default=ENTITY_TYPES)
    parser.add_argument("--tag", default="", help="output-file suffix (e.g. --tag fixed -> graphrag_fixed.jsonl)")
    args = parser.parse_args(argv)

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    corpus = {r["id"]: r for r in read_jsonl(Path("dataset/data/corpus.jsonl"))}
    api_key = _load_api_key()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    teachers = ["graphrag", "twenty"] if args.teacher is None else [args.teacher]
    for teacher in teachers:
        budget = Budget(args.budget)
        transport = http_transport(
            model=MODEL, base_url=BASE_URL, api_key=api_key,
            temperature=0.0, max_tokens=MAX_TOKENS, timeout=120, enable_thinking=True,
        )
        suffix = f"_{args.tag}" if args.tag else ""
        out_path = OUT_DIR / f"{teacher}{suffix}.jsonl"
        done = _existing(out_path)
        remaining = [i for i in args.ids if i not in done]
        print(f"[{teacher}] remaining {len(remaining)} passages (done {len(done)})")

        if teacher == "graphrag":
            for cid in remaining:
                text = corpus[cid]["text"]
                try:
                    row = run_graphrag(transport, budget, text, args.entity_types, MAX_ROUNDS)
                    row["status"] = "ok"
                except (BudgetExhausted, RetryableError) as e:
                    row = {"calls": 0, "status": "failed", "failure_reason": str(e)}
                row.update({"id": cid, "teacher": teacher, "entity_types": args.entity_types})
                row["text"] = text
                _write(out_path, row)
                print(f"[graphrag] {cid}: {row['status']} entities="
                      f"{len(row.get('entities', []))} (raw {row.get('raw_entity_count')}, "
                      f"dropped {row.get('dropped')}) relationships="
                      f"{len(row.get('relationships', []))} calls={row.get('calls')}")
        else:
            client = TeacherClient(transport, system_prompt=render_teacher_prompt(TEACHER_SECTIONS_PROSE), budget=budget)
            for cid in remaining:
                text = corpus[cid]["text"]
                try:
                    row = run_twenty(client, text)
                except BudgetExhausted as e:
                    row = {"status": "failed", "failure_reason": str(e), "triplets": [], "semantic_violations": []}
                row.update({"id": cid, "teacher": teacher})
                row["text"] = text
                _write(out_path, row)
                print(f"[twenty] {cid}: {row['status']} triplets={len(row.get('triplets', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
