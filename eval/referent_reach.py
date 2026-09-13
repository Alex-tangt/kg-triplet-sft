"""Consumer-side reachability axis: embedding retrieval simulation.

The graph-RAG consumer (LightRAG; MS GraphRAG before the 2026-09-11 pivot)
routes a user query to entities by embedding the query against the indexed
entity *descriptions*. This module approximates "is a gold entity
reachable from the student's graph" with a retrieval simulation on a local
embedder (no API, no real queries):

  query   = a gold (teacher) entity's description (+title)
  index   = every student entity description on the sample (one per normalized
            title, longest description kept — the consumer's dedup)
  target  = the student entity referent-matched to that gold (from the pairing)

conditional recall@k : of the gold entities the model FOUND, how many have their
    referent-matched student node among the k nearest index neighbours of the
    gold-description query (isolates description fidelity).
unconditional recall@k: over ALL gold entities; a gold the model never found is
    automatically a miss (end-to-end reachability = finding x fidelity).

The axis is only fielded if a planted fixture proves it discriminates: correct
descriptions must rank their target clearly above empty / wrong-topic sabotage.

Usage:
  python eval/referent_reach.py --fixture           # fixture gate only
  python eval/referent_reach.py                     # full axis on the fixed sample
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from dataset.io import read_jsonl  # noqa: E402

from eval.referent_common import (  # noqa: E402
    DEFAULT_STUDENT,
    DEFAULT_TEACHER,
    OUT_DIR,
    load_pairs,
    normalize,
)

_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _enc(model, texts: list[str]) -> np.ndarray:
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def _desc(e: dict) -> str:
    return (e.get("description") or "").strip()


def _query_text(e: dict) -> str:
    return _desc(e)


def build_index(student_rows: list[dict], model) -> tuple[list[str], list[dict], np.ndarray]:
    """One indexed node per normalized title per passage (longest desc kept)."""
    nodes: dict[tuple[str, str], dict] = {}  # (pid, norm_title) -> chosen student entity
    for r in student_rows:
        for i, e in enumerate(r.get("entities", [])):
            n = normalize(e.get("title"))
            if not n:
                continue
            key = (r["id"], n)
            cur = nodes.get(key)
            if cur is None or len(_desc(e)) > len(_desc(cur["e"])):
                nodes[key] = {"e": e, "idx": i, "passage": r["id"]}
    node_list = list(nodes.values())
    texts = [_desc(n["e"]) for n in node_list]
    em = _enc(model, texts) if texts else np.zeros((0, 384), dtype=np.float32)
    return node_list, texts, em


def _target_position(node_list: list[dict], pid: str, norm_title: str) -> int:
    for j, n in enumerate(node_list):
        if n["passage"] == pid and normalize(n["e"].get("title")) == norm_title:
            return j
    return -1


def recall_at_k(student_rows, teacher_rows, pairs_by_id, model, sample_ids, ks=(1, 5, 10),
                conditional: bool = False) -> dict:
    node_list, _, em = build_index(student_rows, model)
    assert em.shape[0] == len(node_list)
    teacher_by_id = {r["id"]: r for r in teacher_rows}
    student_by_id = {r["id"]: r for r in student_rows}
    hits = {k: 0 for k in ks}
    gold_total = 0
    for pid in sample_ids:
        t = teacher_by_id.get(pid)
        st = student_by_id.get(pid)
        if not t or not st:
            continue
        se = st.get("entities", [])
        for ti, te in enumerate(t.get("entities", [])):
            st_pairs = [s for a, s in pairs_by_id.get(pid, []) if a == ti]
            found = bool(st_pairs)
            if conditional and not found:
                continue
            gold_total += 1
            if not found:
                continue  # unconditional: miss (nothing to reach)
            si = st_pairs[0]
            s_title_norm = normalize(se[si].get("title"))
            q = _query_text(te)
            if not q or not _desc(te):
                continue
            qv = _enc(model, [q])[0]
            scores = em @ qv
            target_pos = _target_position(node_list, pid, s_title_norm)
            if target_pos < 0:
                continue  # no indexed student node under this title
            order = int(np.sum(scores > scores[target_pos])) + 1  # rank of the target by score
            for k in ks:
                if order <= k:
                    hits[k] += 1
    return {f"recall@{k}": round(hits[k] / gold_total, 4) if gold_total else None
            for k in ks}


def _fixture(model) -> str:
    """Planted gate: sabotage on the index side must clearly hurt recall@1, else the axis is not fielded."""
    gold = []
    for pid in range(3):
        for i in range(6):
            e = {"title": f"Concept{pid}_{i}", "type": "CONCEPT",
                 "description": f"The theoretical framework of {chr(97+i)} dynamics for system {pid}."}
            gold.append({"id": f"syn-{pid}", "entities": [e], "relationships": []})
    sabotage = {
        "correct": lambda d: d,
        "empty": lambda d: "",
        "wrong_topic": lambda d: "Economics: supply and demand curves and the inflation rate in a national economy.",
    }
    res = {}
    for mode, fn in sabotage.items():
        # index carries the sabotaged description; the query is always the true gold description
        index_rows = [{"id": r["id"], "entities": [{**e, "description": fn(e["description"])} for e in r["entities"]],
                       "relationships": []} for r in gold]
        node_list, _, em = build_index(index_rows, model)
        hits = miss = 0
        for r in gold:
            for e in r["entities"]:
                qv = _enc(model, [_query_text(e)])[0]
                scores = em @ qv
                target_pos = _target_position(node_list, r["id"], normalize(e["title"]))
                order = int(np.sum(scores > scores[target_pos])) + 1
                hits += int(order == 1)
                miss += 1
        res[mode] = round(hits / miss, 4) if miss else None
    verdict = "PASS" if res["correct"] > res["empty"] and res["correct"] > res["wrong_topic"] and res["correct"] > 0.6 else "FAIL"
    return f"fixture recall@1  correct={res['correct']}  empty={res['empty']}  wrong_topic={res['wrong_topic']}  -> {verdict}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    p.add_argument("--pairs", type=Path, default=OUT_DIR / "pairings_think_699.jsonl")
    p.add_argument("--ids", type=Path, default=OUT_DIR / "sample_200.json")
    p.add_argument("--fixture", action="store_true", help="run only the fixture gate")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(_EMB_MODEL)
    verdict = _fixture(model)
    print(verdict)
    if args.fixture:
        return 0 if verdict.endswith("PASS") else 2

    teacher = read_jsonl(args.teacher)
    student = read_jsonl(args.student)
    pairs_by_id = load_pairs(args.pairs)
    sample_ids = json.loads(args.ids.read_text(encoding="utf-8"))["ids"]
    assert all(r["id"] in sample_ids for r in student if r["id"] in sample_ids), "student file covers the sample"

    report = {"fixture": verdict, "sample_n": len(sample_ids)}
    report["conditional_on_found"] = recall_at_k(student, teacher, pairs_by_id, model, sample_ids, conditional=True)
    report["unconditional"] = recall_at_k(student, teacher, pairs_by_id, model, sample_ids, conditional=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    args.out = args.out or OUT_DIR / f"reach_{args.pairs.stem}.json"
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
