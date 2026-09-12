"""LightRAG consumer-impact e2e — Leg B of ADR-0009.

Fixture mode (`--fixture`) probes the `insert_custom_kg` semantics the graph
constructor depends on, before any real run:

    python eval/lightrag_e2e.py --fixture

It asserts, on a tiny in-memory-ish workspace with a stub LLM that raises if
called (so a pass proves the write path is LLM-free):

  1. entities are last-wins by name
  2. relationships are last-wins by UNORDERED endpoint pair
  3. a relationship missing `keywords` raises (direct subscript)
  4. a self-loop raises
  5. a dangling endpoint becomes an UNKNOWN node
  6. a weight below the evidence floor (1 real source) raises
  7. an explicit weight above the floor is preserved
  8. a known chunk alias resolves to the chunk hash; an unknown one -> UNKNOWN

Exit 0 with "PASS" when every assertion holds.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "outputs" / "lightrag_e2e"
SLICE = OUT_DIR / "slice.json"

EMBED_DIM = 16

LAYER_SOURCES = {
    "gold": ROOT / "dataset" / "data" / "graphrag_labels_full.jsonl",
    "0.6b": ROOT / "outputs" / "capacity_eval" / "qwen3-0.6b-masked" / "predictions.jsonl",
    "4b": ROOT / "outputs" / "capacity_eval" / "qwen3-4b-masked-clean" / "predictions.jsonl",
}


async def _stub_llm(*args, **kwargs):
    raise AssertionError("insert_custom_kg must not call the LLM")


async def _stub_embed(texts, **kwargs):
    vecs = []
    for text in texts:
        seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vecs.append(rng.standard_normal(EMBED_DIM).astype(np.float32))
    return np.vstack(vecs)


async def _new_rag(workdir: Path):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    rag = LightRAG(
        working_dir=str(workdir),
        llm_model_func=_stub_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM, func=_stub_embed, max_token_size=8192
        ),
    )
    await rag.initialize_storages()
    return rag


CHUNKS = [
    {"content": "Alpha fixture text.", "source_id": "p1", "file_path": "fixture"},
    {"content": "Beta fixture text.", "source_id": "p2", "file_path": "fixture"},
]


def _entity(name, desc, type_="CONCEPT", source_id="p1"):
    return {
        "entity_name": name,
        "entity_type": type_,
        "description": desc,
        "source_id": source_id,
    }


def _rel(src, tgt, desc, weight=1.5, source_id="p1", keywords=""):
    row = {
        "src_id": src,
        "tgt_id": tgt,
        "description": desc,
        "weight": weight,
        "source_id": source_id,
    }
    if keywords is not None:
        row["keywords"] = keywords
    return row


async def run_fixture() -> int:
    failures = []

    def check(name, cond, detail=""):
        status = "ok" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        rag = await _new_rag(Path(tmp))
        try:
            await rag.ainsert_custom_kg(
                {
                    "chunks": CHUNKS,
                    "entities": [
                        _entity("ALPHA", "first alpha"),
                        _entity("BETA", "beta desc", type_="PERSON", source_id="p2"),
                        _entity("ALPHA", "second alpha"),
                    ],
                    "relationships": [
                        _rel("ALPHA", "BETA", "rel1", weight=1.8, source_id="p1"),
                        _rel("BETA", "ALPHA", "rel2-last", weight=1.5, source_id="p2"),
                        _rel("ALPHA", "GAMMA", "dangling", weight=1.2, source_id="p1"),
                    ],
                }
            )

            alpha = await rag.chunk_entity_relation_graph.get_node("ALPHA")
            check(
                "entity last-wins by name",
                alpha and alpha.get("description") == "second alpha",
                str(alpha.get("description") if alpha else None),
            )
            check(
                "entity source_id resolves to a chunk hash",
                alpha and str(alpha.get("source_id", "")).startswith("chunk-"),
                str(alpha.get("source_id") if alpha else None),
            )

            edge = await rag.chunk_entity_relation_graph.get_edge("ALPHA", "BETA")
            check(
                "relationship last-wins by unordered pair",
                edge and edge.get("description") == "rel2-last",
                str(edge.get("description") if edge else None),
            )
            check(
                "explicit weight above floor preserved",
                edge and float(edge.get("weight", 0)) == 1.5,
                str(edge.get("weight") if edge else None),
            )

            gamma = await rag.chunk_entity_relation_graph.get_node("GAMMA")
            check(
                "dangling endpoint becomes UNKNOWN node",
                gamma
                and gamma.get("entity_type") == "UNKNOWN"
                and gamma.get("description") == "UNKNOWN",
                str(gamma if gamma else None),
            )

            labels = await rag.chunk_entity_relation_graph.get_all_labels()
            check(
                "no spurious nodes",
                sorted(labels) == ["ALPHA", "BETA", "GAMMA"],
                str(sorted(labels)),
            )
        finally:
            await rag.finalize_storages()

    async def _expect_raise(label, custom_kg, exc_type):
        with tempfile.TemporaryDirectory() as tmp:
            rag = await _new_rag(Path(tmp))
            try:
                try:
                    await rag.ainsert_custom_kg(custom_kg)
                except exc_type as exc:
                    check(label, True, type(exc).__name__)
                except Exception as exc:  # noqa: BLE001
                    check(label, False, f"raised {type(exc).__name__}: {exc}")
                else:
                    check(label, False, "no exception")
            finally:
                await rag.finalize_storages()

    base = {"chunks": CHUNKS, "entities": [_entity("ALPHA", "a")]}

    await _expect_raise(
        "missing keywords raises",
        {**base, "relationships": [_rel("ALPHA", "BETA", "x", keywords=None)]},
        KeyError,
    )
    await _expect_raise(
        "self-loop raises",
        {**base, "relationships": [_rel("ALPHA", "ALPHA", "x")]},
        ValueError,
    )
    await _expect_raise(
        "weight below evidence floor raises",
        {**base, "relationships": [_rel("ALPHA", "BETA", "x", weight=0.5)]},
        ValueError,
    )

    if failures:
        print(f"\nFAIL ({len(failures)}): {', '.join(failures)}")
        return 1
    print("\nPASS")
    return 0


def _norm_title(title) -> str:
    return " ".join(str(title).split()).upper()


def _as_strength(value) -> float | None:
    try:
        s = float(value)
    except (TypeError, ValueError):
        return None
    if s < 0:
        return None
    return min(s, 10.0)


def load_records(path: Path, ids: set[str]) -> dict:
    records = {}
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        if rec["id"] in ids:
            records[rec["id"]] = rec
    missing = ids - set(records)
    if missing:
        raise SystemExit(f"{path.name}: missing {len(missing)} slice ids, e.g. {sorted(missing)[:3]}")
    return records


def build_layer(records: dict, order: list[str], layer: str, texts: dict):
    """Merge one layer's per-passage labels into a LightRAG custom_kg (K3).

    LightRAG is a direct writer (last-wins, no merge), so this does the merging:
    entities by normalized title, relationships by unordered endpoint pair.
    Chunk text comes from the shared slice (`texts`), not the layer record, so
    the three graphs' chunks are byte-identical (student predictions carry no
    passage text).
    """
    ent_desc: dict[str, list[str]] = defaultdict(list)
    ent_type: dict[str, Counter] = defaultdict(Counter)
    ent_source: dict[str, str] = {}
    ent_passages: dict[str, set[str]] = defaultdict(set)
    rel_desc: dict[tuple[str, str], list[str]] = defaultdict(list)
    rel_strength: dict[tuple[str, str], list[float]] = defaultdict(list)
    rel_dir: dict[tuple[str, str], tuple[str, str]] = {}
    rel_source: dict[tuple[str, str], str] = {}
    empty_passages = []
    dropped_self_loops = 0
    missing_strength = 0

    for pid in order:
        rec = records[pid]
        ents = rec.get("entities") or []
        rels = rec.get("relationships") or []
        if not ents and not rels:
            empty_passages.append(pid)
        for e in ents:
            title = _norm_title(e.get("title", ""))
            if not title:
                continue
            desc = " ".join(str(e.get("description", "")).split())
            if desc and desc not in ent_desc[title]:
                ent_desc[title].append(desc)
            ent_type[title][str(e.get("type", "UNKNOWN")).upper()] += 1
            ent_source.setdefault(title, pid)
            ent_passages[title].add(pid)
        for r in rels:
            src, tgt = _norm_title(r.get("source", "")), _norm_title(r.get("target", ""))
            if not src or not tgt:
                continue
            if src == tgt:
                dropped_self_loops += 1
                continue
            key = tuple(sorted((src, tgt)))
            desc = " ".join(str(r.get("description", "")).split())
            if desc and desc not in rel_desc[key]:
                rel_desc[key].append(desc)
            strength = _as_strength(r.get("strength"))
            if strength is None:
                missing_strength += 1
                strength = 5.0
            rel_strength[key].append(strength)
            rel_dir.setdefault(key, (src, tgt))
            rel_source.setdefault(key, pid)

    entities = [
        {
            "entity_name": title,
            "entity_type": ent_type[title].most_common(1)[0][0],
            "description": " | ".join(ent_desc[title]) or title,
            "source_id": ent_source[title],
        }
        for title in sorted(ent_desc)
    ]
    relationships = []
    for key in sorted(rel_desc):
        src, tgt = rel_dir[key]
        mean_strength = sum(rel_strength[key]) / len(rel_strength[key])
        relationships.append(
            {
                "src_id": src,
                "tgt_id": tgt,
                "description": " | ".join(rel_desc[key]) or f"{src} - {tgt}",
                "keywords": "",
                "weight": round(1.0 + mean_strength / 10.0, 4),
                "source_id": rel_source[key],
            }
        )

    known = {e["entity_name"] for e in entities}
    dangling = sum(1 for r in relationships if r["src_id"] not in known or r["tgt_id"] not in known)

    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(known)
    g.add_edges_from((r["src_id"], r["tgt_id"]) for r in relationships)
    components = list(nx.connected_components(g))
    stats = {
        "layer": layer,
        "passages": len(order),
        "entities_raw": sum(len(records[p].get("entities") or []) for p in order),
        "entities": len(entities),
        "relationships_raw": sum(len(records[p].get("relationships") or []) for p in order),
        "relationships": len(relationships),
        "dropped_self_loops": dropped_self_loops,
        "missing_strength": missing_strength,
        "empty_passages": len(empty_passages),
        "dangling_endpoints": dangling,
        "multi_passage_entities": sum(1 for s in ent_passages.values() if len(s) >= 2),
        "isolated_nodes": sum(1 for n in g if g.degree(n) == 0),
        "components": len(components),
        "largest_component": max((len(c) for c in components), default=0),
    }
    custom_kg = {
        "chunks": [
            {"content": texts[p], "source_id": p, "file_path": "e2e"} for p in order
        ],
        "entities": entities,
        "relationships": relationships,
    }
    return custom_kg, stats


def _prf(pred: set, gold: set) -> dict:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "n": len(pred)}


def run_build_graphs() -> int:
    if not SLICE.exists():
        raise SystemExit("run `eval/lightrag_slice.py --build` first")
    slice_data = json.loads(SLICE.read_text(encoding="utf-8"))
    order = [p["id"] for p in slice_data["passages"]]
    ids = set(order)
    texts = {p["id"]: p["text"] for p in slice_data["passages"]}

    records_by_layer = {name: load_records(path, ids) for name, path in LAYER_SOURCES.items()}
    gold = records_by_layer["gold"]

    gold_titles = {_norm_title(e["title"]) for rec in gold.values() for e in rec.get("entities") or []}
    gold_pairs = set()
    for rec in gold.values():
        for r in rec.get("relationships") or []:
            s, t = _norm_title(r.get("source", "")), _norm_title(r.get("target", ""))
            if s and t and s != t:
                gold_pairs.add(tuple(sorted((s, t))))
    df = Counter()
    for rec in gold.values():
        for e in rec.get("entities") or []:
            t = _norm_title(e["title"])
            if t:
                df[t] += 1
    gold_links = {t for t, n in df.items() if n >= 2}

    report = {"slice": {"passages": len(order), "gold_link_entities": len(gold_links)}, "layers": {}}
    for name, records in records_by_layer.items():
        custom_kg, stats = build_layer(records, order, name, texts)
        layer_dir = OUT_DIR / name
        layer_dir.mkdir(parents=True, exist_ok=True)
        (layer_dir / "custom_kg.json").write_text(
            json.dumps(custom_kg, ensure_ascii=False), encoding="utf-8"
        )
        titles = {e["entity_name"] for e in custom_kg["entities"]}
        pairs = {tuple(sorted((r["src_id"], r["tgt_id"]))) for r in custom_kg["relationships"]}
        stats["entity_prf_vs_gold"] = _prf(titles, gold_titles)
        stats["edge_prf_vs_gold"] = _prf(pairs, gold_pairs)
        stats["link_survival"] = round(len(gold_links & titles) / len(gold_links), 4) if gold_links else 0.0
        report["layers"][name] = stats

    def _flat(a, b):
        return all(
            abs(report["layers"][a]["entity_prf_vs_gold"][k] - report["layers"][b]["entity_prf_vs_gold"][k]) < 0.02
            and abs(report["layers"][a]["edge_prf_vs_gold"][k] - report["layers"][b]["edge_prf_vs_gold"][k]) < 0.02
            for k in ("precision", "recall", "f1")
        )

    report["gate"] = {
        "flat_0.6b_vs_4b": _flat("0.6b", "4b"),
        "verdict": "STOP — T0 flat across treatments" if _flat("0.6b", "4b") else "GO — treatments differ at graph formation",
    }
    (OUT_DIR / "t0_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"T0 gate: {report['gate']['verdict']}\n")
    for name, s in report["layers"].items():
        e, g = s["entity_prf_vs_gold"], s["edge_prf_vs_gold"]
        print(
            f"{name:5s} nodes={s['entities']:4d} edges={s['relationships']:4d} "
            f"entity F1={e['f1']:.3f} edge F1={g['f1']:.3f} "
            f"link-survival={s['link_survival']:.3f} multi-passage={s['multi_passage_entities']:3d} "
            f"comps={s['components']:3d} largest={s['largest_component']:3d} iso={s['isolated_nodes']:3d} "
            f"empty={s['empty_passages']} dangling={s['dangling_endpoints']} self-loops={s['dropped_self_loops']}"
        )
    return 0 if not report["gate"]["flat_0.6b_vs_4b"] else 2


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_KEYWORD = "qwen-flash"
MODEL_QUERY = "qwen-max"
MODES = ["local", "global", "hybrid", "naive"]
QUERY_PARAMS = dict(
    top_k=60, chunk_top_k=20, max_entity_tokens=6000, max_relation_tokens=8000, max_total_tokens=30000
)
GROUNDING_PREFIX = (
    "Answer using ONLY the provided context. If the context does not contain the answer, "
    "state that the available information is insufficient. Never use outside knowledge.\n\n"
)

_EMBED_MODEL = None
_RERANK_MODEL = None


def _embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        _EMBED_MODEL = SentenceTransformer("BAAI/bge-m3", device="cpu")
    return _EMBED_MODEL


def _rerank_model():
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import CrossEncoder

        _RERANK_MODEL = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024, device="cpu")
    return _RERANK_MODEL


def _embedding_func():
    from lightrag.utils import EmbeddingFunc

    model = _embed_model()
    dim = model.get_sentence_embedding_dimension()

    async def _embed(texts, **kwargs):
        arr = model.encode(
            list(texts), normalize_embeddings=True, batch_size=16, show_progress_bar=False
        )
        return np.asarray(arr, dtype=np.float32)

    return EmbeddingFunc(embedding_dim=dim, func=_embed, max_token_size=8192)


def _rerank_func():
    model = _rerank_model()

    async def _rerank(query, documents, top_n=None, **kwargs):
        scores = model.predict([(query, d) for d in documents])
        order = list(np.argsort(scores)[::-1])
        if top_n:
            order = order[:top_n]
        return [{"index": int(i), "relevance_score": float(scores[i])} for i in order]

    return _rerank


def _dashscope_key():
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("DASHSCOPE_API_KEY not set (see .env)")
    return key


def _llm_funcs():
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.llm_roles import RoleLLMConfig

    key = _dashscope_key()

    def mk(model_name):
        async def _complete(prompt=None, system_prompt=None, history_messages=None, **kwargs):
            kwargs.pop("model", None)
            kwargs.pop("_priority", None)
            return await openai_complete_if_cache(
                model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                base_url=DASHSCOPE_BASE_URL,
                api_key=key,
                **kwargs,
            )

        return _complete

    query, keyword = mk(MODEL_QUERY), mk(MODEL_KEYWORD)
    roles = {
        "keyword": RoleLLMConfig(func=keyword),
        "query": RoleLLMConfig(func=query),
        "extract": RoleLLMConfig(func=keyword),
    }
    return query, roles


async def _build_rag(workdir: Path, rerank: bool, wipe: bool = True):
    import shutil

    from lightrag import LightRAG

    if wipe and workdir.exists():
        shutil.rmtree(workdir)
    llm, roles = _llm_funcs()
    kwargs = dict(
        working_dir=str(workdir),
        llm_model_func=llm,
        role_llm_configs=roles,
        embedding_func=_embedding_func(),
        user_prompt_prefix=GROUNDING_PREFIX,
    )
    if rerank:
        kwargs["rerank_model_func"] = _rerank_func()
    rag = LightRAG(**kwargs)
    await rag.initialize_storages()
    return rag


async def run_t1(layers, modes, rerank_arms, limit=None, concurrency=4):
    from lightrag import QueryParam

    questions = json.loads((OUT_DIR / "questions.json").read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]
    results = []
    sem = asyncio.Semaphore(concurrency)
    live = (OUT_DIR / "t1_results.jsonl").open("w", encoding="utf-8")

    async def one(rag, layer, rerank, mode, q):
        async with sem:
            param = QueryParam(
                mode=mode, only_need_context=True, enable_rerank=rerank, **QUERY_PARAMS
            )
            ctx = await rag.aquery(q["question"], param=param)
            text = ctx if isinstance(ctx, str) else json.dumps(ctx, ensure_ascii=False)
            up = text.upper()
            hits = [t for t in q["answer_terms"] if _norm_title(t) in up]
            row = {
                "layer": layer,
                "rerank": rerank,
                "mode": mode,
                "qid": q["qid"],
                "type": q["type"],
                "cluster": q["cluster"],
                "recall": round(len(hits) / len(q["answer_terms"]), 4),
                "hits": hits,
                "ctx_chars": len(text),
            }
            results.append(row)
            live.write(json.dumps(row, ensure_ascii=False) + "\n")
            live.flush()
            print(
                f"  {layer} r={int(rerank)} {mode:6s} {q['qid']:3s} recall={row['recall']:.2f}",
                flush=True,
            )

    try:
        for layer in layers:
            kg_path = OUT_DIR / layer / "custom_kg.json"
            if not kg_path.exists():
                raise SystemExit(f"missing {kg_path}; run --build-graphs")
            kg = json.loads(kg_path.read_text(encoding="utf-8"))
            workdir = OUT_DIR / layer / "rag"
            print(f"[{layer}] building + inserting ...", flush=True)
            rag = await _build_rag(workdir, rerank=True)
            try:
                await rag.ainsert_custom_kg(kg)
                print(f"[{layer}] inserted; querying ...", flush=True)
                tasks = [
                    one(rag, layer, rerank, mode, q)
                    for rerank in rerank_arms
                    for mode in modes
                    for q in questions
                ]
                await asyncio.gather(*tasks)
            finally:
                try:
                    await asyncio.wait_for(rag.finalize_storages(), timeout=60)
                except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                    print(f"[{layer}] finalize skipped: {exc}", flush=True)
    finally:
        live.close()

    (OUT_DIR / "t1_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT_DIR / "t1_done.flag").write_text("done", encoding="utf-8")

    print("\n=== T1 recall (mean over questions) ===")
    agg = defaultdict(list)
    for r in results:
        agg[(r["layer"], r["rerank"], r["mode"])].append(r["recall"])
    for layer in layers:
        for rerank in rerank_arms:
            row = " ".join(
                f"{m}={sum(agg[(layer, rerank, m)]) / len(agg[(layer, rerank, m)]):.3f}"
                for m in modes
                if agg[(layer, rerank, m)]
            )
            print(f"{layer:5s} rerank={int(rerank)}  {row}")
    return 0


JUDGE_MODEL = "qwen3.7-flash"
JUDGE_SYSTEM = (
    "You grade whether a generated ANSWER correctly answers a QUESTION, given a REFERENCE answer.\n"
    'Return ONLY JSON: {"verdict": "correct"|"partial"|"incorrect"|"unanswerable"}.\n'
    "- correct: the answer conveys the key facts of the reference (paraphrase OK).\n"
    "- partial: some key facts present, some missing.\n"
    "- incorrect: wrong or contradicts the reference.\n"
    "- unanswerable: the answer says it cannot answer from the context.\n"
)


def _parse_verdict(raw: str) -> str:
    import re

    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return "parse_error"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "parse_error"
    v = str(obj.get("verdict", "")).lower().strip()
    return v if v in {"correct", "partial", "incorrect", "unanswerable"} else "parse_error"


async def run_t2(layers, modes, token_budget=3_000_000, concurrency=4):
    from lightrag import QueryParam

    from dataset import teacher as teacher_api
    from dataset.teacher import http_transport

    questions = json.loads((OUT_DIR / "questions.json").read_text(encoding="utf-8"))
    t1 = json.loads((OUT_DIR / "t1_results.json").read_text(encoding="utf-8"))
    ctx_chars = {(r["layer"], r["mode"], r["qid"]): r["ctx_chars"] for r in t1}

    key = _dashscope_key()
    judge = http_transport(
        model=JUDGE_MODEL,
        base_url=DASHSCOPE_BASE_URL,
        api_key=key,
        temperature=0.0,
        max_tokens=2000,
        timeout=180,
        enable_thinking=True,
    )

    results = []
    spent = {"tokens": 0, "stopped": False}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)
    live = (OUT_DIR / "t2_results.jsonl").open("w", encoding="utf-8")

    async def one(rag, layer, mode, q):
        async with sem:
            async with lock:
                if spent["tokens"] >= token_budget:
                    spent["stopped"] = True
                    return
            param = QueryParam(
                mode=mode, only_need_context=False, enable_rerank=False, **QUERY_PARAMS
            )
            try:
                answer = await rag.aquery(q["question"], param=param)
            except Exception as exc:  # noqa: BLE001
                answer = f"<error: {type(exc).__name__}>"
            answer = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
            user = (
                f"QUESTION: {q['question']}\n"
                f"REFERENCE ANSWER: {q['reference']}\n"
                f"GENERATED ANSWER: {answer}"
            )
            try:
                raw = judge(
                    [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user},
                    ]
                )
                usage = teacher_api.LAST_USAGE or {}
                judge_tokens = int(usage.get("total_tokens") or 0)
            except Exception as exc:  # noqa: BLE001
                raw = ""
                judge_tokens = 0
                print(f"  judge error: {exc}", flush=True)
            gen_tokens = ctx_chars.get((layer, mode, q["qid"]), 0) // 4 + len(answer) // 4
            async with lock:
                spent["tokens"] += gen_tokens + judge_tokens
                budget_now = spent["tokens"]
            row = {
                "layer": layer,
                "mode": mode,
                "qid": q["qid"],
                "type": q["type"],
                "cluster": q["cluster"],
                "verdict": _parse_verdict(raw),
                "answer": answer[:1200],
                "gen_tokens": gen_tokens,
                "judge_tokens": judge_tokens,
                "cum_tokens": budget_now,
            }
            results.append(row)
            live.write(json.dumps(row, ensure_ascii=False) + "\n")
            live.flush()
            print(
                f"  {layer} {mode:6s} {q['qid']:3s} {row['verdict']:12s} cum={budget_now/1e6:.2f}M",
                flush=True,
            )

    try:
        for layer in layers:
            kg = json.loads((OUT_DIR / layer / "custom_kg.json").read_text(encoding="utf-8"))
            workdir = OUT_DIR / layer / "rag"
            rag = await _build_rag(workdir, rerank=False, wipe=False)
            try:
                if not (workdir / "graph_chunk_entity_relation.graphml").exists():
                    print(f"[{layer}] inserting ...", flush=True)
                    await rag.ainsert_custom_kg(kg)
                else:
                    print(f"[{layer}] reusing existing graph", flush=True)
                await asyncio.gather(*[one(rag, layer, mode, q) for mode in modes for q in questions])
            finally:
                try:
                    await asyncio.wait_for(rag.finalize_storages(), timeout=60)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{layer}] finalize skipped: {exc}", flush=True)
    finally:
        live.close()

    (OUT_DIR / "t2_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT_DIR / "t2_done.flag").write_text("done", encoding="utf-8")

    print(f"\n=== T2 verdicts (spent {spent['tokens']/1e6:.2f}M, stopped={spent['stopped']}) ===")
    agg = defaultdict(list)
    for r in results:
        agg[(r["layer"], r["mode"])].append(r["verdict"])
    for layer in layers:
        for mode in modes:
            vs = agg[(layer, mode)]
            if not vs:
                continue
            print(
                f"{layer:5s} {mode:6s} correct={vs.count('correct')/len(vs):.3f} "
                f"partial={vs.count('partial')/len(vs):.3f} n={len(vs)}"
            )
    return 0


def _run_async(coro):
    """Run a coroutine and hard-exit, skipping asyncio's executor shutdown.

    LightRAG's priority worker pools leave non-daemon threads that make
    ``asyncio.run`` hang at shutdown; all work (and result files) is already
    flushed before this returns.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coro)
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result or 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="LightRAG consumer-impact e2e (ADR-0009 Leg B)")
    ap.add_argument("--fixture", action="store_true", help="run the insert_custom_kg probe")
    ap.add_argument("--build-graphs", action="store_true", help="build the three custom KGs + T0 gate")
    ap.add_argument("--run-t1", action="store_true", help="run the T1 retrieval layer")
    ap.add_argument("--run-t2", action="store_true", help="run the T2 answer layer + judge")
    ap.add_argument("--token-budget", type=int, default=3_000_000)
    ap.add_argument("--layers", default="gold,0.6b,4b")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--no-rerank", action="store_true", help="rerank-off arm only")
    ap.add_argument("--limit", type=int, default=None, help="first N questions (smoke)")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    if args.fixture:
        return _run_async(run_fixture())
    if args.build_graphs:
        return run_build_graphs()
    if args.run_t1:
        layers = [s for s in args.layers.split(",") if s]
        modes = [s for s in args.modes.split(",") if s]
        rerank_arms = (False,) if args.no_rerank else (False, True)
        return _run_async(
            run_t1(layers, modes, rerank_arms, limit=args.limit, concurrency=args.concurrency)
        )
    if args.run_t2:
        layers = [s for s in args.layers.split(",") if s]
        modes = [s for s in args.modes.split(",") if s]
        return _run_async(
            run_t2(layers, modes, token_budget=args.token_budget, concurrency=args.concurrency)
        )
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
