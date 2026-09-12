"""Slice builder for the LightRAG consumer-impact e2e (ADR-0009 Leg B).

The e2e runs on a ~18-passage multi-cluster slice of the fixed-200 sample,
chosen for CONCRETE shared entities (low document frequency, appearing in more
than one passage) rather than topical adjacency — a graph consumer can only
transmit a difference where the graph actually connects passages.

    python eval/lightrag_slice.py --analyze   # cluster report (diagnostic)
    python eval/lightrag_slice.py --build      # write outputs/lightrag_e2e/slice.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "outputs" / "referent_eval" / "sample_200.json"
LABELS = ROOT / "dataset" / "data" / "graphrag_labels_full.jsonl"
OUT_DIR = ROOT / "outputs" / "lightrag_e2e"

MAX_DF = 4


def _norm(title: str) -> str:
    return " ".join(str(title).split()).upper()


def load_gold():
    ids = json.loads(SAMPLE.read_text(encoding="utf-8"))["ids"]
    labels = {}
    for line in LABELS.open(encoding="utf-8"):
        rec = json.loads(line)
        if rec["id"] in ids:
            labels[rec["id"]] = rec
    return ids, labels


def entity_index(labels):
    per_passage = {pid: {_norm(e["title"]) for e in rec["entities"]} for pid, rec in labels.items()}
    df = defaultdict(int)
    for titles in per_passage.values():
        for t in titles:
            df[t] += 1
    return per_passage, df


def concrete_entities(per_passage, df):
    return {t for t, n in df.items() if 2 <= n <= MAX_DF}


def components(ids, per_passage, concrete):
    parent = {pid: pid for pid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    owners = defaultdict(list)
    for pid in ids:
        for t in per_passage[pid] & concrete:
            owners[t].append(pid)
    for pids in owners.values():
        for other in pids[1:]:
            union(pids[0], other)

    groups = defaultdict(list)
    for pid in ids:
        groups[find(pid)].append(pid)
    return groups, owners


def report():
    ids, labels = load_gold()
    per_passage, df = entity_index(labels)
    concrete = concrete_entities(per_passage, df)
    groups, owners = components(ids, per_passage, concrete)

    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"{len(ids)} sample passages, {len(concrete)} concrete entities (2..{MAX_DF} df)")
    print(f"{len(multi)} connected components of size >= 2\n")
    for root, pids in sorted(multi.items(), key=lambda kv: -len(kv[1])):
        shared = defaultdict(int)
        for pid in pids:
            for t in per_passage[pid] & concrete:
                shared[t] += 1
        anchors = [t for t, n in sorted(shared.items(), key=lambda kv: -kv[1]) if n >= 2]
        print(f"[{len(pids):2d}] {sorted(pids)}")
        print(f"     anchors: {anchors[:12]}")


SLICE_CLUSTERS = {
    "space": [
        "wikipedia-00912",
        "wikipedia-00928",
        "wikipedia-00940",
        "wikipedia-00954",
        "wikipedia-00967",
        "wikipedia-00974",
        "wikipedia-00980",
        "wikipedia-00982",
    ],
    "kurosawa": [
        "wikipedia-02542",
        "wikipedia-02559",
        "wikipedia-02567",
        "wikipedia-02575",
    ],
    "afroasiatic": [
        "wikipedia-00536",
        "wikipedia-00540",
        "wikipedia-00556",
    ],
    "andorra": [
        "wikipedia-00566",
        "wikipedia-00582",
        "wikipedia-00584",
    ],
    "egypt": [
        "wikipedia-02368",
        "wikipedia-02597",
        "wikipedia-02608",
        "wikipedia-02617",
    ],
}


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids, labels = load_gold()
    passages = {}
    for line in (ROOT / "outputs" / "referent_eval" / "sample_200_passages.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        passages[rec["id"]] = rec["text"]

    chosen = []
    for cluster, pids in SLICE_CLUSTERS.items():
        for pid in pids:
            chosen.append({"id": pid, "cluster": cluster})
    assert chosen, "SLICE_CLUSTERS is empty — run --analyze and curate first"
    missing = [c["id"] for c in chosen if c["id"] not in passages]
    assert not missing, f"ids not in sample_200: {missing}"

    out = {
        "max_df": MAX_DF,
        "clusters": {k: v for k, v in SLICE_CLUSTERS.items() if v},
        "passages": [
            {"id": c["id"], "cluster": c["cluster"], "text": passages[c["id"]]} for c in chosen
        ],
    }
    (OUT_DIR / "slice.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'slice.json'} — {len(chosen)} passages in {len(out['clusters'])} clusters")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        report()
    elif args.build:
        build()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
