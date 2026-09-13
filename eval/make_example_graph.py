"""Pick a discriminative fixed-200 passage and render node-edge graphs for
gold / base / 0.6B / 1.7B / 4B into docs/figures/ (README example).

Deterministic: fixed-200 ids + report metric ranking + seeded layouts.
Requires the local (gitignored) outputs/ artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

TYPE_COLOR = {
    "PERSON": "#e57373",
    "ORGANIZATION": "#64b5f6",
    "GEO": "#81c784",
    "EVENT": "#ffb74d",
    "CONCEPT": "#ba68c8",
}
DEFAULT_COLOR = "#bdbdbd"

MODELS = [
    ("base", "outputs/base_eval/qwen3-0.6b-base/predictions.jsonl", "outputs/referent_eval/report_qwen3-0.6b-base.json", "Qwen3-0.6B base (untrained)"),
    ("0.6b", "outputs/capacity_eval/qwen3-0.6b-masked/predictions.jsonl", "outputs/referent_eval/report_qwen3-0.6b-masked.json", "Qwen3-0.6B masked"),
    ("1.7b", "outputs/capacity_eval/qwen3-1.7b-masked/predictions.jsonl", "outputs/referent_eval/report_qwen3-1.7b-masked.json", "Qwen3-1.7B masked"),
    ("4b", "outputs/capacity_eval/qwen3-4b-masked-clean/predictions.jsonl", "outputs/referent_eval/report_qwen3-4b-masked-clean.json", "Qwen3-4B masked (clean)"),
]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _short(text: str, n: int = 20) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _endpoint_index(titles: list[str]) -> dict[str, int]:
    idx: dict[str, int] = {}
    for i, t in enumerate(titles):
        key = " ".join(str(t).split()).casefold()
        if key and key not in idx:
            idx[key] = i
    return idx


def _draw(ax, label: str, row: dict, seed: int) -> None:
    entities = row.get("entities", []) or []
    rels = row.get("relationships", []) or []
    titles = [e.get("title", "") for e in entities]
    idx = _endpoint_index(titles)

    g = nx.DiGraph()
    for i, e in enumerate(entities):
        g.add_node(i)
    for r in rels:
        s = idx.get(" ".join(str(r.get("source", "")).split()).casefold())
        t = idx.get(" ".join(str(r.get("target", "")).split()).casefold())
        if s is not None and t is not None and s != t:
            w = r.get("strength")
            g.add_edge(s, t, weight=float(w) if isinstance(w, (int, float)) else 5.0)

    ax.set_title(f"{label}\n{len(entities)} entities · {len(rels)} relationships", fontsize=10)
    ax.axis("off")
    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center", color="#9e9e9e")
        return

    n = g.number_of_nodes()
    try:
        pos = nx.kamada_kawai_layout(g) if n > 2 else nx.circular_layout(g)
    except Exception:
        pos = nx.spring_layout(g, seed=seed)

    colors = [TYPE_COLOR.get(str(e.get("type", "")).upper(), DEFAULT_COLOR) for e in entities]
    node_size = 460 if n <= 8 else 300
    font_size = 5.5 if n <= 12 else 5
    widths = [0.6 + 1.6 * min(max(d.get("weight", 5.0), 0.0), 10.0) / 10.0 for _, _, d in g.edges(data=True)]
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#616161", arrows=True, arrowsize=8, width=widths, alpha=0.8)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=node_size, edgecolors="#37474f", linewidths=0.6)
    labels = {i: _short(titles[i]) for i in range(len(titles)) if str(titles[i]).strip()}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=font_size)
    ax.margins(0.18)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", type=Path, default=ROOT / "outputs/referent_eval/sample_200.json")
    p.add_argument("--gold", type=Path, default=ROOT / "outputs/full_eval/test_teacher_with_text.jsonl")
    p.add_argument("--out-dir", type=Path, default=ROOT / "docs/figures")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-gold-entities", type=int, default=6)
    p.add_argument("--max-gold-entities", type=int, default=16)
    p.add_argument("--min-gold-rels", type=int, default=3)
    p.add_argument("--min-model-entities", type=int, default=3)
    p.add_argument("--max-model-entities", type=int, default=20)
    args = p.parse_args(argv)

    gold = {r["id"]: r for r in _read_jsonl(args.gold)}
    sample_ids = json.loads(args.ids.read_text(encoding="utf-8"))["ids"]
    preds = {tag: {r["id"]: r for r in _read_jsonl(ROOT / path)} for tag, path, _, _ in MODELS}
    reports = {
        tag: {r["id"]: r for r in json.loads((ROOT / rpath).read_text(encoding="utf-8"))["per_passage"]}
        for tag, _, rpath, _ in MODELS
    }

    scored = []
    for pid in sample_ids:
        g = gold.get(pid)
        if not g:
            continue
        n_e, n_r = len(g.get("entities", [])), len(g.get("relationships", []))
        if not (args.min_gold_entities <= n_e <= args.max_gold_entities and n_r >= args.min_gold_rels):
            continue
        counts = {tag: len(preds[tag].get(pid, {}).get("entities", [])) for tag, _, _, _ in MODELS}
        if any(c < args.min_model_entities or c > args.max_model_entities for c in counts.values()):
            continue
        f1 = {tag: reports[tag].get(pid, {}).get("entity_f1", 0.0) for tag, _, _, _ in MODELS}
        rec = {tag: reports[tag].get(pid, {}).get("entity_recall", 0.0) for tag, _, _, _ in MODELS}
        scored.append((f1["4b"], rec["4b"] - rec["base"], pid, f1))
    if not scored:
        raise SystemExit("no passage satisfies the readability constraints")
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, recall_gain, pid, f1 = scored[0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    g = gold[pid]
    panels = [("Gold (teacher)", g)] + [(label, preds[tag][pid]) for tag, _, _, label in MODELS]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    axes = axes.ravel()
    for ax, (label, row) in zip(axes, panels):
        _draw(ax, label, row, args.seed)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Extraction example on one fixed-200 passage  (id: {pid})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out_dir / "graph_example.svg", format="svg")
    fig.savefig(args.out_dir / "graph_example.png", format="png", dpi=150)
    plt.close(fig)

    for slug, (label, row) in zip(["gold", "base", "0.6b", "1.7b", "4b"], panels):
        fig1, ax1 = plt.subplots(figsize=(6.5, 5.5))
        _draw(ax1, label, row, args.seed)
        fig1.tight_layout()
        fig1.savefig(args.out_dir / f"graph_example_{slug}.svg", format="svg")
        plt.close(fig1)

    meta = {
        "id": pid,
        "source": pid.split("-")[0],
        "selection": {"criterion": "max 4B entity_f1 (tie-break: recall(4B)-recall(base)); all models 3-15 entities", "recall_gain_4b_minus_base": round(recall_gain, 4)},
        "gold": {"entities": len(g.get("entities", [])), "relationships": len(g.get("relationships", []))},
        "models": {
            tag: {
                "label": label,
                "entities": len(preds[tag][pid].get("entities", [])),
                "relationships": len(preds[tag][pid].get("relationships", [])),
                "entity_f1": f1[tag],
            }
            for tag, _, _, label in MODELS
        },
        "passage_text": g["text"],
    }
    (args.out_dir / "example_graph_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"chosen id: {pid}  (recall gain 4B-base = {recall_gain:.4f})")
    for tag, _, _, label in MODELS:
        print(f"  {label:28s} entities={len(preds[tag][pid].get('entities', [])):3d} rels={len(preds[tag][pid].get('relationships', [])):3d} entity_f1={f1[tag]:.3f}")
    print(f"wrote {args.out_dir}/graph_example.svg (+ per-panel svg, meta json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
