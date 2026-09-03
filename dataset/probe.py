"""Label-quality attribution probes: ontology gap (A) and W1 workflow mode (B).

Two controlled experiments over the same comparison set C (20 clean-baseline
passages) plus the pilot hard negatives H (10):

- **Probe A (ontology gap)**: the teacher runs in *open* mode — any natural
  relation name, no fixed set. Grounded triplets with a relation outside the
  20-set measure how much the frozen ontology fails to express; in-set-but-missed
  triplets measure single-step under-extraction; hard-negative flips measure how
  many `[]` labels are artifacts of the closed set.
- **Probe B (workflow)**: the teacher runs in *W1* two-stage mode (entity spans,
  then 20-set relation judgment) on the same passages; per-passage triplet
  coverage and grounding are compared against the pilot's single-step labels.

Outputs land under ``outputs/``: ``probe_results.jsonl`` (every raw record),
``ontology_gap.json`` (Probe A stats), ``gold_review.html`` (5 passages, union
of all modes, for a human to mark gold triplets) and, once a gold file exists,
precision/recall of both modes against it.

The probes reuse ``TeacherClient`` with injected extractors; only the prompt and
the validation are swapped. The pilot labels are the single-step baseline — no
re-labeling is needed for the control arm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg_contract import ENTITY_TYPES, RELATION_TYPES, extract_json_array, render_teacher_prompt, validate_triplet_list  # noqa: E402
from kg_contract.prompts import TEACHER_SECTIONS_PROSE  # noqa: E402

from dataset.io import read_jsonl  # noqa: E402
from dataset.semantic import check_grounding, normalize  # noqa: E402
from dataset.teacher import (  # noqa: E402
    Budget,
    TeacherClient,
    http_transport,
    label_tasks,
    sample_passages,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"

RELATIONS_ORDERED = (
    "implements", "trained_on", "evaluates", "part_of", "introduces", "extends",
    "depends_on", "contrasts_with", "applied_to", "measured_by", "founded_by",
    "developed_by", "defined_as", "consists_of", "is_type_of", "based_on",
    "used_for", "created_by", "located_in", "predecessor_of",
)

_OPEN_RELATION_SECTION = (
    "You extract knowledge-graph triplets from English text. A triplet is a JSON "
    "object with keys: source, relation, target. source and target are each an "
    "object {title, type}; relation is an object {type, weight}. You may use ANY "
    "natural relation name — a short verb or prepositional phrase such as causes, "
    "successor_of, compares_with, authored_by — you are NOT restricted to a fixed "
    "set; choose the most precise name that fits the text. If the text supports no "
    "triplet at all, return an empty JSON array []."
)

_W1_SYSTEM = (
    "You extract knowledge-graph triplets from English text in two stages.\n\n"
    "Stage A - entity and concept extraction:\n"
    "List every entity (a concrete named thing: person, organization, product, "
    "place, work, event) and every concept (an abstract idea, class, or field) "
    "mentioned in the text. Each entry must be the exact span as it appears in "
    "the text, with a type of exactly entity or concept. Do not invent, "
    "paraphrase, abbreviate, or add labels, numbering, or brackets.\n\n"
    "Stage B - relation judgment:\n"
    "For every pair of extracted nodes that the text supports a relation for, "
    "emit a triplet {source: {title, type}, relation: {type, weight}, target: "
    "{title, type}}. The relation type must be exactly one of these 20: "
    + ", ".join(RELATIONS_ORDERED) + ". If a fact does not fit a listed relation, "
    "emit no triplet for it. Set weight to exactly one of 0.8 (explicitly "
    "stated), 0.5 (indirectly supported), 0.2 (only inferred). Every source and "
    "target title must be grounded in the text: the exact span must appear in "
    "the text; if a referent is only established outside the text, drop the "
    "triplet rather than invent it.\n\n"
    "Output EXACTLY one JSON object with two keys:\n"
    '{"entities": [{"title": "...", "type": "entity"}], "triplets": [{"source": '
    '{...}, "relation": {...}, "target": {...}}]}\n'
    'If no triplet is supported, "triplets" is an empty array.'
)


def open_system_prompt() -> str:
    sections = {**TEACHER_SECTIONS_PROSE, "relation_dictionary": _OPEN_RELATION_SECTION}
    return render_teacher_prompt(sections)


def w1_system_prompt() -> str:
    return _W1_SYSTEM


# --------------------------------------------------------------------------- #
# Extractors (swapped into TeacherClient) and loose validation
# --------------------------------------------------------------------------- #

def extract_json_object(raw) -> dict | None:
    """Tolerantly pull the first JSON object out of model output."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    decoder = json.JSONDecoder()
    start = raw.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            start = raw.find("{", start + 1)
            continue
        return value if isinstance(value, dict) else None
    return None


def validate_open_list(items) -> tuple[bool, list[str]]:
    """Loose validation for open extraction: shape + types + weight, no relation set."""
    if not isinstance(items, list):
        return False, ["output is not a JSON array"]
    errors: list[str] = []
    for i, t in enumerate(items):
        if not isinstance(t, dict):
            errors.append(f"triplets[{i}] must be an object")
            continue
        for side in ("source", "target"):
            node = t.get(side)
            if not isinstance(node, dict) or not isinstance(node.get("title"), str):
                errors.append(f"triplets[{i}].{side} must be {{title, type}}")
            elif node.get("type") not in ENTITY_TYPES:
                errors.append(f"triplets[{i}].{side}.type must be in {sorted(ENTITY_TYPES)}")
        rel = t.get("relation")
        if not isinstance(rel, dict) or not isinstance(rel.get("type"), str) or not rel["type"].strip():
            errors.append(f"triplets[{i}].relation.type must be a non-empty string")
        weight = rel.get("weight") if isinstance(rel, dict) else None
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            errors.append(f"triplets[{i}].relation.weight must be a number")
        elif not 0.0 <= float(weight) <= 1.0:
            errors.append(f"triplets[{i}].relation.weight must be in [0, 1]")
    return not errors, errors


def open_extractor(raw: str) -> tuple[list | None, list[str]]:
    triplets = extract_json_array(raw)
    if triplets is None:
        return None, ["output is not a JSON array"]
    valid, errors = validate_open_list(triplets)
    if not valid:
        return None, errors
    return triplets, []


def w1_extractor(raw: str) -> tuple[list | None, list[str]]:
    obj = extract_json_object(raw)
    if obj is None:
        return None, ["output is not a JSON object"]
    triplets = obj.get("triplets")
    if not isinstance(triplets, list):
        return None, ["'triplets' key missing or not an array"]
    valid, errors = validate_triplet_list(triplets)
    if not valid:
        return None, ["; ".join(f"{e.path}: {e.reason}" for e in errors)]
    return triplets, []


# --------------------------------------------------------------------------- #
# Matching and metrics
# --------------------------------------------------------------------------- #

def triplet_key(t: dict) -> tuple:
    rel = (t.get("relation") or {}).get("type") or ""
    return (normalize((t.get("source") or {}).get("title") or ""), rel, normalize((t.get("target") or {}).get("title") or ""))


def grounded_triplets(triplets: list[dict], passage: str) -> list[dict]:
    bad = {v.triplet_index for v in check_grounding(triplets, passage)}
    return [t for i, t in enumerate(triplets) if i not in bad]


def matched_count(a: list[dict], b: list[dict]) -> int:
    """Matched triplets between two lists (Counter intersection on normalized keys)."""
    return sum((Counter(triplet_key(t) for t in a) & Counter(triplet_key(t) for t in b)).values())


def _mean(xs) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def _sd(xs) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return round((sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5, 3)


def compute_gap(open_rows: list[dict]) -> dict:
    """Relation-vocabulary gap (UPPER BOUND): fraction of grounded open triplets
    whose relation is outside the 20-set. Inflated by near-synonymy — see
    :func:`compute_effective_gap` for the pair-level version."""
    grounded: list[tuple[str, dict]] = []
    for r in open_rows:
        if r["status"] != "ok":
            continue
        grounded.extend((r["id"], t) for t in grounded_triplets(r["triplets"], r["text"]))
    out_of_set = [(pid, t) for pid, t in grounded if (t.get("relation") or {}).get("type") not in RELATION_TYPES]
    return {
        "grounded_triplets": len(grounded),
        "out_of_set": len(out_of_set),
        "gap": round(len(out_of_set) / len(grounded), 3) if grounded else 0,
        "missing_relations": dict(Counter((t.get("relation") or {}).get("type") for _, t in out_of_set).most_common(20)),
        "caveat": "upper bound: inflated by open-mode noise and near-synonymy; use effective_gap + gold for the true number",
    }


def _pair_key(t: dict) -> tuple:
    return ((t.get("source") or {}).get("title", "").strip().lower(), (t.get("target") or {}).get("title", "").strip().lower())


def _pairs(rec: dict | None) -> set:
    if not rec or rec["status"] != "ok":
        return set()
    return {_pair_key(t) for t in rec["triplets"]}


def compute_effective_gap(open_rows: list[dict], pilot_by_id: dict, w1_rows: list[dict]) -> dict:
    """Effective gap at the entity-PAIR level, immune to relation synonymy.

    A pair is 'covered by the 20-set' if single-step (pilot) or W1 emits it with
    ANY relation. The gap = pairs open finds that neither 20-set mode connects.
    Still an upper bound — open mode emits trivial/atomic pairs — so pairs are
    sampled for inspection and the gold review is the ground truth.
    """
    w1_by_id = {r["id"]: r for r in w1_rows}
    open_by_id = {r["id"]: r for r in open_rows}
    ids = [pid for pid in open_by_id if pid in pilot_by_id or pid in w1_by_id]
    open_only: list[tuple] = []
    twenty_only: list[tuple] = []
    per = []
    total = {"open": 0, "twenty": 0, "both": 0, "open_only": 0, "twenty_only": 0}
    for pid in ids:
        open_p = _pairs(open_by_id.get(pid))
        pilot_p = _pairs(pilot_by_id.get(pid))
        w1_p = _pairs(w1_by_id.get(pid))
        twenty = pilot_p | w1_p
        both = open_p & twenty
        g, t = open_p - twenty, twenty - open_p
        open_only.extend((pid, s, tg) for s, tg in g)
        twenty_only.extend((pid, s, tg) for s, tg in t)
        per.append({"id": pid, "open": len(open_p), "twenty": len(twenty), "both": len(both), "open_only": len(g), "twenty_only": len(t)})
        total["open"] += len(open_p)
        total["twenty"] += len(twenty)
        total["both"] += len(both)
        total["open_only"] += len(g)
        total["twenty_only"] += len(t)
    return {
        **total,
        "open_only_rate": round(total["open_only"] / total["open"], 3) if total["open"] else 0,
        "per_passage": per,
        "open_only_samples": open_only[:30],
        "twenty_only_samples": twenty_only[:20],
        "caveat": "upper bound: open-mode noise (trivial/atomic pairs) inflates open_only; gold review is the ground truth",
    }


def compute_flip(open_rows: list[dict]) -> dict:
    flipped = [r["id"] for r in open_rows if r["status"] == "ok" and grounded_triplets(r["triplets"], r["text"])]
    return {"hard_negatives": len(open_rows), "flipped": len(flipped), "flip_rate": round(len(flipped) / len(open_rows), 3) if open_rows else 0}


def compute_w1(w1_rows: list[dict], pilot_by_id: dict) -> dict:
    per = []
    grounded_violations_w1 = 0
    for r in w1_rows:
        if r["status"] != "ok":
            continue
        grounded_violations_w1 += len(r["semantic_violations"])
        pilot = pilot_by_id.get(r["id"])
        if not pilot or pilot["status"] != "ok":
            continue
        pkeys = len(pilot["triplets"])
        wkeys = len(r["triplets"])
        matched = matched_count(r["triplets"], pilot["triplets"])
        per.append({
            "id": r["id"],
            "pilot_triplets": pkeys,
            "w1_triplets": wkeys,
            "matched": matched,
            "baseline_coverage": round(matched / pkeys, 3) if pkeys else 1.0,
            "agreement": round(matched / wkeys, 3) if wkeys else 0.0,
        })
    rows = [p for p in per if "pilot_triplets" in p]
    return {
        "passages": len(rows),
        "pilot_mean": _mean([p["pilot_triplets"] for p in rows]),
        "pilot_sd": _sd([p["pilot_triplets"] for p in rows]),
        "w1_mean": _mean([p["w1_triplets"] for p in rows]),
        "w1_sd": _sd([p["w1_triplets"] for p in rows]),
        "matched_mean": _mean([p["matched"] for p in rows]),
        "baseline_coverage_mean": _mean([p["baseline_coverage"] for p in rows]),
        "agreement_mean": _mean([p["agreement"] for p in rows]),
        "grounding_violations_w1": grounded_violations_w1,
        "per_passage": rows,
    }


def compute_gold(gold_rows: list[dict], probe_results: list[dict], modes=("pilot", "w1")) -> dict:
    """Precision/recall of each mode against human gold triplets."""
    gold_by_id = {r["passage_id"]: [triplet_key(t) for t in r["gold"]] for r in gold_rows}
    by_mode: dict[str, dict[str, list[dict]]] = {"pilot": {}, "w1": {}}
    for rec in probe_results:
        if rec["passage_id"] in gold_by_id and rec["probe"] in by_mode:
            by_mode[rec["probe"]].setdefault(rec["passage_id"], []).append(rec)
    out = {}
    for mode in modes:
        precs, recs = [], []
        for pid, gold_keys in gold_by_id.items():
            mode_rows = by_mode[mode].get(pid, [])
            keys = Counter()
            for rec in mode_rows:
                if rec["status"] == "ok":
                    keys.update(triplet_key(t) for t in rec["triplets"])
            gold_counter = Counter(gold_keys)
            matched = sum((keys & gold_counter).values())
            precs.append(matched / len(keys) if keys else 0.0)
            recs.append(matched / len(gold_counter) if gold_counter else 0.0)
        out[mode] = {"precision": _mean(precs), "recall": _mean(recs), "f1": _mean([2 * p * r / (p + r) if p + r else 0 for p, r in zip(precs, recs)])}
    return out


# --------------------------------------------------------------------------- #
# Gold review material
# --------------------------------------------------------------------------- #

def build_union_triplets(pid: str, pilot_by_id: dict, open_by_id: dict, w1_by_id: dict) -> list[dict]:
    seen: dict[tuple, dict] = {}
    sources = [("pilot", pilot_by_id.get(pid)), ("w1", w1_by_id.get(pid)), ("open", open_by_id.get(pid))]
    for mode, rec in sources:
        if not rec or rec["status"] != "ok":
            continue
        ts = rec["triplets"] if mode != "open" else grounded_triplets(rec["triplets"], rec["text"])
        for t in ts:
            key = triplet_key(t)
            if key not in seen:
                entry = {"src": t["source"]["title"], "rel": t["relation"]["type"], "tgt": t["target"]["title"], "modes": []}
                seen[key] = entry
            seen[key]["modes"].append(mode)
    return list(seen.values())


def render_gold_review(gold_ids: list[str], C: list[dict], pilot_by_id: dict, open_by_id: dict, w1_by_id: dict, out_path: Path) -> Path:
    passages = []
    for pid in gold_ids:
        passage = next(r for r in C if r["id"] == pid)
        passages.append({
            "id": pid,
            "source": passage["source"],
            "text": passage["text"],
            "triplets": build_union_triplets(pid, pilot_by_id, open_by_id, w1_by_id),
        })
    payload = json.dumps({"passages": passages}, ensure_ascii=False)
    html = _GOLD_TEMPLATE.replace("__PAYLOAD__", payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


_GOLD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><title>Gold review — 5 passages</title>
<style>
 body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
 .card { border: 1px solid #e4e4e7; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1.2rem; }
 .meta { color: #71717a; font-size: .8rem; }
 pre { background: #fafafa; border: 1px solid #e4e4e7; border-radius: 8px; padding: .8rem; font-size: .8rem; white-space: pre-wrap; }
 label.t { display: block; margin: .35rem 0; cursor: pointer; }
 label.t span.badge { font-size: .7rem; color: #71717a; margin-left: .4rem; }
 textarea { width: 100%; min-height: 60px; font-size: .8rem; margin-top: .5rem; }
 #export { background: #18181b; color: #fff; border: none; border-radius: 8px; padding: .5rem 1.2rem; cursor: pointer; }
 .gold { background: #dcfce7; }
</style>
</head>
<body>
<h1>Gold triplet review (5 passages)</h1>
<p>Tick every triplet that is TRUE, grounded in the text, <b>and represents key information</b> — not trivial or atomic fragments (e.g. "X eats Y" enumerations, figure labels, or phrase splits). Unticked = not gold. Add missing <i>key</i> triplets via the JSON box per passage. Then export.</p>
<button id="export">Export gold.jsonl</button>
<div id="root"></div>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const root = document.getElementById('root');
const state = {}; // passage_id -> {checked:Set, added:[]}
data.passages.forEach(p => state[p.id] = { checked: new Set(), added: [] });

function render() {
  root.innerHTML = '';
  data.passages.forEach(p => {
    const card = document.createElement('div'); card.className = 'card';
    const meta = document.createElement('div'); meta.className = 'meta';
    meta.textContent = p.id + ' · ' + p.source;
    const details = document.createElement('details');
    const sum = document.createElement('summary'); sum.textContent = 'passage';
    const pre = document.createElement('pre'); pre.textContent = p.text;
    details.append(sum, pre);
    card.append(meta, details);
    p.triplets.forEach((t, i) => {
      const lab = document.createElement('label'); lab.className = 't';
      const cb = document.createElement('input'); cb.type = 'checkbox';
      cb.checked = state[p.id].checked.has(i);
      cb.onchange = () => { state[p.id].checked.has(i) ? state[p.id].checked.delete(i) : state[p.id].checked.add(i); lab.classList.toggle('gold', cb.checked); };
      lab.append(cb);
      lab.append(document.createTextNode(` ${t.src} -[${t.rel}]- ${t.tgt}`));
      const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = t.modes.join(',');
      lab.append(badge);
      if (cb.checked) lab.classList.add('gold');
      card.append(lab);
    });
    const ta = document.createElement('textarea');
    ta.placeholder = 'optional: extra gold triplets as JSON array, e.g. [{"source":{"title":"X","type":"entity"},"relation":{"type":"part_of","weight":0.8},"target":{"title":"Y","type":"concept"}}]';
    ta.oninput = () => { try { state[p.id].added = JSON.parse(ta.value || '[]'); } catch (e) {} };
    card.append(ta);
    root.append(card);
  });
}

document.getElementById('export').onclick = () => {
  const lines = data.passages.map(p => {
    const gold = [];
    state[p.id].checked.forEach(i => { const t = p.triplets[i]; gold.push({source:{title:t.src,type:'entity'},relation:{type:t.rel,weight:0.5},target:{title:t.tgt,type:'entity'}}); });
    state[p.id].added.forEach(t => gold.push(t));
    return JSON.stringify({passage_id: p.id, gold});
  });
  const blob = new Blob([lines.join('\n')], {type: 'application/x-ndjson'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'gold.jsonl'; a.click();
};
render();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def run(args) -> dict:
    data_dir = Path(args.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    pilot = read_jsonl(Path(args.labels))
    pilot_by_id = {r["id"]: r for r in pilot}
    hard_neg_ids = {r["id"] for r in pilot if r["hard_negative"]}

    full = sample_passages(corpus, n=args.sample_size, seed=args.seed)
    C: list[dict] = []
    for r in full[args.offset:]:
        if len(C) >= args.count:
            break
        if r["id"] in hard_neg_ids:
            continue
        C.append(r)
    H = [r for r in full if r["id"] in hard_neg_ids]
    print(f"comparison set C={len(C)} (offset {args.offset}), hard negatives H={len(H)}", flush=True)

    budget = Budget(args.budget)
    transport = http_transport(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        enable_thinking=not args.no_thinking,
    )
    open_client = TeacherClient(transport, system_prompt=open_system_prompt(), extractor=open_extractor, budget=budget, max_tokens=args.max_tokens)
    w1_client = TeacherClient(transport, system_prompt=w1_system_prompt(), extractor=w1_extractor, budget=budget, max_tokens=args.max_tokens)

    started = time.time()
    done = {"n": 0}
    total = len(C) * 2 + len(H)

    def progress(mode: str):
        def cb(rec: dict) -> None:
            done["n"] += 1
            extra = f"{len(rec['triplets'])} triplets" if rec["status"] == "ok" else f"failed: {rec['failure_reason']}"
            print(f"[{mode} {done['n']}/{total}] {rec['id']} {rec['status']} {extra}  budget {budget.used}/{args.budget}  elapsed {time.time() - started:.0f}s", flush=True)
        return cb

    print(f"probe open (C, {len(C)}) …", flush=True)
    open_c, _ = label_tasks(open_client, C, args.workers, progress=progress("openC"))
    print(f"probe open (H, {len(H)}) …", flush=True)
    open_h, _ = label_tasks(open_client, H, args.workers, progress=progress("openH"))
    print(f"probe w1 (C, {len(C)}) …", flush=True)
    w1_c, _ = label_tasks(w1_client, C, args.workers, progress=progress("w1"))

    def tag(probe, rows):
        for r in rows:
            r = dict(r)
            r["probe"] = probe
            yield r

    probe_rows = list(tag("pilot", pilot)) + list(tag("openC", open_c)) + list(tag("openH", open_h)) + list(tag("w1", w1_c))
    outputs = Path(args.outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    with open(outputs / "probe_results.jsonl", "w", encoding="utf-8") as f:
        for r in probe_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "relation_gap": compute_gap(open_c),
        "effective_gap": compute_effective_gap(open_c, pilot_by_id, w1_c),
        "flip": compute_flip(open_h),
        "w1": compute_w1(w1_c, pilot_by_id),
    }
    with open(outputs / "ontology_gap.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    if args.gold_review:
        gold_ids = [r["id"] for r in C[: args.gold_count]]
        open_by_id = {r["id"]: r for r in open_c}
        w1_by_id = {r["id"]: r for r in w1_c}
        gold_path = render_gold_review(gold_ids, C, pilot_by_id, open_by_id, w1_by_id, outputs / "gold_review.html")
        stats["gold_review"] = str(gold_path)

    return {"budget_used": budget.used, "comparison_set": len(C), "hard_negatives": len(H), **stats}


def recompute(args) -> dict:
    """Recompute stats + gold review from existing probe_results.jsonl (no API calls)."""
    outputs = Path(args.outputs)
    probe_rows = read_jsonl(outputs / "probe_results.jsonl")
    by_probe: dict[str, list[dict]] = {}
    for r in probe_rows:
        by_probe.setdefault(r["probe"], []).append(r)
    pilot = read_jsonl(Path(args.labels))
    pilot_by_id = {r["id"]: r for r in pilot}

    stats = {
        "relation_gap": compute_gap(by_probe.get("openC", [])),
        "effective_gap": compute_effective_gap(by_probe.get("openC", []), pilot_by_id, by_probe.get("w1", [])),
        "flip": compute_flip(by_probe.get("openH", [])),
        "w1": compute_w1(by_probe.get("w1", []), pilot_by_id),
    }
    with open(outputs / "ontology_gap.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    if args.gold_review:
        C = [r for r in by_probe.get("openC", []) if r["status"] == "ok"]
        gold_ids = [r["id"] for r in C[: args.gold_count]]
        open_by_id = {r["id"]: r for r in by_probe.get("openC", [])}
        w1_by_id = {r["id"]: r for r in by_probe.get("w1", [])}
        gold_path = render_gold_review(gold_ids, C, pilot_by_id, open_by_id, w1_by_id, outputs / "gold_review.html")
        stats["gold_review"] = str(gold_path)
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--labels", default=str(DEFAULT_DATA_DIR / "03_labels_pilot.jsonl"))
    parser.add_argument("--outputs", default=str(DEFAULT_OUTPUTS))
    parser.add_argument("--offset", type=int, default=10, help="skip first N stratified passages (mixed-provenance pilot batch)")
    parser.add_argument("--count", type=int, default=20, help="comparison set size")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--gold-review", action="store_true", help="generate gold_review.html for the first N passages of C")
    parser.add_argument("--gold-count", type=int, default=5)
    parser.add_argument("--gold", default=None, help="path to gold.jsonl to compute precision/recall")
    parser.add_argument("--stats-only", action="store_true", help="recompute stats + gold review from existing probe_results.jsonl, no API")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass
    if args.api_key is None:
        args.api_key = os.environ.get("DASHSCOPE_API_KEY")
    if args.api_key is None:
        raise SystemExit("DASHSCOPE_API_KEY is required (set it in .env)")
    from dataset.teacher import DEFAULT_BASE_URL, DEFAULT_MODEL

    args.model = args.model or DEFAULT_MODEL
    args.base_url = args.base_url or DEFAULT_BASE_URL

    if args.gold:
        probe_rows = read_jsonl(Path(args.outputs) / "probe_results.jsonl")
        gold = read_jsonl(Path(args.gold))
        print(json.dumps({"gold": compute_gold(gold, probe_rows)}, indent=2))
        return 0

    if args.stats_only:
        print(json.dumps(recompute(args), indent=2, ensure_ascii=False))
        return 0

    print(json.dumps(run(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
