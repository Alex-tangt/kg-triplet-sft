"""Human review for the pilot: a self-contained static report + verdict store.

``report`` renders ``outputs/03_pilot_review.html`` from ``03_labels_pilot.jsonl``:
every flagged case (semantic violations, schema/network failures) plus a seeded
random sample, each as a card with the passage, the teacher's raw output, the
parsed triplets and the violations. The page is fully client-side — verdicts live
in ``localStorage`` and are exported as ``verdicts.jsonl`` via a download button,
so no server ever runs.

``ingest`` loads that file into SQLite (``outputs/review.sqlite3``). The table
schema is deliberately generic (``sample_id``/``verdict``/``notes``/``reviewer``)
so the ticket 11 gold audit — human on 30, LLM on all 100, inter-rater agreement —
reuses the same store with only a different queue source.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.io import read_jsonl  # noqa: E402
from dataset.semantic import violation_counts  # noqa: E402

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def build_review_queue(rows: list[dict], random_sample: int = 20, seed: int = 0) -> list[dict]:
    """Flagged cases first, then a seeded random sample of the rest."""
    flagged = [r for r in rows if r["semantic_violations"] or r["status"] != "ok"]
    flagged_ids = {r["id"] for r in flagged}
    rest = [r for r in rows if r["id"] not in flagged_ids]
    rng = random.Random(seed)
    rng.shuffle(rest)
    return flagged + rest[:random_sample]


def _card_data(rows: list[dict]) -> list[dict]:
    queue = build_review_queue(rows)
    return [
        {
            "id": r["id"],
            "source": r["source"],
            "word_count": r["word_count"],
            "status": r["status"],
            "failure_reason": r["failure_reason"],
            "attempts": r["attempts"],
            "hard_negative": r["hard_negative"],
            "text": r["text"],
            "teacher_raw": r["teacher_raw"],
            "triplets": r["triplets"],
            "schema_errors": r["schema_errors"],
            "violations": r["semantic_violations"],
        }
        for r in queue
    ]


def _stats(rows: list[dict]) -> dict:
    return {
        "total": len(rows),
        "ok": sum(1 for r in rows if r["status"] == "ok"),
        "failed": sum(1 for r in rows if r["status"] != "ok"),
        "failure_reasons": _count(rows, lambda r: r["failure_reason"]),
        "hard_negatives": sum(1 for r in rows if r["hard_negative"]),
        "violations": violation_counts([v for r in rows for v in r["semantic_violations"]]),
        "relation_usage": relation_usage(rows),
    }


def relation_usage(rows: list[dict]) -> dict[str, int]:
    """Histogram of relation types across all parsed triplets ('' = no triplets)."""
    counts: dict[str, int] = {}
    for r in rows:
        if r["status"] != "ok":
            counts[""] = counts.get("", 0) + 1
            continue
        triplets = r.get("triplets") or []
        if not triplets:
            counts[""] = counts.get("", 0) + 1
            continue
        for t in triplets:
            rel = (t.get("relation") or {}).get("type")
            counts[rel if isinstance(rel, str) else ""] = counts.get(rel if isinstance(rel, str) else "", 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _count(rows: list[dict], key) -> dict:
    counts: dict = {}
    for r in rows:
        counts[key(r)] = counts.get(key(r), 0) + 1
    return counts


def render_report(rows: list[dict], out_path: Path) -> Path:
    payload = json.dumps({"stats": _stats(rows), "cards": _card_data(rows)}, ensure_ascii=False)
    html = _TEMPLATE.replace("__PAYLOAD__", payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>KG pilot review</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 900px; padding: 0 1rem; color: #1a1a1a; }
  .stats { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
  .stat { background: #f4f4f5; border-radius: 8px; padding: .5rem .9rem; font-size: .85rem; }
  .stat b { display: block; font-size: 1.2rem; }
  .card { border: 1px solid #e4e4e7; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
  .card.reviewed { border-left: 4px solid #22c55e; }
  .card .meta { color: #71717a; font-size: .8rem; margin-bottom: .5rem; }
  .badge { display: inline-block; border-radius: 999px; padding: .1rem .6rem; font-size: .72rem; margin-right: .3rem; }
  .badge.fail { background: #fee2e2; color: #b91c1c; }
  .badge.hardneg { background: #ede9fe; color: #6d28d9; }
  .badge.ok { background: #dcfce7; color: #15803d; }
  .badge.violation { background: #fef3c7; color: #b45309; }
  details { margin-top: .6rem; }
  summary { cursor: pointer; color: #3b82f6; font-size: .9rem; }
  pre { background: #fafafa; border: 1px solid #e4e4e7; border-radius: 8px; padding: .8rem; overflow-x: auto; font-size: .8rem; }
  .violation { border-left: 3px solid #f59e0b; background: #fffbeb; padding: .5rem .8rem; margin: .4rem 0; font-size: .85rem; }
  .verdict { margin-top: .8rem; display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; }
  .verdict button { border: 1px solid #d4d4d8; background: #fff; border-radius: 8px; padding: .35rem 1rem; cursor: pointer; font-size: .85rem; }
  .verdict button.accept.on { background: #dcfce7; border-color: #22c55e; }
  .verdict button.reject.on { background: #fee2e2; border-color: #ef4444; }
  .verdict input { border: 1px solid #d4d4d8; border-radius: 8px; padding: .3rem .6rem; font-size: .85rem; flex: 1; min-width: 160px; }
  #toolbar { display: flex; gap: .8rem; align-items: center; margin-bottom: 1rem; }
  #toolbar input { border: 1px solid #d4d4d8; border-radius: 8px; padding: .3rem .6rem; font-size: .85rem; }
  #export { background: #18181b; color: #fff; border: none; border-radius: 8px; padding: .45rem 1rem; cursor: pointer; font-size: .85rem; }
</style>
</head>
<body>
<h1>KG-triplet pilot review</h1>
<div class="stats" id="stats"></div>
<div id="toolbar">
  <input id="reviewer" placeholder="reviewer name" value="human">
  <b id="progress">0 / 0 reviewed</b>
  <button id="export">Export verdicts.jsonl</button>
</div>
<div id="cards"></div>
<script id="review-data" type="application/json">__PAYLOAD__</script>
<script>
const data = JSON.parse(document.getElementById('review-data').textContent);
const KEY = 'kg_review_verdicts';
let verdicts = {};
try { verdicts = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { verdicts = {}; }

const statsEl = document.getElementById('stats');
for (const [k, v] of Object.entries(data.stats)) {
  const el = document.createElement('span');
  el.className = 'stat';
  el.innerHTML = '<b>' + (v !== null && typeof v === 'object' ? JSON.stringify(v) : v) + '</b>' + k;
  statsEl.appendChild(el);
}

const cards = document.getElementById('cards');
function save() { localStorage.setItem(KEY, JSON.stringify(verdicts)); updateProgress(); }

function renderCard(c) {
  const el = document.createElement('div');
  el.className = 'card' + (verdicts[c.id] ? ' reviewed' : '');
  const badges = [];
  if (c.status !== 'ok') badges.push('<span class="badge fail">' + c.status + ': ' + (c.failure_reason || '') + '</span>');
  else badges.push('<span class="badge ok">ok</span>');
  if (c.hard_negative) badges.push('<span class="badge hardneg">hard negative</span>');
  const vcount = (c.violations || []).length;
  if (vcount) badges.push('<span class="badge violation">' + vcount + ' violation(s)</span>');
  const viols = (c.violations || []).map(v =>
    '<div class="violation">triplet[' + v.triplet_index + '] ' + v.rule + ' — ' + v.path + ': ' + v.detail + '</div>'
  ).join('');

  el.innerHTML = `
    <div class="meta">${c.id} · ${c.source} · ${c.word_count} words · ${c.attempts} attempt(s) ${badges.join('')}</div>
    <details><summary>passage</summary><pre>${escapeHtml(c.text)}</pre></details>
    <details><summary>teacher raw output</summary><pre>${escapeHtml(c.teacher_raw || '(none)')}</pre></details>
    ${viols}
    <details><summary>parsed triplets (${c.triplets.length})</summary><pre>${escapeHtml(JSON.stringify(c.triplets, null, 1))}</pre></details>
    <div class="verdict">
      <button class="accept${verdicts[c.id]?.verdict === 'accept' ? ' on' : ''}" data-id="${c.id}" data-v="accept">Accept</button>
      <button class="reject${verdicts[c.id]?.verdict === 'reject' ? ' on' : ''}" data-id="${c.id}" data-v="reject">Reject</button>
      <input data-notes="${c.id}" placeholder="notes" value="${verdicts[c.id]?.notes ? escapeAttr(verdicts[c.id].notes) : ''}">
    </div>`;
  cards.appendChild(el);
  el.querySelectorAll('.verdict button').forEach(b => b.onclick = () => {
    verdicts[b.dataset.id] = { verdict: b.dataset.v, notes: el.querySelector('input').value, ts: Date.now() };
    save(); renderCard(c);
  });
  el.querySelector('input').oninput = (e) => {
    if (!verdicts[c.id]) verdicts[c.id] = {};
    verdicts[c.id].notes = e.target.value; save();
  };
}

function escapeHtml(s) { return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function escapeAttr(s) { return escapeHtml(s).replace(/"/g, '&quot;'); }
function updateProgress() {
  const reviewed = Object.values(verdicts).filter(v => v && v.verdict).length;
  document.getElementById('progress').textContent = reviewed + ' / ' + data.cards.length + ' reviewed';
}

document.getElementById('export').onclick = () => {
  const reviewer = document.getElementById('reviewer').value || 'human';
  const lines = data.cards.map(c => {
    const v = verdicts[c.id] || {};
    return JSON.stringify({ sample_id: c.id, verdict: v.verdict || '', notes: v.notes || '', reviewer, created_at: new Date().toISOString() });
  });
  const blob = new Blob([lines.join('\n')], { type: 'application/x-ndjson' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'verdicts.jsonl';
  a.click();
};

data.cards.forEach(renderCard);
updateProgress();
</script>
</body>
</html>
"""


def ingest_verdicts(verdicts_path: Path, db_path: Path) -> dict:
    """Load ``verdicts.jsonl`` into the SQLite verdict store (upsert by sample_id)."""
    rows = read_jsonl(verdicts_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS verdicts (
            sample_id TEXT PRIMARY KEY,
            verdict TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            reviewer TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )"""
    )
    now = datetime.now(timezone.utc).isoformat()
    for r in rows:
        conn.execute(
            """INSERT INTO verdicts (sample_id, verdict, notes, reviewer, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(sample_id) DO UPDATE SET
                 verdict=excluded.verdict, notes=excluded.notes,
                 reviewer=excluded.reviewer, created_at=excluded.created_at""",
            (r["sample_id"], r.get("verdict", ""), r.get("notes", ""), r.get("reviewer", ""), r.get("created_at", now)),
        )
    conn.commit()
    accepted = conn.execute("SELECT COUNT(*) FROM verdicts WHERE verdict='accept'").fetchone()[0]
    rejected = conn.execute("SELECT COUNT(*) FROM verdicts WHERE verdict='reject'").fetchone()[0]
    conn.close()
    return {"verdicts_imported": len(rows), "accepted": accepted, "rejected": rejected}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report")
    p_report.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p_report.add_argument("--labels", default=None, help="03_labels_pilot.jsonl (default <data-dir>)")
    p_report.add_argument("--out", default=str(DEFAULT_OUTPUTS / "03_pilot_review.html"))
    p_report.add_argument("--random-sample", type=int, default=20)
    p_report.add_argument("--seed", type=int, default=0)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--verdicts", default=str(DEFAULT_OUTPUTS / "verdicts.jsonl"))
    p_ingest.add_argument("--db", default=str(DEFAULT_OUTPUTS / "review.sqlite3"))

    args = parser.parse_args(argv)
    if args.command == "report":
        data_dir = Path(args.data_dir)
        labels = Path(args.labels) if args.labels else data_dir / "03_labels_pilot.jsonl"
        rows = read_jsonl(labels)
        out = render_report(rows, Path(args.out))
        print(json.dumps({"report": str(out), "cards": len(build_review_queue(rows, args.random_sample, args.seed))}, indent=2))
    else:
        print(json.dumps({"ingest": ingest_verdicts(Path(args.verdicts), Path(args.db))}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
