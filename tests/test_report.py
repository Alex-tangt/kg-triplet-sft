"""Static review report + SQLite verdict ingest."""

import json
import sqlite3

from dataset.report import build_review_queue, ingest_verdicts, relation_usage, render_report

BASE = {
    "id": "wikipedia-00001",
    "source": "wikipedia",
    "title": "Anarchism",
    "word_count": 10,
    "text": "Anarchism is a political philosophy.",
    "status": "ok",
    "failure_reason": None,
    "attempts": 1,
    "teacher_raw": "[]",
    "triplets": [],
    "hard_negative": True,
    "schema_errors": None,
    "semantic_violations": [],
}


def _rows():
    clean = {**BASE, "id": "wikipedia-00001", "hard_negative": True}
    flagged = {
        **BASE,
        "id": "wikipedia-00002",
        "hard_negative": False,
        "triplets": [{"source": {"title": "Anarchism", "type": "concept"}}],
        "semantic_violations": [
            {"triplet_index": 0, "rule": "grounding", "path": "triplets[0].source.title", "detail": "not grounded"}
        ],
    }
    failed = {**BASE, "id": "wikipedia-00003", "status": "failed", "failure_reason": "schema", "teacher_raw": None, "hard_negative": False}
    return [clean, flagged, failed]


def test_queue_puts_flagged_first_then_seeded_sample():
    queue = build_review_queue(_rows(), random_sample=20, seed=0)
    assert [r["id"] for r in queue[:2]] == ["wikipedia-00002", "wikipedia-00003"]


def test_report_renders_self_contained_html(tmp_path):
    out = render_report(_rows(), tmp_path / "report.html")
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "wikipedia-00002" in html
    assert "grounding" in html
    assert "verdicts.jsonl" in html


def test_report_embeds_all_cards_as_json(tmp_path):
    out = render_report(_rows(), tmp_path / "report.html")
    payload = out.read_text(encoding="utf-8").split('type="application/json">')[1].split("</script>")[0]
    data = json.loads(payload)
    assert len(data["cards"]) == 3
    assert data["stats"]["hard_negatives"] == 1


def test_relation_usage_histogram():
    rows = [
        {**BASE, "id": "a", "triplets": [
            {"relation": {"type": "part_of", "weight": 0.8}},
            {"relation": {"type": "part_of", "weight": 0.8}},
            {"relation": {"type": "used_for", "weight": 0.5}},
        ]},
        {**BASE, "id": "b", "hard_negative": True},
    ]
    usage = relation_usage(rows)
    assert usage["part_of"] == 2
    assert usage["used_for"] == 1
    assert usage[""] == 1


def test_report_embeds_relation_usage(tmp_path):
    out = render_report(_rows(), tmp_path / "report.html")
    payload = out.read_text(encoding="utf-8").split('type="application/json">')[1].split("</script>")[0]
    data = json.loads(payload)
    assert "relation_usage" in data["stats"]


def test_ingest_upserts_and_counts(tmp_path):
    db = tmp_path / "review.sqlite3"
    verdicts = tmp_path / "verdicts.jsonl"
    rows = [
        {"sample_id": "wikipedia-00002", "verdict": "accept", "notes": "", "reviewer": "human"},
        {"sample_id": "wikipedia-00003", "verdict": "reject", "notes": "schema fail", "reviewer": "human"},
    ]
    verdicts.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    first = ingest_verdicts(verdicts, db)
    assert first == {"verdicts_imported": 2, "accepted": 1, "rejected": 1}

    rows[1]["verdict"] = "accept"
    verdicts.write_text(json.dumps(rows[1]) + "\n", encoding="utf-8")
    second = ingest_verdicts(verdicts, db)
    assert second == {"verdicts_imported": 1, "accepted": 2, "rejected": 0}

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0] == 2
    conn.close()


def test_verdict_schema_is_generalizable_to_audit(tmp_path):
    db = tmp_path / "review.sqlite3"
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(
        json.dumps({"sample_id": "audit-0007", "verdict": "accept", "notes": "rubric ok", "reviewer": "qwen3-flash"}) + "\n",
        encoding="utf-8",
    )
    ingest_verdicts(verdicts, db)
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT sample_id, verdict, reviewer FROM verdicts").fetchone()
    assert row == ("audit-0007", "accept", "qwen3-flash")
    conn.close()
