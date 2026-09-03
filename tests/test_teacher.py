"""Teacher tracer: client retry/backoff, budget hard stop, schema-fail semantics,
parallel deterministic labeling, and stratified sampling."""

import json
import time

import pytest

from dataset.teacher import (
    Budget,
    BudgetExhausted,
    NonRetryableError,
    RetryableError,
    TeacherClient,
    build_request_body,
    label_tasks,
    prepare_resume,
    sample_passages,
)

SYSTEM = "system prompt"

VALID = '[{"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "political philosophy", "type": "concept"}}]'
VALID_WITH_WEIGHT_VIOLATION = VALID.replace("0.8", "0.3")
PASSAGE = "Anarchism is a political philosophy and movement."


def client_for(transport, **kw):
    return TeacherClient(transport, system_prompt=SYSTEM, backoff_base=0.01, **kw)


def test_valid_response_parses_to_ok():
    c = client_for(lambda m: VALID)
    rec = c.label(PASSAGE)
    assert rec["status"] == "ok"
    assert len(rec["triplets"]) == 1
    assert rec["schema_errors"] is None
    assert rec["hard_negative"] is False
    assert rec["semantic_violations"] == []


def test_weight_violation_is_reported_not_failed():
    c = client_for(lambda m: VALID_WITH_WEIGHT_VIOLATION)
    rec = c.label(PASSAGE)
    assert rec["status"] == "ok"
    assert rec["semantic_violations"][0]["rule"] == "weight"


def test_empty_array_is_hard_negative():
    c = client_for(lambda m: "[]")
    rec = c.label(PASSAGE)
    assert rec["status"] == "ok"
    assert rec["hard_negative"] is True


def test_markdown_fenced_output_parses():
    c = client_for(lambda m: "```json\n" + VALID + "\n```")
    assert c.label(PASSAGE)["status"] == "ok"


def test_schema_invalid_retried_then_failed_not_converted_to_empty():
    c = client_for(lambda m: '{"not": "an array"}', schema_retries=2)
    rec = c.label(PASSAGE)
    assert rec["status"] == "failed"
    assert rec["failure_reason"] == "schema"
    assert rec["attempts"] == 3
    assert rec["triplets"] == []
    assert rec["hard_negative"] is False  # must not be silently treated as []


def test_schema_recovers_after_transient_bad_output():
    responses = iter(['{"bad": true}', VALID])
    c = client_for(lambda m: next(responses), schema_retries=2)
    rec = c.label(PASSAGE)
    assert rec["status"] == "ok"
    assert rec["attempts"] == 2


def test_network_retries_then_succeeds():
    calls = {"n": 0}

    def flaky(m):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise RetryableError("timeout")
        return VALID

    c = client_for(flaky, network_retries=3)
    rec = c.label(PASSAGE)
    assert rec["status"] == "ok"
    assert rec["attempts"] == 1  # one label round, three internal retries
    assert calls["n"] == 4


def test_network_failure_exhausted_marks_failed():
    c = client_for(lambda m: (_ for _ in ()).throw(RetryableError("down")), network_retries=2)
    rec = c.label(PASSAGE)
    assert rec["status"] == "failed"
    assert rec["failure_reason"] == "network"


def test_non_retryable_http_error_marks_failed():
    c = client_for(lambda m: (_ for _ in ()).throw(NonRetryableError("HTTP 400")), network_retries=3)
    rec = c.label(PASSAGE)
    assert rec["status"] == "failed"
    assert rec["failure_reason"] == "http"


def test_context_guard_fails_on_oversized_input():
    c = client_for(lambda m: VALID, max_context_tokens=100)
    rec = c.label("x" * 10_000)
    assert rec["status"] == "failed"
    assert "context" in rec["schema_errors"][0]


def test_budget_exhausted_raises():
    c = client_for(lambda m: VALID, budget=Budget(1))
    assert c.label(PASSAGE)["status"] == "ok"
    with pytest.raises(BudgetExhausted):
        c.label(PASSAGE)


def test_request_body_enables_thinking_by_default():
    import json

    body = json.loads(build_request_body("m", [{"role": "user", "content": "x"}], 0.0, 2000))
    assert body["enable_thinking"] is True


def test_request_body_can_disable_thinking():
    import json

    body = json.loads(build_request_body("m", [{"role": "user", "content": "x"}], 0.0, 2000, enable_thinking=False))
    assert body["enable_thinking"] is False


def test_label_tasks_appends_incrementally_and_reports_progress(tmp_path):
    out = tmp_path / "labels.jsonl"
    c = client_for(lambda m: VALID)
    tasks = [{"id": f"p{i}", "source": "wikipedia", "title": "t", "word_count": 10, "text": PASSAGE} for i in range(4)]
    seen = []

    rows, _ = label_tasks(c, tasks, workers=2, out=out, progress=lambda rec: seen.append(rec["id"]))

    assert [r["id"] for r in rows] == ["p0", "p1", "p2", "p3"]
    assert seen == ["p0", "p1", "p2", "p3"]
    written = [json.loads(line) for line in open(out, encoding="utf-8")]
    assert [r["id"] for r in written] == ["p0", "p1", "p2", "p3"]


def test_label_tasks_incremental_write_survives_partial_run(tmp_path):
    """A budget-stopped run persists only the completed records; the rest can resume."""
    out = tmp_path / "labels.jsonl"
    c = client_for(lambda m: VALID, budget=Budget(2))
    tasks = [{"id": f"p{i}", "source": "wikipedia", "title": "t", "word_count": 10, "text": PASSAGE} for i in range(5)]

    rows, stopped = label_tasks(c, tasks, workers=2, out=out)

    assert stopped is True
    written = [json.loads(line) for line in open(out, encoding="utf-8")]
    assert len(written) == 2
    assert {r["id"] for r in written} <= {"p0", "p1", "p2", "p3", "p4"}


def test_custom_extractor_injected():
    def extractor(raw):
        return [{"source": {"title": "x", "type": "entity"}, "relation": {"type": "part_of", "weight": 0.5}, "target": {"title": "y", "type": "entity"}}], []

    c = TeacherClient(lambda m: "whatever", system_prompt=SYSTEM, extractor=extractor, backoff_base=0.01)
    assert c.label(PASSAGE)["status"] == "ok"


def test_prepare_resume_keeps_ok_records(tmp_path):
    out = tmp_path / "labels.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in [
        {**{"id": "a", "status": "ok"}, "x": 1},
        {**{"id": "b", "status": "failed"}, "x": 2},
    ]) + "\n", encoding="utf-8")

    existing, ids = prepare_resume(out, retry_failed=True)
    assert {r["id"] for r in existing} == {"a"}
    assert "b" not in ids
    persisted = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert [r["id"] for r in persisted] == ["a"]


def test_prepare_resume_without_retry_keeps_everything(tmp_path):
    out = tmp_path / "labels.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in [
        {**{"id": "a", "status": "ok"}, "x": 1},
        {**{"id": "b", "status": "failed"}, "x": 2},
    ]) + "\n", encoding="utf-8")

    existing, ids = prepare_resume(out, retry_failed=False)
    assert ids == {"a", "b"}


def test_budget_is_thread_safe():
    budget = Budget(50)
    c = client_for(lambda m: (time.sleep(0.001), VALID)[1], budget=budget)
    tasks = [{"id": f"p{i}", "source": "wikipedia", "title": "t", "word_count": 10, "text": PASSAGE} for i in range(200)]
    rows, stopped = label_tasks(c, tasks, workers=8)
    assert budget.used <= 50
    assert len(rows) <= 50
    assert stopped is True


def test_label_tasks_writes_in_deterministic_order():
    c = client_for(lambda m: VALID)
    tasks = [{"id": f"p{i}", "source": "wikipedia", "title": "t", "word_count": 10, "text": PASSAGE} for i in range(5)]
    rows, _ = label_tasks(c, tasks, workers=4)
    assert [r["id"] for r in rows] == [f"p{i}" for i in range(5)]


def _corpus(n_wiki=200, n_arxiv=100):
    rows = []
    for i in range(n_wiki):
        rows.append({"id": f"w{i}", "source": "wikipedia", "title": "t", "word_count": 100 + (i * 3) % 200, "text": "x"})
    for i in range(n_arxiv):
        rows.append({"id": f"a{i}", "source": "arxiv", "title": "t", "word_count": 100 + (i * 5) % 200, "text": "x"})
    return rows


def test_sample_is_60_40_and_deterministic():
    a = sample_passages(_corpus(), n=100, seed=0)
    b = sample_passages(_corpus(), n=100, seed=0)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    sources = [r["source"] for r in a]
    assert sources.count("wikipedia") == 60
    assert sources.count("arxiv") == 40


def test_sample_spans_length_terciles():
    corpus = _corpus()
    a = sample_passages(corpus, n=100, seed=1)
    counts = [r["word_count"] for r in a]
    lo = sum(1 for c in counts if c < 167)
    hi = sum(1 for c in counts if c >= 233)
    assert 15 <= lo <= 85
    assert hi >= 10


def test_incremental_batches_are_prefixes_of_same_draw():
    full = sample_passages(_corpus(), n=100, seed=0)
    assert full[:1][0]["id"] == sample_passages(_corpus(), n=100, seed=0)[:1][0]["id"]
    assert [r["id"] for r in sample_passages(_corpus(), n=100, seed=0)[:10]] == [r["id"] for r in full[:10]]
