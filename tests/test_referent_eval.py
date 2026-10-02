"""Edge-score decomposition: entity-pairing ceiling vs edge-generation loss."""

import json

from eval.referent_eval import main, passage_metrics


def _teacher(rels):
    return {
        "id": "p1",
        "text": "ALPHA relates to BETA and to GAMMA in the passage.",
        "entities": [
            {"title": "ALPHA", "type": "concept", "description": "a"},
            {"title": "BETA", "type": "concept", "description": "b"},
            {"title": "GAMMA", "type": "concept", "description": "c"},
        ],
        "relationships": rels,
    }


def _student(rels):
    return {
        "id": "p1",
        "entities": [
            {"title": "ALPHA", "type": "concept", "description": "a"},
            {"title": "BETA", "type": "concept", "description": "b"},
            {"title": "XGAMMA", "type": "concept", "description": "x"},
        ],
        "relationships": rels,
    }


def test_decomposition_splits_pairing_and_generation_loss():
    teacher = _teacher([
        {"source": "ALPHA", "target": "BETA", "description": "ab", "strength": 5},
        {"source": "BETA", "target": "GAMMA", "description": "bc", "strength": 5},
        {"source": "ALPHA", "target": "GAMMA", "description": "ac", "strength": 5},
    ])
    # student reproduces ALPHA-BETA; GAMMA is a referent miss (XGAMMA unmatched)
    student = _student([
        {"source": "ALPHA", "target": "BETA", "description": "ab", "strength": 5},
        {"source": "ALPHA", "target": "XGAMMA", "description": "ax", "strength": 5},
    ])
    m = passage_metrics(teacher, student, [[0, 0], [1, 1]])

    assert m["n_teacher_rels"] == 3
    assert m["n_student_rels"] == 2
    assert m["edge_eligible_teacher"] == 1
    assert m["edge_eligible_student"] == 1
    assert m["edge_tp_t"] == 1
    assert m["edge_tp_s"] == 1


def test_summary_attribution_is_exact_and_pooled(tmp_path):
    teacher = _teacher([
        {"source": "ALPHA", "target": "BETA", "description": "ab", "strength": 5},
        {"source": "BETA", "target": "GAMMA", "description": "bc", "strength": 5},
        {"source": "ALPHA", "target": "GAMMA", "description": "ac", "strength": 5},
    ])
    student = _student([
        {"source": "ALPHA", "target": "BETA", "description": "ab", "strength": 5},
        {"source": "ALPHA", "target": "XGAMMA", "description": "ax", "strength": 5},
    ])
    t = tmp_path / "teacher.jsonl"
    s = tmp_path / "student.jsonl"
    pr = tmp_path / "pairs.jsonl"
    out = tmp_path / "report.json"
    t.write_text(json.dumps(teacher) + "\n", encoding="utf-8")
    s.write_text(json.dumps(student) + "\n", encoding="utf-8")
    pr.write_text(json.dumps({"id": "p1", "status": "ok", "pairs": [[0, 0], [1, 1]]}) + "\n", encoding="utf-8")

    assert main(["--teacher", str(t), "--student", str(s), "--pairs", str(pr), "--out", str(out)]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))["summary"]

    micro = summary["micro"]
    assert micro["edge_recall"] == 0.3333
    assert micro["edge_recall_ceiling"] == 0.3333
    assert micro["edge_conditional_recall"] == 1.0
    assert micro["edge_conditional_precision"] == 1.0
    assert micro["edge_student_endpoint_match_rate"] == 0.5

    attr = summary["edge_attribution"]
    assert attr["recall"] == 0.3333
    assert abs(attr["loss_from_entity_pairing"] + attr["loss_from_edge_generation"] - attr["loss_total"]) < 1e-9


def test_edge_generation_is_the_only_loss_when_endpoints_match(tmp_path):
    teacher = _teacher([{"source": "ALPHA", "target": "BETA", "description": "ab", "strength": 5}])
    student = _student([])
    t = tmp_path / "teacher.jsonl"
    s = tmp_path / "student.jsonl"
    pr = tmp_path / "pairs.jsonl"
    out = tmp_path / "report.json"
    t.write_text(json.dumps(teacher) + "\n", encoding="utf-8")
    s.write_text(json.dumps(student) + "\n", encoding="utf-8")
    pr.write_text(json.dumps({"id": "p1", "status": "ok", "pairs": [[0, 0], [1, 1]]}) + "\n", encoding="utf-8")

    assert main(["--teacher", str(t), "--student", str(s), "--pairs", str(pr), "--out", str(out)]) == 0
    attr = json.loads(out.read_text(encoding="utf-8"))["summary"]["edge_attribution"]

    assert attr["loss_total"] == 1.0
    assert attr["loss_from_entity_pairing"] == 0.0
    assert attr["loss_from_edge_generation"] == 1.0
