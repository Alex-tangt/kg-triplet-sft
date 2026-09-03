"""Probe module: extractors, loose validation, matching, metrics, gold review."""

import json

from dataset.probe import (
    compute_effective_gap,
    compute_flip,
    compute_gap,
    compute_gold,
    compute_w1,
    extract_json_object,
    grounded_triplets,
    matched_count,
    open_extractor,
    open_system_prompt,
    render_gold_review,
    triplet_key,
    validate_open_list,
    w1_extractor,
    w1_system_prompt,
)
from kg_contract import RELATION_TYPES

OPEN_OK = '[{"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "successor_of", "weight": 0.5}, "target": {"title": "communism", "type": "concept"}}]'
W1_OK = '{"entities": [{"title": "Anarchism", "type": "concept"}], "triplets": [{"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "political philosophy", "type": "concept"}}]}'
PASSAGE = "Anarchism is a political philosophy. Successor of communism."


def test_extract_json_object_plain_fenced_prose():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('here: {"a": 1} then text') == {"a": 1}
    assert extract_json_object("no object here") is None


def test_open_extractor_accepts_novel_relations():
    triplets, errs = open_extractor(OPEN_OK)
    assert errs == []
    assert triplets[0]["relation"]["type"] == "successor_of"


def test_open_extractor_rejects_bad_shape():
    triplets, errs = open_extractor('[{"source": {"title": "x", "type": "work"}, "relation": {"type": "a", "weight": 1}, "target": {"title": "y", "type": "entity"}}]')
    assert triplets is None
    assert any("source" in e for e in errs)


def test_validate_open_list_loose_on_relation_set():
    ok, _ = validate_open_list([{"source": {"title": "a", "type": "entity"}, "relation": {"type": "successor_of", "weight": 0.5}, "target": {"title": "b", "type": "entity"}}])
    assert ok
    bad, errs = validate_open_list([{"source": {"title": "a", "type": "entity"}, "relation": {"type": "successor_of", "weight": 2.0}, "target": {"title": "b", "type": "entity"}}])
    assert not bad
    assert any("weight" in e for e in errs)


def test_w1_extractor_parses_two_section_output():
    triplets, errs = w1_extractor(W1_OK)
    assert errs == []
    assert len(triplets) == 1


def test_w1_extractor_rejects_missing_triplets_key():
    triplets, errs = w1_extractor('{"entities": []}')
    assert triplets is None
    assert any("triplets" in e for e in errs)


def test_w1_extractor_enforces_20_set():
    triplets, errs = w1_extractor('[{"source": {"title": "a", "type": "entity"}, "relation": {"type": "successor_of", "weight": 0.5}, "target": {"title": "b", "type": "entity"}}]')
    assert triplets is None


def test_triplet_key_normalizes():
    a = {"source": {"title": "Anarchism"}, "relation": {"type": "part_of"}, "target": {"title": "Political Philosophy"}}
    b = {"source": {"title": "anarchism,"}, "relation": {"type": "part_of"}, "target": {"title": "political-philosophy"}}
    assert triplet_key(a) == triplet_key(b)


def test_matched_count_intersects():
    a = [{"source": {"title": "X", "type": "entity"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "Y", "type": "entity"}}]
    b = [{"source": {"title": "X", "type": "entity"}, "relation": {"type": "part_of", "weight": 0.5}, "target": {"title": "Y", "type": "entity"}}]
    c = [{"source": {"title": "Z", "type": "entity"}, "relation": {"type": "part_of", "weight": 0.5}, "target": {"title": "Y", "type": "entity"}}]
    assert matched_count(a, b) == 1
    assert matched_count(a, c) == 0


def test_grounded_triplets_filters_ungrounded():
    triplets = [
        {"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "political philosophy", "type": "concept"}},
        {"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "Quantum Physics", "type": "concept"}},
    ]
    kept = grounded_triplets(triplets, PASSAGE)
    assert len(kept) == 1


def _ok_row(pid, triplets, text=PASSAGE):
    return {"id": pid, "source": "wikipedia", "status": "ok", "triplets": triplets, "text": text, "semantic_violations": []}


def test_compute_effective_gap_pair_level():
    # open uses a near-synonym (subdivided_into) — pair still covered by 20-set mode
    open_rows = [
        _ok_row("c1", [
            {"source": {"title": "Albania", "type": "entity"}, "relation": {"type": "subdivided_into", "weight": 0.8}, "target": {"title": "forests", "type": "concept"}},
            {"source": {"title": "Albania", "type": "entity"}, "relation": {"type": "party_to", "weight": 0.8}, "target": {"title": "CBD", "type": "concept"}},
        ]),
    ]
    pilot_by_id = {
        "c1": {"id": "c1", "status": "ok", "triplets": [
            {"source": {"title": "Albania", "type": "entity"}, "relation": {"type": "consists_of", "weight": 0.8}, "target": {"title": "forests", "type": "concept"}},
        ]},
    }
    w1_rows = []
    gap = compute_effective_gap(open_rows, pilot_by_id, w1_rows)
    assert gap["both"] == 1  # subdivided_into ~ consists_of does NOT inflate the gap
    assert gap["open_only"] == 1  # party_to->CBD genuinely not expressible
    assert gap["open_only_rate"] == 0.5


def test_compute_gap_counts_out_of_set():
    rows = [
        _ok_row("c1", [
            {"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "successor_of", "weight": 0.5}, "target": {"title": "communism", "type": "concept"}},
            {"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "political philosophy", "type": "concept"}},
        ]),
    ]
    gap = compute_gap(rows)
    assert gap["grounded_triplets"] == 2
    assert gap["out_of_set"] == 1
    assert gap["gap"] == 0.5
    assert "successor_of" in gap["missing_relations"]


def test_compute_flip():
    rows = [
        _ok_row("h1", [{"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "successor_of", "weight": 0.5}, "target": {"title": "communism", "type": "concept"}}]),
        {"id": "h2", "status": "ok", "triplets": [], "text": PASSAGE, "semantic_violations": []},
    ]
    flip = compute_flip(rows)
    assert flip == {"hard_negatives": 2, "flipped": 1, "flip_rate": 0.5}


def test_compute_w1_compares_coverage():
    w1_rows = [
        _ok_row("c1", [
            {"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "political philosophy", "type": "concept"}},
            {"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "communism", "type": "concept"}},
        ]),
    ]
    pilot_by_id = {
        "c1": {"id": "c1", "status": "ok", "triplets": [
            {"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "political philosophy", "type": "concept"}},
        ]},
    }
    stats = compute_w1(w1_rows, pilot_by_id)
    assert stats["w1_mean"] == 2.0
    assert stats["baseline_coverage_mean"] == 1.0
    assert stats["agreement_mean"] == 0.5


def test_compute_gold_precision_recall():
    gold = [{"passage_id": "c1", "gold": [
        {"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "political philosophy", "type": "concept"}},
        {"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "communism", "type": "concept"}},
    ]}]
    probe = [
        {"passage_id": "c1", "probe": "pilot", "status": "ok", "triplets": [
            {"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "political philosophy", "type": "concept"}},
        ]},
        {"passage_id": "c1", "probe": "w1", "status": "ok", "triplets": [
            {"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "political philosophy", "type": "concept"}},
            {"source": {"title": "Anarchism", "type": "concept"}, "relation": {"type": "part_of", "weight": 0.8}, "target": {"title": "communism", "type": "concept"}},
        ]},
    ]
    out = compute_gold(gold, probe)
    assert out["pilot"]["precision"] == 1.0
    assert out["pilot"]["recall"] == 0.5
    assert out["w1"]["recall"] == 1.0


def test_render_gold_review_embeds_payload(tmp_path):
    C = [_ok_row("c1", []), _ok_row("c2", [])]
    out = render_gold_review(["c1"], C, {}, {}, {}, tmp_path / "gold.html")
    html = out.read_text(encoding="utf-8")
    assert "gold.jsonl" in html
    assert "c1" in html


def test_open_system_prompt_allows_free_relations():
    p = open_system_prompt()
    assert "NOT restricted" in p
    assert "relation_dictionary" not in p.split("\n\n")[0][:10]


def test_w1_system_prompt_lists_all_20_relations():
    p = w1_system_prompt()
    for rel in RELATION_TYPES:
        assert rel in p
    assert "Stage A" in p and "Stage B" in p
