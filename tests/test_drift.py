"""Drift guard: the contract's vocabularies must stay byte-identical to the
vendored harness constants. Loading the harness by path keeps it zero-change."""

import importlib.util
import sys
from pathlib import Path

from kg_contract import ENTITY_TYPES, RELATION_TYPES

HARNESS_SCRIPT = (
    Path(__file__).resolve().parent.parent / "eval" / "harness" / "scripts" / "evaluation.py"
)


def _load_harness_module():
    # Register in sys.modules: the harness uses `from __future__ import
    # annotations`, and its dataclasses resolve string annotations via the
    # class's registered module.
    spec = importlib.util.spec_from_file_location("hgr_evaluation", HARNESS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hgr_evaluation"] = module
    spec.loader.exec_module(module)
    return module


def test_harness_is_vendored():
    assert HARNESS_SCRIPT.exists()


def test_relation_set_matches_harness_constants():
    harness = _load_harness_module()
    assert RELATION_TYPES == harness.VALID_RELATION_TYPES


def test_entity_types_match_harness_constants():
    harness = _load_harness_module()
    assert ENTITY_TYPES == harness.VALID_ENTITY_TYPES
