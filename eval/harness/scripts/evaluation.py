"""Knowledge-graph triplet evaluation pipeline.

This module scores a model's predicted knowledge-graph triplets against a gold
reference set.  A *triplet* has the shape::

    {
      "source":   {"title": str, "type": str},
      "relation": {"type": str, "weight": float},  # weight in [0, 1]
      "target":   {"title": str, "type": str},
    }

The pipeline (see the per-function docstrings for the original specification):

1. ``validate_schema``  - is each predicted triplet well formed? (highest-weighted axis)
2. ``normalize_triplets`` - strip whitespace / company suffixes so titles compare cleanly
3. ``hungarian_algorithm_for_one_one_matching`` - optimal 1-1 alignment of pred->gold
4. ``embedding_matching`` - rigid string match first, then a *lite & fast* embedding
   model for fuzzy entity/relation matching (accepted only when similarity >= 0.80),
   plus weight validation (|dw| <= 0.10 -> full, <= 0.20 -> partial, else 0)
5. ``final_hallucination_check`` - flag predicted triplets not grounded in gold/prompt,
   drop self-referential (source == target) triplets
6. ``scoring`` - multi-axis score; schema is weighted highest because the generating
   model is small and schema adherence is the cheapest signal to get right.

The embedding backend is pluggable and degrades gracefully: it prefers ``fastembed``
(ONNX, lightweight), then ``sentence-transformers``, and finally a pure-stdlib lexical
similarity so the pipeline always runs even with no model / no network.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# --------------------------------------------------------------------------- #
# Constants / configuration
# --------------------------------------------------------------------------- #

# Any relation type a model emits that is NOT in this set is treated as a
# hallucination: the model was never instructed to produce it.

VALID_RELATION_TYPES: frozenset[str] = frozenset({
    "implements", "trained_on", "evaluates", "part_of", "introduces",
    "extends", "depends_on", "contrasts_with", "applied_to", "measured_by",
    "founded_by", "developed_by", "defined_as", "consists_of", "is_type_of",
    "based_on", "used_for", "created_by", "located_in", "predecessor_of",
})

# The two entity types that appear in the data.  Kept for reference / reporting.
VALID_ENTITY_TYPES: frozenset[str] = frozenset({"entity", "concept"})

# Default dataset location (……/KG_triplet_evaluation/data/predict.jsonl).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "predict.jsonl"

# Fuzzy-match acceptance: an embedding (or lexical) similarity only counts as a
# match when it is at least this high (per the embedding_matching spec, 0.80).
DEFAULT_SIM_THRESHOLD = 0.80

# A Hungarian assignment is only treated as a real (gold, pred) pairing when the
# entity-level alignment similarity reaches this floor; below it the gold triplet
# is considered missed and the predicted triplet extra.
DEFAULT_ALIGN_ACCEPT = 0.50

# Weight-closeness bands (|pred_weight - gold_weight|).
WEIGHT_FULL_BAND = 0.10   # within 0.10  -> 1.0
WEIGHT_PARTIAL_BAND = 0.20  # within 0.20 -> 0.5, else 0.0

# A *lite and fast* embedding model, as the spec requests.  fastembed's default.
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Multi-axis scoring weights.  Schema is highest (small generator -> schema is the
# most important, cheapest-to-verify signal).  Axes that are not applicable for an
# entry (e.g. relation accuracy when nothing matched) are dropped and the rest are
# renormalised, so the composite always lives in [0, 1].
SCORE_WEIGHTS: dict[str, float] = {
    "schema": 0.30,       # highest weight
    "entity_f1": 0.25,    # do the endpoints (source/target) line up?
    "relation_acc": 0.20,  # given matched endpoints, is the relation right?
    "weight": 0.10,       # is the confidence weight close enough?
    "grounding": 0.15,    # 1 - hallucination_rate
}

# Company / legal suffixes stripped during normalisation ("removes any Ltd/Inc marks").
_COMPANY_SUFFIXES = [
    "incorporated", "corporation", "limited", "company",
    "inc", "llc", "ltd", "co", "corp", "plc", "gmbh", "ag", "sa", "nv", "bv",
]
_COMPANY_RE = re.compile(
    r"[\s,]+(?:" + "|".join(_COMPANY_SUFFIXES) + r")\.?$",
    flags=re.IGNORECASE,
)
_PUNCT_STRIP_RE = re.compile(r"^[\s\"'`\(\[\{.,;:]+|[\s\"'`\)\]\}.,;:]+$")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_entries(file_path: str | Path) -> list[dict]:
    """Load the JSONL dataset into a list of entry dicts.

    Each entry has keys: ``prompt``, ``gold_raw``, ``gold_parsed``,
    ``pred_raw``, ``pred_parsed``.
    """
    path = Path(file_path)
    entries: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, row in enumerate(f, start=1):
            row = row.strip()
            if not row:
                continue
            try:
                entries.append(json.loads(row))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                print(f"[warn] skipping malformed JSON on line {line_no}: {exc}",
                      file=sys.stderr)
    return entries


def get_pred_list(file_path: str | Path) -> list[list]:
    """Return the predicted triplet list for every entry (one inner list each).

    Fixes the original bugs: it opened the literal string ``"file_path"`` and
    double-nested every entry's list.
    """
    return [entry.get("pred_parsed") or [] for entry in load_entries(file_path)]


def get_gold_list(file_path: str | Path) -> list[list]:
    """Return the gold triplet list for every entry (one inner list each)."""
    return [entry.get("gold_parsed") or [] for entry in load_entries(file_path)]


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

def normalize_title(title: Any) -> str:
    """Normalise an entity title for comparison.

    Removes surrounding whitespace/punctuation, collapses internal whitespace,
    drops trailing company/legal suffixes (Ltd, Inc, Corp, LLC, ...), applies
    Unicode NFKC folding and lowercases.  Non-string titles normalise to "".
    """
    if not isinstance(title, str):
        return ""
    text = unicodedata.normalize("NFKC", title)
    text = text.replace("_", " ")
    text = _WS_RE.sub(" ", text).strip()
    text = text.lower()
    # strip trailing company suffix possibly more than once ("Foo Co., Ltd")
    prev = None
    while prev != text:
        prev = text
        text = _COMPANY_RE.sub("", text).strip()
    text = _PUNCT_STRIP_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_triplet(triplet: dict) -> dict:
    """Return a copy of a triplet with normalised source/target titles.

    The original ``title``/``type`` values are preserved under ``_raw`` so the
    report can still show what the model produced.
    """
    src = triplet.get("source") or {}
    tgt = triplet.get("target") or {}
    rel = triplet.get("relation") or {}
    return {
        "source": {
            "title": normalize_title(src.get("title")),
            "type": (src.get("type") or "").strip().lower(),
            "_raw": src.get("title"),
        },
        "target": {
            "title": normalize_title(tgt.get("title")),
            "type": (tgt.get("type") or "").strip().lower(),
            "_raw": tgt.get("title"),
        },
        "relation": {
            "type": (rel.get("type") or "").strip().lower(),
            "weight": rel.get("weight"),
        },
    }


def normalize_triplets(triplets: Sequence[dict]) -> list[dict]:
    """Normalise a list of (already schema-valid) triplet dicts.

    The original spec said: "passes the kg_triplets and normalize them, removes
    white spaces and removes any Ltd/Inc marks as well."  The original
    implementation took a file path and had an inverted ``.jsonl`` suffix guard
    that rejected exactly the files it should accept; normalisation operates on
    triplet dicts, which is what every downstream stage needs.
    """
    if not isinstance(triplets, (list, tuple)):
        return []
    return [normalize_triplet(t) for t in triplets if isinstance(t, dict)]


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #

def is_valid_triplet(item: Any) -> bool:
    """True when a single triplet conforms to the required schema."""
    if not isinstance(item, dict):
        return False
    for key in ("source", "target"):
        sub = item.get(key)
        if not isinstance(sub, dict):
            return False
        if not isinstance(sub.get("title"), str) or not isinstance(sub.get("type"), str):
            return False
    relation = item.get("relation")
    if not isinstance(relation, dict):
        return False
    if not isinstance(relation.get("type"), str):
        return False
    w = relation.get("weight")
    # bool is a subclass of int - exclude it explicitly.
    if isinstance(w, bool) or not isinstance(w, (int, float)):
        return False
    return 0.0 <= float(w) <= 1.0


def validate_schema(pred_raw: Any) -> tuple[bool, list[dict]]:
    """Strict, all-or-nothing schema check (kept from the original spec).

    Returns ``(True, triplets)`` only when *every* triplet is well formed,
    otherwise ``(False, [])``.  An empty list is valid (the prompt explicitly
    allows returning an empty array).
    """
    if not isinstance(pred_raw, list):
        return False, []
    for item in pred_raw:
        if not is_valid_triplet(item):
            return False, []
    return True, pred_raw


def valid_triplets(pred_raw: Any) -> tuple[list[dict], int, int]:
    """Partial-credit variant: keep only the well-formed triplets.

    Returns ``(kept, n_total, n_valid)`` where ``n_total`` is the number of
    list items the model produced and ``kept`` are the ones that pass schema.
    """
    if not isinstance(pred_raw, list):
        return [], 0, 0
    kept = [t for t in pred_raw if is_valid_triplet(t)]
    return kept, len(pred_raw), len(kept)


def schema_score(pred_raw: Any) -> float:
    """Fraction of produced items that are well formed (1.0 for an empty list)."""
    if not isinstance(pred_raw, list):
        return 0.0
    if not pred_raw:
        return 1.0
    n_valid = sum(1 for t in pred_raw if is_valid_triplet(t))
    return n_valid / len(pred_raw)


# --------------------------------------------------------------------------- #
# Embedding backend (pluggable, lite & fast, with lexical fallback)
# --------------------------------------------------------------------------- #

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def lexical_similarity(a: str, b: str) -> float:
    """Pure-stdlib similarity in [0, 1]: blends char ratio and token Jaccard."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(_TOKEN_RE.findall(a)), set(_TOKEN_RE.findall(b))
    if ta or tb:
        jacc = len(ta & tb) / len(ta | tb)
    else:
        jacc = 0.0
    return 0.5 * ratio + 0.5 * jacc


class EmbeddingBackend:
    """Lazy, cached text-similarity backend.

    Tries ``fastembed`` then ``sentence-transformers``; if neither is available
    (or the model fails to load), falls back to :func:`lexical_similarity` so the
    pipeline still runs.  ``backend`` reports which path is active.
    """

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self.backend = "lexical"
        self._model: Any = None
        self._loaded = False
        self._cache: dict[str, list[float]] = {}

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.enabled:
            self.backend = "lexical"
            return
        # 1) fastembed (ONNX, lightweight) ---------------------------------- #
        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
            self.backend = f"fastembed:{self.model_name}"
            return
        except Exception as exc:  # noqa: BLE001 - any failure -> try next backend
            print(f"[info] fastembed unavailable ({exc!r}); trying sentence-transformers",
                  file=sys.stderr)
        # 2) sentence-transformers ------------------------------------------ #
        try:
            from sentence_transformers import SentenceTransformer

            st_name = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
            self._model = SentenceTransformer(st_name)
            self.backend = f"sentence-transformers:{st_name}"
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[info] sentence-transformers unavailable ({exc!r}); "
                  f"falling back to lexical similarity", file=sys.stderr)
        # 3) lexical fallback ----------------------------------------------- #
        self._model = None
        self.backend = "lexical"

    # -- embedding ------------------------------------------------------------ #
    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        if self.backend.startswith("fastembed"):
            return [list(map(float, v)) for v in self._model.embed(texts)]
        if self.backend.startswith("sentence-transformers"):
            arr = self._model.encode(texts, normalize_embeddings=False)
            return [list(map(float, v)) for v in arr]
        return []  # lexical: no vectors

    def warm(self, texts: Iterable[str]) -> None:
        """Pre-compute and cache embeddings for a batch of texts (fast path)."""
        self._load()
        if self.backend == "lexical":
            return
        todo = sorted({t for t in texts if t and t not in self._cache})
        if not todo:
            return
        for text, vec in zip(todo, self._embed_uncached(todo)):
            self._cache[text] = vec

    def _vector(self, text: str) -> Optional[list[float]]:
        if self.backend == "lexical":
            return None
        if text not in self._cache:
            self.warm([text])
        return self._cache.get(text)

    # -- public similarity ---------------------------------------------------- #
    def similarity(self, a: str, b: str) -> float:
        """Semantic similarity in [0, 1] between two normalised strings."""
        self._load()
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0
        if self.backend == "lexical":
            return lexical_similarity(a, b)
        va, vb = self._vector(a), self._vector(b)
        if va is None or vb is None:
            return lexical_similarity(a, b)
        # cosine is in [-1, 1]; clamp negatives to 0 for a [0, 1] score.
        return max(0.0, _cosine(va, vb))


# --------------------------------------------------------------------------- #
# Pure-python Hungarian algorithm (optimal assignment, minimisation)
# --------------------------------------------------------------------------- #

def hungarian_min(cost: list[list[float]]) -> list[int]:
    """Solve the square assignment problem (minimise total cost).

    Classic O(n^3) Kuhn-Munkres (Jonker-Volgenant style potentials).  Returns
    ``assignment`` where ``assignment[row] = col``.  Avoids a SciPy dependency so
    the module is self-contained.
    """
    n = len(cost)
    if n == 0:
        return []
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)   # p[col] = row assigned to col (1-indexed; 0 = none)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


# --------------------------------------------------------------------------- #
# Triplet-level similarity + embedding matching (incl. weight validation)
# --------------------------------------------------------------------------- #

def relation_text(rel_type: str) -> str:
    """Turn a relation type into natural words for embedding ("based_on" -> "based on")."""
    return rel_type.replace("_", " ").strip()


def field_similarity(a: str, b: str, embedder: EmbeddingBackend,
                     threshold: float) -> float:
    """Rigid string match first, then embedding fuzzy match gated at ``threshold``.

    Returns 1.0 for an exact (normalised) match; otherwise the semantic
    similarity if it clears ``threshold``; otherwise 0.0.
    """
    if a == b and a != "":
        return 1.0
    sim = embedder.similarity(a, b)
    return sim if sim >= threshold else 0.0


def weight_score(gold_w: Any, pred_w: Any) -> float:
    """Weight validation: |dw| <= 0.10 -> 1.0, <= 0.20 -> 0.5, else 0.0."""
    if not isinstance(gold_w, (int, float)) or not isinstance(pred_w, (int, float)):
        return 0.0
    diff = abs(float(gold_w) - float(pred_w))
    if diff <= WEIGHT_FULL_BAND:
        return 1.0
    if diff <= WEIGHT_PARTIAL_BAND:
        return 0.5
    return 0.0


def embedding_matching(gold_t: dict, pred_t: dict, embedder: EmbeddingBackend,
                       threshold: float = DEFAULT_SIM_THRESHOLD) -> dict:
    """Score a single (gold, pred) triplet pair.

    Rigid python string matching is tried first for source/target/relation; any
    field that does not match exactly falls back to the embedding model and only
    counts if its similarity is >= ``threshold``.  Weight validation is performed
    here too.  Returns a dict of sub-scores used by both alignment and scoring.
    """
    g_src, g_tgt, g_rel = gold_t["source"], gold_t["target"], gold_t["relation"]
    p_src, p_tgt, p_rel = pred_t["source"], pred_t["target"], pred_t["relation"]

    src_sim = field_similarity(g_src["title"], p_src["title"], embedder, threshold)
    tgt_sim = field_similarity(g_tgt["title"], p_tgt["title"], embedder, threshold)

    # relation: exact type match, else embed the natural-language form.
    if g_rel["type"] and g_rel["type"] == p_rel["type"]:
        rel_sim = 1.0
    else:
        rel_sim = field_similarity(relation_text(g_rel["type"]),
                                   relation_text(p_rel["type"]),
                                   embedder, threshold)

    entity_sim = 0.5 * (src_sim + tgt_sim)
    entity_match = src_sim > 0.0 and tgt_sim > 0.0
    relation_match = rel_sim > 0.0
    return {
        "src_sim": src_sim,
        "tgt_sim": tgt_sim,
        "rel_sim": rel_sim,
        "entity_sim": entity_sim,
        "entity_match": entity_match,
        "relation_match": relation_match,
        "full_match": entity_match and relation_match,
        "type_agree": (g_src["type"] == p_src["type"]) and (g_tgt["type"] == p_tgt["type"]),
        "weight_score": weight_score(g_rel.get("weight"), p_rel.get("weight")),
        "align_sim": 0.45 * src_sim + 0.45 * tgt_sim + 0.10 * rel_sim,
    }


# --------------------------------------------------------------------------- #
# Hungarian 1-1 matching of pred -> gold for one entry
# --------------------------------------------------------------------------- #

@dataclass
class MatchResult:
    pairs: list[tuple[int, int, dict]] = field(default_factory=list)  # (gold_i, pred_j, scores)
    unmatched_gold: list[int] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)


def hungarian_algorithm_for_one_one_matching(
    gold: Sequence[dict],
    pred: Sequence[dict],
    embedder: EmbeddingBackend,
    threshold: float = DEFAULT_SIM_THRESHOLD,
    accept: float = DEFAULT_ALIGN_ACCEPT,
) -> MatchResult:
    """Optimal 1-1 alignment of predicted triplets to gold triplets.

    For one prompt we compare ``gold`` vs ``pred`` (both normalised).  A full
    pairwise similarity matrix is built (rigid string match first, embeddings for
    the fuzzy remainder) and the Hungarian algorithm finds the assignment that
    maximises total similarity.  Pairs whose entity-level alignment similarity is
    below ``accept`` are rejected: that gold triplet counts as missed and the
    predicted one as extra (a hallucination candidate handled downstream).
    """
    ng, npd = len(gold), len(pred)
    result = MatchResult()
    if ng == 0 and npd == 0:
        return result
    if ng == 0:
        result.unmatched_pred = list(range(npd))
        return result
    if npd == 0:
        result.unmatched_gold = list(range(ng))
        return result

    # Pre-compute pairwise scores.
    scores: list[list[dict]] = [[None] * npd for _ in range(ng)]  # type: ignore
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            scores[i][j] = embedding_matching(g, p, embedder, threshold)

    # Build a square cost matrix (pad with no-match dummies). cost = 1 - align_sim.
    n = max(ng, npd)
    cost = [[1.0] * n for _ in range(n)]
    for i in range(ng):
        for j in range(npd):
            cost[i][j] = 1.0 - scores[i][j]["align_sim"]

    assignment = hungarian_min(cost)

    matched_gold: set[int] = set()
    matched_pred: set[int] = set()
    for gi in range(ng):
        pj = assignment[gi]
        if pj < 0 or pj >= npd:
            continue
        sc = scores[gi][pj]
        if sc["align_sim"] >= accept:
            result.pairs.append((gi, pj, sc))
            matched_gold.add(gi)
            matched_pred.add(pj)

    result.unmatched_gold = [i for i in range(ng) if i not in matched_gold]
    result.unmatched_pred = [j for j in range(npd) if j not in matched_pred]
    return result


# --------------------------------------------------------------------------- #
# Hallucination detection
# --------------------------------------------------------------------------- #

def _entity_in_prompt(title: str, prompt_norm: str, prompt_tokens: set[str]) -> bool:
    """Is a (normalised) entity title grounded in the prompt text?"""
    if not title:
        return False
    if title in prompt_norm:
        return True
    toks = set(_TOKEN_RE.findall(title))
    if not toks:
        return False
    overlap = len(toks & prompt_tokens) / len(toks)
    return overlap >= 0.6


def final_hallucination_check(
    pred: Sequence[dict],
    gold: Sequence[dict],
    unmatched_pred_idx: Sequence[int],
    prompt: str,
) -> dict:
    """Classify unmatched predicted triplets and detect self-loops.

    Per the spec: check whether predicted entities appear among gold entities;
    for a novel entity with a *valid* relation, look for it in the prompt text;
    and drop triplets where source == target (an entity "matched to itself").

    Returns counts plus the indices of triplets that should be dropped/penalised.
    """
    gold_entities = set()
    for g in gold:
        gold_entities.add(g["source"]["title"])
        gold_entities.add(g["target"]["title"])

    prompt_norm = normalize_title(prompt) if prompt else ""
    prompt_tokens = set(_TOKEN_RE.findall(prompt_norm))

    self_loops: list[int] = []
    hallucinated_relation: list[int] = []   # relation type not in VALID set
    ungrounded: list[int] = []              # entity neither in gold nor prompt
    novel_grounded: list[int] = []          # new but supported by the prompt

    for j in unmatched_pred_idx:
        t = pred[j]
        src, tgt = t["source"]["title"], t["target"]["title"]
        rel = t["relation"]["type"]

        # self-referential triplet -> remove.
        if src and src == tgt:
            self_loops.append(j)
            continue

        # invalid / never-instructed relation -> hallucination.
        if rel not in VALID_RELATION_TYPES:
            hallucinated_relation.append(j)
            continue

        src_known = src in gold_entities or _entity_in_prompt(src, prompt_norm, prompt_tokens)
        tgt_known = tgt in gold_entities or _entity_in_prompt(tgt, prompt_norm, prompt_tokens)
        if src_known and tgt_known:
            # valid relation, entities supported by gold/prompt -> plausible novel edge.
            novel_grounded.append(j)
        else:
            ungrounded.append(j)

    # "Hard" hallucinations are the clear errors: bad relations, ungrounded
    # entities, and self-loops.  novel_grounded edges are extra (cost precision)
    # but not counted as hard hallucinations because the prompt supports them.
    hard = sorted(set(self_loops) | set(hallucinated_relation) | set(ungrounded))
    return {
        "self_loops": self_loops,
        "hallucinated_relation": hallucinated_relation,
        "ungrounded": ungrounded,
        "novel_grounded": novel_grounded,
        "hard_hallucinations": hard,
        "drop": sorted(set(self_loops)),  # self-loops are removed outright
    }


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2 * p * r, p + r) if (p + r) else 0.0


def _weighted_composite(axes: dict[str, Optional[float]]) -> float:
    """Combine available axes using SCORE_WEIGHTS, renormalising over present ones."""
    num = den = 0.0
    for name, weight in SCORE_WEIGHTS.items():
        val = axes.get(name)
        if val is None:
            continue
        num += weight * val
        den += weight
    return _safe_div(num, den)


def score_entry(entry: dict, embedder: EmbeddingBackend,
                threshold: float = DEFAULT_SIM_THRESHOLD,
                accept: float = DEFAULT_ALIGN_ACCEPT) -> dict:
    """Run the full pipeline on a single entry and return its metrics."""
    gold_raw = entry.get("gold_parsed") or []
    pred_raw = entry.get("pred_parsed") or []
    prompt = entry.get("prompt") or ""

    # --- schema (highest axis) ------------------------------------------- #
    sch = schema_score(pred_raw)
    gold_valid, _, _ = valid_triplets(gold_raw)
    pred_valid, n_pred_total, n_pred_valid = valid_triplets(pred_raw)

    # --- normalise -------------------------------------------------------- #
    gold = normalize_triplets(gold_valid)
    pred = normalize_triplets(pred_valid)

    # --- self-loop removal happens before matching (entity matched to itself) #
    pred_kept_idx = [j for j, t in enumerate(pred)
                     if not (t["source"]["title"] and t["source"]["title"] == t["target"]["title"])]
    dropped_self_loops = len(pred) - len(pred_kept_idx)
    pred_matchable = [pred[j] for j in pred_kept_idx]

    n_gold = len(gold)
    n_pred = len(pred_matchable)

    # --- Hungarian 1-1 matching ------------------------------------------ #
    match = hungarian_algorithm_for_one_one_matching(
        gold, pred_matchable, embedder, threshold, accept)

    entity_tp = sum(1 for _, _, s in match.pairs if s["entity_match"])
    full_tp = sum(1 for _, _, s in match.pairs if s["full_match"])
    rel_tp_given_entity = sum(1 for _, _, s in match.pairs
                              if s["entity_match"] and s["relation_match"])
    type_agree = sum(1 for _, _, s in match.pairs if s["entity_match"] and s["type_agree"])

    weight_scores = [s["weight_score"] for _, _, s in match.pairs if s["full_match"]]

    # --- precision / recall / F1 ----------------------------------------- #
    # Precision denominator is everything the model produced (invalid items and
    # self-loops included) so schema errors are penalised here too.
    pred_denom = n_pred_total
    entity_precision = _safe_div(entity_tp, pred_denom)
    entity_recall = _safe_div(entity_tp, n_gold)
    entity_f1 = _f1(entity_precision, entity_recall)

    triplet_precision = _safe_div(full_tp, pred_denom)
    triplet_recall = _safe_div(full_tp, n_gold)
    triplet_f1 = _f1(triplet_precision, triplet_recall)

    relation_acc = _safe_div(rel_tp_given_entity, entity_tp) if entity_tp else None
    weight_axis = (sum(weight_scores) / len(weight_scores)) if weight_scores else None

    # --- hallucination check on the unmatched predictions ---------------- #
    halluc = final_hallucination_check(pred_matchable, gold, match.unmatched_pred, prompt)
    n_hard = len(halluc["hard_hallucinations"]) + dropped_self_loops
    halluc_rate = _safe_div(n_hard, n_pred_total) if n_pred_total else 0.0
    grounding = 1.0 - halluc_rate

    # --- composite (schema highest; drop N/A axes, renormalise) ---------- #
    # Schema and grounding are only meaningful when the model actually produced
    # items: an empty array trivially "validates" and trivially "doesn't
    # hallucinate", which must NOT be rewarded.  They are N/A when n_pred == 0.
    axes = {
        "schema": sch if n_pred_total > 0 else None,
        "entity_f1": entity_f1,
        "relation_acc": relation_acc,
        "weight": weight_axis,
        "grounding": grounding if n_pred_total > 0 else None,
    }
    if n_gold == 0:
        # The reference says nothing should be extracted: perfect iff pred is
        # also empty, otherwise everything predicted is a false positive.
        composite = 1.0 if n_pred_total == 0 else 0.0
    else:
        # Non-empty gold with an empty/zero-match prediction collapses to the
        # recall-driven entity_f1 (0.0), as it should.
        composite = _weighted_composite(axes)

    return {
        "n_gold": n_gold,
        "n_pred": n_pred,
        "n_pred_raw_items": n_pred_total,
        "n_pred_valid": n_pred_valid,
        "schema_valid_all": n_pred_valid == n_pred_total,
        "schema_score": sch,
        "entity_tp": entity_tp,
        "full_tp": full_tp,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
        "triplet_precision": triplet_precision,
        "triplet_recall": triplet_recall,
        "triplet_f1": triplet_f1,
        "relation_accuracy": relation_acc,
        "weight_score": weight_axis,
        "type_agreement": _safe_div(type_agree, entity_tp) if entity_tp else None,
        "self_loops_removed": dropped_self_loops,
        "hard_hallucinations": n_hard,
        "novel_grounded": len(halluc["novel_grounded"]),
        "hallucination_rate": halluc_rate,
        "grounding": grounding,
        "composite": composite,
    }


def scoring(entries: Sequence[dict], embedder: EmbeddingBackend,
            threshold: float = DEFAULT_SIM_THRESHOLD,
            accept: float = DEFAULT_ALIGN_ACCEPT,
            progress: bool = False) -> dict:
    """Multi-axis scoring across the whole dataset.

    Computes per-entry metrics, then both a macro-average (mean of per-entry
    composites) and micro-averaged corpus precision/recall/F1.  Schema is the
    most heavily weighted axis because the generating model is small and schema
    adherence is the cheapest correctness signal.
    """
    per_entry: list[dict] = []

    # Warm the embedding cache once over every title for speed.
    if embedder.enabled:
        titles: set[str] = set()
        for e in entries:
            for t in (e.get("gold_parsed") or []) + (e.get("pred_parsed") or []):
                if isinstance(t, dict):
                    for k in ("source", "target"):
                        sub = t.get(k) or {}
                        titles.add(normalize_title(sub.get("title")))
                    rel = (t.get("relation") or {}).get("type")
                    if isinstance(rel, str):
                        titles.add(relation_text(rel.strip().lower()))
        titles.discard("")
        embedder.warm(titles)

    n = len(entries)
    for idx, entry in enumerate(entries, start=1):
        per_entry.append(score_entry(entry, embedder, threshold, accept))
        if progress and (idx % 50 == 0 or idx == n):
            print(f"  scored {idx}/{n} entries", file=sys.stderr)

    # ---- corpus aggregates ---------------------------------------------- #
    def msum(key: str) -> int:
        return sum(int(m[key]) for m in per_entry)

    tot_gold = msum("n_gold")
    tot_pred_items = msum("n_pred_raw_items")
    tot_entity_tp = msum("entity_tp")
    tot_full_tp = msum("full_tp")
    tot_hard = msum("hard_hallucinations")

    micro_entity_p = _safe_div(tot_entity_tp, tot_pred_items)
    micro_entity_r = _safe_div(tot_entity_tp, tot_gold)
    micro_triplet_p = _safe_div(tot_full_tp, tot_pred_items)
    micro_triplet_r = _safe_div(tot_full_tp, tot_gold)

    def mean(key: str) -> float:
        vals = [m[key] for m in per_entry if m.get(key) is not None]
        return _safe_div(sum(vals), len(vals))

    summary = {
        "n_entries": n,
        "embedding_backend": embedder.backend,
        "sim_threshold": threshold,
        "align_accept": accept,
        "totals": {
            "gold_triplets": tot_gold,
            "pred_triplet_items": tot_pred_items,
            "entity_true_positives": tot_entity_tp,
            "full_triplet_true_positives": tot_full_tp,
            "hard_hallucinations": tot_hard,
            "empty_predictions": sum(1 for m in per_entry if m["n_pred_raw_items"] == 0),
            "schema_invalid_entries": sum(1 for m in per_entry if not m["schema_valid_all"]),
        },
        "micro": {
            "entity_precision": micro_entity_p,
            "entity_recall": micro_entity_r,
            "entity_f1": _f1(micro_entity_p, micro_entity_r),
            "triplet_precision": micro_triplet_p,
            "triplet_recall": micro_triplet_r,
            "triplet_f1": _f1(micro_triplet_p, micro_triplet_r),
            "hallucination_rate": _safe_div(tot_hard, tot_pred_items),
        },
        "macro": {
            "schema_score": mean("schema_score"),
            "entity_f1": mean("entity_f1"),
            "triplet_f1": mean("triplet_f1"),
            "relation_accuracy": mean("relation_accuracy"),
            "weight_score": mean("weight_score"),
            "type_agreement": mean("type_agreement"),
            "grounding": mean("grounding"),
            "composite": mean("composite"),
        },
        "score_weights": dict(SCORE_WEIGHTS),
    }
    return {"summary": summary, "per_entry": per_entry}


# --------------------------------------------------------------------------- #
# Reporting / CLI
# --------------------------------------------------------------------------- #

def _fmt(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{x:6.3f}"


def print_summary(summary: dict) -> None:
    s, t, mi, ma = summary, summary["totals"], summary["micro"], summary["macro"]
    print("\n" + "=" * 64)
    print(" KG TRIPLET EVALUATION SUMMARY")
    print("=" * 64)
    print(f" entries              : {s['n_entries']}")
    print(f" embedding backend    : {s['embedding_backend']}")
    print(f" sim threshold        : {s['sim_threshold']}   align accept: {s['align_accept']}")
    print("-" * 64)
    print(" corpus totals")
    print(f"   gold triplets          : {t['gold_triplets']}")
    print(f"   predicted items        : {t['pred_triplet_items']}")
    print(f"   entity TP / triplet TP : {t['entity_true_positives']} / {t['full_triplet_true_positives']}")
    print(f"   empty predictions      : {t['empty_predictions']}")
    print(f"   schema-invalid entries : {t['schema_invalid_entries']}")
    print(f"   hard hallucinations    : {t['hard_hallucinations']}")
    print("-" * 64)
    print(" micro-averaged (corpus-level)")
    print(f"   entity   P/R/F1 : {_fmt(mi['entity_precision'])} {_fmt(mi['entity_recall'])} {_fmt(mi['entity_f1'])}")
    print(f"   triplet  P/R/F1 : {_fmt(mi['triplet_precision'])} {_fmt(mi['triplet_recall'])} {_fmt(mi['triplet_f1'])}")
    print(f"   hallucination rate : {_fmt(mi['hallucination_rate'])}")
    print("-" * 64)
    print(" macro-averaged (per-entry mean) -- scoring axes")
    print(f"   schema_score        : {_fmt(ma['schema_score'])}   (weight {SCORE_WEIGHTS['schema']})")
    print(f"   entity_f1           : {_fmt(ma['entity_f1'])}   (weight {SCORE_WEIGHTS['entity_f1']})")
    print(f"   relation_accuracy   : {_fmt(ma['relation_accuracy'])}   (weight {SCORE_WEIGHTS['relation_acc']})")
    print(f"   weight_score        : {_fmt(ma['weight_score'])}   (weight {SCORE_WEIGHTS['weight']})")
    print(f"   grounding           : {_fmt(ma['grounding'])}   (weight {SCORE_WEIGHTS['grounding']})")
    print(f"   triplet_f1 (info)   : {_fmt(ma['triplet_f1'])}")
    print(f"   type_agreement(info): {_fmt(ma['type_agreement'])}")
    print("=" * 64)
    print(f"  >>> COMPOSITE SCORE : {ma['composite']:.4f}  <<<")
    print("=" * 64 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate predicted KG triplets against gold triplets.")
    p.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT,
                   help=f"Path to the JSONL dataset (default: {DEFAULT_INPUT}).")
    p.add_argument("--report", "-r", type=Path, default=None,
                   help="Optional path to write the full JSON report.")
    p.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL,
                   help=f"Embedding model name (default: {DEFAULT_EMBED_MODEL}).")
    p.add_argument("--no-embeddings", action="store_true",
                   help="Disable the embedding model; use lexical similarity only.")
    p.add_argument("--embedding-threshold", type=float, default=DEFAULT_SIM_THRESHOLD,
                   help=f"Fuzzy-match acceptance threshold (default: {DEFAULT_SIM_THRESHOLD}).")
    p.add_argument("--align-accept", type=float, default=DEFAULT_ALIGN_ACCEPT,
                   help=f"Min alignment similarity for a match (default: {DEFAULT_ALIGN_ACCEPT}).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate the first N entries (useful for a quick check).")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.input.exists():
        print(f"[error] input file not found: {args.input}", file=sys.stderr)
        return 2

    entries = load_entries(args.input)
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        print("[error] no entries to evaluate.", file=sys.stderr)
        return 2

    embedder = EmbeddingBackend(model_name=args.embedding_model,
                                enabled=not args.no_embeddings)
    if not args.quiet:
        print(f"[info] loaded {len(entries)} entries from {args.input}", file=sys.stderr)

    report = scoring(entries, embedder,
                     threshold=args.embedding_threshold,
                     accept=args.align_accept,
                     progress=not args.quiet)

    print_summary(report["summary"])

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[info] full report written to {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
