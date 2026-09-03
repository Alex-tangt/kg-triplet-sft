"""GraphRAG-format extraction parser.

Parses the ``("entity"<|>...)`` / ``("relationship"<|>...)`` records that the
official GraphRAG extraction prompt (``kg_contract/graphrag_prompts.py``)
produces into a canonical structure that maps 1:1 onto the MS GraphRAG graph
schema — entities {title, type, description} and relationships
{source, target, description, strength(=rank)}.

The parser is deliberately robust to the teacher's structural sloppiness:
records are split on the ``("entity"<|>`` / ``("relationship"<|>`` openers (NOT
on ``)``, which legitimately occurs inside descriptions), fields are split on
``<|>``, and a relationship whose strength field is missing defaults to 1.0
instead of swallowing the following record.
"""

from __future__ import annotations

import re

_RECORD_OPEN = re.compile(r'\("(entity|relationship)"<\|>', re.DOTALL)


def _clean(s: str) -> str:
    return s.strip().strip('"').strip()


_TAIL_JUNK_RE = re.compile(r"(?:<\|COMPLETE\|>|[#*]+|\)|>|\s)*$")


def _clean_tail(s: str) -> str:
    """Strip a record's trailing junk so only the intended content remains.

    The teacher ends each record with `)` and separates records with `##` or
    `**##**` lines (the prompt: "Use **##** as the list delimiter"); the final
    record is followed by `<|COMPLETE|>`. A record body sliced to the next
    opener therefore carries trailing `)`, separator blocks, and/or the COMPLETE
    marker, and a relationship may omit its strength field (leaving just
    `DESC)`). This removes that tail (whitespace / `#` `*` runs / `)` / the
    COMPLETE marker) from the end, left to right.
    """
    return _TAIL_JUNK_RE.sub("", s).strip()


def _norm(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_graphrag_output(text: str) -> dict | None:
    """Parse GraphRAG serialization into {entities, relationships}.

    Returns None when nothing parseable was found. A relationship whose final
    field is not numeric (the teacher sometimes omits strength) keeps that field
    as part of the description and defaults strength to 1.0.
    """
    entities: list[dict] = []
    relationships: list[dict] = []
    opens = list(_RECORD_OPEN.finditer(text))
    for idx, m in enumerate(opens):
        end = opens[idx + 1].start() if idx + 1 < len(opens) else len(text)
        body = text[m.end() : end]
        fields = [_clean(f) for f in body.split("<|>")]
        if m.group(1) == "entity":
            if len(fields) < 2 or not fields[0] or not fields[1]:
                continue
            desc = _clean_tail("<|>".join(fields[2:]))
            entities.append({"title": fields[0], "type": fields[1], "description": desc})
        else:
            if len(fields) < 3 or not fields[0] or not fields[1]:
                continue
            src, tgt = fields[0], fields[1]
            rest = list(fields[2:])
            strength = 1.0
            if len(rest) >= 2:
                tail = _clean_tail(rest[-1])
                if _is_number(tail):
                    strength = max(0.0, min(10.0, float(tail)))
                    rest = rest[:-1]
            desc = _clean_tail("<|>".join(rest))
            relationships.append(
                {"source": src, "target": tgt, "description": desc, "strength": strength}
            )

    if not entities:
        return None
    return {"entities": entities, "relationships": relationships}


def dedup_graphrag(parsed: dict, text: str) -> dict:
    """Deduplicate GraphRAG records and drop ungrounded entities.

    Entities are keyed by the normalized title (case / punctuation / whitespace /
    dash-variant insensitive, via ``semantic.normalize``); the longest description
    wins. Relationships are keyed by the (normalized source, normalized target)
    pair; the longest description wins and self-loops are dropped. An entity whose
    normalized title is not a substring of the normalized passage is dropped, along
    with any relationship touching it — this filters fabricated names while the
    continuation loop keeps re-emitting near-identical variants across rounds.
    """
    from dataset.semantic import normalize

    ntext = normalize(text)
    entities: dict[str, dict] = {}
    for e in parsed["entities"]:
        key = normalize(e["title"])
        if not key or len(key) < 3 or key not in ntext:
            continue
        prev = entities.get(key)
        if prev is None or len(e["description"]) > len(prev["description"]):
            entities[key] = e

    relationships: list[dict] = []
    best: dict[tuple[str, str], dict] = {}
    for r in parsed["relationships"]:
        s, t = normalize(r["source"]), normalize(r["target"])
        if not s or not t or s == t or s not in entities or t not in entities:
            continue
        key = (s, t)
        prev = best.get(key)
        if prev is None or len(r["description"]) > len(prev["description"]):
            best[key] = r
    relationships = list(best.values())

    return {"entities": list(entities.values()), "relationships": relationships}
