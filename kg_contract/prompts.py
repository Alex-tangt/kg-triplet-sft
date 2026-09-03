"""Prompt templates: the one source of truth for what the model sees.

The target prompt is reproduced character-for-character from the original
project (ADR-0001): the 20-relation ontology stays implicit, never listed in the
instruction. One template constant renders every consumer — the Alpaca dataset
projection, evaluation prediction generation, and serving — so drift between the
three code paths is structurally impossible.

The teacher prompt (what qwen3-flash sees when generating gold labels) is a
separate, richer instruction. Its skeleton fixes the five elements and their
order; the actual prose is filled in by the labeling tickets.
"""

from __future__ import annotations

TARGET_PROMPT_PREFIX = (
    "Extract knowledge graph triplets from the following text. Return a JSON "
    "array of objects with keys: source (title, type), relation (type, weight), "
    "target (title, type). If no valid triplets can be extracted, return an "
    "empty array.\n\nText:\n"
)

TARGET_PROMPT_TEMPLATE = TARGET_PROMPT_PREFIX + "{text}"


def render_target_prompt(text: str) -> str:
    """Render the verbatim target prompt for one passage."""
    return TARGET_PROMPT_TEMPLATE.format(text=text)


WEIGHT_ANCHORS: tuple[float, ...] = (0.2, 0.5, 0.8)

#: The five required elements of the teacher instruction, in rendering order.
TEACHER_SECTIONS: tuple[str, ...] = (
    "relation_dictionary",
    "entity_naming_rules",
    "exhaustiveness_demand",
    "weight_rubric",
    "in_chunk_grounding",
)

#: The actual teacher prose per section, rendered in fixed order by
#: :func:`render_teacher_prompt`. Consumed by the labeling pipeline (dataset) and
#: the gold audit rubric (eval, ticket 11) — both must score the same instruction.
TEACHER_SECTIONS_PROSE: dict[str, str] = {
    "relation_dictionary": (
        "You extract knowledge-graph triplets from English text. A triplet is a JSON "
        "object {source: {title, type}, relation: {type, weight}, target: {title, type}}. "
        "Use ONLY these 20 relations, exactly as written: implements, trained_on, "
        "evaluates, part_of, introduces, extends, depends_on, contrasts_with, applied_to, "
        "measured_by, founded_by, developed_by, defined_as, consists_of, is_type_of, "
        "based_on, used_for, created_by, located_in, predecessor_of. If a fact has no "
        "faithful listed relation, emit no triplet for it. part_of covers membership and "
        "part-whole inclusion: a member of an institution, journal, or academy; a "
        "component of a larger structure; a region's constituent ecoregion; a named "
        "constituent of a theory (e.g. 'the three-gluon vertex ... in nonabelian gauge "
        "theory'). Connect an ecoregion to the realm it belongs to ('ecoregions of the "
        "Palearctic realm' → 'Illyrian deciduous forests part_of Palearctic realm'); the "
        "country whose territory merely contains it is located_in, never part_of. "
        "part_of does NOT cover treaty or convention participation (party to CBD, "
        "signatory to Ramsar), which has no faithful relation here. Spatial placement "
        "is located_in, never part_of."
    ),
    "entity_naming_rules": (
        "title is a string exactly as it appears in the text; type is exactly entity or "
        "concept. Never invent or paraphrase a title, never add labels or numbering, and "
        "never create a category or aggregate node to hold a list ('national parks', "
        "'the four wetlands', 'the one-loop vertex' when its members are named). When a "
        "statement names several distinct entities in a list, emit one triplet PER "
        "member, connecting it to the statement's actual subject: '...protected areas "
        "within its boundaries, encompassing 12 national parks among others Butrint, "
        "Karaburun-Sazan, ...' becomes 'Butrint located_in Albania', 'Karaburun-Sazan "
        "located_in Albania' — never 'national parks consists_of Butrint'. Likewise "
        "split 'the scalar, spinor and gluon loop contributions to the one-loop vertex' "
        "into 'scalar loop contributions part_of one-loop vertex', 'spinor loop "
        "contributions part_of one-loop vertex', 'gluon loop contributions part_of "
        "one-loop vertex'. Never merge distinct names into one node: 'found by binger "
        "and brodsky' is two people — one triplet per person. When a phrase is a "
        "derivative-of-base construction, split it instead of keeping a compound node: "
        "'an off-shell extension of the bern-kosower replacement rules' becomes "
        "'off-shell extension extends bern-kosower replacement rules'."
    ),
    "exhaustiveness_demand": (
        "Extract every key fact the text supports; never pick a subset. Do not omit a "
        "fact because it appears inside a list, and do not stop after a few members — "
        "every member of a list is a separate triplet, and if a person or group holds "
        "several memberships or affiliations, capture each one. When the text states "
        "the same fact more than once, emit it only once. SKIP lists that merely "
        "enumerate a set's constituents without asserting a relation for the members — "
        "e.g. 'the trees within the forests are primarily fir, oak, beech and pine'. A "
        "list IS extractable when the passage asserts a relation per member: 'plants "
        "are used in the preparation of herbs and medicines' → 'plants used_for herbs', "
        "'plants used_for medicines'. Prefer the relation the text states directly over "
        "an inferred reading; when the text situates a named place within a territory, "
        "emit located_in. When the text gives both a provenance verb and a usage verb "
        "for the same artifact, emit the provenance relation ('Albania has developed "
        "and implemented NBSAP' → 'NBSAP developed_by Albania'). Keep derivation chains "
        "as separate links: when X follows from Y, and Y is itself an extension of Z, "
        "emit X based_on Y and Y extends Z — do not collapse to X based_on Z. If the "
        "text supports no triplet at all, return an empty JSON array []."
    ),
    "weight_rubric": (
        "Set relation.weight to exactly one of 0.8 (explicitly stated), 0.5 (indirectly "
        "supported), or 0.2 (only inferred). Use only these three values — never any "
        "other number."
    ),
    "in_chunk_grounding": (
        "Every source and target title, or a normal form of it, must appear in the "
        "provided text. If a referent is only established outside the chunk, drop the "
        "triplet rather than invent the entity. Never output an entity that does not "
        "occur in the text. Before finalizing, re-read each title against the text and "
        "fix any spelling, spacing, or hyphenation drift."
    ),
}


def render_teacher_prompt(sections: dict[str, str]) -> str:
    """Assemble a teacher prompt from its five named sections, in fixed order."""
    missing = [name for name in TEACHER_SECTIONS if name not in sections]
    if missing:
        raise ValueError(f"missing teacher sections: {', '.join(missing)}")
    unknown = [name for name in sections if name not in TEACHER_SECTIONS]
    if unknown:
        raise ValueError(f"unknown teacher sections: {', '.join(unknown)}")
    return "\n\n".join(sections[name] for name in TEACHER_SECTIONS)
