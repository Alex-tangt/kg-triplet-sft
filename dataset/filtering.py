"""Passage quality filters: language, length, and boilerplate.

Cheap, conservative heuristics — the corpus is built with margin (we produce
more passages than we label), so a false positive costs a passage, not a label.
"""

from __future__ import annotations

import re

_WORDS = re.compile(r"[a-z]+")

# arXiv-source artifacts: math/citation placeholders and LaTeX table directives
# that survive the dataset's text extraction.
_ARTIFACT_RE = re.compile(r"@x(?:math\d*|cite)|\[[^\]]*?(?:cols|options)=[^\]]*\]")

_STOPWORDS = frozenset(
    "the a an and of to in is that for on with as was by at from this are it or "
    "be which not but have has he she they we you his her their will can would "
    "could should may more most about between after before when where what who "
    "how also its all some such there these those than then them because into "
    "through during including over under up out off again once many much so "
    "very just only own same other each both few no"
    .split()
)

# Wikipedia section headings and template boilerplate that survive the markup
# strip; chunks dominated by these are not prose we want to label.
_BOILERPLATE_MARKERS = (
    "this article is a stub",
    "this article about",
    "this article needs additional citations",
    "this article relies largely or entirely",
    "this biography of a living person",
    "citation needed",
    "references ^",
    "external links",
    "see also",
    "further reading",
    "wikimedia commons",
    "is a disambiguation page",
    "may be in need of reorganization",
    "to meet wikipedia's quality standards",
)


def is_english(text: str) -> bool:
    """Heuristic: mostly Latin letters plus a minimum of common English words."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    latin = sum(1 for c in letters if ord(c) < 0x250)
    if latin / len(letters) < 0.9:
        return False
    words = set(_WORDS.findall(text.lower()))
    return len(words & _STOPWORDS) >= 3


def is_boilerplate(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(marker in normalized for marker in _BOILERPLATE_MARKERS)


def clean_text(text: str) -> str:
    """Strip arXiv annotation artifacts (``@xmath``/``@xcite`` placeholders and
    table directives), collapsing the leftover whitespace."""
    cleaned = _ARTIFACT_RE.sub(" ", text)
    return " ".join(cleaned.split())
