"""Sentence-boundary chunking: split articles into 100-300 word passages, cutting
only between sentences — never mid-sentence. This is the deliberate improvement
over the original project's hard mid-sentence truncation (recorded in the README
as a pipeline difference).
"""

from __future__ import annotations

import re

_WORD = re.compile(r"\S+")

_KNOWN_ABBREV = frozenset(
    "dr mr mrs ms prof st sr jr vs etc no fig cf al approx dept est gen govt max "
    "min mt pres pt sec sgt vol un co inc ltd corp ref ed rev trans n o ex eq pp "
    "ch a.m p.m u.s b.c a.d c.e"
    .split()
)


def word_count(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    """Split text into complete sentences.

    A sentence ends only after ``.``/``!``/``?`` that is neither a known
    abbreviation nor part of a number. A trailing fragment with no ending
    punctuation is dropped, so every returned sentence is complete.
    """
    sentences: list[str] = []
    current: list[str] = []
    for token in _WORD.findall(text):
        current.append(token)
        if _ends_sentence(token):
            sentences.append(" ".join(current))
            current = []
    return sentences


def ends_at_sentence_boundary(text: str) -> bool:
    """True when ``text`` is a sequence of complete sentences (the last token
    ends with . ! ? before any closing quote or bracket)."""
    return text.rstrip().rstrip('"\')]}').endswith((".", "!", "?"))


def chunk_sentences(
    sentences: list[str],
    max_words: int = 300,
) -> list[str]:
    """Greedily pack complete sentences into chunks capped at ``max_words``.

    A chunk is closed whenever the next sentence would push it past
    ``max_words`` (never exceeding the cap just because the running total is
    still under the 100-word target), so a long sentence never drags a short
    tail over the cap. A single sentence longer than ``max_words`` is kept
    whole rather than split mid-sentence (callers log the oversize). The lower
    bound is enforced downstream by the filter stage.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = word_count(sentence)
        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text(
    text: str,
    max_words: int = 300,
) -> list[str]:
    """Sentence-split then chunk a raw article text (never mid-sentence)."""
    return chunk_sentences(split_sentences(text), max_words)


def _ends_sentence(token: str) -> bool:
    stripped = token.rstrip('"\')]}')
    if not stripped.endswith((".", "!", "?")):
        return False
    if _is_abbreviation(stripped):
        return False
    return True


def _is_abbreviation(token: str) -> bool:
    core = token.rstrip(".!?")
    if not core:
        return False
    if re.fullmatch(r"(?:[A-Za-z]\.){2,}", token):  # e.g., i.e., U.S., Ph.D.
        return True
    return core.lower() in _KNOWN_ABBREV
