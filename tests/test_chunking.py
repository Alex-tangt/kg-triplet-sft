from dataset.chunking import (
    chunk_sentences,
    chunk_text,
    ends_at_sentence_boundary,
    split_sentences,
    word_count,
)


def test_split_sentences_basic():
    assert split_sentences("Hello world. Foo bar. Baz qux!") == [
        "Hello world.",
        "Foo bar.",
        "Baz qux!",
    ]


def test_split_sentences_protects_abbreviations():
    text = "Dr. Smith went home. He slept well. The U.S. economy grew. It was fine."
    assert split_sentences(text) == [
        "Dr. Smith went home.",
        "He slept well.",
        "The U.S. economy grew.",
        "It was fine.",
    ]


def test_split_sentences_drops_trailing_fragment():
    assert split_sentences("Complete sentence. Dangling fragment") == ["Complete sentence."]


def test_split_sentences_handles_decimal_and_quotes():
    assert split_sentences('Pi is about 3.14. He said "stop." Then he left.') == [
        'Pi is about 3.14.',
        'He said "stop."',
        "Then he left.",
    ]


def test_ends_at_sentence_boundary():
    assert ends_at_sentence_boundary("A full sentence.")
    assert ends_at_sentence_boundary('A quote ends here."')
    assert not ends_at_sentence_boundary("A dangling fragment")
    assert not ends_at_sentence_boundary("")


def test_chunk_sentences_packs_100_300_words():
    sentences = ["word " * 25 + "S."] * 20  # 20 sentences of 26 words each
    chunks = chunk_sentences(sentences, max_words=300)
    assert all(100 <= word_count(c) <= 300 for c in chunks)
    # 520 words -> 2-3 chunks (260 each is too tight with 26-word granularity)
    assert 2 <= len(chunks) <= 3


def test_chunk_sentences_keeps_oversized_sentence_whole():
    giant = ("tiny " * 350) + "S."
    chunks = chunk_sentences(["First sentence.", giant, "Last sentence."], max_words=300)
    assert any(giant in chunk for chunk in chunks)


def test_chunk_text_never_cuts_mid_sentence():
    article = "One complete sentence. Another complete sentence. " * 40
    for chunk in chunk_text(article):
        assert ends_at_sentence_boundary(chunk)
