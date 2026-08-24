from dataset.minhash import deduplicate, estimated_jaccard, shingles, signature

TEXT_A = (
    "The telegraph was the first electrical means of long-distance communication "
    "and its development spanned a century of incremental invention."
)
TEXT_B = (
    "The telegraph was the first electrical means of long-distance communication "
    "and its development spanned a century of incremental invention."
)
TEXT_A_TWEAKED = TEXT_A.replace("first", "very first")
TEXT_C = (
    "Quantum computing uses qubits to perform computations that are infeasible "
    "for classical machines on certain problems like integer factorization."
)


def test_identical_signatures_have_jaccard_one():
    sig_a = signature(shingles(TEXT_A))
    sig_b = signature(shingles(TEXT_B))
    assert estimated_jaccard(sig_a, sig_b) == 1.0


def test_deduplicate_keeps_first_occurrence_of_exact_duplicates():
    keep = deduplicate([TEXT_A, TEXT_B, TEXT_C])
    assert keep == [True, False, True]


def test_deduplicate_removes_near_duplicate_one_word_change():
    keep = deduplicate([TEXT_A, TEXT_A_TWEAKED, TEXT_C], threshold=0.85)
    assert keep == [True, False, True]


def test_deduplicate_keeps_distinct_texts():
    keep = deduplicate([TEXT_A, TEXT_C], threshold=0.85)
    assert keep == [True, True]


def test_deduplicate_empty():
    assert deduplicate([]) == []
