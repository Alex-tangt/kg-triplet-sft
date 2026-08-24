"""MinHash near-duplicate detection with LSH banding (mmh3 + numpy).

Character k-shingles → a MinHash signature per text; LSH buckets group
candidate near-duplicates; a bucket pair is a duplicate when the estimated
Jaccard reaches the threshold. Deterministic: iteration order decides which
occurrence is kept (the first).

The signature uses the universal linear hash family ``h_i(x) = a_i * h(x) + b_i
mod 2^32`` (``a_i`` odd, so each row is a permutation of the shingle hashes);
the per-row min over shingles is aggregated with numpy, keeping the work
C-level while staying a sound MinHash estimate (the textbook per-seed
``mmh3.hash(x, seed)`` variant is O(num_hashes) hashes per shingle and was the
bottleneck).
"""

from __future__ import annotations

import re

import mmh3
import numpy as np

_WS = re.compile(r"\s+")
_HASH_ROWS: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _rows(num_hashes: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic (a, b) coefficient arrays for the linear hash family."""
    cached = _HASH_ROWS.get(num_hashes)
    if cached is None:
        rng = np.random.default_rng(0xC0FFEE)
        a = rng.integers(1, 2**32, size=num_hashes, dtype=np.uint64) | 1  # odd -> permutation
        b = rng.integers(0, 2**32, size=num_hashes, dtype=np.uint64)
        cached = (a, b)
        _HASH_ROWS[num_hashes] = cached
    return cached


def shingles(text: str, k: int = 5) -> list[str]:
    normalized = _WS.sub(" ", text).strip().lower()
    return [normalized[i : i + k] for i in range(len(normalized) - k + 1)]


def signature(shingles: list[str], num_hashes: int = 128) -> list[int]:
    """MinHash signature vector of ``num_hashes`` rows for one shingle set."""
    if not shingles:
        return [0] * num_hashes
    h = np.fromiter((mmh3.hash(s, 0, signed=False) for s in shingles), dtype=np.uint64)
    a, b = _rows(num_hashes)
    values = (h[:, None] * a[None, :] + b[None, :]) & 0xFFFFFFFF
    return [int(v) for v in np.min(values, axis=0)]


def estimated_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def deduplicate(
    texts: list[str],
    threshold: float = 0.85,
    num_hashes: int = 128,
    rows_per_band: int = 8,
) -> list[bool]:
    """Return a ``keep`` flag per text; ``False`` marks a near-duplicate of an
    earlier kept text. Keeps the first occurrence of each duplicate group."""
    if not texts:
        return []
    bands = max(1, num_hashes // rows_per_band)
    signatures = [signature(shingles(t), num_hashes) for t in texts]
    keep = [True] * len(texts)
    buckets: dict[tuple, list[int]] = {}
    for idx, sig in enumerate(signatures):
        for band in range(bands):
            key = (band, tuple(sig[band * rows_per_band : (band + 1) * rows_per_band]))
            buckets.setdefault(key, []).append(idx)
    for indices in buckets.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                first, second = indices[i], indices[j]
                if not keep[first] or not keep[second]:
                    continue
                if estimated_jaccard(signatures[first], signatures[second]) >= threshold:
                    keep[second] = False
    return keep
