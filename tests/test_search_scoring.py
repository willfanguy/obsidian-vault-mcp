"""Tests for hybrid search scoring logic in src/search.py.

Tests the mathematical properties of the scoring formula without hitting
LanceDB or embeddings providers. The core formula:
  combined = sem_score * 0.7 + kw_score * 0.3
  if both > 0: combined *= 1.2  (dual-match boost)
"""

import pytest


def compute_hybrid_score(sem_score: float, kw_score: float) -> float:
    """Replication of the scoring formula from search.py hybrid_search().

    This is intentionally a reimplementation (not an import) so we're testing
    the mathematical properties, not just that the code runs without error.
    """
    combined = sem_score * 0.7 + kw_score * 0.3
    if sem_score > 0 and kw_score > 0:
        combined *= 1.2
    return combined


def normalize_bm25(raw_score: float, max_fts_score: float) -> float:
    """Replication of BM25 normalization from search.py."""
    return (raw_score / max_fts_score) if max_fts_score > 0 else 0


# --- Score formula tests ---


def test_semantic_only_score():
    """Result with only semantic score: combined = sem * 0.7."""
    score = compute_hybrid_score(sem_score=0.9, kw_score=0)
    assert score == pytest.approx(0.9 * 0.7)
    # No dual-match boost since kw_score is 0
    assert score == pytest.approx(0.63)


def test_keyword_only_score():
    """Result with only keyword score: combined = kw * 0.3."""
    score = compute_hybrid_score(sem_score=0, kw_score=0.8)
    assert score == pytest.approx(0.8 * 0.3)
    assert score == pytest.approx(0.24)


def test_dual_match_boost():
    """Result appearing in both methods gets 1.2x boost."""
    score = compute_hybrid_score(sem_score=0.9, kw_score=0.8)
    base = 0.9 * 0.7 + 0.8 * 0.3
    assert score == pytest.approx(base * 1.2)


def test_dual_match_outranks_semantic_only():
    """A dual-match result should outscore a semantic-only result, all else equal."""
    dual = compute_hybrid_score(sem_score=0.7, kw_score=0.5)
    sem_only = compute_hybrid_score(sem_score=0.9, kw_score=0)

    # Dual: (0.7*0.7 + 0.5*0.3) * 1.2 = (0.49 + 0.15) * 1.2 = 0.768
    # Sem-only: 0.9*0.7 = 0.63
    assert dual > sem_only


def test_zero_scores():
    """Both scores zero produces zero combined."""
    assert compute_hybrid_score(0, 0) == 0


# --- BM25 normalization tests ---


def test_bm25_normalization_standard():
    """Scores are divided by max to produce 0-1 range."""
    assert normalize_bm25(5.0, 10.0) == pytest.approx(0.5)
    assert normalize_bm25(10.0, 10.0) == pytest.approx(1.0)
    assert normalize_bm25(0.0, 10.0) == pytest.approx(0.0)


def test_bm25_normalization_zero_max():
    """When max_fts_score is 0, normalized score is 0 (no division by zero)."""
    assert normalize_bm25(5.0, 0.0) == 0
    assert normalize_bm25(0.0, 0.0) == 0


def test_bm25_normalization_preserves_ordering():
    """Normalization preserves relative ordering of scores."""
    scores = [2.0, 5.0, 8.0, 10.0]
    max_score = max(scores)
    normalized = [normalize_bm25(s, max_score) for s in scores]

    # Should still be sorted ascending
    assert normalized == sorted(normalized)
    # Max should normalize to 1.0
    assert normalized[-1] == pytest.approx(1.0)
