"""Regression tests: a failing embedding provider must never put zero vectors in the index.

Background (2026-10-01): the OpenAI account ran out of credits. `_embed_openai` caught
every failure and substituted `[0.0] * 3072`, so:

- every query embedded to the zero vector, which is equidistant from every chunk, and
  vault_search returned the same three unrelated chunks at score 1.000 for any query;
- every note re-indexed during the outage was stored with zero vectors and marked done
  in the manifest, so it stayed invisible to search after credits were restored.
  111 chunks across 11 files had to be repaired by hand.

These tests drive the real indexer and search against a real LanceDB in a temp dir.
Only the OpenAI client is faked, at the import boundary `_embed_openai` uses, so the
production error-handling path is the one under test.
"""

import os
import sys
import types
import zlib

import numpy as np
import pytest

from src import embeddings
from src.indexer import TABLE_NAME, full_index, get_db, incremental_index, load_manifest
from src.search import semantic_search

QUOTA_ERROR = "Error code: 429 - insufficient_quota"
POISON = "POISON-PILL"


def _unit_vector(text: str, dims: int) -> list[float]:
    vec = [0.0] * dims
    vec[zlib.crc32(text.encode()) % dims] = 1.0
    return vec


@pytest.fixture
def fake_openai(monkeypatch):
    """Install a fake `openai` module. Flip state["fail"] to simulate an outage.

    Any input containing POISON always fails, to model one bad chunk in a healthy batch.
    """
    state = {"fail": False, "calls": 0}

    class _Embeddings:
        def create(self, model, input, dimensions):
            state["calls"] += 1
            if state["fail"]:
                raise RuntimeError(QUOTA_ERROR)
            if any(POISON in t for t in input):
                raise RuntimeError("Error code: 400 - invalid input")
            return types.SimpleNamespace(
                data=[types.SimpleNamespace(embedding=_unit_vector(t, dimensions)) for t in input]
            )

    class _OpenAI:
        def __init__(self, api_key=None):
            self.embeddings = _Embeddings()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return state


@pytest.fixture
def vault(tmp_path, monkeypatch):
    root = tmp_path / "vault"
    root.mkdir()
    db_path = str(tmp_path / "data" / "vault.lance")
    monkeypatch.setenv("LANCE_DB_PATH", db_path)
    _write(root / "alpha.md", "Alpha", "The transaxle takes GL-4 gear oil, never GL-5.")
    _write(root / "beta.md", "Beta", "Resting voltage must clear 12.4 V before a load test.")
    return root, db_path


def _write(path, title, body, bump=0.0):
    path.write_text(f"---\ntitle: {title}\n---\n\n## Notes\n\n{body}\n", encoding="utf-8")
    if bump:
        st = path.stat()
        os.utime(path, (st.st_atime + bump, st.st_mtime + bump))


def _rows(db_path):
    table = get_db(db_path).open_table(TABLE_NAME)
    at = table.search().select(["file_path", "content", "vector"]).limit(table.count_rows()).to_arrow()
    vecs = np.stack(at.column("vector").to_numpy(zero_copy_only=False))
    norms = np.linalg.norm(vecs, axis=1)
    return list(zip(at.column("file_path").to_pylist(), at.column("content").to_pylist(), norms))


def _zero_vector_rows(db_path):
    return [(p, c) for p, c, n in _rows(db_path) if not np.isfinite(n) or n < 1e-6]


# --- The provider layer ---


def test_embed_texts_raises_when_provider_fails(fake_openai):
    fake_openai["fail"] = True
    with pytest.raises(Exception, match="insufficient_quota") as exc:
        embeddings.embed_texts(["chunk one", "chunk two"])
    assert isinstance(exc.value, embeddings.EmbeddingError)


def test_one_bad_text_fails_the_call_instead_of_zero_filling(fake_openai):
    with pytest.raises(Exception, match="invalid input"):
        embeddings.embed_texts(["fine text", f"broken {POISON} text", "also fine"])


def test_healthy_provider_returns_one_unit_vector_per_text(fake_openai):
    vectors = embeddings.embed_texts(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(abs(np.linalg.norm(v) - 1.0) < 1e-9 for v in vectors)


# --- Search: reproduces the score-1.000 junk results ---


def test_search_during_outage_raises_instead_of_returning_junk(fake_openai, vault):
    root, db_path = vault
    full_index(str(root), db_path)

    fake_openai["fail"] = True
    with pytest.raises(Exception, match="insufficient_quota"):
        semantic_search("transaxle oil")


# --- Incremental reindex: reproduces the 111 zero-vector chunks ---


def test_incremental_reindex_during_outage_stores_no_zero_vectors(fake_openai, vault):
    root, db_path = vault
    full_index(str(root), db_path)
    old_mtime = load_manifest(db_path)["alpha.md"]

    _write(root / "alpha.md", "Alpha", "Updated: drain and fill plugs are 29-43 ft-lb.", bump=10)
    fake_openai["fail"] = True
    incremental_index(str(root), db_path)

    assert _zero_vector_rows(db_path) == []
    # The note must stay pending so the next run retries it...
    assert load_manifest(db_path).get("alpha.md", 0.0) <= old_mtime
    # ...and keep its previous chunks instead of vanishing from the index.
    assert any(p == "alpha.md" and "GL-4" in c for p, c, _ in _rows(db_path))


def test_failed_file_is_retried_once_the_provider_recovers(fake_openai, vault):
    root, db_path = vault
    full_index(str(root), db_path)
    _write(root / "alpha.md", "Alpha", "Updated: drain and fill plugs are 29-43 ft-lb.", bump=10)

    fake_openai["fail"] = True
    incremental_index(str(root), db_path)
    fake_openai["fail"] = False
    incremental_index(str(root), db_path)

    alpha = [(c, n) for p, c, n in _rows(db_path) if p == "alpha.md"]
    assert alpha and all("29-43 ft-lb" in c for c, _ in alpha)
    assert all(n > 0.99 for _, n in alpha)
    assert load_manifest(db_path)["alpha.md"] == (root / "alpha.md").stat().st_mtime


def test_one_bad_file_does_not_block_or_corrupt_the_others(fake_openai, vault):
    root, db_path = vault
    full_index(str(root), db_path)

    _write(root / "alpha.md", "Alpha", "Updated: lug nuts are 65-87 ft-lb.", bump=10)
    _write(root / "beta.md", "Beta", f"Updated with a {POISON} the provider rejects.", bump=10)
    incremental_index(str(root), db_path)

    assert _zero_vector_rows(db_path) == []
    rows = _rows(db_path)
    assert any(p == "alpha.md" and "65-87 ft-lb" in c for p, c, _ in rows)
    assert load_manifest(db_path)["alpha.md"] == (root / "alpha.md").stat().st_mtime
    assert load_manifest(db_path)["beta.md"] < (root / "beta.md").stat().st_mtime


def test_zero_chunk_note_settles_even_during_outage(fake_openai, vault):
    """An empty note needs no embedding, so an outage must not leave it pending forever."""
    root, db_path = vault
    full_index(str(root), db_path)

    (root / "empty.md").write_text("", encoding="utf-8")
    fake_openai["fail"] = True
    incremental_index(str(root), db_path)

    assert load_manifest(db_path)["empty.md"] == (root / "empty.md").stat().st_mtime


# --- Full reindex ---


def test_full_reindex_failure_keeps_the_existing_index(fake_openai, vault):
    root, db_path = vault
    full_index(str(root), db_path)
    before = sorted((p, c) for p, c, _ in _rows(db_path))
    old_mtime = load_manifest(db_path)["alpha.md"]

    _write(root / "alpha.md", "Alpha", "Changed after the last good index.", bump=10)
    fake_openai["fail"] = True
    with pytest.raises(Exception, match="insufficient_quota"):
        full_index(str(root), db_path)

    assert sorted((p, c) for p, c, _ in _rows(db_path)) == before
    assert _zero_vector_rows(db_path) == []
    # The manifest must not claim the changed note was indexed, or the next
    # incremental run would skip it.
    assert load_manifest(db_path)["alpha.md"] == old_mtime
