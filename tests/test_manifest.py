"""Tests for the processed-file manifest that fixes the perpetual-pending bug.

Background: the chunk table only stores rows for files that produced >=1 chunk.
Empty / heading-only notes yield zero chunks, so they never appear in the table
and were therefore counted as "pending reindex" on every run, forever. The
manifest records the mtime of every processed file (chunked or not) so those
notes settle.

The manifest I/O helpers (load_manifest / save_manifest) are pure filesystem
functions and are tested directly. The pending-detection predicate is replicated
here (as test_search_scoring.py does for the scoring formula) so we assert the
mathematical invariant, not just that the code runs.
"""

import json

from src.indexer import load_manifest, save_manifest, _manifest_path


# --- Manifest I/O round-trip ---


def test_manifest_round_trip(tmp_path):
    db_path = str(tmp_path / "vault.lance")
    mtimes = {"a.md": 1.0, "dir/b.md": 2.5, "empty-note.md": 3.25}

    save_manifest(mtimes, db_path)
    loaded = load_manifest(db_path)

    assert loaded == mtimes


def test_manifest_missing_returns_none(tmp_path):
    db_path = str(tmp_path / "vault.lance")
    assert load_manifest(db_path) is None


def test_manifest_corrupt_returns_none(tmp_path):
    db_path = str(tmp_path / "vault.lance")
    manifest_path = _manifest_path(db_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("{ this is not valid json")

    assert load_manifest(db_path) is None


def test_manifest_path_is_sidecar_to_db(tmp_path):
    db_path = str(tmp_path / "nested" / "vault.lance")
    assert _manifest_path(db_path) == str(tmp_path / "nested" / "index_manifest.json")


def test_manifest_values_coerced_to_float(tmp_path):
    """mtimes serialized as ints/strings load back as floats."""
    db_path = str(tmp_path / "vault.lance")
    manifest_path = _manifest_path(db_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"a.md": 5, "b.md": "7.5"}, f)

    loaded = load_manifest(db_path)
    assert loaded == {"a.md": 5.0, "b.md": 7.5}
    assert all(isinstance(v, float) for v in loaded.values())


def test_save_manifest_overwrites(tmp_path):
    db_path = str(tmp_path / "vault.lance")
    save_manifest({"a.md": 1.0}, db_path)
    save_manifest({"b.md": 2.0}, db_path)
    assert load_manifest(db_path) == {"b.md": 2.0}


# --- Pending-detection predicate (replicated from indexer/search) ---


def is_pending(rel_path: str, mtime: float, known_mtimes: dict[str, float]) -> bool:
    """Replication of the pending / to_reindex predicate.

    A file is pending if we've never processed it, or if it's changed since.
    """
    return rel_path not in known_mtimes or mtime > known_mtimes[rel_path]


def test_zero_chunk_file_in_manifest_is_not_pending():
    """The core regression: a note with no chunk rows but recorded in the manifest
    must NOT be considered pending. Before the fix, known_mtimes came from chunk
    rows only, so this file was absent and pending forever."""
    known_mtimes = {"empty-note.md": 100.0}
    assert is_pending("empty-note.md", 100.0, known_mtimes) is False


def test_new_file_is_pending():
    assert is_pending("brand-new.md", 100.0, {}) is True


def test_modified_file_is_pending():
    known_mtimes = {"note.md": 100.0}
    assert is_pending("note.md", 150.0, known_mtimes) is True


def test_unchanged_file_is_not_pending():
    known_mtimes = {"note.md": 100.0}
    assert is_pending("note.md", 100.0, known_mtimes) is False
