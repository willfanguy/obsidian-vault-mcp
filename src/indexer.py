"""Vault indexing: scan files, chunk, embed, store in LanceDB."""

import os
import json
import time
import logging
from pathlib import Path

import lancedb

from . import embeddings
from .chunker import chunk_markdown

logger = logging.getLogger(__name__)

SKIP_DIRS = {".obsidian", ".git", ".trash", "6. Media", "TaskNotes/Views", "_backups"}
SKIP_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".mp3", ".mp4", ".m4a", ".wav"}

TABLE_NAME = "vault_chunks"
MANIFEST_NAME = "index_manifest.json"


def _manifest_path(db_path: str | None = None) -> str:
    """Path to the processed-file manifest (sidecar to the LanceDB directory).

    The chunk table only holds rows for files that produced at least one chunk,
    so it can't answer "have we already processed this file?" for empty or
    heading-only notes (they yield zero chunks). Without a separate record, those
    files are re-scanned on every run and counted as perpetually "pending." The
    manifest records the mtime of every file we've processed — regardless of
    chunk count — so zero-chunk notes settle instead of looping forever.
    """
    path = db_path or os.getenv("LANCE_DB_PATH", "./data/vault.lance")
    return os.path.join(os.path.dirname(path) or ".", MANIFEST_NAME)


def load_manifest(db_path: str | None = None) -> dict[str, float] | None:
    """Load the processed-file manifest. Returns None if it hasn't been written yet."""
    manifest_path = _manifest_path(db_path)
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): float(v) for k, v in data.items()}
    except Exception as e:
        logger.warning(f"Could not read index manifest ({manifest_path}): {e}")
        return None


def save_manifest(mtimes: dict[str, float], db_path: str | None = None) -> None:
    """Atomically write the processed-file manifest (path -> mtime)."""
    manifest_path = _manifest_path(db_path)
    try:
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        tmp_path = f"{manifest_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(mtimes, f)
        os.replace(tmp_path, manifest_path)
    except Exception as e:
        logger.warning(f"Could not write index manifest ({manifest_path}): {e}")


def create_or_rebuild_fts_index(table: lancedb.table.Table) -> None:
    """Create or rebuild the FTS index on the content column.

    LanceDB native FTS only supports single-field indexes.
    Title/tag matching is handled by the semantic side via metadata-enriched embeddings.
    """
    try:
        table.create_fts_index("content", replace=True)
        logger.info("FTS index created/rebuilt on 'content' column")
    except Exception as e:
        logger.warning(f"Failed to create FTS index: {e}")


def get_db(db_path: str | None = None) -> lancedb.DBConnection:
    path = db_path or os.getenv("LANCE_DB_PATH", "./data/vault.lance")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return lancedb.connect(path)


def get_table(db: lancedb.DBConnection) -> lancedb.table.Table | None:
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return None


def scan_vault(vault_path: str) -> list[tuple[str, float]]:
    """Scan vault and return list of (relative_path, mtime) for .md files."""
    vault = Path(vault_path)
    results = []

    for path in vault.rglob("*.md"):
        rel = str(path.relative_to(vault))

        # Skip excluded directories
        if any(rel.startswith(d) or f"/{d}/" in f"/{rel}" for d in SKIP_DIRS):
            continue

        results.append((rel, path.stat().st_mtime))

    return results


def full_index(vault_path: str, db_path: str | None = None, batch_size: int = 50) -> dict:
    """Build a complete index from scratch."""
    start = time.time()
    vault = Path(vault_path)
    db = get_db(db_path)
    embeddings.get_dimensions()  # warm up embedding provider

    # Drop existing table
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)

    files = scan_vault(vault_path)
    logger.info(f"Scanning {len(files)} markdown files...")

    all_chunks = []
    processed_mtimes: dict[str, float] = {}
    for rel_path, mtime in files:
        full_path = vault / rel_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Could not read {rel_path}: {e}")
            continue

        # Record every file we successfully read, even if it produces zero
        # chunks, so empty/heading-only notes aren't flagged as pending forever.
        processed_mtimes[rel_path] = mtime
        chunks = chunk_markdown(rel_path, content, file_mtime=mtime)
        for chunk in chunks:
            chunk["file_mtime"] = mtime
        all_chunks.extend(chunks)

    # Persist the manifest regardless of whether any chunks were produced.
    save_manifest(processed_mtimes, db_path)

    if not all_chunks:
        return {
            "files_indexed": len(processed_mtimes),
            "chunks_created": 0,
            "files_removed": 0,
            "duration_seconds": round(time.time() - start, 2),
        }

    logger.info(f"Embedding {len(all_chunks)} chunks...")

    # Embed in batches
    all_vectors = []
    texts = [c["text_to_embed"] for c in all_chunks]
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = embeddings.embed_texts(batch)
        all_vectors.extend(vectors)
        logger.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    # Build records for LanceDB
    records = []
    for chunk, vector in zip(all_chunks, all_vectors):
        records.append(
            {
                "file_path": chunk["file_path"],
                "chunk_index": chunk["chunk_index"],
                "heading": chunk["heading"],
                "content": chunk["content"],
                "title": chunk["title"],
                "tags": ",".join(str(t) for t in chunk["tags"] if t is not None) if isinstance(chunk["tags"], list) else str(chunk["tags"] or ""),
                "projects": ",".join(str(p) for p in chunk["projects"] if p is not None) if isinstance(chunk["projects"], list) else str(chunk["projects"] or ""),
                "area": str(chunk["area"]) if chunk["area"] is not None else "",
                "status": str(chunk["status"]) if chunk["status"] is not None else "",
                "source": str(chunk["source"]) if chunk["source"] is not None else "",
                "file_mtime": chunk["file_mtime"],
                "vector": vector,
            }
        )

    table = db.create_table(TABLE_NAME, data=records)
    create_or_rebuild_fts_index(table)

    duration = time.time() - start
    unique_files = len(set(r["file_path"] for r in records))
    logger.info(f"Indexed {unique_files} files, {len(records)} chunks in {duration:.1f}s")

    return {
        "files_indexed": unique_files,
        "chunks_created": len(records),
        "files_removed": 0,
        "duration_seconds": round(duration, 2),
    }


def incremental_index(vault_path: str, db_path: str | None = None, batch_size: int = 50) -> dict:
    """Update index with only changed/new files."""
    start = time.time()
    vault = Path(vault_path)
    db = get_db(db_path)

    table = get_table(db)
    if table is None:
        return full_index(vault_path, db_path, batch_size)

    # Get current file states
    current_files = dict(scan_vault(vault_path))

    # Determine what we've already processed and at which mtime. Prefer the
    # manifest (records every processed file, including zero-chunk notes); fall
    # back to chunk-derived mtimes for the first run after upgrading, before a
    # manifest exists.
    known_mtimes = load_manifest(db_path)
    manifest_missing = known_mtimes is None
    if manifest_missing:
        df = table.to_pandas()
        known_mtimes = {}
        for _, row in df[["file_path", "file_mtime"]].drop_duplicates("file_path").iterrows():
            known_mtimes[row["file_path"]] = row["file_mtime"]

    # Find files that need reindexing
    to_reindex = []
    for rel_path, mtime in current_files.items():
        if rel_path not in known_mtimes or mtime > known_mtimes[rel_path]:
            to_reindex.append((rel_path, mtime))

    # Find files that were deleted
    deleted = set(known_mtimes.keys()) - set(current_files.keys())

    if not to_reindex and not deleted:
        # Nothing to do. Seed the manifest on the first run after upgrading so
        # the chunk-derived fallback doesn't recur (and zero-chunk files settle).
        if manifest_missing:
            save_manifest(dict(current_files), db_path)
        return {
            "files_indexed": 0,
            "chunks_created": 0,
            "files_removed": 0,
            "duration_seconds": round(time.time() - start, 2),
        }

    # Remove old chunks for files being reindexed or deleted
    paths_to_remove = set(p for p, _ in to_reindex) | deleted
    if paths_to_remove:
        filter_expr = " OR ".join(f'file_path = "{p}"' for p in paths_to_remove)
        table.delete(filter_expr)

    # Index new/changed files
    new_chunks = []
    failed: set[str] = set()
    for rel_path, mtime in to_reindex:
        full_path = vault / rel_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            failed.add(rel_path)
            continue
        chunks = chunk_markdown(rel_path, content, file_mtime=mtime)
        for chunk in chunks:
            chunk["file_mtime"] = mtime
        new_chunks.extend(chunks)

    if new_chunks:
        texts = [c["text_to_embed"] for c in new_chunks]
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_vectors.extend(embeddings.embed_texts(batch))

        records = []
        for chunk, vector in zip(new_chunks, all_vectors):
            records.append(
                {
                    "file_path": chunk["file_path"],
                    "chunk_index": chunk["chunk_index"],
                    "heading": chunk["heading"],
                    "content": chunk["content"],
                    "title": chunk["title"],
                    "tags": ",".join(str(t) for t in chunk["tags"] if t is not None) if isinstance(chunk["tags"], list) else str(chunk["tags"] or ""),
                    "projects": ",".join(str(p) for p in chunk["projects"] if p is not None) if isinstance(chunk["projects"], list) else str(chunk["projects"] or ""),
                    "area": chunk["area"] or "",
                    "status": chunk["status"] or "",
                    "source": chunk["source"] or "",
                    "file_mtime": chunk["file_mtime"],
                    "vector": vector,
                }
            )
        table.add(records)

    # Rebuild FTS index after any modifications (adds or deletes)
    create_or_rebuild_fts_index(table)

    # Update the manifest to reflect the current on-disk state. After a run every
    # current file is processed at its current mtime (unchanged-and-known, or just
    # reindexed). Files that failed to read keep their prior state so they retry.
    new_manifest = dict(current_files)
    for rel_path in failed:
        if rel_path in known_mtimes:
            new_manifest[rel_path] = known_mtimes[rel_path]
        else:
            new_manifest.pop(rel_path, None)
    save_manifest(new_manifest, db_path)

    duration = time.time() - start
    return {
        "files_indexed": len(to_reindex),
        "chunks_created": len(new_chunks),
        "files_removed": len(deleted),
        "duration_seconds": round(duration, 2),
    }
