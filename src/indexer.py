"""Vault indexing: scan files, chunk, embed, store in LanceDB."""

import os
import time
import logging
from pathlib import Path

import lancedb
import pyarrow as pa

from . import embeddings
from .chunker import chunk_markdown

logger = logging.getLogger(__name__)

SKIP_DIRS = {".obsidian", ".git", ".trash", "6. Media", "TaskNotes/Views", "_backups"}
SKIP_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".mp3", ".mp4", ".m4a", ".wav"}

TABLE_NAME = "vault_chunks"


def create_or_rebuild_fts_index(table: lancedb.table.Table) -> None:
    """Create or rebuild the full-text search index on content, title, and tags."""
    try:
        table.create_fts_index(["content", "title", "tags"], replace=True)
        logger.info("FTS index created/rebuilt on content, title, tags columns")
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
    dim = embeddings.get_dimensions()

    # Drop existing table
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)

    files = scan_vault(vault_path)
    logger.info(f"Scanning {len(files)} markdown files...")

    all_chunks = []
    for rel_path, mtime in files:
        full_path = vault / rel_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Could not read {rel_path}: {e}")
            continue

        chunks = chunk_markdown(rel_path, content, file_mtime=mtime)
        for chunk in chunks:
            chunk["file_mtime"] = mtime
        all_chunks.extend(chunks)

    if not all_chunks:
        return {"files_indexed": 0, "chunks_created": 0, "files_removed": 0, "duration_seconds": 0}

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

    # Get indexed file states from LanceDB
    df = table.to_pandas()
    indexed_mtimes = {}
    for _, row in df[["file_path", "file_mtime"]].drop_duplicates("file_path").iterrows():
        indexed_mtimes[row["file_path"]] = row["file_mtime"]

    # Find files that need reindexing
    to_reindex = []
    for rel_path, mtime in current_files.items():
        if rel_path not in indexed_mtimes or mtime > indexed_mtimes[rel_path]:
            to_reindex.append((rel_path, mtime))

    # Find files that were deleted
    deleted = set(indexed_mtimes.keys()) - set(current_files.keys())

    if not to_reindex and not deleted:
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
    for rel_path, mtime in to_reindex:
        full_path = vault / rel_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
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
                    "tags": ",".join(chunk["tags"]) if isinstance(chunk["tags"], list) else str(chunk["tags"]),
                    "projects": ",".join(chunk["projects"]) if isinstance(chunk["projects"], list) else str(chunk["projects"]),
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

    duration = time.time() - start
    return {
        "files_indexed": len(to_reindex),
        "chunks_created": len(new_chunks),
        "files_removed": len(deleted),
        "duration_seconds": round(duration, 2),
    }
