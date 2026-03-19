"""Search operations over the LanceDB index."""

import os
import logging

import lancedb

from . import embeddings
from .indexer import get_db, get_table, TABLE_NAME, scan_vault
from .models import SearchResult, NoteContent, NoteMetadata, IndexStatus

logger = logging.getLogger(__name__)


def semantic_search(
    query: str,
    top_k: int = 10,
    tags: list[str] | None = None,
) -> list[SearchResult]:
    """Search vault by semantic similarity."""
    db = get_db()
    table = get_table(db)
    if table is None:
        return []

    query_vector = embeddings.embed_query(query)

    results = table.search(query_vector).limit(top_k * 3).to_pandas()

    if results.empty:
        return []

    # Filter by tags if specified
    if tags:
        tag_set = set(tags)
        mask = results["tags"].apply(
            lambda t: bool(tag_set & set(t.split(","))) if t else False
        )
        results = results[mask]

    results = results.head(top_k)

    return [
        SearchResult(
            title=row.get("title", ""),
            file_path=row["file_path"],
            heading=row.get("heading") or None,
            snippet=row["content"][:300],
            score=float(1 - row.get("_distance", 0)),
            tags=row.get("tags", "").split(",") if row.get("tags") else [],
            projects=row.get("projects", "").split(",") if row.get("projects") else [],
            area=row.get("area") or None,
        )
        for _, row in results.iterrows()
    ]


def hybrid_search(
    query: str,
    top_k: int = 10,
) -> list[SearchResult]:
    """Combine semantic search with keyword matching."""
    db = get_db()
    table = get_table(db)
    if table is None:
        return []

    # Semantic results
    query_vector = embeddings.embed_query(query)
    sem_results = table.search(query_vector).limit(top_k * 2).to_pandas()

    # Keyword results (full-text search on content)
    try:
        kw_results = table.search(query, query_type="fts").limit(top_k * 2).to_pandas()
    except Exception as e:
        logger.warning(f"FTS search failed (index may not exist): {e}")
        kw_results = sem_results.iloc[0:0]

    # Merge and deduplicate, boost items that appear in both
    seen = {}
    for _, row in sem_results.iterrows():
        key = f"{row['file_path']}:{row['chunk_index']}"
        seen[key] = {
            "row": row,
            "sem_score": 1 - row.get("_distance", 0),
            "kw_score": 0,
        }

    for _, row in kw_results.iterrows():
        key = f"{row['file_path']}:{row['chunk_index']}"
        if key in seen:
            seen[key]["kw_score"] = 1.0
        else:
            seen[key] = {
                "row": row,
                "sem_score": 0,
                "kw_score": 1.0,
            }

    # Combined score: 70% semantic + 30% keyword, boost if both match
    ranked = []
    for key, data in seen.items():
        combined = data["sem_score"] * 0.7 + data["kw_score"] * 0.3
        if data["sem_score"] > 0 and data["kw_score"] > 0:
            combined *= 1.2  # Boost items found by both methods
        ranked.append((combined, data["row"]))

    ranked.sort(key=lambda x: x[0], reverse=True)

    return [
        SearchResult(
            title=row.get("title", ""),
            file_path=row["file_path"],
            heading=row.get("heading") or None,
            snippet=row["content"][:300],
            score=round(score, 4),
            tags=row.get("tags", "").split(",") if row.get("tags") else [],
            projects=row.get("projects", "").split(",") if row.get("projects") else [],
            area=row.get("area") or None,
        )
        for score, row in ranked[:top_k]
    ]


def get_note(vault_path: str, file_path: str) -> NoteContent | None:
    """Read a full note from the vault."""
    from pathlib import Path
    from .chunker import parse_frontmatter

    full_path = Path(vault_path) / file_path
    if not full_path.exists():
        return None

    content = full_path.read_text(encoding="utf-8", errors="replace")
    fm, body = parse_frontmatter(content)

    return NoteContent(
        file_path=file_path,
        title=fm.get("title", full_path.stem),
        content=body,
        frontmatter=fm,
    )


def list_by_metadata(
    tags: list[str] | None = None,
    projects: list[str] | None = None,
    status: str | None = None,
    area: str | None = None,
) -> list[NoteMetadata]:
    """Query notes by frontmatter metadata (no embeddings needed)."""
    db = get_db()
    table = get_table(db)
    if table is None:
        return []

    df = table.to_pandas()

    # Deduplicate to file level
    df = df.drop_duplicates(subset=["file_path"])

    if tags:
        tag_set = set(tags)
        df = df[df["tags"].apply(lambda t: bool(tag_set & set(t.split(","))) if t else False)]

    if projects:
        proj_set = set(projects)
        df = df[df["projects"].apply(lambda p: bool(proj_set & set(p.split(","))) if p else False)]

    if status:
        df = df[df["status"] == status]

    if area:
        df = df[df["area"].str.contains(area, case=False, na=False)]

    return [
        NoteMetadata(
            file_path=row["file_path"],
            title=row.get("title", ""),
            tags=row.get("tags", "").split(",") if row.get("tags") else [],
            projects=row.get("projects", "").split(",") if row.get("projects") else [],
            status=row.get("status") or None,
            area=row.get("area") or None,
        )
        for _, row in df.iterrows()
    ]


def index_status() -> IndexStatus:
    """Get current index status."""
    db = get_db()
    table = get_table(db)

    if table is None:
        return IndexStatus(total_chunks=0, total_files=0)

    df = table.to_pandas()
    total_chunks = len(df)
    total_files = df["file_path"].nunique()

    # Check for pending reindex
    vault_path = os.getenv("VAULT_PATH", "")
    pending = 0
    if vault_path:
        current_files = dict(scan_vault(vault_path))
        indexed_mtimes = {}
        for _, row in df[["file_path", "file_mtime"]].drop_duplicates("file_path").iterrows():
            indexed_mtimes[row["file_path"]] = row["file_mtime"]

        for rel_path, mtime in current_files.items():
            if rel_path not in indexed_mtimes or mtime > indexed_mtimes[rel_path]:
                pending += 1

    # DB size
    db_path = os.getenv("LANCE_DB_PATH", "./data/vault.lance")
    db_size = 0.0
    if os.path.exists(db_path):
        for dirpath, _, filenames in os.walk(db_path):
            for f in filenames:
                db_size += os.path.getsize(os.path.join(dirpath, f))
    db_size_mb = round(db_size / (1024 * 1024), 2)

    return IndexStatus(
        total_chunks=total_chunks,
        total_files=total_files,
        pending_reindex=pending,
        db_size_mb=db_size_mb,
    )
