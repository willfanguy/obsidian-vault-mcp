"""Pydantic models for the vault search MCP server."""

from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    file_path: str
    heading: str | None = None
    snippet: str
    score: float
    tags: list[str] = []
    projects: list[str] = []
    area: str | None = None


class NoteContent(BaseModel):
    file_path: str
    title: str
    content: str
    frontmatter: dict = {}


class NoteMetadata(BaseModel):
    file_path: str
    title: str
    tags: list[str] = []
    projects: list[str] = []
    status: str | None = None
    area: str | None = None
    created: str | None = None


class IndexStatus(BaseModel):
    total_chunks: int
    total_files: int
    last_index_time: str | None = None
    pending_reindex: int = 0
    db_size_mb: float = 0.0


class ReindexResult(BaseModel):
    files_indexed: int
    chunks_created: int
    files_removed: int
    duration_seconds: float
