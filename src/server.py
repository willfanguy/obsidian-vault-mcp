"""FastMCP server exposing vault semantic search tools."""

import os
import logging

from dotenv import load_dotenv
from fastmcp import FastMCP

from .search import semantic_search, hybrid_search, get_note, list_by_metadata, index_status
from .indexer import full_index, incremental_index

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("obsidian-vault-search")

VAULT_PATH = os.getenv("VAULT_PATH", "")
API_KEY = os.getenv("VAULT_API_KEY", "")


class APIKeyMiddleware:
    """Pure ASGI middleware that rejects requests without a valid API key.

    Uses raw ASGI instead of BaseHTTPMiddleware to avoid conflicts
    with SSE streaming responses.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        path = scope.get("path", "")

        # Allow without auth: health check, SSE stream (read-only until messages are sent)
        if scope["type"] == "http" and path == "/":
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"status": "ok", "service": "obsidian-vault-search"}',
            })
            return

        # SSE stream endpoint — allow without auth for connector validation.
        # The stream is inert until authenticated POST /messages/ calls execute tools.
        if path == "/sse":
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()

        if auth == f"Bearer {API_KEY}":
            return await self.app(scope, receive, send)

        # Reject with 401
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({
            "type": "http.response.body",
            "body": b'{"error": "unauthorized"}',
        })


@mcp.tool()
def vault_search(query: str, top_k: int = 10, tags: list[str] | None = None) -> str:
    """Search the Obsidian vault by semantic similarity.

    Args:
        query: Natural language search query (e.g., "decisions about SuperFit voice tone")
        top_k: Maximum number of results to return (default 10)
        tags: Optional list of tags to filter by (e.g., ["task"], ["reference", "cooking"])

    Returns:
        Ranked search results with title, path, snippet, score, and metadata.
    """
    results = semantic_search(query, top_k=top_k, tags=tags)
    if not results:
        return "No results found."

    lines = [f"Found {len(results)} results:\n"]
    for i, r in enumerate(results, 1):
        heading = f" > {r.heading}" if r.heading else ""
        tags_str = f" [{', '.join(r.tags)}]" if r.tags else ""
        lines.append(f"**{i}. {r.title or r.file_path}**{heading}{tags_str}")
        lines.append(f"   Path: {r.file_path}")
        lines.append(f"   Score: {r.score:.3f}")
        lines.append(f"   {r.snippet}...")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def vault_search_hybrid(query: str, top_k: int = 10) -> str:
    """Search the vault combining semantic similarity with keyword matching.

    Better for queries that mix concepts with specific terms
    (e.g., "SuperFit alpha timeline decisions").

    Args:
        query: Search query mixing natural language and keywords
        top_k: Maximum number of results

    Returns:
        Ranked results combining semantic and keyword relevance.
    """
    results = hybrid_search(query, top_k=top_k)
    if not results:
        return "No results found."

    lines = [f"Found {len(results)} results (hybrid search):\n"]
    for i, r in enumerate(results, 1):
        heading = f" > {r.heading}" if r.heading else ""
        lines.append(f"**{i}. {r.title or r.file_path}**{heading}")
        lines.append(f"   Path: {r.file_path}")
        lines.append(f"   Score: {r.score:.3f}")
        lines.append(f"   {r.snippet}...")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def vault_get_note(path: str) -> str:
    """Retrieve the full content of a vault note.

    Use after search to read a complete note that appeared in results.

    Args:
        path: Relative path in the vault (e.g., "4. Resources/Work Log/Tasks/Some Task.md")

    Returns:
        Full note content with parsed frontmatter.
    """
    note = get_note(VAULT_PATH, path)
    if note is None:
        return f"Note not found: {path}"

    lines = [f"# {note.title}", f"Path: {note.file_path}", ""]
    if note.frontmatter:
        lines.append("**Frontmatter:**")
        for k, v in note.frontmatter.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    lines.append("**Content:**")
    lines.append(note.content)

    return "\n".join(lines)


@mcp.tool()
def vault_list_by_metadata(
    tags: list[str] | None = None,
    projects: list[str] | None = None,
    status: str | None = None,
    area: str | None = None,
) -> str:
    """Query vault notes by frontmatter metadata (no semantic search needed).

    Args:
        tags: Filter by tags (e.g., ["task"], ["reference", "cooking"])
        projects: Filter by project names (e.g., ["SuperFit", "AI-Foundations"])
        status: Filter by status (e.g., "open", "done", "in-progress")
        area: Filter by area name (e.g., "Health", "Professional Development")

    Returns:
        List of matching notes with their metadata.
    """
    results = list_by_metadata(tags=tags, projects=projects, status=status, area=area)
    if not results:
        return "No matching notes found."

    lines = [f"Found {len(results)} notes:\n"]
    for r in results:
        tags_str = f" [{', '.join(r.tags)}]" if r.tags else ""
        status_str = f" ({r.status})" if r.status else ""
        lines.append(f"- **{r.title or r.file_path}**{status_str}{tags_str}")
        lines.append(f"  {r.file_path}")

    return "\n".join(lines)


@mcp.tool()
def vault_index_status() -> str:
    """Check the current state of the vault search index.

    Returns:
        Index statistics: total files/chunks indexed, pending reindex count, DB size.
    """
    status = index_status()
    return (
        f"Vault Index Status:\n"
        f"  Files indexed: {status.total_files}\n"
        f"  Total chunks: {status.total_chunks}\n"
        f"  Pending reindex: {status.pending_reindex}\n"
        f"  DB size: {status.db_size_mb} MB"
    )


@mcp.tool()
def vault_reindex(path: str | None = None) -> str:
    """Reindex the vault (or a single file).

    Args:
        path: Optional file path to reindex. If omitted, does incremental reindex of all changed files.

    Returns:
        Reindex results: files processed, chunks created, duration.
    """
    vault_path = VAULT_PATH
    if path:
        # Single file reindex - just do incremental (it handles the diff)
        result = incremental_index(vault_path)
    else:
        result = incremental_index(vault_path)

    return (
        f"Reindex complete:\n"
        f"  Files indexed: {result['files_indexed']}\n"
        f"  Chunks created: {result['chunks_created']}\n"
        f"  Files removed: {result['files_removed']}\n"
        f"  Duration: {result['duration_seconds']}s"
    )


def main():
    """Entry point for the MCP server."""
    port = int(os.getenv("MCP_PORT", "3789"))
    transport = os.getenv("MCP_TRANSPORT", "sse")

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif API_KEY:
        # Run with API key auth middleware (pure ASGI, SSE-safe)
        import uvicorn

        sse_app = mcp.http_app(transport="sse")
        app = APIKeyMiddleware(sse_app)
        logger.info(f"Starting with API key auth on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        mcp.run(transport="sse", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
