"""FastMCP server exposing vault semantic search tools."""

import hmac
import os
import logging

from dotenv import load_dotenv
from fastmcp import FastMCP

from .search import semantic_search, hybrid_search, get_note, list_by_metadata, index_status
from .indexer import incremental_index

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("obsidian-vault-search")

VAULT_PATH = os.getenv("VAULT_PATH", "")
API_KEY = os.getenv("VAULT_API_KEY", "")
FUNNEL_PREFIX = os.getenv("VAULT_FUNNEL_PREFIX", "")

# Tool annotations. Everything here reads the vault or the index; only
# vault_reindex writes (to the index — never to the vault itself).
READ_ONLY = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
MUTATING = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


class APIKeyMiddleware:
    """Pure ASGI middleware that rejects requests without a valid credential.

    Uses raw ASGI instead of BaseHTTPMiddleware to avoid conflicts with
    streaming responses. The root path "/" is an unauthenticated health check.
    Every other path needs ONE of two credentials:

    1. `Authorization: Bearer <VAULT_API_KEY>` — what LAN clients (Claude Code,
       Claude Desktop via the mcp-remote shim) use.
    2. A leading secret path segment matching VAULT_FUNNEL_PREFIX, for clients
       that cannot set headers at all (Claude custom connectors). The prefix is
       stripped before the request reaches FastMCP, so the app below still sees
       "/mcp" and existing header-based clients are entirely unaffected.

    ⚠️ Option 2 is deliberately weaker: it trades header secrecy for URL
    secrecy, and a capability URL leaks through proxy logs, referrers and
    browser history in ways a header does not. Unlike service-manuals — where
    the same mechanism guards a freely-downloadable manual — the corpus here is
    Will's PRIVATE vault, and `vault_reindex` both mutates the index and spends
    OpenAI credit. Anyone holding the URL gets all of that, and rotating the
    prefix is the only revocation. Enabled by explicit decision (2026-08-04);
    leave `VAULT_FUNNEL_PREFIX` blank to keep the server bearer-only.
    """

    def __init__(self, app):
        self.app = app

    @staticmethod
    def _match(candidate: str, secret: str) -> bool:
        """Constant-time equality that never raises on odd input."""
        if not secret:
            return False
        return hmac.compare_digest(
            candidate.encode("utf-8", "replace"), secret.encode("utf-8")
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        path = scope.get("path", "")

        # Allow without auth: health check only.
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

        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()

        if self._match(auth, f"Bearer {API_KEY}" if API_KEY else ""):
            return await self.app(scope, receive, send)

        # Split the first path segment and compare it in constant time, rather
        # than str.startswith, so the comparison itself leaks nothing and a
        # longer segment that merely begins with the secret cannot match.
        if FUNNEL_PREFIX:
            head, _, tail = path.lstrip("/").partition("/")
            if self._match("/" + head, FUNNEL_PREFIX):
                # Drop a trailing slash so FastMCP (which mounts at "/mcp", NOT
                # "/mcp/" — verified on 3.1.1 and 3.2.4) answers directly
                # instead of issuing a 307. That redirect is a trap here:
                # Starlette builds the Location from the rewritten path only,
                # so the secret prefix is lost and the client follows it
                # straight into a 401. Normalising here means both
                # "<prefix>/mcp" and "<prefix>/mcp/" work.
                rewritten = "/" + tail
                if len(rewritten) > 1:
                    rewritten = rewritten.rstrip("/")
                scope = dict(scope)
                scope["path"] = rewritten
                if scope.get("raw_path"):
                    scope["raw_path"] = rewritten.encode()
                return await self.app(scope, receive, send)

        # Anonymous requests get 404, not 401 — deliberately.
        #
        # MCP clients treat a 401 as "this resource uses OAuth" and begin a
        # discovery + Dynamic Client Registration flow. Claude probes
        # /.well-known/oauth-authorization-server and
        # /.well-known/oauth-protected-resource at the ORIGIN ROOT when adding a
        # custom connector; answering those with 401 while sending no
        # WWW-Authenticate header advertises an authorization server that does
        # not exist here, and setup fails with "Couldn't register with <name>'s
        # sign-in service."
        #
        # A caller that DID send an Authorization header still gets 401, since
        # "your token is wrong" is real, useful information for a misconfigured
        # client. Auth strength is unchanged either way — anonymous requests are
        # still rejected, just with a status that doesn't advertise OAuth.
        if auth:
            status, body = 401, b'{"error": "unauthorized"}'
        else:
            status, body = 404, b'{"error": "not found"}'

        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": body})


@mcp.tool(annotations=READ_ONLY)
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


@mcp.tool(annotations=READ_ONLY)
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


@mcp.tool(annotations=READ_ONLY)
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


@mcp.tool(annotations=READ_ONLY)
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


@mcp.tool(annotations=READ_ONLY)
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


@mcp.tool(annotations=MUTATING)
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
    """Entry point for the MCP server. Transport + auth chosen from env."""
    port = int(os.getenv("MCP_PORT", "3789"))
    # Default is streamable HTTP. SSE is gone: it raced the client init
    # handshake ("Received request before initialization was complete" /
    # -32602 on every tool call) and the server no longer serves it at all.
    transport = os.getenv("MCP_TRANSPORT", "http")

    if transport == "stdio":
        mcp.run(transport="stdio")
        return

    if API_KEY:
        # Auth middleware wraps the app as raw ASGI. Served at /mcp.
        import uvicorn

        http_app = mcp.http_app(transport="http")
        app = APIKeyMiddleware(http_app)
        logger.info(f"Serving streamable HTTP with API-key auth on :{port}")
        if FUNNEL_PREFIX:
            # Never log the prefix itself — it is a credential.
            logger.info(
                "Secret-path credential also enabled (%d chars)", len(FUNNEL_PREFIX)
            )
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        logger.warning("VAULT_API_KEY unset — serving WITHOUT auth.")
        mcp.run(transport="http", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
