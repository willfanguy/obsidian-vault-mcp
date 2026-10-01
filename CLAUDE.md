# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Python FastMCP server providing semantic search over an Obsidian vault using LanceDB vector database with OpenAI or Ollama embeddings. Exposes MCP tools for hybrid search (semantic + full-text), note retrieval, metadata queries, and reindexing.

## Common Commands

```bash
# Install in development mode
pip install -e .

# Run the server (streamable HTTP on MCP_PORT, default 3789)
vault-mcp

# Run in stdio mode (MCP Inspector, Claude Desktop direct)
MCP_TRANSPORT=stdio vault-mcp

# Full reindex of vault
.venv/bin/python scripts/full_reindex.py

# Incremental reindex (only changed/new files)
.venv/bin/python scripts/incremental_reindex.py
```

## Testing

```bash
# Run all tests
.venv/bin/pytest tests/ -v

# Install test deps (if venv exists)
uv pip install --python .venv/bin/python -e ".[test]"
```

- **Frameworks**: pytest + hypothesis (property-based testing)
- **Test location**: `tests/` directory
- **Policy**: Follows workspace-level Testing Standards. Pure functions tested directly; search scoring tested via formula replication, not live LanceDB. Two deliberate exceptions hit a scratch LanceDB table. The delete-filter test does, because the bug it guards was a wrong assumption about LanceDB's SQL dialect, and a replicated formula would have reproduced the same wrong assumption. The embedding-failure tests do, because the bug they guard lived in what got *stored* (zero vectors, an advanced manifest), which only a real table and manifest can show. They fake only the `openai` module, at the import `_embed_openai` uses.

### Test coverage

- `chunker.py` — frontmatter parsing, wikilink cleaning, metadata headers, HTML stripping, heading splitting, paragraph overlap chunking, full integration
- `search.py` — hybrid scoring formula (70/30 weighting, dual-match 1.2x boost), BM25 normalization
- `indexer.py` — manifest I/O and the pending-reindex predicate; `sql_string_literal` /
  `build_path_delete_filter` quoting and escaping, plus one live scratch-table
  delete that pins the LanceDB filter dialect
- `server.py` — `APIKeyMiddleware` only: both credentials, 404-vs-401 split,
  prefix stripping, trailing-slash normalisation, and the hostile cases
  (segment merely *starting* with the secret, secret in a later segment, blank
  prefix, non-ASCII header/path, lifespan passthrough). Driven as raw ASGI via
  `asyncio.run`, so no pytest-asyncio dependency. Both the `startswith` bug and
  the missing-normalisation bug were mutation-checked to confirm the tests bite.
- Embedding failures (`tests/test_embedding_failures.py`) — a failing provider
  must raise `EmbeddingError`, never return a placeholder vector. End to end
  over a temp vault: search raises during an outage; an incremental run stores
  no zero vectors, keeps a failed file's old chunks and manifest entry, and
  retries it once the provider recovers; one rejected chunk fails only its own
  file; a failed full reindex leaves the previous table and manifest untouched.
  Mutation-checked (restoring the zero-fill fails 7; deleting old chunks before
  embedding fails 1). Guards the 2026-10-01 outage that stored 111 zero chunks.

### Not yet tested

- `embeddings.py` — truncation, and the Ollama path beyond error wrapping
- `indexer.py` — `scan_vault` skip-dir rules
- `server.py` — MCP tool registrations (the middleware *is* covered, above)

## Embedding failures — never zero-fill

`embeddings.embed_texts` raises `EmbeddingError` when the provider can't embed a
text. Don't catch it and substitute anything: a zero vector is equidistant from
every chunk, so it turns search into identical junk results and turns indexed
notes into rows that look done but can never match. `incremental_index` handles
it per file (keep old chunks, keep the old manifest entry, retry next run);
`full_index` lets it propagate before touching the existing table.

## Architecture

```
src/
  server.py      — MCP server (FastMCP tools + two-credential ASGI auth middleware)
  indexer.py     — Vault scanning, full/incremental indexing, FTS table creation
  search.py      — Semantic search, full-text search, hybrid (70/30 weighting)
  embeddings.py  — Embedding provider abstraction (OpenAI API / Ollama local)
  chunker.py     — Heading-aware markdown splitting with overlap, metadata-enriched text
  models.py      — Pydantic types (SearchResult, NoteMetadata, IndexStats)
```

**Data flow:** Vault markdown files → chunker (heading-aware split) → embeddings provider → LanceDB vector table. Search queries go through the same embedding path, then LanceDB ANN + optional FTS reranking.

## MCP Tools

- `vault_search` — Pure semantic search (cosine similarity)
- `vault_search_hybrid` — Semantic + full-text fusion (default, best results)
- `vault_get_note` — Retrieve full note content by path
- `vault_list_by_metadata` — Query notes by frontmatter fields
- `vault_reindex` — Trigger incremental reindex
- `vault_index_status` — Check index health and stats

## Configuration

All config via environment variables (see `.env.example`):

- `VAULT_PATH` — Path to Obsidian vault (required)
- `EMBEDDING_PROVIDER` — `openai` or `ollama` (default: openai)
- `OPENAI_API_KEY` — Required if using OpenAI embeddings
- `OLLAMA_HOST` — Ollama server URL (default: http://localhost:11434)
- `LANCE_DB_PATH` — Where to store the vector database (default: ./data/lancedb)
- `MCP_PORT` — HTTP port (default: 3789)
- `MCP_TRANSPORT` — `http` (streamable, default) | `stdio`
- `VAULT_API_KEY` — Bearer token for the HTTP server; blank = **no auth**
- `VAULT_FUNNEL_PREFIX` — optional *second* credential: a secret first path
  segment (e.g. `/v-<32 hex>`) that authenticates by itself, for clients that
  cannot send headers. Blank = disabled. See "Two credentials" below.
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — Chunking parameters

## Two credentials (and why)

`APIKeyMiddleware` accepts **either** `Authorization: Bearer <VAULT_API_KEY>`
**or** a leading secret path segment matching `VAULT_FUNNEL_PREFIX`. The prefix
is stripped before the request reaches FastMCP, so the app below always sees
`/mcp` and header-based clients are unaffected.

The second path exists because **Claude custom connectors cannot send request
headers** — the connection is made by Anthropic's servers, not your device, and
the request-header feature is a beta that isn't enabled on this account. So a
funnelled connector has no way to present a Bearer token. Same mechanism as
`service_manuals_mcp`, which is where this pattern came from.

⚠️ **This is a weaker credential guarding stronger content.** In
service-manuals the trade was easy: read-only tools over a freely downloadable
manual. Here the corpus is a private vault and `vault_reindex` both mutates the
index and spends OpenAI credit. Anyone holding the URL gets all of it, and
rotating the prefix is the only revocation. Enabled by explicit decision
(2026-08-04). If this ever needs real revocation or per-client access, replace
the branch with OAuth rather than layering onto it.

⚠️ **FastMCP mounts at `/mcp`, NOT `/mcp/`** (verified on 3.1.1 and 3.2.4). A
request to `/mcp/` gets a 307 whose `Location` Starlette rebuilds from the
rewritten path only — so the secret prefix is dropped and the client follows it
into a 401. The middleware therefore normalises away a trailing slash, so both
`<prefix>/mcp` and `<prefix>/mcp/` work.

Anonymous requests get **404, not 401** — a 401 without `WWW-Authenticate` makes
Claude infer an OAuth server, attempt Dynamic Client Registration, and fail
connector setup. A request that *did* send an `Authorization` header still gets
401, since "your token is wrong" is useful to a misconfigured LAN client.
