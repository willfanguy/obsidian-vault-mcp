# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Python FastMCP server providing semantic search over an Obsidian vault using LanceDB vector database with OpenAI or Ollama embeddings. Exposes MCP tools for hybrid search (semantic + full-text), note retrieval, metadata queries, and reindexing.

## Common Commands

```bash
# Install in development mode
pip install -e .

# Run the server (stdio mode for Claude Desktop)
vault-mcp

# Run as HTTP server with SSE (for remote access)
vault-mcp --transport sse --host 0.0.0.0 --port 8765

# Full reindex of vault
.venv/bin/python scripts/full_reindex.py

# Incremental reindex (only changed/new files)
.venv/bin/python scripts/incremental_reindex.py
```

## Architecture

```
src/
  server.py      — MCP server (FastMCP tools + API key auth middleware for SSE)
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
- `MCP_API_KEY` — API key for SSE transport authentication
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — Chunking parameters
