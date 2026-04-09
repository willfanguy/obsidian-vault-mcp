# obsidian-vault-mcp

An MCP (Model Context Protocol) server that provides semantic search over an Obsidian vault using LanceDB vector storage and OpenAI or Ollama embeddings.

## Features

- **Semantic search** — find notes by meaning, not just keywords
- **Hybrid search** — combines vector similarity with keyword matching
- **Incremental indexing** — automatically tracks vault changes via filesystem watcher
- **OpenAI or Ollama** — use cloud embeddings (`text-embedding-3-large`) or run fully local with Ollama
- **Bearer token auth** — optional API key middleware for safe public exposure (e.g. via Tailscale Funnel)
- **MCP-compatible** — works with Claude Desktop, Cursor, and any MCP client

## Requirements

- Python 3.10+
- An Obsidian vault
- OpenAI API key **or** a running Ollama instance

## Setup

```bash
# Clone and install
git clone https://github.com/willfanguy/obsidian-vault-mcp
cd obsidian-vault-mcp
pip install -e .

# Configure
cp .env.example .env
# Edit .env — set VAULT_PATH and your embedding provider credentials

# Index your vault (first run)
python scripts/full_reindex.py

# Start the server
vault-mcp
```

## Configuration

Copy `.env.example` to `.env` and set:

| Variable | Description | Default |
|----------|-------------|---------|
| `VAULT_PATH` | Absolute path to your Obsidian vault | *(required)* |
| `EMBEDDING_PROVIDER` | `openai` or `ollama` | `openai` |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) | — |
| `OLLAMA_URL` | Ollama server URL (if using Ollama) | `http://localhost:11434` |
| `LANCE_DB_PATH` | Where to store the vector database | `./data/vault.lance` |
| `MCP_PORT` | HTTP port for the MCP server | `3789` |
| `VAULT_API_KEY` | Bearer token for auth (optional) | *(disabled)* |

## MCP Tools

| Tool | Description |
|------|-------------|
| `semantic_search` | Find notes by semantic similarity |
| `hybrid_search` | Semantic + keyword combined search |
| `get_note` | Retrieve a specific note by path |
| `list_by_metadata` | Filter notes by frontmatter tags/fields |
| `index_status` | Check indexing state |
| `full_index` | Trigger a full re-index |
| `incremental_index` | Index only changed files |

## Claude Desktop Integration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "obsidian-vault": {
      "command": "vault-mcp",
      "env": {
        "VAULT_PATH": "/path/to/your/vault"
      }
    }
  }
}
```

## License

MIT — see [LICENSE](LICENSE)
