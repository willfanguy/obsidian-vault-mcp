#!/usr/bin/env python3
"""One-shot full reindex of the vault."""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.indexer import full_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

vault_path = os.getenv("VAULT_PATH")
if not vault_path:
    print("Error: VAULT_PATH environment variable not set (set in .env or export)")
    sys.exit(1)
print(f"Full reindex of: {vault_path}")
print(f"Embedding provider: {os.getenv('EMBEDDING_PROVIDER', 'openai')}")

result = full_index(vault_path)
print(f"\nDone!")
print(f"  Files indexed: {result['files_indexed']}")
print(f"  Chunks created: {result['chunks_created']}")
print(f"  Duration: {result['duration_seconds']}s")
