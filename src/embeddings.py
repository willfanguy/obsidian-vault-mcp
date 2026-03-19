"""Embedding generation with OpenAI (primary) and Ollama (fallback)."""

import os
import logging

logger = logging.getLogger(__name__)

OPENAI_MODEL = "text-embedding-3-large"
OPENAI_DIMENSIONS = 3072
OPENAI_MAX_TOKENS = 8000  # model limit is 8192, leave headroom
OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_DIMENSIONS = 768


def get_provider() -> str:
    return os.getenv("EMBEDDING_PROVIDER", "openai")


def get_dimensions() -> int:
    return OPENAI_DIMENSIONS if get_provider() == "openai" else OLLAMA_DIMENSIONS


def embed_texts(texts: list[str]) -> list[list[float]]:
    provider = get_provider()
    if provider == "openai":
        return _embed_openai(texts)
    return _embed_ollama(texts)


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def _truncate_for_openai(text: str) -> str:
    """Truncate text to fit within OpenAI's token limit. Rough estimate: 1 token ~ 4 chars."""
    max_chars = OPENAI_MAX_TOKENS * 4
    if len(text) > max_chars:
        logger.debug(f"Truncating text from {len(text)} to {max_chars} chars")
        return text[:max_chars]
    return text


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    all_embeddings = []
    for i in range(0, len(texts), 50):
        batch = [_truncate_for_openai(t) for t in texts[i : i + 50]]
        try:
            response = client.embeddings.create(
                model=OPENAI_MODEL,
                input=batch,
                dimensions=OPENAI_DIMENSIONS,
            )
            all_embeddings.extend([item.embedding for item in response.data])
        except Exception as e:
            logger.warning(f"Batch embedding failed at index {i}: {e}. Embedding individually.")
            for text in batch:
                try:
                    resp = client.embeddings.create(
                        model=OPENAI_MODEL,
                        input=[_truncate_for_openai(text[:16000])],
                        dimensions=OPENAI_DIMENSIONS,
                    )
                    all_embeddings.append(resp.data[0].embedding)
                except Exception as e2:
                    logger.error(f"Single embedding failed: {e2}. Using zero vector.")
                    all_embeddings.append([0.0] * OPENAI_DIMENSIONS)
    return all_embeddings


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    import ollama

    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    client = ollama.Client(host=url)
    embeddings = []
    for text in texts:
        response = client.embed(model=OLLAMA_MODEL, input=text)
        embeddings.append(response["embeddings"][0])
    return embeddings
