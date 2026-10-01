"""Embedding generation with OpenAI (primary) and Ollama (fallback)."""

import os
import logging

logger = logging.getLogger(__name__)

OPENAI_MODEL = "text-embedding-3-large"
OPENAI_DIMENSIONS = 3072
OPENAI_MAX_TOKENS = 8000  # model limit is 8192, leave headroom
OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_DIMENSIONS = 768


class EmbeddingError(RuntimeError):
    """The embedding provider could not embed a text.

    Callers must handle this, never paper over it. A placeholder vector is worse
    than no vector: zeros are equidistant from everything, so a zero query vector
    returns the same junk for every search, and a zero chunk vector is stored,
    marked done, and stays unsearchable after the provider recovers (2026-10-01).
    """


def get_provider() -> str:
    return os.getenv("EMBEDDING_PROVIDER", "openai")


def get_dimensions() -> int:
    return OPENAI_DIMENSIONS if get_provider() == "openai" else OLLAMA_DIMENSIONS


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in order. Raises EmbeddingError if any text can't be embedded."""
    provider = get_provider()
    try:
        if provider == "openai":
            return _embed_openai(texts)
        return _embed_ollama(texts)
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(f"{provider} embedding failed: {e}") from e


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
            # Retrying one at a time salvages a batch that failed because of a
            # single oversized or rejected text.
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
                    # Stop at the first failure. During an outage every call fails,
                    # so retrying the rest only burns requests.
                    raise EmbeddingError(f"OpenAI embedding failed: {e2}") from e2
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
