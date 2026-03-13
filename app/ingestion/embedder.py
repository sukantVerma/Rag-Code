"""Embedding wrapper using fastembed (lightweight, local, no API key needed)."""

import logging

from fastembed import TextEmbedding

from app.config import settings

logger = logging.getLogger(__name__)

# Truncate input text to this character limit before embedding
_MAX_CHARS = 8000


class Embedder:
    """Generates embeddings using fastembed (ONNX-based, lightweight)."""

    def __init__(self) -> None:
        self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        logger.info("Fastembed embedder initialised (BAAI/bge-small-en-v1.5, dim=384)")

    def _truncate(self, text: str) -> str:
        """Truncate text to fit within a safe character limit."""
        if len(text) > _MAX_CHARS:
            logger.warning(
                "Text has %d chars; truncating to %d", len(text), _MAX_CHARS
            )
            text = text[:_MAX_CHARS]
        return text

    def embed_single(self, text: str) -> list[float]:
        """Embed a single piece of text."""
        text = self._truncate(text)
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []
        truncated = [self._truncate(t) for t in texts]
        embeddings = list(self._model.embed(truncated))
        return [e.tolist() for e in embeddings]
