"""FAISS vector store for PDF document chunks with a JSON metadata sidecar."""

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class FAISSStore:
    """Manages a FAISS IndexFlatL2 index with a parallel metadata list."""

    def __init__(self) -> None:
        self._index_path = Path(settings.faiss_index_path) / "index.faiss"
        self._meta_path = Path(settings.faiss_index_path) / "metadata.json"
        self._dimension = settings.embedding_dimension
        self._index: faiss.IndexFlatL2 | None = None
        self._metadata: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load existing index and metadata from disk, or create new ones."""
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info(
                "Loaded FAISS index with %d vectors", self._index.ntotal
            )
        else:
            self._index = faiss.IndexFlatL2(self._dimension)
            self._metadata = []
            logger.info("Created new FAISS index (dim=%d)", self._dimension)

    @property
    def count(self) -> int:
        """Return the number of vectors in the index."""
        return self._index.ntotal if self._index else 0

    def add_chunks(
        self,
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        """Add chunks to the FAISS index and metadata sidecar.

        Args:
            embeddings: Embedding vectors.
            metadatas: Metadata dicts for each chunk.
            documents: Raw text for each chunk (stored in metadata sidecar).
        """
        if not embeddings:
            return

        vectors = np.array(embeddings, dtype=np.float32)
        self._index.add(vectors)

        for i, meta in enumerate(metadatas):
            entry = {**meta, "document": documents[i]}
            self._metadata.append(entry)

        self.save()
        logger.info("Added %d vectors to FAISS index", len(embeddings))

    def query(
        self,
        embedding: list[float],
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Find the most similar vectors to the query embedding.

        Args:
            embedding: Query embedding vector.
            n_results: Number of results to return.

        Returns:
            List of dicts with keys: document, metadata, distance.
        """
        if self._index.ntotal == 0:
            return []

        query_vec = np.array([embedding], dtype=np.float32)
        k = min(n_results, self._index.ntotal)
        distances, indices = self._index.search(query_vec, k)

        results: list[dict[str, Any]] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append(
                {
                    "document": meta.get("document", ""),
                    "metadata": {
                        k: v for k, v in meta.items() if k != "document"
                    },
                    "distance": float(distances[0][i]),
                }
            )

        return results

    def save(self) -> None:
        """Persist the FAISS index and metadata sidecar to disk."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False)
        logger.info("FAISS index saved (%d vectors)", self._index.ntotal)

    def reset(self) -> None:
        """Clear the index and metadata completely."""
        self._index = faiss.IndexFlatL2(self._dimension)
        self._metadata = []
        self.save()
        logger.info("FAISS index reset")

    def health_check(self) -> dict[str, Any]:
        """Return health status of the FAISS store."""
        return {
            "store": "faiss",
            "status": "ok",
            "vector_count": self.count,
            "index_path": str(self._index_path),
            "metadata_synced": self.count == len(self._metadata),
        }
