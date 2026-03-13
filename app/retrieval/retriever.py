"""Multi-source retriever: queries ChromaDB and FAISS, merges and deduplicates."""

import asyncio
import hashlib
import logging
from typing import Any

from app.ingestion.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """Return a SHA-256 hex digest for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MultiSourceRetriever:
    """Queries both ChromaDB (code) and FAISS (PDF) and merges results."""

    def __init__(
        self,
        chroma_store: ChromaStore,
        faiss_store: FAISSStore,
        embedder: Embedder,
    ) -> None:
        self._chroma = chroma_store
        self._faiss = faiss_store
        self._embedder = embedder

    def _query_chroma(
        self, embedding: list[float], n_results: int
    ) -> list[dict[str, Any]]:
        """Query ChromaDB and tag results with source='github'."""
        results = self._chroma.query(embedding, n_results=n_results)
        for r in results:
            r["source"] = "github"
            # ChromaDB returns cosine distance; lower is better
            r["score"] = 1.0 - r.get("distance", 0.0)
        return results

    def _query_faiss(
        self, embedding: list[float], n_results: int
    ) -> list[dict[str, Any]]:
        """Query FAISS and tag results with source='pdf'."""
        results = self._faiss.query(embedding, n_results=n_results)
        for r in results:
            r["source"] = "pdf"
            # FAISS returns L2 distance; convert to a similarity-like score
            distance = r.get("distance", 0.0)
            r["score"] = 1.0 / (1.0 + distance)
        return results

    async def aretrieve(
        self, query: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Asynchronously retrieve from both stores and merge results.

        Args:
            query: Natural language question.
            top_k: Total number of results to return.

        Returns:
            Merged, deduplicated, and ranked list of chunk dicts.
        """
        embedding = self._embedder.embed_single(query)

        loop = asyncio.get_event_loop()
        chroma_task = loop.run_in_executor(
            None, self._query_chroma, embedding, top_k
        )
        faiss_task = loop.run_in_executor(
            None, self._query_faiss, embedding, top_k
        )

        chroma_results, faiss_results = await asyncio.gather(
            chroma_task, faiss_task
        )

        combined = chroma_results + faiss_results

        # Deduplicate by content hash
        seen_hashes: set[str] = set()
        unique: list[dict[str, Any]] = []
        for item in combined:
            h = _content_hash(item.get("document", ""))
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(item)

        # Sort by score descending (higher = more relevant)
        unique.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        results = unique[:top_k]
        logger.info(
            "Retrieved %d chunks (chroma=%d, faiss=%d, after dedup=%d)",
            len(results),
            len(chroma_results),
            len(faiss_results),
            len(unique),
        )
        return results

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Synchronous wrapper around aretrieve for non-async contexts."""
        return asyncio.run(self.aretrieve(query, top_k))
