"""ChromaDB persistent vector store for GitHub codebase chunks."""

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

logger = logging.getLogger(__name__)


class ChromaStore:
    """Manages a persistent ChromaDB collection for code embeddings."""

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d items)",
            settings.chroma_collection_name,
            self._collection.count(),
        )

    @property
    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def add_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert chunks into the ChromaDB collection.

        Args:
            ids: Unique IDs for each chunk.
            embeddings: Embedding vectors.
            documents: Raw text for each chunk.
            metadatas: Metadata dicts (file_path, language, chunk_index, etc.).
        """
        if not ids:
            return

        # ChromaDB has a batch limit; insert in batches of 5000
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info("Added %d chunks to ChromaDB", len(ids))

    def delete_by_file(self, file_path: str) -> None:
        """Remove all chunks belonging to a specific file.

        Args:
            file_path: The relative file path to match against metadata.
        """
        self._collection.delete(where={"file_path": file_path})
        logger.info("Deleted chunks for file: %s", file_path)

    def delete_by_repo(self, repo_name: str) -> None:
        """Remove all chunks belonging to a specific repository.

        Args:
            repo_name: Name of the repository.
        """
        self._collection.delete(where={"repo_name": repo_name})
        logger.info("Deleted all chunks for repo: %s", repo_name)

    def query(
        self,
        embedding: list[float],
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Query the collection for the most similar chunks.

        Args:
            embedding: Query embedding vector.
            n_results: Number of results to return.

        Returns:
            List of dicts with keys: id, document, metadata, distance.
        """
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, max(self.count, 1)),
            include=["documents", "metadatas", "distances"],
        )

        items: list[dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return items

        for i, doc_id in enumerate(results["ids"][0]):
            items.append(
                {
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return items

    def health_check(self) -> dict[str, Any]:
        """Return health status of the ChromaDB store."""
        return {
            "store": "chromadb",
            "status": "ok",
            "collection": settings.chroma_collection_name,
            "document_count": self.count,
        }
