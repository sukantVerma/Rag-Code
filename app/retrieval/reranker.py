"""Optional cross-encoder reranker for improving retrieval precision."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    results: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Rerank retrieval results using a cross-encoder model.

    Falls back to the original order if sentence-transformers is not
    available or the reranker fails.

    Args:
        query: The user's query string.
        results: List of retrieval result dicts with a 'document' key.
        top_k: Number of results to return after reranking.

    Returns:
        Reranked list of result dicts.
    """
    if not results:
        return results

    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, r.get("document", "")) for r in results]
        scores = model.predict(pairs)

        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)

        results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        logger.info("Reranked %d results", len(results))
    except Exception:
        logger.warning("Reranker unavailable; returning results in original order")

    return results[:top_k]
