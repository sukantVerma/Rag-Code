"""Query API route: retrieve context and generate an answer with Gemini."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.schemas import QueryRequest, QueryResponse, SourceInfo
from app.llm.claude_client import GeminiClient
from app.retrieval.retriever import MultiSourceRetriever

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])

# Singletons — initialised lazily via dependency injection in main.py
_retriever: MultiSourceRetriever | None = None
_claude: GeminiClient | None = None


def init_query_dependencies(
    retriever: MultiSourceRetriever, claude: GeminiClient
) -> None:
    """Inject shared singletons into the query module."""
    global _retriever, _claude
    _retriever = retriever
    _claude = claude


def _build_source_info(chunk: dict[str, Any]) -> SourceInfo:
    """Convert a raw retrieval result dict into a SourceInfo schema."""
    meta = chunk.get("metadata", {})
    source_type = chunk.get("source", "unknown")

    if source_type == "github":
        file_path = meta.get("file_path")
    else:
        file_path = meta.get("source_file")

    return SourceInfo(
        source=source_type,
        file_path=file_path,
        page_number=meta.get("page_number"),
        chunk_index=meta.get("chunk_index"),
        score=chunk.get("score"),
    )


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    """Accept a natural language question, retrieve context, and answer via Claude."""
    try:
        chunks = await _retriever.aretrieve(body.question, top_k=body.top_k)
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    try:
        answer = _claude.query(body.question, chunks)
    except Exception as exc:
        logger.exception("LLM query failed")
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    sources = [_build_source_info(c) for c in chunks]

    return QueryResponse(
        answer=answer,
        sources=sources,
        retrieval_count=len(chunks),
    )
