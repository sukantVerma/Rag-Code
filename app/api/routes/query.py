"""Query API route: retrieve context and generate an answer with Claude."""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.schemas import QueryRequest, QueryResponse, SourceInfo
from app.llm.claude_client import ClaudeClient
from app.retrieval.retriever import MultiSourceRetriever

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])

# Singletons — initialised lazily via dependency injection in main.py
_retriever: MultiSourceRetriever | None = None
_claude: ClaudeClient | None = None


def init_query_dependencies(
    retriever: MultiSourceRetriever, claude: ClaudeClient
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

    # Run sync Anthropic HTTP call in thread pool to avoid blocking the event loop
    try:
        answer = await asyncio.to_thread(_claude.query, body.question, chunks)
    except Exception as exc:
        logger.exception("LLM query failed")
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    sources = [_build_source_info(c) for c in chunks]

    return QueryResponse(
        answer=answer,
        sources=sources,
        retrieval_count=len(chunks),
    )
