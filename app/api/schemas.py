"""Pydantic request/response schemas for all API endpoints."""

from pydantic import BaseModel, Field


# ── Ingestion Schemas ────────────────────────────────────────────────────────


class GithubIngestRequest(BaseModel):
    """Request body for POST /ingest/github."""

    repo_url: str = Field(
        ..., description="HTTPS URL of the GitHub repository to ingest."
    )
    pat_token: str | None = Field(
        None, description="Optional personal access token for private repos."
    )


class IngestResponse(BaseModel):
    """Generic response after an ingestion operation."""

    status: str
    source: str
    chunks_processed: int
    message: str


# ── Query Schemas ────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(..., description="Natural language question.")
    top_k: int = Field(
        10, ge=1, le=50, description="Number of chunks to retrieve."
    )


class SourceInfo(BaseModel):
    """Metadata about a single source chunk used in the answer."""

    source: str = Field(..., description="Source type: 'github' or 'pdf'.")
    file_path: str | None = Field(None, description="File path or PDF name.")
    page_number: int | None = Field(None, description="PDF page number, if applicable.")
    chunk_index: int | None = Field(None, description="Chunk index within the source.")
    score: float | None = Field(None, description="Similarity score.")


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str
    sources: list[SourceInfo]
    retrieval_count: int


# ── Health Check ─────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    chroma: dict
    faiss: dict
