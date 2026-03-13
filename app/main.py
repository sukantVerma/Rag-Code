"""FastAPI application entrypoint for the CodeDoc RAG system."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import ingest, query
from app.api.schemas import HealthResponse
from app.config import settings
from app.ingestion.embedder import Embedder
from app.llm.claude_client import GeminiClient
from app.retrieval.retriever import MultiSourceRetriever
from app.vectorstore.chroma_store import ChromaStore
from app.vectorstore.faiss_store import FAISSStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Module-level singletons (initialised in lifespan)
_chroma: ChromaStore | None = None
_faiss_store: FAISSStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup and clean up on shutdown."""
    global _chroma, _faiss_store

    logger.info("Starting CodeDoc RAG system …")
    settings.ensure_directories()

    # Build singletons
    embedder = Embedder()
    _chroma = ChromaStore()
    _faiss_store = FAISSStore()
    llm_client = GeminiClient()
    retriever = MultiSourceRetriever(_chroma, _faiss_store, embedder)

    # Inject dependencies into route modules
    ingest.init_ingest_dependencies(embedder, _chroma, _faiss_store)
    query.init_query_dependencies(retriever, llm_client)

    logger.info("All components initialised — ready to serve requests")
    yield
    logger.info("Shutting down CodeDoc RAG system")


app = FastAPI(
    title="CodeDoc RAG System",
    description=(
        "Multi-source Retrieval-Augmented Generation system that ingests "
        "GitHub repositories and PDF documents, and answers natural language "
        "questions using Claude."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Return the health status of both vector stores."""
    chroma_health = _chroma.health_check() if _chroma else {"status": "not initialised"}
    faiss_health = (
        _faiss_store.health_check() if _faiss_store else {"status": "not initialised"}
    )
    return HealthResponse(
        status="ok",
        chroma=chroma_health,
        faiss=faiss_health,
    )
