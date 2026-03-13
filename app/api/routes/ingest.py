"""Ingestion API routes: GitHub repos and PDF uploads."""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.api.schemas import GithubIngestRequest, IngestResponse
from app.config import settings
from app.ingestion.chunker import chunk_code_file
from app.ingestion.embedder import Embedder
from app.ingestion.github_ingestor import ingest_github_repo
from app.ingestion.pdf_ingestor import chunk_pdf
from app.vectorstore.chroma_store import ChromaStore
from app.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Singletons — initialised lazily via dependency injection in main.py
_embedder: Embedder | None = None
_chroma: ChromaStore | None = None
_faiss: FAISSStore | None = None


def init_ingest_dependencies(
    embedder: Embedder, chroma: ChromaStore, faiss_store: FAISSStore
) -> None:
    """Inject shared singletons into the ingest module."""
    global _embedder, _chroma, _faiss
    _embedder = embedder
    _chroma = chroma
    _faiss = faiss_store


# ── POST /ingest/github ─────────────────────────────────────────────────────


@router.post("/github", response_model=IngestResponse)
async def ingest_github(body: GithubIngestRequest) -> IngestResponse:
    """Clone a GitHub repo and ingest its code into ChromaDB."""
    try:
        code_files = ingest_github_repo(body.repo_url, body.pat_token)
    except Exception as exc:
        logger.exception("GitHub ingest failed")
        raise HTTPException(status_code=500, detail=str(exc))

    if not code_files:
        return IngestResponse(
            status="ok",
            source="github",
            chunks_processed=0,
            message="No supported files found in the repository.",
        )

    # Chunk all files
    all_chunks = []
    for cf in code_files:
        chunks = chunk_code_file(cf.content, cf.file_path, cf.language, cf.repo_name)
        all_chunks.extend(chunks)

    if not all_chunks:
        return IngestResponse(
            status="ok",
            source="github",
            chunks_processed=0,
            message="Files found but produced no chunks.",
        )

    # Embed
    texts = [c.text for c in all_chunks]
    embeddings = _embedder.embed_batch(texts)

    # Store in ChromaDB
    ids = [
        f"{c.repo_name}::{c.file_path}::chunk_{c.chunk_index}"
        for c in all_chunks
    ]
    metadatas = [
        {
            "file_path": c.file_path,
            "language": c.language,
            "chunk_index": c.chunk_index,
            "start_line": c.start_line,
            "repo_name": c.repo_name,
            "source": "github",
        }
        for c in all_chunks
    ]

    _chroma.add_chunks(ids, embeddings, texts, metadatas)

    return IngestResponse(
        status="ok",
        source="github",
        chunks_processed=len(all_chunks),
        message=f"Ingested {len(code_files)} files into {len(all_chunks)} chunks.",
    )


# ── POST /ingest/pdf ────────────────────────────────────────────────────────


@router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    """Upload a PDF file and ingest it into FAISS."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save uploaded file to the PDF data directory
    settings.ensure_directories()
    dest = Path(settings.pdf_data_path) / file.filename
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        logger.exception("Failed to save uploaded PDF")
        raise HTTPException(status_code=500, detail=str(exc))

    # Chunk
    try:
        chunks = chunk_pdf(str(dest))
    except Exception as exc:
        logger.exception("PDF chunking failed")
        raise HTTPException(status_code=500, detail=str(exc))

    if not chunks:
        return IngestResponse(
            status="ok",
            source="pdf",
            chunks_processed=0,
            message="PDF produced no text chunks.",
        )

    # Embed
    texts = [c.text for c in chunks]
    embeddings = _embedder.embed_batch(texts)

    # Store in FAISS
    metadatas = [
        {
            "source_file": c.source_file,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "source": "pdf",
        }
        for c in chunks
    ]
    _faiss.add_chunks(embeddings, metadatas, texts)

    return IngestResponse(
        status="ok",
        source="pdf",
        chunks_processed=len(chunks),
        message=f"Ingested {file.filename} into {len(chunks)} chunks.",
    )
