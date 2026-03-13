#!/usr/bin/env python3
"""CLI script for ingesting changed files — called by GitHub Actions CI/CD.

Usage:
    python scripts/ingest.py --repo-url <URL> --changed-files changed_files.txt
    python scripts/ingest.py --repo-url <URL> --full
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion.chunker import chunk_code_file
from app.ingestion.embedder import Embedder
from app.ingestion.github_ingestor import (
    ingest_github_repo,
    incremental_ingest,
)
from app.vectorstore.chroma_store import ChromaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_full_ingest(repo_url: str, pat_token: str | None) -> None:
    """Perform a full ingestion of the repository into ChromaDB."""
    logger.info("Running full ingest for %s", repo_url)
    settings.ensure_directories()

    code_files = ingest_github_repo(repo_url, pat_token)
    if not code_files:
        logger.warning("No supported files found — nothing to ingest")
        return

    all_chunks = []
    for cf in code_files:
        chunks = chunk_code_file(cf.content, cf.file_path, cf.language, cf.repo_name)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No chunks produced — nothing to store")
        return

    embedder = Embedder()
    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_batch(texts)

    chroma = ChromaStore()
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

    chroma.add_chunks(ids, embeddings, texts, metadatas)
    logger.info("Full ingest complete: %d chunks stored", len(all_chunks))


def run_incremental_ingest(
    repo_url: str,
    changed_files_path: str,
    pat_token: str | None,
) -> None:
    """Ingest only the files listed in the changed-files text file."""
    path = Path(changed_files_path)
    if not path.exists():
        logger.error("Changed-files list not found: %s", changed_files_path)
        sys.exit(1)

    changed = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not changed:
        logger.info("No changed files — nothing to ingest")
        return

    logger.info("Incremental ingest: %d changed files", len(changed))
    settings.ensure_directories()

    code_files = incremental_ingest(repo_url, changed, pat_token)
    if not code_files:
        logger.info("None of the changed files are supported — skipping")
        return

    chroma = ChromaStore()
    embedder = Embedder()

    # Delete old chunks for changed files, then re-add
    for cf in code_files:
        chroma.delete_by_file(cf.file_path)

    all_chunks = []
    for cf in code_files:
        chunks = chunk_code_file(cf.content, cf.file_path, cf.language, cf.repo_name)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("Changed files produced no chunks")
        return

    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_batch(texts)

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

    chroma.add_chunks(ids, embeddings, texts, metadatas)
    logger.info("Incremental ingest complete: %d chunks updated", len(all_chunks))


def main() -> None:
    """Parse CLI arguments and run the appropriate ingest mode."""
    parser = argparse.ArgumentParser(
        description="CodeDoc RAG — CLI ingestion script"
    )
    parser.add_argument(
        "--repo-url", required=True, help="HTTPS URL of the GitHub repository"
    )
    parser.add_argument(
        "--changed-files",
        default=None,
        help="Path to a text file listing changed file paths (one per line)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run a full re-ingestion instead of incremental",
    )
    parser.add_argument(
        "--pat-token",
        default=None,
        help="GitHub PAT (defaults to GITHUB_PAT env var)",
    )

    args = parser.parse_args()
    pat = args.pat_token or settings.github_pat or None

    if args.full or not args.changed_files:
        run_full_ingest(args.repo_url, pat)
    else:
        run_incremental_ingest(args.repo_url, args.changed_files, pat)


if __name__ == "__main__":
    main()
