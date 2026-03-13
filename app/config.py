"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the CodeDoc RAG system."""

    # API keys
    github_pat: str = ""

    # Gemini config
    gemini_api_key: str = ""
    gemini_embed_model: str = "gemini-2.0-flash"
    gemini_llm_model: str = "gemini-2.0-flash"
    embedding_dimension: int = 384
    llm_max_tokens: int = 4096

    # Paths
    chroma_db_path: str = "./data/chroma_db"
    faiss_index_path: str = "./data/faiss_index"
    pdf_data_path: str = "./data/pdfs"
    repo_clone_path: str = "./data/repos"

    # Chunking config
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval config
    default_top_k: int = 10

    # ChromaDB
    chroma_collection_name: str = "codebase"

    # Supported file extensions for code ingestion
    supported_extensions: set[str] = {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".go", ".rs", ".cpp", ".c",
        ".md", ".yaml", ".yml",
    }

    # Directories to exclude during code ingestion
    excluded_dirs: set[str] = {
        "node_modules", ".git", "__pycache__", "dist",
        "build", ".venv", "venv", ".tox", ".mypy_cache",
    }

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def ensure_directories(self) -> None:
        """Create required data directories if they don't exist."""
        for path_str in (
            self.chroma_db_path,
            self.faiss_index_path,
            self.pdf_data_path,
            self.repo_clone_path,
        ):
            Path(path_str).mkdir(parents=True, exist_ok=True)


settings = Settings()
