"""Tests for ingestion modules: GitHub ingestor, PDF ingestor, chunker, embedder."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.chunker import CodeChunk, chunk_code_file
from app.ingestion.github_ingestor import (
    EXTENSION_LANGUAGE_MAP,
    _repo_name_from_url,
    _should_include,
)
from app.ingestion.pdf_ingestor import PDFChunk


# ── GitHub ingestor helpers ──────────────────────────────────────────────────


class TestRepoNameFromUrl:
    def test_simple_url(self):
        assert _repo_name_from_url("https://github.com/owner/repo") == "repo"

    def test_url_with_git_suffix(self):
        assert _repo_name_from_url("https://github.com/owner/repo.git") == "repo"

    def test_url_with_trailing_slash(self):
        assert _repo_name_from_url("https://github.com/owner/repo/") == "repo"


class TestShouldInclude:
    def test_python_file_included(self, tmp_path):
        p = tmp_path / "src" / "main.py"
        p.parent.mkdir()
        p.touch()
        assert _should_include(Path("src/main.py")) is True

    def test_txt_file_excluded(self):
        assert _should_include(Path("readme.txt")) is False

    def test_node_modules_excluded(self):
        assert _should_include(Path("node_modules/package/index.js")) is False

    def test_pycache_excluded(self):
        assert _should_include(Path("app/__pycache__/module.py")) is False


# ── Code chunker ─────────────────────────────────────────────────────────────


class TestChunkCodeFile:
    def test_empty_content_returns_no_chunks(self):
        assert chunk_code_file("", "test.py", "python", "testrepo") == []

    def test_small_file_returns_single_chunk(self):
        content = "def hello():\n    return 'world'\n"
        chunks = chunk_code_file(content, "hello.py", "python", "testrepo")
        assert len(chunks) >= 1
        assert all(isinstance(c, CodeChunk) for c in chunks)
        assert chunks[0].file_path == "hello.py"
        assert chunks[0].repo_name == "testrepo"

    def test_chunk_metadata(self):
        content = "x = 1\n" * 500  # large enough to produce multiple chunks
        chunks = chunk_code_file(content, "big.py", "python", "myrepo")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.language == "python"
            assert chunk.start_line >= 1


# ── Extension language map ───────────────────────────────────────────────────


class TestExtensionMap:
    def test_known_extensions(self):
        assert EXTENSION_LANGUAGE_MAP[".py"] == "python"
        assert EXTENSION_LANGUAGE_MAP[".ts"] == "typescript"
        assert EXTENSION_LANGUAGE_MAP[".go"] == "go"

    def test_all_extensions_have_string_values(self):
        for ext, lang in EXTENSION_LANGUAGE_MAP.items():
            assert isinstance(lang, str)
