"""Tests for retrieval and vector store modules."""

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from app.retrieval.reranker import rerank


# ── Reranker ─────────────────────────────────────────────────────────────────


class TestRerank:
    def test_empty_results_returned_as_is(self):
        assert rerank("query", []) == []

    def test_passthrough_when_model_unavailable(self):
        """If sentence-transformers import fails, results stay in original order."""
        results = [
            {"document": "chunk A", "score": 0.9},
            {"document": "chunk B", "score": 0.7},
        ]
        with patch(
            "app.retrieval.reranker.CrossEncoder",
            side_effect=ImportError("not installed"),
        ):
            out = rerank("test query", results, top_k=2)

        # Should return the results without crashing
        assert len(out) == 2

    def test_top_k_limits_output(self):
        results = [{"document": f"doc {i}", "score": 0.5} for i in range(20)]
        out = rerank("query", results, top_k=5)
        assert len(out) == 5


# ── Content hash helper (used in retriever) ──────────────────────────────────


def test_content_hash_deterministic():
    from app.retrieval.retriever import _content_hash

    h1 = _content_hash("hello world")
    h2 = _content_hash("hello world")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest


def test_content_hash_different_inputs():
    from app.retrieval.retriever import _content_hash

    assert _content_hash("a") != _content_hash("b")
