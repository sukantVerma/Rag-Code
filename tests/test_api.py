"""Tests for FastAPI API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock

from app.main import app


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    # Mock the singletons that lifespan would create
    with (
        patch("app.main.Embedder"),
        patch("app.main.ChromaStore") as mock_chroma_cls,
        patch("app.main.FAISSStore") as mock_faiss_cls,
        patch("app.main.ClaudeClient"),
        patch("app.main.MultiSourceRetriever"),
    ):
        mock_chroma = mock_chroma_cls.return_value
        mock_chroma.health_check.return_value = {
            "store": "chromadb",
            "status": "ok",
            "collection": "codebase",
            "document_count": 0,
        }
        mock_faiss = mock_faiss_cls.return_value
        mock_faiss.health_check.return_value = {
            "store": "faiss",
            "status": "ok",
            "vector_count": 0,
            "index_path": "./data/faiss_index/index.faiss",
            "metadata_synced": True,
        }

        with TestClient(app) as tc:
            yield tc


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "chroma" in data
        assert "faiss" in data


class TestIngestEndpoints:
    def test_pdf_rejects_non_pdf(self, client):
        """Uploading a non-PDF file should return 400."""
        response = client.post(
            "/ingest/pdf",
            files={"file": ("readme.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 400


class TestQueryEndpoint:
    def test_query_missing_question_returns_422(self, client):
        """Omitting the required 'question' field should return 422."""
        response = client.post("/query", json={})
        assert response.status_code == 422
