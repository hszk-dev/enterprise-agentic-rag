"""Unit tests for health endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create a test client with mocked lifespan dependencies."""
    # Create mock repository and vector store for lifespan
    mock_repo = AsyncMock()
    mock_repo.initialize = AsyncMock()
    mock_repo.close = AsyncMock()

    mock_vector_store = AsyncMock()
    mock_vector_store.initialize = AsyncMock()

    # Patch functions called directly in lifespan (not through Depends)
    with (
        patch(
            "src.main.get_document_repository",
            return_value=mock_repo,
        ),
        patch(
            "src.main.get_vector_store",
            return_value=mock_vector_store,
        ),
        TestClient(app) as test_client,
    ):
        yield test_client


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"

    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe."""
        response = client.get("/api/v1/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe."""
        response = client.get("/api/v1/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert "checks" in data
        assert isinstance(data["checks"], dict)

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Enterprise Agentic RAG"
        assert data["status"] == "running"
