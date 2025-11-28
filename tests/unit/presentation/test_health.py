"""Unit tests for health endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.presentation.api.dependencies import (
    get_document_repository,
    get_vector_store,
)


@pytest.fixture
def mock_document_repo():
    """Create mock document repository."""
    mock = AsyncMock()
    mock.initialize = AsyncMock()
    mock.close = AsyncMock()
    mock.count = AsyncMock(return_value=0)
    return mock


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    mock = AsyncMock()
    mock.initialize = AsyncMock()
    mock.get_collection_stats = AsyncMock(return_value={"points_count": 0})
    return mock


@pytest.fixture
def client(mock_document_repo, mock_vector_store):
    """Create a test client with mocked lifespan dependencies."""
    # Override DI dependencies
    app.dependency_overrides[get_document_repository] = lambda: mock_document_repo
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store

    # Patch functions called directly in lifespan (not through Depends)
    with (
        patch(
            "src.main.get_document_repository",
            return_value=mock_document_repo,
        ),
        patch(
            "src.main.get_vector_store",
            return_value=mock_vector_store,
        ),
        TestClient(app) as test_client,
    ):
        yield test_client

    app.dependency_overrides.clear()


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
        assert data["checks"]["database"] is True
        assert data["checks"]["vector_store"] is True

    def test_readiness_probe_database_failure(self, mock_vector_store):
        """Test readiness probe with database failure."""
        mock_repo = AsyncMock()
        mock_repo.initialize = AsyncMock()
        mock_repo.close = AsyncMock()
        mock_repo.count = AsyncMock(side_effect=Exception("Database error"))

        app.dependency_overrides[get_document_repository] = lambda: mock_repo
        app.dependency_overrides[get_vector_store] = lambda: mock_vector_store

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
            response = test_client.get("/api/v1/health/ready")

        app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is False
        assert data["checks"]["database"] is False
        assert data["checks"]["vector_store"] is True

    def test_readiness_probe_vector_store_failure(self, mock_document_repo):
        """Test readiness probe with vector store failure."""
        mock_vs = AsyncMock()
        mock_vs.initialize = AsyncMock()
        mock_vs.get_collection_stats = AsyncMock(
            side_effect=Exception("Vector store error")
        )

        app.dependency_overrides[get_document_repository] = lambda: mock_document_repo
        app.dependency_overrides[get_vector_store] = lambda: mock_vs

        with (
            patch(
                "src.main.get_document_repository",
                return_value=mock_document_repo,
            ),
            patch(
                "src.main.get_vector_store",
                return_value=mock_vs,
            ),
            TestClient(app) as test_client,
        ):
            response = test_client.get("/api/v1/health/ready")

        app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is False
        assert data["checks"]["database"] is True
        assert data["checks"]["vector_store"] is False

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Enterprise Agentic RAG"
        assert data["status"] == "running"
