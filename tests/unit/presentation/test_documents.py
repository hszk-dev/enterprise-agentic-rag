"""Unit tests for document endpoints."""

import io
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.domain.entities import Document
from src.domain.exceptions import StorageError
from src.domain.value_objects import ContentType, DocumentStatus
from src.main import app
from src.presentation.api.dependencies import (
    get_document_repository,
    get_ingestion_service,
)


class TestDocumentEndpoints:
    """Test document management endpoints."""

    @pytest.fixture
    def mock_document_repo(self):
        """Create mock document repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_ingestion_service(self):
        """Create mock ingestion service."""
        return AsyncMock()

    @pytest.fixture
    def sample_document(self):
        """Create a sample document."""
        return Document(
            id=uuid4(),
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
            status=DocumentStatus.COMPLETED,
            metadata={"author": "Test"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            chunk_count=5,
            file_path="documents/test.pdf",
        )

    @pytest.fixture
    def client(self, mock_document_repo, mock_ingestion_service):
        """Create test client with mocked dependencies."""
        # Create mock repository and vector store for lifespan
        mock_lifespan_repo = AsyncMock()
        mock_lifespan_repo.initialize = AsyncMock()
        mock_lifespan_repo.close = AsyncMock()

        mock_vector_store = AsyncMock()
        mock_vector_store.initialize = AsyncMock()

        # Mock service dependencies
        app.dependency_overrides[get_document_repository] = lambda: mock_document_repo
        app.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion_service

        # Patch functions called directly in lifespan (not through Depends)
        with (
            patch(
                "src.main.get_document_repository",
                return_value=mock_lifespan_repo,
            ),
            patch(
                "src.main.get_vector_store",
                return_value=mock_vector_store,
            ),
            TestClient(app) as test_client,
        ):
            yield test_client

        app.dependency_overrides.clear()

    def test_upload_document_success(
        self, client, mock_ingestion_service, sample_document
    ):
        """Test successful document upload."""
        # Setup mock
        mock_ingestion_service.ingest_document.return_value = sample_document

        # Create test file
        file_content = b"Test PDF content"
        files = {"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")}

        response = client.post("/api/v1/documents", files=files)

        assert response.status_code == 202
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert data["status"] == "completed"
        assert mock_ingestion_service.ingest_document.called

    def test_upload_document_unsupported_type(self, client):
        """Test upload with unsupported file type."""
        file_content = b"Test content"
        files = {
            "file": ("test.xyz", io.BytesIO(file_content), "application/octet-stream")
        }

        response = client.post("/api/v1/documents", files=files)

        assert response.status_code == 415

    def test_upload_document_no_filename(self, client):
        """Test upload without filename."""
        file_content = b"Test content"
        files = {"file": ("", io.BytesIO(file_content), "application/pdf")}

        response = client.post("/api/v1/documents", files=files)

        # FastAPI returns 422 for validation errors (no file provided)
        # or 400 if our validation catches it first
        assert response.status_code in (400, 422)

    def test_list_documents(self, client, mock_document_repo, sample_document):
        """Test listing documents."""
        mock_document_repo.list.return_value = [sample_document]
        mock_document_repo.count.return_value = 1

        response = client.get("/api/v1/documents")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["total"] == 1
        assert data["limit"] == 20
        assert data["offset"] == 0

    def test_list_documents_with_pagination(
        self, client, mock_document_repo, sample_document
    ):
        """Test listing documents with pagination."""
        mock_document_repo.list.return_value = [sample_document]
        mock_document_repo.count.return_value = 100

        response = client.get("/api/v1/documents?limit=10&offset=20")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 20
        mock_document_repo.list.assert_called_once_with(
            status=None, limit=10, offset=20
        )

    def test_list_documents_with_status_filter(
        self, client, mock_document_repo, sample_document
    ):
        """Test listing documents with status filter."""
        mock_document_repo.list.return_value = [sample_document]
        mock_document_repo.count.return_value = 1

        response = client.get("/api/v1/documents?status=completed")

        assert response.status_code == 200
        mock_document_repo.list.assert_called_once_with(
            status=DocumentStatus.COMPLETED, limit=20, offset=0
        )

    def test_get_document_success(self, client, mock_document_repo, sample_document):
        """Test getting document by ID."""
        mock_document_repo.get_by_id.return_value = sample_document

        response = client.get(f"/api/v1/documents/{sample_document.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_document.id)
        assert data["filename"] == "test.pdf"

    def test_get_document_not_found(self, client, mock_document_repo):
        """Test getting non-existent document."""
        mock_document_repo.get_by_id.return_value = None
        doc_id = uuid4()

        response = client.get(f"/api/v1/documents/{doc_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_document_success(self, client, mock_ingestion_service):
        """Test successful document deletion."""
        mock_ingestion_service.delete_document.return_value = True
        doc_id = uuid4()

        response = client.delete(f"/api/v1/documents/{doc_id}")

        assert response.status_code == 204
        mock_ingestion_service.delete_document.assert_called_once_with(doc_id)

    def test_delete_document_not_found(self, client, mock_ingestion_service):
        """Test deleting non-existent document."""
        mock_ingestion_service.delete_document.return_value = False
        doc_id = uuid4()

        response = client.delete(f"/api/v1/documents/{doc_id}")

        assert response.status_code == 404

    def test_delete_document_storage_error(self, client, mock_ingestion_service):
        """Test deletion with storage error."""
        mock_ingestion_service.delete_document.side_effect = StorageError(
            "Failed to delete from storage"
        )
        doc_id = uuid4()

        response = client.delete(f"/api/v1/documents/{doc_id}")

        assert response.status_code == 500
