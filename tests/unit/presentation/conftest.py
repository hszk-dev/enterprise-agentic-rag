"""Fixtures for presentation layer tests."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.domain.entities import Chunk, Document, GenerationResult, Query, SearchResult
from src.domain.value_objects import ContentType, DocumentStatus, TokenUsage
from src.main import app
from src.presentation.api.dependencies import (
    get_document_repository,
    get_generation_service,
    get_ingestion_service,
    get_search_service,
)


@pytest.fixture
def mock_document_repository():
    """Create a mock document repository."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_blob_storage():
    """Create a mock blob storage."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_ingestion_service():
    """Create a mock ingestion service."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_search_service():
    """Create a mock search service."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_generation_service():
    """Create a mock generation service."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id=uuid4(),
        filename="test.pdf",
        content_type=ContentType.PDF,
        size_bytes=1024,
        status=DocumentStatus.COMPLETED,
        metadata={"author": "Test Author"},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        chunk_count=5,
        file_path="documents/test.pdf",
    )


@pytest.fixture
def sample_chunk(sample_document):
    """Create a sample chunk for testing."""
    return Chunk(
        id=uuid4(),
        document_id=sample_document.id,
        content="This is a test chunk content.",
        chunk_index=0,
        start_char=0,
        end_char=30,
        metadata={"filename": "test.pdf"},
    )


@pytest.fixture
def sample_search_result(sample_chunk):
    """Create a sample search result for testing."""
    return SearchResult(
        chunk=sample_chunk,
        score=0.95,
        rerank_score=0.98,
        rank=1,
    )


@pytest.fixture
def sample_generation_result(sample_search_result):
    """Create a sample generation result for testing."""
    query = Query.create(text="What is the test about?")
    return GenerationResult.create(
        query=query,
        answer="This is a test answer based on the retrieved context.",
        sources=[sample_search_result],
        usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4o",
        ),
        model="gpt-4o",
        latency_ms=500.0,
    )


@pytest.fixture
def client(
    mock_document_repository,
    mock_ingestion_service,
    mock_search_service,
    mock_generation_service,
):
    """Create a test client with mocked dependencies."""
    # Override dependencies with mocks
    app.dependency_overrides[get_document_repository] = lambda: mock_document_repository
    app.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion_service
    app.dependency_overrides[get_search_service] = lambda: mock_search_service
    app.dependency_overrides[get_generation_service] = lambda: mock_generation_service

    with TestClient(app) as test_client:
        yield test_client

    # Clear overrides after test
    app.dependency_overrides.clear()
