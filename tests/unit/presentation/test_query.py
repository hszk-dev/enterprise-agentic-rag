"""Unit tests for query endpoints."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.application.services.generation_service import GenerationMetrics
from src.application.services.search_service import SearchMetrics
from src.domain.entities import Chunk, GenerationResult, Query, SearchResult
from src.domain.exceptions import LLMError, RateLimitError, SearchError
from src.domain.value_objects import TokenUsage
from src.main import app
from src.presentation.api.dependencies import (
    get_generation_service,
    get_search_service,
)


class TestQueryEndpoints:
    """Test query and RAG generation endpoints."""

    @pytest.fixture
    def mock_search_service(self):
        """Create mock search service."""
        return AsyncMock()

    @pytest.fixture
    def mock_generation_service(self):
        """Create mock generation service."""
        return AsyncMock()

    @pytest.fixture
    def sample_chunk(self):
        """Create a sample chunk."""
        return Chunk(
            id=uuid4(),
            document_id=uuid4(),
            content="This is test content for RAG.",
            chunk_index=0,
            start_char=0,
            end_char=30,
            metadata={"filename": "test.pdf"},
        )

    @pytest.fixture
    def sample_search_result(self, sample_chunk):
        """Create a sample search result."""
        return SearchResult(
            chunk=sample_chunk,
            score=0.95,
            rerank_score=0.98,
            rank=1,
        )

    @pytest.fixture
    def sample_search_metrics(self):
        """Create sample search metrics."""
        return SearchMetrics(
            total_latency_ms=100.0,
            embedding_latency_ms=30.0,
            search_latency_ms=50.0,
            rerank_latency_ms=20.0,
            initial_results_count=10,
            final_results_count=5,
        )

    @pytest.fixture
    def sample_generation_result(self, sample_search_result):
        """Create a sample generation result."""
        query = Query.create(text="What is this about?")
        return GenerationResult.create(
            query=query,
            answer="This is a generated answer based on the context.",
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
    def sample_generation_metrics(self):
        """Create sample generation metrics."""
        return GenerationMetrics(
            total_latency_ms=500.0,
            search_latency_ms=0.0,
            llm_latency_ms=500.0,
            context_tokens_estimate=100,
            sources_count=1,
        )

    @pytest.fixture
    def client(self, mock_search_service, mock_generation_service):
        """Create test client with mocked dependencies."""
        # Create mock repository and vector store for lifespan
        mock_repo = AsyncMock()
        mock_repo.initialize = AsyncMock()
        mock_repo.close = AsyncMock()

        mock_vector_store = AsyncMock()
        mock_vector_store.initialize = AsyncMock()

        # Mock service dependencies
        app.dependency_overrides[get_search_service] = lambda: mock_search_service
        app.dependency_overrides[get_generation_service] = (
            lambda: mock_generation_service
        )

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

        app.dependency_overrides.clear()

    def test_execute_query_success(
        self,
        client,
        mock_search_service,
        mock_generation_service,
        sample_search_result,
        sample_search_metrics,
        sample_generation_result,
        sample_generation_metrics,
    ):
        """Test successful RAG query execution."""
        mock_search_service.search.return_value = (
            [sample_search_result],
            sample_search_metrics,
        )
        mock_generation_service.generate.return_value = (
            sample_generation_result,
            sample_generation_metrics,
        )

        response = client.post(
            "/api/v1/query",
            json={"query": "What is this about?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) == 1
        assert data["model"] == "gpt-4o"
        assert "usage" in data

    def test_execute_query_with_parameters(
        self,
        client,
        mock_search_service,
        mock_generation_service,
        sample_search_result,
        sample_search_metrics,
        sample_generation_result,
        sample_generation_metrics,
    ):
        """Test query with custom parameters."""
        mock_search_service.search.return_value = (
            [sample_search_result],
            sample_search_metrics,
        )
        mock_generation_service.generate.return_value = (
            sample_generation_result,
            sample_generation_metrics,
        )

        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is this about?",
                "top_k": 20,
                "rerank_top_n": 10,
                "alpha": 0.7,
            },
        )

        assert response.status_code == 200

    def test_execute_query_without_sources(
        self,
        client,
        mock_search_service,
        mock_generation_service,
        sample_search_result,
        sample_search_metrics,
        sample_generation_result,
        sample_generation_metrics,
    ):
        """Test query with include_sources=False."""
        mock_search_service.search.return_value = (
            [sample_search_result],
            sample_search_metrics,
        )
        mock_generation_service.generate.return_value = (
            sample_generation_result,
            sample_generation_metrics,
        )

        response = client.post(
            "/api/v1/query",
            json={"query": "What is this about?", "include_sources": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == []

    def test_execute_query_no_results(
        self,
        client,
        mock_search_service,
        mock_generation_service,
        sample_search_metrics,
        sample_generation_result,
        sample_generation_metrics,
    ):
        """Test query with no search results."""
        mock_search_service.search.return_value = ([], sample_search_metrics)
        mock_generation_service.generate_with_no_context.return_value = (
            sample_generation_result,
            sample_generation_metrics,
        )

        response = client.post(
            "/api/v1/query",
            json={"query": "What is this about?"},
        )

        assert response.status_code == 200
        mock_generation_service.generate_with_no_context.assert_called_once()

    def test_execute_query_empty_query(self, client):
        """Test query with empty query string."""
        response = client.post(
            "/api/v1/query",
            json={"query": ""},
        )

        assert response.status_code == 422  # Validation error

    def test_execute_query_search_error(self, client, mock_search_service):
        """Test query with search error."""
        mock_search_service.search.side_effect = SearchError("Search failed")

        response = client.post(
            "/api/v1/query",
            json={"query": "What is this about?"},
        )

        assert response.status_code == 500
        assert "Search failed" in response.json()["detail"]

    def test_execute_query_llm_error(
        self,
        client,
        mock_search_service,
        mock_generation_service,
        sample_search_result,
        sample_search_metrics,
    ):
        """Test query with LLM error."""
        mock_search_service.search.return_value = (
            [sample_search_result],
            sample_search_metrics,
        )
        mock_generation_service.generate.side_effect = LLMError("LLM failed")

        response = client.post(
            "/api/v1/query",
            json={"query": "What is this about?"},
        )

        assert response.status_code == 500
        assert "Generation failed" in response.json()["detail"]

    def test_execute_query_rate_limit(self, client, mock_search_service):
        """Test query with rate limit error."""
        mock_search_service.search.side_effect = RateLimitError("Rate limited")

        response = client.post(
            "/api/v1/query",
            json={"query": "What is this about?"},
        )

        assert response.status_code == 429

    def test_execute_stream_query_success(
        self,
        client,
        mock_search_service,
        mock_generation_service,
        sample_search_result,
        sample_search_metrics,
    ):
        """Test successful streaming query execution."""
        mock_search_service.search.return_value = (
            [sample_search_result],
            sample_search_metrics,
        )

        # Mock async generator for streaming
        async def mock_stream():
            yield "This is "
            yield "a streaming "
            yield "response."

        mock_generation_service.generate_stream.return_value = mock_stream()

        response = client.post(
            "/api/v1/query/stream",
            json={"query": "What is this about?"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Check that we got SSE events
        content = response.text
        assert "data:" in content
