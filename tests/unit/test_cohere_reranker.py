"""Unit tests for Cohere reranker service with mocked dependencies."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import SecretStr

from config.settings import CohereSettings
from src.domain.entities import Chunk, SearchResult
from src.domain.exceptions import RerankError
from src.infrastructure.rerankers import CohereReranker


@pytest.fixture
def mock_cohere_settings() -> CohereSettings:
    """Create Cohere settings for unit tests."""
    return CohereSettings(
        api_key=SecretStr("test-api-key"),  # pragma: allowlist secret
        rerank_model="rerank-v3.5",
        max_retries=3,
        timeout=30.0,
    )


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Create sample search results for testing."""
    doc_id = uuid4()
    results = []
    for i in range(5):
        chunk = Chunk(
            id=uuid4(),
            document_id=doc_id,
            content=f"Sample content {i}",
            chunk_index=i,
            start_char=i * 100,
            end_char=(i + 1) * 100,
        )
        results.append(SearchResult(chunk=chunk, score=0.5 - i * 0.1))
    return results


@pytest.mark.unit
class TestCohereRerankerUnit:
    """Unit tests for CohereReranker class."""

    def test_model_name_property(self, mock_cohere_settings: CohereSettings) -> None:
        """Test that model_name property returns configured model."""
        with patch("src.infrastructure.rerankers.cohere_reranker.cohere.AsyncClientV2"):
            reranker = CohereReranker(mock_cohere_settings)
            assert reranker.model_name == "rerank-v3.5"

    async def test_rerank_success(
        self,
        mock_cohere_settings: CohereSettings,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test rerank returns reordered results with scores."""
        with patch(
            "src.infrastructure.rerankers.cohere_reranker.cohere.AsyncClientV2"
        ) as mock_client_class:
            # Setup mock response - reverse order to test reranking
            mock_results = []
            for i in range(3):
                mock_result = MagicMock()
                mock_result.index = 4 - i  # Reverse order: 4, 3, 2
                mock_result.relevance_score = 0.9 - i * 0.1  # 0.9, 0.8, 0.7
                mock_results.append(mock_result)

            mock_response = MagicMock()
            mock_response.results = mock_results

            mock_client = AsyncMock()
            mock_client.rerank = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            reranker = CohereReranker(mock_cohere_settings)
            results = await reranker.rerank(
                query="test query",
                results=sample_search_results,
                top_n=3,
            )

            assert len(results) == 3
            # Check rerank scores are set
            assert results[0].rerank_score == 0.9
            assert results[1].rerank_score == 0.8
            assert results[2].rerank_score == 0.7
            # Check ranks are set correctly
            assert results[0].rank == 1
            assert results[1].rank == 2
            assert results[2].rank == 3
            # Verify the chunks are from the correct original indices
            assert results[0].chunk.content == "Sample content 4"
            assert results[1].chunk.content == "Sample content 3"
            assert results[2].chunk.content == "Sample content 2"

    async def test_rerank_empty_results(
        self, mock_cohere_settings: CohereSettings
    ) -> None:
        """Test rerank returns empty list for empty input."""
        with patch("src.infrastructure.rerankers.cohere_reranker.cohere.AsyncClientV2"):
            reranker = CohereReranker(mock_cohere_settings)
            results = await reranker.rerank(
                query="test query",
                results=[],
                top_n=5,
            )
            assert results == []

    async def test_rerank_empty_query_raises_error(
        self,
        mock_cohere_settings: CohereSettings,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test rerank raises RerankError for empty query."""
        with patch("src.infrastructure.rerankers.cohere_reranker.cohere.AsyncClientV2"):
            reranker = CohereReranker(mock_cohere_settings)

            with pytest.raises(RerankError) as exc_info:
                await reranker.rerank(
                    query="",
                    results=sample_search_results,
                    top_n=5,
                )

            assert "empty query" in str(exc_info.value).lower()

    async def test_rerank_whitespace_query_raises_error(
        self,
        mock_cohere_settings: CohereSettings,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test rerank raises RerankError for whitespace-only query."""
        with patch("src.infrastructure.rerankers.cohere_reranker.cohere.AsyncClientV2"):
            reranker = CohereReranker(mock_cohere_settings)

            with pytest.raises(RerankError):
                await reranker.rerank(
                    query="   ",
                    results=sample_search_results,
                    top_n=5,
                )

    async def test_rerank_top_n_limited_to_available(
        self,
        mock_cohere_settings: CohereSettings,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test rerank limits top_n to available results."""
        with patch(
            "src.infrastructure.rerankers.cohere_reranker.cohere.AsyncClientV2"
        ) as mock_client_class:
            # Only return 3 results even though we have 5
            mock_results = []
            for i in range(3):
                mock_result = MagicMock()
                mock_result.index = i
                mock_result.relevance_score = 0.9 - i * 0.1
                mock_results.append(mock_result)

            mock_response = MagicMock()
            mock_response.results = mock_results

            mock_client = AsyncMock()
            mock_client.rerank = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            reranker = CohereReranker(mock_cohere_settings)
            # Request 10 but only 5 available
            _ = await reranker.rerank(
                query="test query",
                results=sample_search_results,  # Only 5 results
                top_n=10,
            )

            # Verify the API was called with min(10, 5) = 5
            call_args = mock_client.rerank.call_args
            assert call_args.kwargs["top_n"] == 5

    async def test_rerank_preserves_original_score(
        self,
        mock_cohere_settings: CohereSettings,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test rerank preserves original search score."""
        with patch(
            "src.infrastructure.rerankers.cohere_reranker.cohere.AsyncClientV2"
        ) as mock_client_class:
            mock_result = MagicMock()
            mock_result.index = 0
            mock_result.relevance_score = 0.95

            mock_response = MagicMock()
            mock_response.results = [mock_result]

            mock_client = AsyncMock()
            mock_client.rerank = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            reranker = CohereReranker(mock_cohere_settings)
            results = await reranker.rerank(
                query="test query",
                results=sample_search_results,
                top_n=1,
            )

            # Original score should be preserved
            assert results[0].score == sample_search_results[0].score
            # Rerank score should be set
            assert results[0].rerank_score == 0.95

    async def test_rerank_api_error_raises_rerank_error(
        self,
        mock_cohere_settings: CohereSettings,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test rerank raises RerankError on API failure."""
        with patch(
            "src.infrastructure.rerankers.cohere_reranker.cohere.AsyncClientV2"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.rerank = AsyncMock(side_effect=Exception("API Error"))
            mock_client_class.return_value = mock_client

            reranker = CohereReranker(mock_cohere_settings)

            with pytest.raises(RerankError) as exc_info:
                await reranker.rerank(
                    query="test query",
                    results=sample_search_results,
                    top_n=5,
                )

            assert "failed" in str(exc_info.value).lower()

    async def test_close_is_noop(self, mock_cohere_settings: CohereSettings) -> None:
        """Test close method is a no-op (for interface consistency)."""
        with patch("src.infrastructure.rerankers.cohere_reranker.cohere.AsyncClientV2"):
            reranker = CohereReranker(mock_cohere_settings)
            # Should not raise any errors
            await reranker.close()
