"""Unit tests for search service with mocked dependencies."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.application.services import SearchService
from src.domain.entities import Chunk, Query, SearchResult
from src.domain.exceptions import SearchError
from src.domain.value_objects import SparseVector


@pytest.fixture
def mock_dense_embedding() -> AsyncMock:
    """Create mock dense embedding service."""
    mock = AsyncMock()
    mock.embed_text = AsyncMock(return_value=[0.1] * 1536)
    return mock


@pytest.fixture
def mock_sparse_embedding() -> AsyncMock:
    """Create mock sparse embedding service."""
    mock = AsyncMock()
    mock.embed_text = AsyncMock(
        return_value=SparseVector(indices=(1, 5, 10), values=(0.5, 0.8, 0.3))
    )
    return mock


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    """Create mock vector store."""
    doc_id = uuid4()
    mock_results = []
    for i in range(5):
        chunk = Chunk(
            id=uuid4(),
            document_id=doc_id,
            content=f"Sample content {i}",
            chunk_index=i,
            start_char=i * 100,
            end_char=(i + 1) * 100,
        )
        mock_results.append(SearchResult(chunk=chunk, score=0.9 - i * 0.1))

    mock = AsyncMock()
    mock.hybrid_search = AsyncMock(return_value=mock_results)
    mock.search = AsyncMock(return_value=mock_results)
    return mock


@pytest.fixture
def mock_reranker() -> AsyncMock:
    """Create mock reranker."""

    async def rerank_side_effect(
        query: str, results: list[SearchResult], top_n: int
    ) -> list[SearchResult]:
        # Return top_n results with rerank scores
        reranked = []
        for rank, result in enumerate(results[:top_n], start=1):
            reranked.append(
                SearchResult(
                    chunk=result.chunk,
                    score=result.score,
                    rerank_score=0.95 - (rank - 1) * 0.1,
                    rank=rank,
                )
            )
        return reranked

    mock = AsyncMock()
    mock.rerank = AsyncMock(side_effect=rerank_side_effect)
    return mock


@pytest.fixture
def sample_query() -> Query:
    """Create sample query for testing."""
    return Query.create(
        text="What is RAG?",
        top_k=10,
        rerank_top_n=5,
        alpha=0.5,
    )


@pytest.mark.unit
class TestSearchServiceUnit:
    """Unit tests for SearchService class."""

    def test_has_reranker_true(
        self,
        mock_vector_store: AsyncMock,
        mock_dense_embedding: AsyncMock,
        mock_sparse_embedding: AsyncMock,
        mock_reranker: AsyncMock,
    ) -> None:
        """Test has_reranker returns True when reranker is configured."""
        service = SearchService(
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
            reranker=mock_reranker,
        )
        assert service.has_reranker is True

    def test_has_reranker_false(
        self,
        mock_vector_store: AsyncMock,
        mock_dense_embedding: AsyncMock,
        mock_sparse_embedding: AsyncMock,
    ) -> None:
        """Test has_reranker returns False when no reranker configured."""
        service = SearchService(
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
            reranker=None,
        )
        assert service.has_reranker is False

    async def test_search_with_rerank(
        self,
        mock_vector_store: AsyncMock,
        mock_dense_embedding: AsyncMock,
        mock_sparse_embedding: AsyncMock,
        mock_reranker: AsyncMock,
        sample_query: Query,
    ) -> None:
        """Test search with reranking returns reranked results."""
        service = SearchService(
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
            reranker=mock_reranker,
        )

        results, metrics = await service.search(sample_query)

        # Verify embeddings were generated
        mock_dense_embedding.embed_text.assert_called_once_with(sample_query.text)
        mock_sparse_embedding.embed_text.assert_called_once_with(sample_query.text)

        # Verify hybrid search was called
        mock_vector_store.hybrid_search.assert_called_once()

        # Verify reranking was called
        mock_reranker.rerank.assert_called_once()

        # Verify results have rerank scores
        assert len(results) == sample_query.rerank_top_n
        assert all(r.rerank_score is not None for r in results)
        assert all(r.rank > 0 for r in results)

        # Verify metrics
        assert metrics.total_latency_ms > 0
        assert metrics.embedding_latency_ms > 0
        assert metrics.search_latency_ms > 0
        assert metrics.rerank_latency_ms >= 0
        assert metrics.initial_results_count == 5
        assert metrics.final_results_count == 5

    async def test_search_skip_rerank(
        self,
        mock_vector_store: AsyncMock,
        mock_dense_embedding: AsyncMock,
        mock_sparse_embedding: AsyncMock,
        mock_reranker: AsyncMock,
        sample_query: Query,
    ) -> None:
        """Test search with skip_rerank=True skips reranking."""
        service = SearchService(
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
            reranker=mock_reranker,
        )

        results, _ = await service.search(sample_query, skip_rerank=True)

        # Verify reranking was not called
        mock_reranker.rerank.assert_not_called()

        # Results should not have rerank scores
        assert all(r.rerank_score is None for r in results)
        # But should have ranks assigned
        assert all(r.rank > 0 for r in results)

    async def test_search_without_reranker(
        self,
        mock_vector_store: AsyncMock,
        mock_dense_embedding: AsyncMock,
        mock_sparse_embedding: AsyncMock,
        sample_query: Query,
    ) -> None:
        """Test search without reranker assigns ranks but no rerank scores."""
        service = SearchService(
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
            reranker=None,
        )

        results, _ = await service.search(sample_query)

        # Results should be limited to rerank_top_n
        assert len(results) == sample_query.rerank_top_n
        # No rerank scores
        assert all(r.rerank_score is None for r in results)
        # Ranks should be assigned
        assert [r.rank for r in results] == [1, 2, 3, 4, 5]

    async def test_search_empty_results(
        self,
        mock_dense_embedding: AsyncMock,
        mock_sparse_embedding: AsyncMock,
        mock_reranker: AsyncMock,
        sample_query: Query,
    ) -> None:
        """Test search handles empty results."""
        mock_vector_store = AsyncMock()
        mock_vector_store.hybrid_search = AsyncMock(return_value=[])

        service = SearchService(
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
            reranker=mock_reranker,
        )

        results, metrics = await service.search(sample_query)

        assert results == []
        # Reranker should not be called for empty results
        mock_reranker.rerank.assert_not_called()
        assert metrics.final_results_count == 0

    async def test_search_embedding_error_raises_search_error(
        self,
        mock_vector_store: AsyncMock,
        mock_sparse_embedding: AsyncMock,
        sample_query: Query,
    ) -> None:
        """Test search raises SearchError on embedding failure."""
        mock_dense_embedding = AsyncMock()
        mock_dense_embedding.embed_text = AsyncMock(
            side_effect=Exception("Embedding Error")
        )

        service = SearchService(
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
        )

        with pytest.raises(SearchError) as exc_info:
            await service.search(sample_query)

        assert "failed" in str(exc_info.value).lower()

    async def test_search_vector_store_error_raises_search_error(
        self,
        mock_dense_embedding: AsyncMock,
        mock_sparse_embedding: AsyncMock,
        sample_query: Query,
    ) -> None:
        """Test search raises SearchError on vector store failure."""
        mock_vector_store = AsyncMock()
        mock_vector_store.hybrid_search = AsyncMock(
            side_effect=Exception("Vector Store Error")
        )

        service = SearchService(
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
        )

        with pytest.raises(SearchError) as exc_info:
            await service.search(sample_query)

        assert "failed" in str(exc_info.value).lower()

    async def test_search_dense_only(
        self,
        mock_vector_store: AsyncMock,
        mock_dense_embedding: AsyncMock,
        mock_sparse_embedding: AsyncMock,
        sample_query: Query,
    ) -> None:
        """Test search_dense_only uses only dense embeddings."""
        service = SearchService(
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
        )

        results, metrics = await service.search_dense_only(sample_query)

        # Verify only dense embedding was generated
        mock_dense_embedding.embed_text.assert_called_once_with(sample_query.text)
        mock_sparse_embedding.embed_text.assert_not_called()

        # Verify dense search was called (not hybrid)
        mock_vector_store.search.assert_called_once()
        mock_vector_store.hybrid_search.assert_not_called()

        # Verify results
        assert len(results) == sample_query.rerank_top_n
        assert metrics.rerank_latency_ms == 0.0

    async def test_search_passes_filters(
        self,
        mock_vector_store: AsyncMock,
        mock_dense_embedding: AsyncMock,
        mock_sparse_embedding: AsyncMock,
    ) -> None:
        """Test search passes filters to vector store."""
        filters = {"document_id": str(uuid4())}
        query = Query.create(
            text="test query",
            filters=filters,
        )

        service = SearchService(
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
        )

        await service.search(query)

        # Verify filters were passed to hybrid search
        call_kwargs = mock_vector_store.hybrid_search.call_args.kwargs
        assert call_kwargs["filters"] == filters

    async def test_search_passes_alpha(
        self,
        mock_vector_store: AsyncMock,
        mock_dense_embedding: AsyncMock,
        mock_sparse_embedding: AsyncMock,
    ) -> None:
        """Test search passes alpha parameter to vector store."""
        query = Query.create(
            text="test query",
            alpha=0.7,  # Custom alpha
        )

        service = SearchService(
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
        )

        await service.search(query)

        # Verify alpha was passed to hybrid search
        call_kwargs = mock_vector_store.hybrid_search.call_args.kwargs
        assert call_kwargs["alpha"] == 0.7
