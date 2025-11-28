"""Integration tests for Search Service (E2E).

These tests require:
- Running Qdrant instance (localhost:6333)
- OPENAI_API_KEY environment variable
- COHERE_API_KEY environment variable

Run with:
    OPENAI_API_KEY=xxx COHERE_API_KEY=xxx pytest tests/integration/test_search_service.py -m integration -v

These tests validate the full search pipeline:
1. Query embedding generation (Dense + Sparse)
2. Hybrid search in Qdrant
3. Reranking with Cohere
"""

import asyncio
import contextlib
import os
import time
from uuid import uuid4

import pytest

from config import CohereSettings, OpenAISettings, QdrantSettings
from src.application.services import SearchService
from src.domain.entities import Chunk, Query
from src.infrastructure.embeddings import (
    FastEmbedSparseEmbeddingService,
    OpenAIEmbeddingService,
)
from src.infrastructure.rerankers import CohereReranker
from src.infrastructure.vectorstores import QdrantVectorStore

# Skip if required env vars or services are not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    ),
    pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY"),
        reason="COHERE_API_KEY not set",
    ),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def openai_settings():
    """Create OpenAI settings with real API key."""
    return OpenAISettings(
        api_key=os.environ.get("OPENAI_API_KEY", ""),  # pragma: allowlist secret
        model="gpt-4o",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        max_retries=3,
        timeout=30.0,
    )


@pytest.fixture
def cohere_settings():
    """Create Cohere settings with real API key."""
    return CohereSettings(
        api_key=os.environ.get("COHERE_API_KEY", ""),  # pragma: allowlist secret
        rerank_model="rerank-v3.5",
        max_retries=3,
        timeout=30.0,
    )


@pytest.fixture
def qdrant_settings():
    """Create Qdrant settings with unique collection name."""
    return QdrantSettings(
        host="localhost",
        port=6333,
        collection_name=f"test-search-e2e-{uuid4().hex[:8]}",
        use_grpc=False,
    )


@pytest.fixture(scope="module")
def sparse_embedding_service():
    """Create sparse embedding service (module-scoped for model reuse)."""
    return FastEmbedSparseEmbeddingService()


@pytest.fixture
async def dense_embedding_service(openai_settings: OpenAISettings):
    """Create dense embedding service."""
    service = OpenAIEmbeddingService(openai_settings)
    yield service
    await service.close()


@pytest.fixture
async def vector_store(qdrant_settings: QdrantSettings):
    """Create and initialize vector store."""
    store = QdrantVectorStore(qdrant_settings, embedding_dim=1536)
    await store.initialize()
    yield store
    # Cleanup: delete collection
    with contextlib.suppress(Exception):
        await store._client.delete_collection(qdrant_settings.collection_name)
    await store.close()


@pytest.fixture
def reranker(cohere_settings: CohereSettings):
    """Create Cohere reranker."""
    return CohereReranker(cohere_settings)


@pytest.fixture
async def indexed_vector_store(
    vector_store: QdrantVectorStore,
    dense_embedding_service: OpenAIEmbeddingService,
    sparse_embedding_service: FastEmbedSparseEmbeddingService,
):
    """Vector store with pre-indexed sample documents."""
    await _index_sample_documents(
        vector_store, dense_embedding_service, sparse_embedding_service
    )
    return vector_store


@pytest.fixture
async def search_service(
    indexed_vector_store: QdrantVectorStore,
    dense_embedding_service: OpenAIEmbeddingService,
    sparse_embedding_service: FastEmbedSparseEmbeddingService,
    reranker: CohereReranker,
):
    """Create fully configured SearchService with indexed data."""
    return SearchService(
        vector_store=indexed_vector_store,
        dense_embedding=dense_embedding_service,
        sparse_embedding=sparse_embedding_service,
        reranker=reranker,
    )


@pytest.fixture
async def search_service_no_reranker(
    indexed_vector_store: QdrantVectorStore,
    dense_embedding_service: OpenAIEmbeddingService,
    sparse_embedding_service: FastEmbedSparseEmbeddingService,
):
    """Create SearchService without reranker."""
    return SearchService(
        vector_store=indexed_vector_store,
        dense_embedding=dense_embedding_service,
        sparse_embedding=sparse_embedding_service,
        reranker=None,
    )


# =============================================================================
# Helper Functions
# =============================================================================


async def _index_sample_documents(
    vector_store: QdrantVectorStore,
    dense_embedding: OpenAIEmbeddingService,
    sparse_embedding: FastEmbedSparseEmbeddingService,
):
    """Index sample documents for testing."""
    doc_id = uuid4()
    contents = [
        # English - Python/AI related
        "Python is a versatile programming language used in web development, data science, and AI.",
        "Machine learning algorithms can learn patterns from data without explicit programming.",
        "RAG (Retrieval Augmented Generation) combines retrieval with language models for better answers.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers.",
        "Python's pandas library is widely used for data manipulation and analysis.",
        # English - Unrelated
        "Tokyo is the capital city of Japan with a population of over 13 million people.",
        "Italian cuisine includes pasta, pizza, and various regional specialties.",
        "The weather forecast predicts sunny skies for the weekend.",
        # Japanese
        "Pythonは機械学習やデータサイエンスで広く使われているプログラミング言語です。",
        "製品コード A2849 は最新のスマートフォンモデルです。",
        # Product codes / Technical
        "The iPhone 15 Pro features the A17 Pro chip with improved GPU performance.",
        "Model XR-5000 specifications: 16GB RAM, 512GB storage, OLED display.",
    ]

    chunks = []
    for i, content in enumerate(contents):
        chunk = Chunk.create(
            document_id=doc_id,
            content=content,
            chunk_index=i,
            start_char=0,
            end_char=len(content),
            metadata={"source": "test", "index": i},
        )
        chunks.append(chunk)

    # Generate embeddings
    texts = [c.content for c in chunks]
    dense_embeddings = await dense_embedding.embed_texts(texts)
    sparse_embeddings = await sparse_embedding.embed_texts(texts)

    for chunk, dense, sparse in zip(
        chunks, dense_embeddings, sparse_embeddings, strict=True
    ):
        chunk.dense_embedding = dense
        chunk.sparse_embedding = sparse

    await vector_store.upsert_chunks(chunks)


# =============================================================================
# Test Classes
# =============================================================================


class TestSearchServiceE2E:
    """End-to-end integration tests for SearchService."""

    async def test_full_search_pipeline(self, search_service: SearchService) -> None:
        """Test full search pipeline: Embed → Hybrid Search → Rerank."""
        query = Query.create(
            text="What is Python programming language?",
            top_k=10,
            rerank_top_n=3,
        )
        results, metrics = await search_service.search(query)

        # Verify results structure
        assert len(results) == 3
        assert all(r.rerank_score is not None for r in results)
        assert all(r.rank > 0 for r in results)

        # Verify ranking order
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3

        # Verify metrics
        assert metrics.total_latency_ms > 0
        assert metrics.embedding_latency_ms > 0
        assert metrics.search_latency_ms > 0
        assert metrics.rerank_latency_ms > 0
        assert metrics.initial_results_count >= metrics.final_results_count
        assert metrics.final_results_count == 3

        # Python-related content should be in top results
        contents = [r.chunk.content for r in results]
        python_count = sum(1 for c in contents if "Python" in c)
        assert python_count >= 1, f"Expected Python content in results: {contents}"

    async def test_search_with_skip_rerank(self, search_service: SearchService) -> None:
        """Test search with reranking explicitly skipped."""
        query = Query.create(
            text="machine learning algorithms",
            top_k=5,
            rerank_top_n=3,
        )
        results, metrics = await search_service.search(query, skip_rerank=True)

        assert len(results) == 3
        # No rerank scores when skipped
        assert all(r.rerank_score is None for r in results)
        # Rerank latency should be minimal
        assert metrics.rerank_latency_ms < 10.0

    async def test_search_without_reranker_configured(
        self, search_service_no_reranker: SearchService
    ) -> None:
        """Test search when no reranker is configured."""
        query = Query.create(
            text="data science",
            top_k=5,
            rerank_top_n=3,
        )
        results, _metrics = await search_service_no_reranker.search(query)

        assert len(results) == 3
        assert all(r.rerank_score is None for r in results)
        assert not search_service_no_reranker.has_reranker

    async def test_search_dense_only(self, search_service: SearchService) -> None:
        """Test dense-only search mode."""
        query = Query.create(
            text="artificial intelligence neural networks",
            top_k=5,
            rerank_top_n=3,
        )
        results, metrics = await search_service.search_dense_only(query)

        assert len(results) == 3
        assert metrics.rerank_latency_ms == 0.0
        # Should still return relevant results
        assert all(r.chunk.content for r in results)

    async def test_hybrid_search_alpha_dense_heavy(
        self, search_service: SearchService
    ) -> None:
        """Test hybrid search with dense-heavy weighting (alpha=0.9)."""
        query = Query.create(
            text="programming language for AI",
            top_k=5,
            rerank_top_n=3,
            alpha=0.9,  # Dense-heavy
        )
        results, _ = await search_service.search(query, skip_rerank=True)

        assert len(results) == 3
        # Results should be returned (semantic similarity focused)
        assert all(r.score is not None for r in results)

    async def test_hybrid_search_alpha_sparse_heavy(
        self, search_service: SearchService
    ) -> None:
        """Test hybrid search with sparse-heavy weighting (alpha=0.1)."""
        query = Query.create(
            text="Python pandas",
            top_k=5,
            rerank_top_n=3,
            alpha=0.1,  # Sparse-heavy (keyword focused)
        )
        results, _ = await search_service.search(query, skip_rerank=True)

        assert len(results) == 3
        # Keyword-focused search
        contents = [r.chunk.content for r in results]
        # At least some results should contain the keywords
        keyword_matches = sum(1 for c in contents if "Python" in c or "pandas" in c)
        assert keyword_matches >= 1

    async def test_keyword_exact_match_product_code(
        self, search_service: SearchService
    ) -> None:
        """Test exact keyword matching for product codes."""
        query = Query.create(
            text="A2849",
            top_k=5,
            rerank_top_n=3,
            alpha=0.3,  # Sparse-heavy for keyword matching
        )
        results, _ = await search_service.search(query)

        # Product code document should be found
        contents = [r.chunk.content for r in results]
        found_product = any("A2849" in c for c in contents)
        assert found_product, f"Product code A2849 not found in: {contents}"

    async def test_keyword_model_number(self, search_service: SearchService) -> None:
        """Test keyword matching for model numbers."""
        query = Query.create(
            text="XR-5000",
            top_k=5,
            rerank_top_n=3,
            alpha=0.3,
        )
        results, _ = await search_service.search(query)

        contents = [r.chunk.content for r in results]
        found_model = any("XR-5000" in c for c in contents)
        assert found_model, f"Model XR-5000 not found in: {contents}"

    async def test_japanese_query_search(self, search_service: SearchService) -> None:
        """Test search with Japanese query."""
        query = Query.create(
            text="Pythonでデータサイエンス",
            top_k=5,
            rerank_top_n=3,
        )
        results, metrics = await search_service.search(query)

        assert len(results) > 0
        assert metrics.total_latency_ms > 0

        # Should find relevant content (Japanese or English about Python/data science)
        contents = [r.chunk.content for r in results]
        relevant = any(
            "Python" in c or "データサイエンス" in c or "data science" in c.lower()
            for c in contents
        )
        assert relevant, f"No relevant content found in: {contents}"

    async def test_search_metrics_timing_accuracy(
        self, search_service: SearchService
    ) -> None:
        """Test that metrics timing is accurate."""
        query = Query.create(
            text="retrieval augmented generation RAG",
            top_k=5,
            rerank_top_n=3,
        )
        results, metrics = await search_service.search(query)

        # Component times should sum approximately to total
        component_sum = (
            metrics.embedding_latency_ms
            + metrics.search_latency_ms
            + metrics.rerank_latency_ms
        )
        # Allow for small overhead (10%)
        assert metrics.total_latency_ms >= component_sum * 0.9, (
            f"Total ({metrics.total_latency_ms}ms) < components ({component_sum}ms)"
        )

        # Counts should be accurate
        assert metrics.initial_results_count >= metrics.final_results_count
        assert metrics.final_results_count == len(results)

    async def test_search_with_different_top_k(
        self, search_service: SearchService
    ) -> None:
        """Test search with different top_k values."""
        for top_k, _expected in [(3, 3), (5, 5), (10, 10)]:
            query = Query.create(
                text="programming",
                top_k=top_k,
                rerank_top_n=top_k,  # Same as top_k
            )
            results, metrics = await search_service.search(query)

            # Should return up to top_k results (or all if fewer docs)
            assert len(results) <= top_k
            assert metrics.final_results_count == len(results)

    async def test_concurrent_searches(self, search_service: SearchService) -> None:
        """Test concurrent search execution."""
        queries = [
            Query.create("Python programming", top_k=3, rerank_top_n=2),
            Query.create("machine learning", top_k=3, rerank_top_n=2),
            Query.create("Tokyo Japan", top_k=3, rerank_top_n=2),
        ]

        # Execute concurrently
        tasks = [search_service.search(q) for q in queries]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for search_results, metrics in results:
            assert len(search_results) > 0
            assert metrics.total_latency_ms > 0

    async def test_has_reranker_property(
        self,
        search_service: SearchService,
        search_service_no_reranker: SearchService,
    ) -> None:
        """Test has_reranker property."""
        assert search_service.has_reranker is True
        assert search_service_no_reranker.has_reranker is False


class TestSearchServicePerformance:
    """Performance benchmark tests for SearchService."""

    async def test_search_latency_benchmark(
        self, search_service: SearchService
    ) -> None:
        """Benchmark: Full search should complete in < 3 seconds."""
        query = Query.create(
            text="Python AI machine learning",
            top_k=10,
            rerank_top_n=5,
        )

        start = time.perf_counter()
        results, metrics = await search_service.search(query)
        elapsed = time.perf_counter() - start

        assert len(results) > 0
        assert elapsed < 3.0, f"Search took {elapsed:.2f}s (expected < 3s)"

        print(
            f"\n  [PERF] Full search: {metrics.total_latency_ms:.0f}ms "
            f"(embed: {metrics.embedding_latency_ms:.0f}ms, "
            f"search: {metrics.search_latency_ms:.0f}ms, "
            f"rerank: {metrics.rerank_latency_ms:.0f}ms)"
        )

    async def test_dense_only_search_latency(
        self, search_service: SearchService
    ) -> None:
        """Benchmark: Dense-only search should be faster."""
        query = Query.create(
            text="neural networks",
            top_k=5,
            rerank_top_n=3,
        )

        start = time.perf_counter()
        results, metrics = await search_service.search_dense_only(query)
        elapsed = time.perf_counter() - start

        assert len(results) > 0
        # Should be faster without rerank
        assert elapsed < 2.0, f"Dense search took {elapsed:.2f}s (expected < 2s)"

        print(f"\n  [PERF] Dense-only search: {metrics.total_latency_ms:.0f}ms")

    async def test_skip_rerank_latency(self, search_service: SearchService) -> None:
        """Benchmark: Skipping rerank should reduce latency."""
        query = Query.create(
            text="data analysis",
            top_k=5,
            rerank_top_n=3,
        )

        # With rerank
        _, metrics_with = await search_service.search(query, skip_rerank=False)

        # Without rerank
        _, metrics_without = await search_service.search(query, skip_rerank=True)

        # Skip rerank should be faster
        assert metrics_without.total_latency_ms < metrics_with.total_latency_ms

        print(
            f"\n  [PERF] With rerank: {metrics_with.total_latency_ms:.0f}ms, "
            f"Without: {metrics_without.total_latency_ms:.0f}ms"
        )


class TestSearchServiceEdgeCases:
    """Edge case tests for SearchService."""

    async def test_search_very_short_query(self, search_service: SearchService) -> None:
        """Test search with very short query."""
        query = Query.create(text="AI", top_k=3, rerank_top_n=2)
        results, _ = await search_service.search(query)

        # Should still return results
        assert len(results) > 0

    async def test_search_long_query(self, search_service: SearchService) -> None:
        """Test search with longer query."""
        long_query = (
            "I'm looking for information about Python programming language, "
            "specifically how it's used in machine learning, data science, "
            "and artificial intelligence applications."
        )
        query = Query.create(text=long_query, top_k=5, rerank_top_n=3)
        results, _ = await search_service.search(query)

        assert len(results) > 0
        # Should find Python/ML related content
        contents = [r.chunk.content for r in results]
        assert any("Python" in c or "machine learning" in c.lower() for c in contents)

    async def test_search_special_characters(
        self, search_service: SearchService
    ) -> None:
        """Test search with special characters."""
        query = Query.create(
            text="iPhone 15 Pro A17",
            top_k=5,
            rerank_top_n=3,
        )
        results, _ = await search_service.search(query)

        assert len(results) > 0
        # Should find iPhone content
        contents = [r.chunk.content for r in results]
        assert any("iPhone" in c or "A17" in c for c in contents)

    async def test_search_no_matching_content(
        self, search_service: SearchService
    ) -> None:
        """Test search with query unlikely to match well."""
        query = Query.create(
            text="quantum physics black holes",
            top_k=3,
            rerank_top_n=2,
        )
        results, _ = await search_service.search(query)

        # Should still return some results (nearest neighbors)
        # but scores might be lower
        assert len(results) > 0

    async def test_search_rerank_top_n_equals_top_k(
        self, search_service: SearchService
    ) -> None:
        """Test when rerank_top_n equals top_k."""
        query = Query.create(
            text="Python",
            top_k=5,
            rerank_top_n=5,  # Same as top_k
        )
        results, metrics = await search_service.search(query)

        # Should return up to rerank_top_n results
        assert len(results) <= 5
        assert metrics.final_results_count <= metrics.initial_results_count
