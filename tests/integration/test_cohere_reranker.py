"""Integration tests for Cohere reranker service.

These tests require a valid COHERE_API_KEY environment variable.
Run with: COHERE_API_KEY=xxx pytest tests/integration/test_cohere_reranker.py -m integration -v

To run these tests, set the COHERE_API_KEY environment variable.
"""

import os
import time
from uuid import uuid4

import pytest

from config import CohereSettings
from src.domain.entities import Chunk, SearchResult
from src.infrastructure.rerankers import CohereReranker

# Skip all tests in this module if COHERE_API_KEY is not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY"),
        reason="COHERE_API_KEY environment variable not set",
    ),
]


@pytest.fixture
def cohere_settings():
    """Create Cohere settings with real API key."""
    api_key = os.environ.get("COHERE_API_KEY", "")
    return CohereSettings(
        api_key=api_key,  # pragma: allowlist secret
        rerank_model="rerank-v3.5",
        max_retries=3,
        timeout=30.0,
    )


@pytest.fixture
def reranker(cohere_settings: CohereSettings):
    """Create Cohere reranker for testing."""
    return CohereReranker(cohere_settings)


@pytest.fixture
def sample_search_results():
    """Create sample search results for reranking."""
    doc_id = uuid4()
    contents = [
        "Python is a programming language known for its simplicity and readability.",
        "The weather forecast predicts rain tomorrow in Tokyo.",
        "Machine learning uses algorithms to learn patterns from data.",
        "Python is widely used in data science, AI, and web development.",
        "Italian pasta recipes include spaghetti carbonara and penne arrabbiata.",
        "Deep learning is a subset of machine learning using neural networks.",
        "Python's pandas library is essential for data manipulation.",
    ]
    results = []
    for i, content in enumerate(contents):
        chunk = Chunk.create(
            document_id=doc_id,
            content=content,
            chunk_index=i,
            start_char=0,
            end_char=len(content),
            metadata={"source": "test", "index": i},
        )
        # Simulate initial search scores
        results.append(SearchResult(chunk=chunk, score=0.5 - i * 0.05, rank=i + 1))
    return results


class TestCohereRerankerIntegration:
    """Integration tests for CohereReranker."""

    async def test_rerank_returns_sorted_results(
        self,
        reranker: CohereReranker,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test that rerank returns results sorted by relevance."""
        query = "What is Python programming language?"
        reranked = await reranker.rerank(query, sample_search_results, top_n=3)

        assert len(reranked) == 3
        # Should be sorted by rerank_score descending
        scores = [r.rerank_score for r in reranked]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"

    async def test_rerank_top_n_limit(
        self,
        reranker: CohereReranker,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test that rerank respects top_n limit."""
        query = "Python programming"

        # Request different top_n values
        for top_n in [1, 2, 3, 5]:
            reranked = await reranker.rerank(query, sample_search_results, top_n=top_n)
            assert len(reranked) == top_n, (
                f"Expected {top_n} results, got {len(reranked)}"
            )

    async def test_rerank_top_n_exceeds_results(
        self,
        reranker: CohereReranker,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test that top_n > len(results) returns all results."""
        query = "Python"
        reranked = await reranker.rerank(query, sample_search_results, top_n=100)

        # Should return all available results
        assert len(reranked) == len(sample_search_results)

    async def test_rerank_score_range(
        self,
        reranker: CohereReranker,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test that rerank scores are in valid range [0, 1]."""
        query = "Python programming language"
        reranked = await reranker.rerank(query, sample_search_results, top_n=5)

        for result in reranked:
            assert result.rerank_score is not None
            assert 0.0 <= result.rerank_score <= 1.0, (
                f"Score {result.rerank_score} out of range [0, 1]"
            )

    async def test_rerank_preserves_original_score(
        self,
        reranker: CohereReranker,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test that original search score is preserved after reranking."""
        query = "Python"
        reranked = await reranker.rerank(query, sample_search_results, top_n=3)

        for result in reranked:
            # Original score should be preserved
            assert result.score is not None
            # Rerank score should be added
            assert result.rerank_score is not None

    async def test_rerank_assigns_ranks(
        self,
        reranker: CohereReranker,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test that ranks are correctly assigned (1-indexed)."""
        query = "machine learning"
        reranked = await reranker.rerank(query, sample_search_results, top_n=5)

        expected_ranks = list(range(1, len(reranked) + 1))
        actual_ranks = [r.rank for r in reranked]
        assert actual_ranks == expected_ranks, (
            f"Ranks {actual_ranks} != {expected_ranks}"
        )

    async def test_rerank_relevant_first(
        self,
        reranker: CohereReranker,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test that relevant documents are ranked higher."""
        query = "Python programming language"
        reranked = await reranker.rerank(query, sample_search_results, top_n=5)

        # Python-related docs should be in top results
        top_3_contents = [r.chunk.content for r in reranked[:3]]
        python_in_top_3 = sum(1 for c in top_3_contents if "Python" in c)
        assert python_in_top_3 >= 2, (
            f"Expected >=2 Python docs in top 3, got {python_in_top_3}"
        )

    async def test_rerank_irrelevant_lower(
        self,
        reranker: CohereReranker,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test that irrelevant documents are ranked lower."""
        query = "Python data science programming"
        reranked = await reranker.rerank(query, sample_search_results, top_n=7)

        # Weather and pasta (irrelevant) should be ranked lower
        bottom_2_contents = [r.chunk.content for r in reranked[-2:]]
        irrelevant_count = sum(
            1
            for c in bottom_2_contents
            if "weather" in c.lower() or "pasta" in c.lower()
        )
        assert irrelevant_count >= 1, "Expected irrelevant docs in bottom results"

    async def test_rerank_japanese_query(
        self,
        reranker: CohereReranker,
    ) -> None:
        """Test reranking with Japanese query and documents."""
        doc_id = uuid4()
        contents = [
            "Pythonは機械学習でよく使われるプログラミング言語です。",
            "今日の東京の天気は晴れです。",
            "データサイエンスにはPythonが人気です。",
            "イタリア料理のパスタレシピを紹介します。",
        ]
        results = [
            SearchResult(
                chunk=Chunk.create(
                    document_id=doc_id,
                    content=c,
                    chunk_index=i,
                    start_char=0,
                    end_char=len(c),
                ),
                score=0.5,
                rank=i + 1,
            )
            for i, c in enumerate(contents)
        ]

        query = "Pythonでの機械学習とデータサイエンス"
        reranked = await reranker.rerank(query, results, top_n=4)

        assert len(reranked) == 4
        # Python/ML related should rank higher
        top_content = reranked[0].chunk.content
        assert (
            "Python" in top_content
            or "機械学習" in top_content
            or "データサイエンス" in top_content
        )

    async def test_rerank_mixed_language(
        self,
        reranker: CohereReranker,
    ) -> None:
        """Test reranking with mixed English/Japanese content."""
        doc_id = uuid4()
        contents = [
            "Python is great for machine learning applications.",
            "Pythonは機械学習に最適な言語です。",
            "The weather in Tokyo is sunny today.",
            "東京の天気は晴れです。",
        ]
        results = [
            SearchResult(
                chunk=Chunk.create(
                    document_id=doc_id,
                    content=c,
                    chunk_index=i,
                    start_char=0,
                    end_char=len(c),
                ),
                score=0.5,
                rank=i + 1,
            )
            for i, c in enumerate(contents)
        ]

        query = "Python machine learning"
        reranked = await reranker.rerank(query, results, top_n=4)

        # Both Python docs (English and Japanese) should rank high
        top_2_contents = [r.chunk.content for r in reranked[:2]]
        python_count = sum(1 for c in top_2_contents if "Python" in c)
        assert python_count == 2, (
            f"Expected both Python docs in top 2, got {python_count}"
        )

    async def test_rerank_empty_results(
        self,
        reranker: CohereReranker,
    ) -> None:
        """Test reranking with empty results."""
        reranked = await reranker.rerank("test query", [], top_n=5)
        assert reranked == []

    async def test_rerank_single_result(
        self,
        reranker: CohereReranker,
    ) -> None:
        """Test reranking with single result."""
        doc_id = uuid4()
        chunk = Chunk.create(
            document_id=doc_id,
            content="Python programming language.",
            chunk_index=0,
            start_char=0,
            end_char=30,
        )
        results = [SearchResult(chunk=chunk, score=0.5, rank=1)]

        reranked = await reranker.rerank("Python", results, top_n=5)

        assert len(reranked) == 1
        assert reranked[0].rank == 1
        assert reranked[0].rerank_score is not None

    async def test_model_name_property(
        self,
        reranker: CohereReranker,
    ) -> None:
        """Test model_name property returns expected value."""
        assert reranker.model_name == "rerank-v3.5"

    async def test_rerank_performance(
        self,
        reranker: CohereReranker,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Benchmark: Reranking 7 docs should complete in < 3 seconds."""
        query = "Python programming"

        start = time.perf_counter()
        await reranker.rerank(query, sample_search_results, top_n=5)
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0, f"Reranking took {elapsed:.2f}s (expected < 3s)"
        print(f"\n  [PERF] Reranking 7 docs in {elapsed:.2f}s")

    async def test_rerank_with_long_documents(
        self,
        reranker: CohereReranker,
    ) -> None:
        """Test reranking with longer document content."""
        doc_id = uuid4()
        # Create longer documents (simulating chunks)
        long_content = " ".join(["Python is a versatile programming language."] * 20)
        short_content = "Weather forecast for tomorrow."

        results = [
            SearchResult(
                chunk=Chunk.create(
                    document_id=doc_id,
                    content=long_content,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(long_content),
                ),
                score=0.5,
                rank=1,
            ),
            SearchResult(
                chunk=Chunk.create(
                    document_id=doc_id,
                    content=short_content,
                    chunk_index=1,
                    start_char=0,
                    end_char=len(short_content),
                ),
                score=0.5,
                rank=2,
            ),
        ]

        query = "Python programming"
        reranked = await reranker.rerank(query, results, top_n=2)

        assert len(reranked) == 2
        # Long Python doc should rank higher
        assert "Python" in reranked[0].chunk.content

    async def test_close_is_safe(
        self,
        reranker: CohereReranker,
    ) -> None:
        """Test that close() can be called safely."""
        await reranker.close()
        # Should be able to call multiple times
        await reranker.close()
