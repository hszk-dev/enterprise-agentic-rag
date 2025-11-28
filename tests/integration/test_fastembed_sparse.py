"""Integration tests for FastEmbed sparse embedding service.

These tests require the FastEmbed model to be downloaded (happens automatically on first use).
The SPLADE model is downloaded to the cache directory on initialization.

Run with: pytest tests/integration/test_fastembed_sparse.py -m integration -v
"""

import time

import pytest

from src.domain.value_objects import SparseVector
from src.infrastructure.embeddings import FastEmbedSparseEmbeddingService

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sparse_embedding_service():
    """Create sparse embedding service (module-scoped for model reuse).

    Module scope ensures the model is loaded only once per test module,
    significantly speeding up test execution.
    """
    return FastEmbedSparseEmbeddingService()


class TestFastEmbedSparseIntegration:
    """Integration tests for FastEmbedSparseEmbeddingService."""

    async def test_embed_text_returns_sparse_vector(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Test that embed_text returns a valid sparse vector."""
        text = "Machine learning is a subset of artificial intelligence."
        sparse = await sparse_embedding_service.embed_text(text)

        # Verify sparse vector structure
        assert isinstance(sparse, SparseVector)
        assert len(sparse) > 0  # Non-zero elements exist
        assert all(isinstance(idx, int) for idx in sparse.indices)
        assert all(isinstance(val, float) for val in sparse.values)
        # SPLADE values are positive (term expansion weights)
        assert all(val > 0 for val in sparse.values)

    async def test_embed_texts_batch(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Test batch embedding generation."""
        texts = [
            "First document about machine learning.",
            "Second document about deep learning.",
            "Third document about natural language processing.",
        ]
        embeddings = await sparse_embedding_service.embed_texts(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert isinstance(emb, SparseVector)
            assert len(emb) > 0

    async def test_embed_texts_empty_list(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Test that empty list returns empty list."""
        embeddings = await sparse_embedding_service.embed_texts([])
        assert embeddings == []

    async def test_model_lazy_loading(self) -> None:
        """Test that model is lazily loaded on first use."""
        service = FastEmbedSparseEmbeddingService()
        # Model should not be loaded yet
        assert service._model is None

        # After embedding, model should be loaded
        await service.embed_text("Test text")
        assert service._model is not None

    async def test_similar_texts_have_overlapping_terms(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Test that similar texts share common sparse terms."""
        text1 = "Python is a popular programming language."
        text2 = "Python programming is widely used."
        text3 = "The weather is sunny today."

        emb1 = await sparse_embedding_service.embed_text(text1)
        emb2 = await sparse_embedding_service.embed_text(text2)
        emb3 = await sparse_embedding_service.embed_text(text3)

        # Calculate term overlap
        overlap_1_2 = len(set(emb1.indices) & set(emb2.indices))
        overlap_1_3 = len(set(emb1.indices) & set(emb3.indices))

        # Similar texts (both about Python) should have more overlapping terms
        assert overlap_1_2 > overlap_1_3, (
            f"Expected overlap_1_2 ({overlap_1_2}) > overlap_1_3 ({overlap_1_3})"
        )

    async def test_different_texts_have_different_vectors(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Test that different texts produce different sparse vectors."""
        text1 = "The iPhone 15 Pro features A17 chip."
        text2 = "The Galaxy S24 Ultra has advanced camera."

        emb1 = await sparse_embedding_service.embed_text(text1)
        emb2 = await sparse_embedding_service.embed_text(text2)

        # Different products should have different sparse representations
        # They may share some common terms, but not be identical
        assert emb1.indices != emb2.indices or emb1.values != emb2.values

    async def test_japanese_text_embedding(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Test embedding generation for Japanese text.

        Note: SPLADE model (prithvida/Splade_PP_en_v1) is English-focused,
        but should not raise errors for Japanese text.
        Performance/quality may be limited for non-English text.
        """
        text = "機械学習は人工知能の一分野です。"
        sparse = await sparse_embedding_service.embed_text(text)

        # Should return some sparse representation (even if suboptimal)
        assert isinstance(sparse, SparseVector)
        assert len(sparse) > 0

    async def test_mixed_language_text(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Test embedding for mixed English/Japanese text."""
        text = "Pythonは機械学習 (machine learning) でよく使われます。"
        sparse = await sparse_embedding_service.embed_text(text)

        assert isinstance(sparse, SparseVector)
        assert len(sparse) > 0

    async def test_special_characters_handling(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Test handling of special characters and product codes."""
        text = "Product code: A2849-XYZ, version 3.14.159"
        sparse = await sparse_embedding_service.embed_text(text)

        assert isinstance(sparse, SparseVector)
        assert len(sparse) > 0

    async def test_long_text_embedding(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Test embedding for longer text (chunk-sized)."""
        # Simulate a typical chunk size (~1000 chars)
        long_text = " ".join(["This is a test sentence about machine learning."] * 50)
        assert len(long_text) > 500  # Ensure it's reasonably long

        sparse = await sparse_embedding_service.embed_text(long_text)

        assert isinstance(sparse, SparseVector)
        assert len(sparse) > 0

    async def test_short_text_embedding(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Test embedding for very short text."""
        short_text = "Python"
        sparse = await sparse_embedding_service.embed_text(short_text)

        assert isinstance(sparse, SparseVector)
        assert len(sparse) > 0

    async def test_model_name_property(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Test model_name property returns expected value."""
        assert sparse_embedding_service.model_name == "prithvida/Splade_PP_en_v1"

    async def test_sparse_vector_to_dict_conversion(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Test that sparse vectors can be converted to dict for storage."""
        text = "Test document for conversion."
        sparse = await sparse_embedding_service.embed_text(text)

        # Convert to dict (for Qdrant storage)
        sparse_dict = sparse.to_dict()

        assert isinstance(sparse_dict, dict)
        assert len(sparse_dict) == len(sparse)
        assert all(isinstance(k, int) for k in sparse_dict)
        assert all(isinstance(v, float) for v in sparse_dict.values())

    async def test_batch_consistency(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Test that single and batch embedding produce consistent results."""
        text = "Test document for consistency check."

        # Single embedding
        single_result = await sparse_embedding_service.embed_text(text)

        # Batch embedding with single text
        batch_result = await sparse_embedding_service.embed_texts([text])

        # Results should be identical
        assert single_result.indices == batch_result[0].indices
        assert single_result.values == batch_result[0].values

    async def test_performance_benchmark_small_batch(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Benchmark: 10 texts should complete quickly."""
        texts = [f"Document number {i} with some content about AI." for i in range(10)]

        start = time.perf_counter()
        embeddings = await sparse_embedding_service.embed_texts(texts)
        elapsed = time.perf_counter() - start

        assert len(embeddings) == 10
        # Should be fast for small batches (< 5 seconds)
        assert elapsed < 5.0, f"Batch embedding took {elapsed:.2f}s (expected < 5s)"

    @pytest.mark.slow
    async def test_performance_benchmark_large_batch(
        self, sparse_embedding_service: FastEmbedSparseEmbeddingService
    ) -> None:
        """Benchmark: 100 texts should complete in < 30 seconds."""
        texts = [
            f"Document number {i} with content about machine learning."
            for i in range(100)
        ]

        start = time.perf_counter()
        embeddings = await sparse_embedding_service.embed_texts(texts)
        elapsed = time.perf_counter() - start

        assert len(embeddings) == 100
        assert elapsed < 30.0, f"Large batch took {elapsed:.2f}s (expected < 30s)"
        print(
            f"\n  [PERF] 100 texts embedded in {elapsed:.2f}s ({elapsed / 100 * 1000:.1f}ms/text)"
        )
