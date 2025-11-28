"""Integration tests for OpenAI embedding service.

These tests require a valid OpenAI API key.
Run with: pytest tests/integration/test_openai_embedding.py -m integration

To run these tests, set the OPENAI_API_KEY environment variable.
"""

import os

import pytest

from config import OpenAISettings
from src.infrastructure.embeddings import OpenAIEmbeddingService

# Skip all tests in this module if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)


@pytest.fixture
def real_openai_settings():
    """Create OpenAI settings with real API key."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return OpenAISettings(
        api_key=api_key,  # pragma: allowlist secret
        model="gpt-4o",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        max_retries=3,
        timeout=30.0,
    )


@pytest.fixture
async def embedding_service(real_openai_settings: OpenAISettings):
    """Create embedding service for testing."""
    service = OpenAIEmbeddingService(real_openai_settings)
    yield service
    await service.close()


@pytest.mark.integration
class TestOpenAIEmbeddingServiceIntegration:
    """Integration tests for OpenAIEmbeddingService."""

    async def test_embed_text_returns_vector(
        self, embedding_service: OpenAIEmbeddingService
    ) -> None:
        """Test that embed_text returns a valid embedding vector."""
        text = "This is a test sentence for embedding."
        embedding = await embedding_service.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
        # Embeddings should be normalized (unit length)
        magnitude = sum(x**2 for x in embedding) ** 0.5
        assert 0.99 < magnitude < 1.01

    async def test_embed_texts_batch(
        self, embedding_service: OpenAIEmbeddingService
    ) -> None:
        """Test that embed_texts returns embeddings for multiple texts."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence with different content.",
        ]
        embeddings = await embedding_service.embed_texts(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 1536

    async def test_similar_texts_have_similar_embeddings(
        self, embedding_service: OpenAIEmbeddingService
    ) -> None:
        """Test that similar texts produce similar embeddings."""
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "A fast brown fox leaps over a sleepy dog."
        text3 = "Python is a popular programming language."

        embeddings = await embedding_service.embed_texts([text1, text2, text3])

        # Calculate cosine similarity
        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            mag_a = sum(x**2 for x in a) ** 0.5
            mag_b = sum(x**2 for x in b) ** 0.5
            return dot / (mag_a * mag_b)

        # Similar sentences should have higher similarity
        sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
        sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])

        assert sim_1_2 > sim_1_3, "Similar sentences should have higher similarity"

    async def test_dimension_property(
        self, embedding_service: OpenAIEmbeddingService
    ) -> None:
        """Test that dimension property returns correct value."""
        assert embedding_service.dimension == 1536

    async def test_embed_long_text(
        self, embedding_service: OpenAIEmbeddingService
    ) -> None:
        """Test embedding generation for longer text."""
        # Create a longer text (but still within token limits)
        long_text = " ".join(["This is a test sentence."] * 100)
        embedding = await embedding_service.embed_text(long_text)

        assert len(embedding) == 1536

    async def test_embed_japanese_text(
        self, embedding_service: OpenAIEmbeddingService
    ) -> None:
        """Test embedding generation for Japanese text."""
        text = "これは日本語のテスト文です。自然言語処理のテストを行っています。"
        embedding = await embedding_service.embed_text(text)

        assert len(embedding) == 1536
        # Verify it's a valid embedding
        magnitude = sum(x**2 for x in embedding) ** 0.5
        assert 0.99 < magnitude < 1.01
