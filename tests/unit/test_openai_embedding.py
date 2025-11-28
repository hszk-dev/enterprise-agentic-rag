"""Unit tests for OpenAI embedding service with mocked dependencies."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import OpenAISettings
from src.domain.exceptions import EmbeddingError
from src.infrastructure.embeddings import OpenAIEmbeddingService


@pytest.fixture
def mock_openai_settings():
    """Create OpenAI settings for unit tests."""
    return OpenAISettings(
        api_key="test-api-key",  # pragma: allowlist secret
        model="gpt-4o",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        max_retries=3,
        timeout=30.0,
    )


@pytest.mark.unit
class TestOpenAIEmbeddingServiceUnit:
    """Unit tests for OpenAIEmbeddingService class."""

    def test_dimension_property(self, mock_openai_settings: OpenAISettings) -> None:
        """Test that dimension property returns configured dimensions."""
        with patch("src.infrastructure.embeddings.openai_embedding.AsyncOpenAI"):
            service = OpenAIEmbeddingService(mock_openai_settings)
            assert service.dimension == 1536

    async def test_embed_text_success(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test embed_text returns embedding on success."""
        with patch(
            "src.infrastructure.embeddings.openai_embedding.AsyncOpenAI"
        ) as mock_client_class:
            # Setup mock response
            mock_embedding = [0.1] * 1536
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=mock_embedding)]
            mock_response.usage.total_tokens = 10

            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            service = OpenAIEmbeddingService(mock_openai_settings)
            result = await service.embed_text("Hello, world!")

            assert result == mock_embedding
            assert len(result) == 1536
            mock_client.embeddings.create.assert_called_once()

    async def test_embed_text_empty_raises_error(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test embed_text raises EmbeddingError for empty text."""
        with patch("src.infrastructure.embeddings.openai_embedding.AsyncOpenAI"):
            service = OpenAIEmbeddingService(mock_openai_settings)

            with pytest.raises(EmbeddingError) as exc_info:
                await service.embed_text("")

            assert "empty" in str(exc_info.value).lower()

    async def test_embed_text_whitespace_raises_error(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test embed_text raises EmbeddingError for whitespace-only text."""
        with patch("src.infrastructure.embeddings.openai_embedding.AsyncOpenAI"):
            service = OpenAIEmbeddingService(mock_openai_settings)

            with pytest.raises(EmbeddingError):
                await service.embed_text("   ")

    async def test_embed_text_api_error_raises_embedding_error(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test embed_text raises EmbeddingError on API failure."""
        with patch(
            "src.infrastructure.embeddings.openai_embedding.AsyncOpenAI"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            mock_client_class.return_value = mock_client

            service = OpenAIEmbeddingService(mock_openai_settings)

            with pytest.raises(EmbeddingError) as exc_info:
                await service.embed_text("Test text")

            assert "failed" in str(exc_info.value).lower()

    async def test_embed_texts_success(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test embed_texts returns embeddings for multiple texts."""
        with patch(
            "src.infrastructure.embeddings.openai_embedding.AsyncOpenAI"
        ) as mock_client_class:
            # Setup mock response
            mock_embeddings = [
                MagicMock(embedding=[0.1] * 1536),
                MagicMock(embedding=[0.2] * 1536),
            ]
            mock_response = MagicMock()
            mock_response.data = mock_embeddings
            mock_response.usage.total_tokens = 20

            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            service = OpenAIEmbeddingService(mock_openai_settings)
            result = await service.embed_texts(["Hello", "World"])

            assert len(result) == 2
            assert len(result[0]) == 1536
            assert len(result[1]) == 1536

    async def test_embed_texts_empty_list_returns_empty(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test embed_texts returns empty list for empty input."""
        with patch("src.infrastructure.embeddings.openai_embedding.AsyncOpenAI"):
            service = OpenAIEmbeddingService(mock_openai_settings)
            result = await service.embed_texts([])
            assert result == []

    async def test_embed_texts_all_empty_raises_error(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test embed_texts raises error when all texts are empty."""
        with patch("src.infrastructure.embeddings.openai_embedding.AsyncOpenAI"):
            service = OpenAIEmbeddingService(mock_openai_settings)

            with pytest.raises(EmbeddingError) as exc_info:
                await service.embed_texts(["", "  ", ""])

            assert "empty" in str(exc_info.value).lower()

    async def test_embed_texts_mixed_empty_returns_zero_vectors(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test embed_texts handles mixed empty and non-empty texts."""
        with patch(
            "src.infrastructure.embeddings.openai_embedding.AsyncOpenAI"
        ) as mock_client_class:
            # Setup mock response for non-empty texts
            mock_embedding = MagicMock(embedding=[0.5] * 1536)
            mock_response = MagicMock()
            mock_response.data = [mock_embedding]
            mock_response.usage.total_tokens = 10

            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            service = OpenAIEmbeddingService(mock_openai_settings)
            result = await service.embed_texts(["", "Valid text", ""])

            assert len(result) == 3
            # Empty texts get zero vectors
            assert result[0] == [0.0] * 1536
            assert result[2] == [0.0] * 1536
            # Non-empty text gets real embedding
            assert result[1] == [0.5] * 1536

    async def test_close_closes_client(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test close method closes the OpenAI client."""
        with patch(
            "src.infrastructure.embeddings.openai_embedding.AsyncOpenAI"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            service = OpenAIEmbeddingService(mock_openai_settings)
            await service.close()

            mock_client.close.assert_called_once()
