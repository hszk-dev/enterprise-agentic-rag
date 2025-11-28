"""Unit tests for FastEmbed sparse embedding service with mocked dependencies."""

from unittest.mock import MagicMock, patch

import pytest

from src.domain.exceptions import EmbeddingError
from src.domain.value_objects import SparseVector
from src.infrastructure.embeddings import FastEmbedSparseEmbeddingService


@pytest.mark.unit
class TestFastEmbedSparseEmbeddingServiceUnit:
    """Unit tests for FastEmbedSparseEmbeddingService class."""

    def test_model_name_default(self) -> None:
        """Test that default model name is set correctly."""
        service = FastEmbedSparseEmbeddingService()
        assert service.model_name == "prithvida/Splade_PP_en_v1"

    def test_model_name_custom(self) -> None:
        """Test that custom model name is set correctly."""
        service = FastEmbedSparseEmbeddingService(model_name="custom/model")
        assert service.model_name == "custom/model"

    async def test_embed_text_success(self) -> None:
        """Test embed_text returns sparse vector on success."""
        with patch("fastembed.SparseTextEmbedding") as mock_model_class:
            # Setup mock response
            mock_sparse_result = MagicMock()
            mock_sparse_result.indices.tolist.return_value = [1, 5, 10, 100]
            mock_sparse_result.values.tolist.return_value = [0.5, 0.8, 0.3, 0.9]

            mock_model = MagicMock()
            mock_model.embed.return_value = [mock_sparse_result]
            mock_model_class.return_value = mock_model

            service = FastEmbedSparseEmbeddingService()
            result = await service.embed_text("Hello, world!")

            assert isinstance(result, SparseVector)
            assert len(result) == 4
            assert result.indices == (1, 5, 10, 100)
            assert result.values == (0.5, 0.8, 0.3, 0.9)

    async def test_embed_text_empty_raises_error(self) -> None:
        """Test embed_text raises EmbeddingError for empty text."""
        service = FastEmbedSparseEmbeddingService()

        with pytest.raises(EmbeddingError) as exc_info:
            await service.embed_text("")

        assert "empty" in str(exc_info.value).lower()

    async def test_embed_text_whitespace_raises_error(self) -> None:
        """Test embed_text raises EmbeddingError for whitespace-only text."""
        service = FastEmbedSparseEmbeddingService()

        with pytest.raises(EmbeddingError):
            await service.embed_text("   ")

    async def test_embed_text_model_error_raises_embedding_error(self) -> None:
        """Test embed_text raises EmbeddingError on model failure."""
        with patch("fastembed.SparseTextEmbedding") as mock_model_class:
            mock_model = MagicMock()
            mock_model.embed.side_effect = RuntimeError("Model Error")
            mock_model_class.return_value = mock_model

            service = FastEmbedSparseEmbeddingService()

            with pytest.raises(EmbeddingError) as exc_info:
                await service.embed_text("Test text")

            assert "failed" in str(exc_info.value).lower()

    async def test_embed_texts_success(self) -> None:
        """Test embed_texts returns sparse vectors for multiple texts."""
        with patch("fastembed.SparseTextEmbedding") as mock_model_class:
            # Setup mock responses
            mock_sparse_result1 = MagicMock()
            mock_sparse_result1.indices.tolist.return_value = [1, 2, 3]
            mock_sparse_result1.values.tolist.return_value = [0.1, 0.2, 0.3]

            mock_sparse_result2 = MagicMock()
            mock_sparse_result2.indices.tolist.return_value = [4, 5]
            mock_sparse_result2.values.tolist.return_value = [0.4, 0.5]

            mock_model = MagicMock()
            mock_model.embed.return_value = [mock_sparse_result1, mock_sparse_result2]
            mock_model_class.return_value = mock_model

            service = FastEmbedSparseEmbeddingService()
            result = await service.embed_texts(["Hello", "World"])

            assert len(result) == 2
            assert isinstance(result[0], SparseVector)
            assert isinstance(result[1], SparseVector)
            assert len(result[0]) == 3
            assert len(result[1]) == 2

    async def test_embed_texts_empty_list_returns_empty(self) -> None:
        """Test embed_texts returns empty list for empty input."""
        service = FastEmbedSparseEmbeddingService()
        result = await service.embed_texts([])
        assert result == []

    async def test_embed_texts_with_empty_text_raises_error(self) -> None:
        """Test embed_texts raises error when any text is empty."""
        service = FastEmbedSparseEmbeddingService()

        with pytest.raises(EmbeddingError) as exc_info:
            await service.embed_texts(["Valid text", "", "Another text"])

        assert "index 1" in str(exc_info.value)

    async def test_embed_texts_count_mismatch_raises_error(self) -> None:
        """Test embed_texts raises error when embedding count doesn't match input."""
        with patch("fastembed.SparseTextEmbedding") as mock_model_class:
            # Return wrong number of embeddings
            mock_sparse_result = MagicMock()
            mock_sparse_result.indices.tolist.return_value = [1]
            mock_sparse_result.values.tolist.return_value = [0.1]

            mock_model = MagicMock()
            mock_model.embed.return_value = [
                mock_sparse_result
            ]  # Only 1 result for 2 texts
            mock_model_class.return_value = mock_model

            service = FastEmbedSparseEmbeddingService()

            with pytest.raises(EmbeddingError) as exc_info:
                await service.embed_texts(["Hello", "World"])

            assert "Expected 2" in str(exc_info.value)

    async def test_lazy_model_initialization(self) -> None:
        """Test that model is lazily initialized on first use."""
        with patch("fastembed.SparseTextEmbedding") as mock_model_class:
            mock_sparse_result = MagicMock()
            mock_sparse_result.indices.tolist.return_value = [1]
            mock_sparse_result.values.tolist.return_value = [0.5]

            mock_model = MagicMock()
            mock_model.embed.return_value = [mock_sparse_result]
            mock_model_class.return_value = mock_model

            service = FastEmbedSparseEmbeddingService()

            # Model should not be initialized yet
            mock_model_class.assert_not_called()

            # First embed call should initialize the model
            await service.embed_text("Test")
            mock_model_class.assert_called_once()

            # Second call should reuse the same model
            await service.embed_text("Test 2")
            mock_model_class.assert_called_once()  # Still only 1 call
