"""OpenAI embedding service implementation.

This module provides dense embedding generation using OpenAI's text-embedding-3-small model.
"""

import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.domain.exceptions import EmbeddingError

if TYPE_CHECKING:
    from config.settings import OpenAISettings

logger = logging.getLogger(__name__)


class OpenAIEmbeddingService:
    """OpenAI embedding service for dense vector generation.

    Implements the EmbeddingService protocol using OpenAI's API.

    Example:
        >>> service = OpenAIEmbeddingService(settings)
        >>> embedding = await service.embed_text("Hello, world!")
        >>> len(embedding)
        1536
    """

    def __init__(self, settings: "OpenAISettings") -> None:
        """Initialize the OpenAI embedding service.

        Args:
            settings: OpenAI configuration settings.
        """
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(),
            timeout=settings.timeout,
            max_retries=0,  # We handle retries with tenacity
        )
        self._model = settings.embedding_model
        self._dimension = settings.embedding_dimensions

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension.

        Returns:
            Embedding dimension (1536 for text-embedding-3-small).
        """
        return self._dimension

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Dense embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not text or not text.strip():
            msg = "Cannot embed empty text"
            raise EmbeddingError(msg)

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=text,
                dimensions=self._dimension,
            )
            embedding = response.data[0].embedding
            logger.debug(
                f"Generated embedding for text ({len(text)} chars), "
                f"tokens: {response.usage.total_tokens}"
            )
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            msg = f"Embedding generation failed: {e}"
            raise EmbeddingError(msg) from e

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (batch).

        Uses OpenAI's batch embedding API for efficiency.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of dense embedding vectors in the same order as input.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not texts:
            return []

        # Filter empty texts but track their positions
        non_empty_texts: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append((i, text))

        if not non_empty_texts:
            msg = "All texts are empty"
            raise EmbeddingError(msg)

        try:
            # Batch texts (OpenAI has a limit of ~2048 texts per request)
            batch_size = 2048
            all_embeddings: dict[int, list[float]] = {}

            for batch_start in range(0, len(non_empty_texts), batch_size):
                batch = non_empty_texts[batch_start : batch_start + batch_size]
                batch_texts = [t[1] for t in batch]
                batch_indices = [t[0] for t in batch]

                response = await self._client.embeddings.create(
                    model=self._model,
                    input=batch_texts,
                    dimensions=self._dimension,
                )

                # Map embeddings back to original indices
                for embedding_data, original_idx in zip(
                    response.data, batch_indices, strict=True
                ):
                    all_embeddings[original_idx] = embedding_data.embedding

                logger.debug(
                    f"Generated batch embeddings for {len(batch_texts)} texts, "
                    f"tokens: {response.usage.total_tokens}"
                )

            # Build result list in original order
            # For empty texts, we return zero vectors
            result: list[list[float]] = []
            for i in range(len(texts)):
                if i in all_embeddings:
                    result.append(all_embeddings[i])
                else:
                    # Empty text gets zero vector
                    result.append([0.0] * self._dimension)

            return result

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            msg = f"Batch embedding generation failed: {e}"
            raise EmbeddingError(msg) from e

    async def close(self) -> None:
        """Close the OpenAI client.

        Releases any resources held by the client.
        """
        await self._client.close()
