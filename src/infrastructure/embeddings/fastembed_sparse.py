"""FastEmbed sparse embedding service implementation.

This module provides sparse embedding generation using FastEmbed's SPLADE models
for keyword-based hybrid search.
"""

import logging
from typing import TYPE_CHECKING

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.domain.exceptions import EmbeddingError
from src.domain.value_objects import SparseVector

if TYPE_CHECKING:
    from fastembed import SparseTextEmbedding

logger = logging.getLogger(__name__)


class FastEmbedSparseEmbeddingService:
    """FastEmbed sparse embedding service for keyword search.

    Implements the SparseEmbeddingService protocol using FastEmbed's SPLADE model.
    SPLADE (SParse Lexical AnD Expansion) provides sparse representations that
    combine lexical matching with learned term expansion.

    Example:
        >>> service = FastEmbedSparseEmbeddingService()
        >>> sparse = await service.embed_text("Hello, world!")
        >>> len(sparse)  # Number of non-zero terms
        42
    """

    # Default SPLADE model - good balance of quality and speed
    DEFAULT_MODEL = "prithvida/Splade_PP_en_v1"

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the FastEmbed sparse embedding service.

        Args:
            model_name: SPLADE model name. Defaults to prithvida/Splade_PP_en_v1.
            cache_dir: Directory to cache downloaded models.
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._cache_dir = cache_dir
        self._model: SparseTextEmbedding | None = None

    def _get_model(self) -> "SparseTextEmbedding":
        """Lazily initialize and return the embedding model.

        Returns:
            Initialized SparseTextEmbedding model.

        Raises:
            EmbeddingError: If model initialization fails.
        """
        if self._model is None:
            try:
                from fastembed import SparseTextEmbedding

                self._model = SparseTextEmbedding(
                    model_name=self._model_name,
                    cache_dir=self._cache_dir,
                )
                logger.info(f"Initialized FastEmbed SPLADE model: {self._model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize FastEmbed model: {e}")
                msg = f"Failed to initialize sparse embedding model: {e}"
                raise EmbeddingError(msg) from e
        return self._model

    @retry(
        retry=retry_if_exception_type((RuntimeError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def embed_text(self, text: str) -> SparseVector:
        """Generate sparse embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Sparse vector representation.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not text or not text.strip():
            msg = "Cannot embed empty text"
            raise EmbeddingError(msg)

        try:
            model = self._get_model()
            # FastEmbed returns a generator, we need to consume it
            embeddings = list(model.embed([text]))

            if not embeddings:
                msg = "No embedding returned from model"
                raise EmbeddingError(msg)

            sparse_embedding = embeddings[0]

            # Convert FastEmbed sparse format to SparseVector
            sparse_dict = dict(
                zip(
                    sparse_embedding.indices.tolist(),
                    sparse_embedding.values.tolist(),
                    strict=True,
                )
            )
            sparse_vector = SparseVector.from_dict(sparse_dict)

            logger.debug(
                f"Generated sparse embedding for text ({len(text)} chars), "
                f"non-zero terms: {len(sparse_vector)}"
            )
            return sparse_vector

        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate sparse embedding: {e}")
            msg = f"Sparse embedding generation failed: {e}"
            raise EmbeddingError(msg) from e

    @retry(
        retry=retry_if_exception_type((RuntimeError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        """Generate sparse embeddings for multiple texts (batch).

        Args:
            texts: List of input texts to embed.

        Returns:
            List of sparse vector representations in the same order as input.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not texts:
            return []

        # Validate all texts are non-empty
        for i, text in enumerate(texts):
            if not text or not text.strip():
                msg = f"Cannot embed empty text at index {i}"
                raise EmbeddingError(msg)

        try:
            model = self._get_model()
            # FastEmbed returns a generator
            embeddings = list(model.embed(texts))

            if len(embeddings) != len(texts):
                msg = f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                raise EmbeddingError(msg)

            sparse_vectors: list[SparseVector] = []
            for sparse_embedding in embeddings:
                sparse_dict = dict(
                    zip(
                        sparse_embedding.indices.tolist(),
                        sparse_embedding.values.tolist(),
                        strict=True,
                    )
                )
                sparse_vectors.append(SparseVector.from_dict(sparse_dict))

            logger.debug(
                f"Generated batch sparse embeddings for {len(texts)} texts, "
                f"avg non-zero terms: {sum(len(sv) for sv in sparse_vectors) / len(sparse_vectors):.1f}"
            )
            return sparse_vectors

        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate batch sparse embeddings: {e}")
            msg = f"Batch sparse embedding generation failed: {e}"
            raise EmbeddingError(msg) from e

    @property
    def model_name(self) -> str:
        """Return the model name being used.

        Returns:
            Model name string.
        """
        return self._model_name
