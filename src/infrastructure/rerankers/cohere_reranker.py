"""Cohere reranker service implementation.

This module provides reranking functionality using Cohere's rerank API
to improve search result relevance.
"""

import logging
from typing import TYPE_CHECKING

import cohere
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.domain.entities import SearchResult
from src.domain.exceptions import RerankError

if TYPE_CHECKING:
    from config.settings import CohereSettings

logger = logging.getLogger(__name__)

# Retryable Cohere exceptions
RETRYABLE_EXCEPTIONS = (
    cohere.errors.TooManyRequestsError,
    cohere.errors.InternalServerError,
    cohere.errors.ServiceUnavailableError,
)


class CohereReranker:
    """Cohere reranker service for improving search relevance.

    Implements the Reranker protocol using Cohere's rerank API.
    Uses cross-encoder architecture to score query-document pairs.

    Example:
        >>> reranker = CohereReranker(settings)
        >>> reranked = await reranker.rerank("query", results, top_n=5)
        >>> reranked[0].rerank_score  # Highest relevance score
        0.95
    """

    def __init__(self, settings: "CohereSettings") -> None:
        """Initialize the Cohere reranker.

        Args:
            settings: Cohere configuration settings.
        """
        self._settings = settings
        self._client = cohere.AsyncClientV2(
            api_key=settings.api_key.get_secret_value(),
            timeout=settings.timeout,
        )
        self._model = settings.rerank_model

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = 5,
    ) -> list[SearchResult]:
        """Rerank search results based on relevance to query.

        Args:
            query: Original query text.
            results: Search results to rerank.
            top_n: Number of top results to return.

        Returns:
            Reranked results with updated scores and ranks.

        Raises:
            RerankError: If reranking fails.
        """
        if not results:
            return []

        if not query or not query.strip():
            msg = "Cannot rerank with empty query"
            raise RerankError(msg)

        # Limit top_n to available results
        top_n = min(top_n, len(results))

        try:
            # Extract document texts for reranking
            documents = [result.chunk.content for result in results]

            response = await self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=top_n,
            )

            # Build reranked results
            reranked_results: list[SearchResult] = []
            for rank, rerank_result in enumerate(response.results, start=1):
                original_index = rerank_result.index
                original_result = results[original_index]

                # Create new SearchResult with rerank score
                reranked = SearchResult(
                    chunk=original_result.chunk,
                    score=original_result.score,
                    rerank_score=rerank_result.relevance_score,
                    rank=rank,
                )
                reranked_results.append(reranked)

            logger.debug(
                f"Reranked {len(results)} results to top {top_n}, "
                f"top score: {reranked_results[0].rerank_score:.4f}"
                if reranked_results
                else "No results"
            )

            return reranked_results

        except RETRYABLE_EXCEPTIONS:
            raise
        except cohere.errors.BadRequestError as e:
            logger.error(f"Invalid rerank request: {e}")
            msg = f"Invalid rerank request: {e}"
            raise RerankError(msg) from e
        except Exception as e:
            logger.error(f"Failed to rerank results: {e}")
            msg = f"Reranking failed: {e}"
            raise RerankError(msg) from e

    async def close(self) -> None:
        """Close the Cohere client.

        This is a no-op as the AsyncClientV2 doesn't require explicit cleanup,
        but is provided for interface consistency.
        """
        # AsyncClientV2 doesn't have a close method
        pass

    @property
    def model_name(self) -> str:
        """Return the rerank model name being used.

        Returns:
            Model name string.
        """
        return self._model
