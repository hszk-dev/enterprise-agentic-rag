"""Search service implementation.

This module provides the search service that orchestrates hybrid search
and reranking for RAG queries.
"""

import logging
import time
from dataclasses import dataclass

from src.domain.entities import Query, SearchResult
from src.domain.exceptions import SearchError
from src.domain.interfaces import (
    EmbeddingService,
    Reranker,
    SparseEmbeddingService,
    VectorStore,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchMetrics:
    """Metrics collected during search execution.

    Attributes:
        total_latency_ms: Total search time in milliseconds.
        embedding_latency_ms: Time spent generating embeddings.
        search_latency_ms: Time spent in vector store search.
        rerank_latency_ms: Time spent reranking results.
        initial_results_count: Number of results before reranking.
        final_results_count: Number of results after reranking.
    """

    total_latency_ms: float
    embedding_latency_ms: float
    search_latency_ms: float
    rerank_latency_ms: float
    initial_results_count: int
    final_results_count: int


class SearchService:
    """Search service for hybrid search with reranking.

    Orchestrates the search pipeline:
    1. Generate dense and sparse embeddings for query
    2. Execute hybrid search (dense + sparse)
    3. Rerank results for improved relevance

    Example:
        >>> service = SearchService(
        ...     vector_store=qdrant_store,
        ...     dense_embedding=openai_embedding,
        ...     sparse_embedding=fastembed_sparse,
        ...     reranker=cohere_reranker,
        ... )
        >>> query = Query.create("What is RAG?")
        >>> results, metrics = await service.search(query)
        >>> results[0].chunk.content
        "RAG (Retrieval Augmented Generation) is..."
    """

    def __init__(
        self,
        vector_store: VectorStore,
        dense_embedding: EmbeddingService,
        sparse_embedding: SparseEmbeddingService,
        reranker: Reranker | None = None,
    ) -> None:
        """Initialize the search service.

        Args:
            vector_store: Vector store for similarity search.
            dense_embedding: Dense embedding service.
            sparse_embedding: Sparse embedding service.
            reranker: Optional reranker for result refinement.
        """
        self._vector_store = vector_store
        self._dense_embedding = dense_embedding
        self._sparse_embedding = sparse_embedding
        self._reranker = reranker

    async def search(
        self,
        query: Query,
        skip_rerank: bool = False,
    ) -> tuple[list[SearchResult], SearchMetrics]:
        """Execute hybrid search with optional reranking.

        Args:
            query: Search query with parameters.
            skip_rerank: Skip reranking even if reranker is configured.

        Returns:
            Tuple of (search results, metrics).

        Raises:
            SearchError: If search fails.
        """
        start_time = time.perf_counter()
        embedding_start = start_time

        try:
            # Step 1: Generate embeddings (dense + sparse in parallel would be ideal,
            # but we do them sequentially for simplicity)
            logger.debug(f"Generating embeddings for query: {query.text[:50]}...")

            dense_embedding = await self._dense_embedding.embed_text(query.text)
            sparse_embedding = await self._sparse_embedding.embed_text(query.text)

            embedding_end = time.perf_counter()
            embedding_latency_ms = (embedding_end - embedding_start) * 1000

            # Step 2: Execute hybrid search
            search_start = time.perf_counter()

            results = await self._vector_store.hybrid_search(
                query_text=query.text,
                query_dense_embedding=dense_embedding,
                query_sparse_embedding=sparse_embedding,
                top_k=query.top_k,
                alpha=query.alpha,
                filters=query.filters,
            )

            search_end = time.perf_counter()
            search_latency_ms = (search_end - search_start) * 1000
            initial_results_count = len(results)

            logger.debug(
                f"Hybrid search returned {initial_results_count} results "
                f"in {search_latency_ms:.1f}ms"
            )

            # Step 3: Rerank results (if reranker is configured and not skipped)
            rerank_start = time.perf_counter()

            if self._reranker and not skip_rerank and results:
                results = await self._reranker.rerank(
                    query=query.text,
                    results=results,
                    top_n=query.rerank_top_n,
                )
                logger.debug(
                    f"Reranked to {len(results)} results, "
                    f"top score: {results[0].rerank_score:.4f}"
                    if results
                    else "No results after rerank"
                )
            else:
                # Assign ranks if not reranked
                for rank, result in enumerate(results[: query.rerank_top_n], start=1):
                    result.rank = rank
                results = results[: query.rerank_top_n]

            rerank_end = time.perf_counter()
            rerank_latency_ms = (rerank_end - rerank_start) * 1000

            # Calculate total latency
            total_latency_ms = (time.perf_counter() - start_time) * 1000

            metrics = SearchMetrics(
                total_latency_ms=total_latency_ms,
                embedding_latency_ms=embedding_latency_ms,
                search_latency_ms=search_latency_ms,
                rerank_latency_ms=rerank_latency_ms,
                initial_results_count=initial_results_count,
                final_results_count=len(results),
            )

            logger.info(
                f"Search completed in {total_latency_ms:.1f}ms "
                f"(embed: {embedding_latency_ms:.1f}ms, "
                f"search: {search_latency_ms:.1f}ms, "
                f"rerank: {rerank_latency_ms:.1f}ms), "
                f"results: {initial_results_count} -> {len(results)}"
            )

            return results, metrics

        except Exception as e:
            logger.error(f"Search failed for query '{query.text[:50]}...': {e}")
            if isinstance(e, SearchError):
                raise
            msg = f"Search failed: {e}"
            raise SearchError(msg) from e

    async def search_dense_only(
        self,
        query: Query,
    ) -> tuple[list[SearchResult], SearchMetrics]:
        """Execute dense-only search (no sparse, no rerank).

        Useful for comparison or when sparse embedding is unavailable.

        Args:
            query: Search query with parameters.

        Returns:
            Tuple of (search results, metrics).

        Raises:
            SearchError: If search fails.
        """
        start_time = time.perf_counter()

        try:
            # Generate dense embedding
            dense_embedding = await self._dense_embedding.embed_text(query.text)
            embedding_latency_ms = (time.perf_counter() - start_time) * 1000

            # Execute dense search
            search_start = time.perf_counter()
            results = await self._vector_store.search(
                query_embedding=dense_embedding,
                top_k=query.top_k,
                filters=query.filters,
            )
            search_latency_ms = (time.perf_counter() - search_start) * 1000

            # Assign ranks
            for rank, result in enumerate(results[: query.rerank_top_n], start=1):
                result.rank = rank
            results = results[: query.rerank_top_n]

            total_latency_ms = (time.perf_counter() - start_time) * 1000

            metrics = SearchMetrics(
                total_latency_ms=total_latency_ms,
                embedding_latency_ms=embedding_latency_ms,
                search_latency_ms=search_latency_ms,
                rerank_latency_ms=0.0,
                initial_results_count=len(results),
                final_results_count=len(results),
            )

            return results, metrics

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            if isinstance(e, SearchError):
                raise
            msg = f"Dense search failed: {e}"
            raise SearchError(msg) from e

    @property
    def has_reranker(self) -> bool:
        """Check if reranker is configured.

        Returns:
            True if reranker is available.
        """
        return self._reranker is not None
