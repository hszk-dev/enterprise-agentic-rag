"""Qdrant vector store implementation.

This module provides vector storage and hybrid search using Qdrant.
Supports both dense (OpenAI) and sparse (SPLADE) vectors for hybrid retrieval.
"""

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from qdrant_client.models import (
    SparseVector as QdrantSparseVector,
)

from src.domain.entities import Chunk, SearchResult
from src.domain.exceptions import SearchError
from src.domain.value_objects import SparseVector

if TYPE_CHECKING:
    from config.settings import QdrantSettings

logger = logging.getLogger(__name__)

# Vector names in Qdrant collection
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantVectorStore:
    """Qdrant vector store for hybrid search.

    Implements the VectorStore protocol using Qdrant's native hybrid search
    capabilities with dense (OpenAI) and sparse (SPLADE) vectors.

    Example:
        >>> store = QdrantVectorStore(settings, embedding_dim=1536)
        >>> await store.initialize()
        >>> await store.upsert_chunks(chunks)
        >>> results = await store.hybrid_search(...)
    """

    def __init__(
        self,
        settings: "QdrantSettings",
        embedding_dim: int = 1536,
    ) -> None:
        """Initialize the Qdrant vector store.

        Args:
            settings: Qdrant configuration settings.
            embedding_dim: Dimension of dense embedding vectors.
        """
        self._settings = settings
        self._embedding_dim = embedding_dim
        self._collection_name = settings.collection_name

        # Initialize client based on settings
        if settings.use_grpc:
            self._client = AsyncQdrantClient(
                host=settings.host,
                grpc_port=settings.grpc_port,
                prefer_grpc=True,
                api_key=(
                    settings.api_key.get_secret_value() if settings.api_key else None
                ),
            )
        else:
            self._client = AsyncQdrantClient(
                host=settings.host,
                port=settings.port,
                api_key=(
                    settings.api_key.get_secret_value() if settings.api_key else None
                ),
            )

    async def initialize(self) -> None:
        """Initialize the vector store collection.

        Creates the collection if it doesn't exist with proper vector configuration.

        Raises:
            SearchError: If initialization fails.
        """
        try:
            collections = await self._client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self._collection_name not in collection_names:
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: VectorParams(
                            size=self._embedding_dim,
                            distance=Distance.COSINE,
                        ),
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: models.SparseVectorParams(
                            modifier=models.Modifier.IDF,
                        ),
                    },
                )
                logger.info(f"Created collection: {self._collection_name}")

                # Create payload index for document_id filtering
                await self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name="document_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info("Created payload index for document_id")
            else:
                logger.info(f"Collection already exists: {self._collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
            msg = f"Vector store initialization failed: {e}"
            raise SearchError(msg) from e

    async def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks in the vector store.

        Args:
            chunks: List of chunks with embeddings set.

        Raises:
            SearchError: If upsert fails.
        """
        if not chunks:
            return

        try:
            points = []
            for chunk in chunks:
                if chunk.dense_embedding is None:
                    msg = f"Chunk {chunk.id} missing dense embedding"
                    raise SearchError(msg)

                # Build vectors dict
                vectors: dict[str, Any] = {
                    DENSE_VECTOR_NAME: chunk.dense_embedding,
                }

                # Add sparse vector if available
                if chunk.sparse_embedding is not None:
                    vectors[SPARSE_VECTOR_NAME] = QdrantSparseVector(
                        indices=list(chunk.sparse_embedding.indices),
                        values=list(chunk.sparse_embedding.values),
                    )

                # Build payload
                payload = {
                    "document_id": str(chunk.document_id),
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "metadata": chunk.metadata,
                }

                points.append(
                    PointStruct(
                        id=str(chunk.id),
                        vector=vectors,
                        payload=payload,
                    )
                )

            await self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            logger.info(f"Upserted {len(chunks)} chunks to Qdrant")

        except SearchError:
            raise
        except Exception as e:
            logger.error(f"Failed to upsert chunks: {e}")
            msg = f"Vector upsert failed: {e}"
            raise SearchError(msg) from e

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Dense vector similarity search.

        Args:
            query_embedding: Query dense embedding vector.
            top_k: Number of results to return.
            filters: Metadata filters.

        Returns:
            List of search results sorted by score (descending).

        Raises:
            SearchError: If search fails.
        """
        try:
            qdrant_filter = self._build_filter(filters) if filters else None

            results = await self._client.query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                using=DENSE_VECTOR_NAME,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )

            return self._convert_query_results(results.points)

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            msg = f"Search failed: {e}"
            raise SearchError(msg) from e

    async def hybrid_search(
        self,
        query_text: str,
        query_dense_embedding: list[float],
        query_sparse_embedding: SparseVector,
        top_k: int = 10,
        alpha: float = 0.5,  # noqa: ARG002 - Reserved for future custom fusion
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Hybrid search combining dense and sparse vectors.

        Uses Qdrant's query API with prefetch for hybrid retrieval.
        The alpha parameter controls the weight between dense and sparse:
        - alpha=0: Only sparse (keyword) search
        - alpha=1: Only dense (semantic) search
        - alpha=0.5: Equal weight (default)

        Note: Currently using RRF (Reciprocal Rank Fusion) which doesn't use alpha.
        Alpha is reserved for future custom fusion implementations.

        Args:
            query_text: Original query text (for logging/debugging).
            query_dense_embedding: Query dense embedding.
            query_sparse_embedding: Query sparse embedding.
            top_k: Number of results to return.
            alpha: Weight for dense vs sparse (reserved for future use).
            filters: Metadata filters.

        Returns:
            List of search results sorted by combined score.

        Raises:
            SearchError: If search fails.
        """
        try:
            qdrant_filter = self._build_filter(filters) if filters else None

            # Use Reciprocal Rank Fusion (RRF) for hybrid search
            # Note: RRF doesn't use alpha weight - reserved for future custom fusion
            # Prefetch more results from each method for better fusion
            prefetch_limit = top_k * 2

            results = await self._client.query_points(
                collection_name=self._collection_name,
                prefetch=[
                    # Dense search prefetch
                    models.Prefetch(
                        query=query_dense_embedding,
                        using=DENSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=qdrant_filter,
                    ),
                    # Sparse search prefetch
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=list(query_sparse_embedding.indices),
                            values=list(query_sparse_embedding.values),
                        ),
                        using=SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=qdrant_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

            logger.debug(
                f"Hybrid search for '{query_text[:50]}...' "
                f"returned {len(results.points)} results"
            )

            return self._convert_query_results(results.points)

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            msg = f"Hybrid search failed: {e}"
            raise SearchError(msg) from e

    async def delete_by_document_id(self, document_id: UUID) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: Parent document UUID.

        Returns:
            Number of chunks deleted.

        Raises:
            SearchError: If deletion fails.
        """
        try:
            # First count how many will be deleted
            count_result = await self._client.count(
                collection_name=self._collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id)),
                        )
                    ]
                ),
            )
            count = count_result.count

            if count > 0:
                await self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=models.FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="document_id",
                                    match=MatchValue(value=str(document_id)),
                                )
                            ]
                        )
                    ),
                )
                logger.info(f"Deleted {count} chunks for document {document_id}")

            return count

        except Exception as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {e}")
            msg = f"Delete failed: {e}"
            raise SearchError(msg) from e

    async def get_collection_stats(self) -> dict[str, Any]:
        """Get vector store collection statistics.

        Returns:
            Dictionary with stats (vectors_count, indexed_vectors_count, etc.).
        """
        try:
            info = await self._client.get_collection(self._collection_name)
            return {
                "collection_name": self._collection_name,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
                "optimizer_status": info.optimizer_status,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self._client.close()

    def _build_filter(self, filters: dict[str, Any]) -> Filter:
        """Build Qdrant filter from dictionary.

        Args:
            filters: Dictionary of field -> value mappings.

        Returns:
            Qdrant Filter object.
        """
        conditions = []
        for key, value in filters.items():
            # Ensure document_id is string
            filter_value = str(value) if key == "document_id" else value
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=filter_value))
            )
        return Filter(must=conditions) if conditions else Filter()

    def _convert_query_results(
        self, results: list[models.ScoredPoint]
    ) -> list[SearchResult]:
        """Convert Qdrant query results to domain SearchResults.

        Args:
            results: Qdrant ScoredPoint results from query_points.

        Returns:
            List of domain SearchResult objects.
        """
        search_results = []
        for rank, point in enumerate(results, start=1):
            chunk = self._payload_to_chunk(point.id, point.payload)
            search_results.append(
                SearchResult(
                    chunk=chunk,
                    score=point.score,
                    rank=rank,
                )
            )
        return search_results

    def _payload_to_chunk(
        self, point_id: str | int | UUID, payload: dict[str, Any] | None
    ) -> Chunk:
        """Convert Qdrant payload to Chunk entity.

        Args:
            point_id: Qdrant point ID (chunk UUID).
            payload: Qdrant payload dictionary.

        Returns:
            Chunk entity.
        """
        if payload is None:
            payload = {}

        return Chunk(
            id=UUID(str(point_id)),
            document_id=UUID(payload.get("document_id", "")),
            content=payload.get("content", ""),
            chunk_index=payload.get("chunk_index", 0),
            start_char=payload.get("start_char", 0),
            end_char=payload.get("end_char", 0),
            metadata=payload.get("metadata", {}),
            # Embeddings not returned from search
            dense_embedding=None,
            sparse_embedding=None,
        )
