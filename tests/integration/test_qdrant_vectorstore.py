"""Integration tests for Qdrant vector store.

These tests require a running Qdrant instance.
Run with: pytest tests/integration/test_qdrant_vectorstore.py -m integration
"""

import contextlib
from uuid import uuid4

import pytest

from config import QdrantSettings
from src.domain.entities import Chunk
from src.domain.value_objects import SparseVector
from src.infrastructure.vectorstores import QdrantVectorStore


@pytest.fixture
def integration_qdrant_settings():
    """Create Qdrant settings for integration tests with unique collection."""
    return QdrantSettings(
        host="localhost",
        port=6333,
        grpc_port=6334,
        collection_name=f"test-integration-{uuid4().hex[:8]}",
        use_grpc=False,
        api_key=None,
    )


@pytest.fixture
async def qdrant_store(integration_qdrant_settings: QdrantSettings):
    """Create and initialize Qdrant store for testing."""
    store = QdrantVectorStore(integration_qdrant_settings, embedding_dim=1536)
    try:
        await store.initialize()
        yield store
    finally:
        # Cleanup: delete collection
        with contextlib.suppress(Exception):
            await store._client.delete_collection(
                integration_qdrant_settings.collection_name
            )
        await store.close()


@pytest.fixture
def sample_chunks():
    """Create sample chunks with embeddings for testing."""
    doc_id = uuid4()
    chunks = []
    for i in range(3):
        chunk = Chunk.create(
            document_id=doc_id,
            content=f"This is test content for chunk {i}. It contains some text.",
            chunk_index=i,
            start_char=i * 100,
            end_char=(i + 1) * 100,
            metadata={"source": "test", "page": i + 1},
        )
        # Add embeddings
        chunk.dense_embedding = [0.1 * (i + 1)] * 1536
        chunk.sparse_embedding = SparseVector(
            indices=(1, 5, 10 + i),
            values=(0.5, 0.3, 0.2),
        )
        chunks.append(chunk)
    return chunks


@pytest.mark.integration
class TestQdrantVectorStoreIntegration:
    """Integration tests for QdrantVectorStore."""

    async def test_initialize_creates_collection(
        self, qdrant_store: QdrantVectorStore
    ) -> None:
        """Test that initialize creates the collection."""
        stats = await qdrant_store.get_collection_stats()
        assert "indexed_vectors_count" in stats
        assert stats["points_count"] == 0

    async def test_upsert_and_search(
        self,
        qdrant_store: QdrantVectorStore,
        sample_chunks: list[Chunk],
    ) -> None:
        """Test upserting chunks and searching."""
        # Upsert chunks
        await qdrant_store.upsert_chunks(sample_chunks)

        # Verify stats
        stats = await qdrant_store.get_collection_stats()
        assert stats["points_count"] == 3

        # Search with query similar to first chunk
        query_embedding = [0.1] * 1536
        results = await qdrant_store.search(query_embedding, top_k=5)

        assert len(results) == 3
        # First result should be closest to query (chunk 0)
        assert results[0].rank == 1
        assert results[0].chunk.chunk_index == 0

    async def test_hybrid_search(
        self,
        qdrant_store: QdrantVectorStore,
        sample_chunks: list[Chunk],
    ) -> None:
        """Test hybrid search with dense and sparse vectors."""
        # Upsert chunks
        await qdrant_store.upsert_chunks(sample_chunks)

        # Hybrid search
        query_dense = [0.2] * 1536
        query_sparse = SparseVector(indices=(1, 5, 11), values=(0.5, 0.3, 0.2))

        results = await qdrant_store.hybrid_search(
            query_text="test query",
            query_dense_embedding=query_dense,
            query_sparse_embedding=query_sparse,
            top_k=3,
            alpha=0.5,
        )

        assert len(results) == 3
        # Results should be ranked
        for i, result in enumerate(results, start=1):
            assert result.rank == i

    async def test_delete_by_document_id(
        self,
        qdrant_store: QdrantVectorStore,
        sample_chunks: list[Chunk],
    ) -> None:
        """Test deleting chunks by document ID."""
        # Upsert chunks
        await qdrant_store.upsert_chunks(sample_chunks)

        # Verify chunks exist
        stats = await qdrant_store.get_collection_stats()
        assert stats["points_count"] == 3

        # Delete by document ID
        doc_id = sample_chunks[0].document_id
        deleted_count = await qdrant_store.delete_by_document_id(doc_id)

        assert deleted_count == 3

        # Verify deletion
        stats = await qdrant_store.get_collection_stats()
        assert stats["points_count"] == 0

    async def test_search_with_filters(
        self,
        qdrant_store: QdrantVectorStore,
        sample_chunks: list[Chunk],
    ) -> None:
        """Test search with metadata filters."""
        # Upsert chunks
        await qdrant_store.upsert_chunks(sample_chunks)

        # Search with filter for specific document
        query_embedding = [0.1] * 1536
        doc_id = sample_chunks[0].document_id

        results = await qdrant_store.search(
            query_embedding,
            top_k=10,
            filters={"document_id": str(doc_id)},
        )

        assert len(results) == 3
        for result in results:
            assert result.chunk.document_id == doc_id

    async def test_upsert_updates_existing(
        self,
        qdrant_store: QdrantVectorStore,
        sample_chunks: list[Chunk],
    ) -> None:
        """Test that upsert updates existing chunks."""
        # Initial upsert
        await qdrant_store.upsert_chunks(sample_chunks)

        # Modify and re-upsert first chunk
        sample_chunks[0].dense_embedding = [0.9] * 1536

        await qdrant_store.upsert_chunks([sample_chunks[0]])

        # Verify still only 3 points
        stats = await qdrant_store.get_collection_stats()
        assert stats["points_count"] == 3
