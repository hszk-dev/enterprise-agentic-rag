"""Unit tests for Qdrant vector store with mocked dependencies."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from config import QdrantSettings
from src.domain.entities import Chunk
from src.domain.exceptions import SearchError
from src.domain.value_objects import SparseVector
from src.infrastructure.vectorstores import QdrantVectorStore


@pytest.fixture
def mock_qdrant_settings():
    """Create Qdrant settings for unit tests."""
    return QdrantSettings(
        host="localhost",
        port=6333,
        grpc_port=6334,
        collection_name="test-documents",
        use_grpc=False,
        api_key=None,
    )


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing."""
    return Chunk.create(
        document_id=uuid4(),
        content="This is test content for the chunk.",
        chunk_index=0,
        start_char=0,
        end_char=35,
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_chunk_with_embeddings(sample_chunk: Chunk):
    """Create a sample chunk with embeddings."""
    sample_chunk.dense_embedding = [0.1] * 1536
    sample_chunk.sparse_embedding = SparseVector(
        indices=(1, 5, 10), values=(0.5, 0.3, 0.2)
    )
    return sample_chunk


@pytest.mark.unit
class TestQdrantVectorStoreUnit:
    """Unit tests for QdrantVectorStore class."""

    def test_init_creates_client_rest(
        self, mock_qdrant_settings: QdrantSettings
    ) -> None:
        """Test that __init__ creates REST client when use_grpc is False."""
        with patch(
            "src.infrastructure.vectorstores.qdrant_vectorstore.AsyncQdrantClient"
        ) as mock_client_class:
            QdrantVectorStore(mock_qdrant_settings, embedding_dim=1536)

            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args.kwargs
            assert call_kwargs["host"] == "localhost"
            assert call_kwargs["port"] == 6333
            assert "grpc_port" not in call_kwargs

    def test_init_creates_client_grpc(
        self, mock_qdrant_settings: QdrantSettings
    ) -> None:
        """Test that __init__ creates gRPC client when use_grpc is True."""
        mock_qdrant_settings.use_grpc = True
        with patch(
            "src.infrastructure.vectorstores.qdrant_vectorstore.AsyncQdrantClient"
        ) as mock_client_class:
            QdrantVectorStore(mock_qdrant_settings, embedding_dim=1536)

            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args.kwargs
            assert call_kwargs["host"] == "localhost"
            assert call_kwargs["grpc_port"] == 6334
            assert call_kwargs["prefer_grpc"] is True

    async def test_initialize_creates_collection_if_not_exists(
        self, mock_qdrant_settings: QdrantSettings
    ) -> None:
        """Test initialize creates collection when it doesn't exist."""
        with patch(
            "src.infrastructure.vectorstores.qdrant_vectorstore.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_collections = MagicMock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            mock_client_class.return_value = mock_client

            store = QdrantVectorStore(mock_qdrant_settings, embedding_dim=1536)
            await store.initialize()

            mock_client.get_collections.assert_called_once()
            mock_client.create_collection.assert_called_once()
            mock_client.create_payload_index.assert_called_once()

    async def test_initialize_skips_creation_if_exists(
        self, mock_qdrant_settings: QdrantSettings
    ) -> None:
        """Test initialize doesn't create collection when it exists."""
        with patch(
            "src.infrastructure.vectorstores.qdrant_vectorstore.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_collection = MagicMock()
            mock_collection.name = "test-documents"
            mock_collections = MagicMock()
            mock_collections.collections = [mock_collection]
            mock_client.get_collections.return_value = mock_collections
            mock_client_class.return_value = mock_client

            store = QdrantVectorStore(mock_qdrant_settings, embedding_dim=1536)
            await store.initialize()

            mock_client.get_collections.assert_called_once()
            mock_client.create_collection.assert_not_called()

    async def test_upsert_chunks_success(
        self,
        mock_qdrant_settings: QdrantSettings,
        sample_chunk_with_embeddings: Chunk,
    ) -> None:
        """Test upsert_chunks successfully upserts chunks."""
        with patch(
            "src.infrastructure.vectorstores.qdrant_vectorstore.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            store = QdrantVectorStore(mock_qdrant_settings, embedding_dim=1536)
            await store.upsert_chunks([sample_chunk_with_embeddings])

            mock_client.upsert.assert_called_once()
            call_kwargs = mock_client.upsert.call_args.kwargs
            assert call_kwargs["collection_name"] == "test-documents"
            assert len(call_kwargs["points"]) == 1

    async def test_upsert_chunks_empty_list(
        self, mock_qdrant_settings: QdrantSettings
    ) -> None:
        """Test upsert_chunks does nothing for empty list."""
        with patch(
            "src.infrastructure.vectorstores.qdrant_vectorstore.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            store = QdrantVectorStore(mock_qdrant_settings, embedding_dim=1536)
            await store.upsert_chunks([])

            mock_client.upsert.assert_not_called()

    async def test_upsert_chunks_missing_embedding_raises_error(
        self, mock_qdrant_settings: QdrantSettings, sample_chunk: Chunk
    ) -> None:
        """Test upsert_chunks raises error when embedding is missing."""
        with patch(
            "src.infrastructure.vectorstores.qdrant_vectorstore.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            store = QdrantVectorStore(mock_qdrant_settings, embedding_dim=1536)

            with pytest.raises(SearchError) as exc_info:
                await store.upsert_chunks([sample_chunk])

            assert "missing dense embedding" in str(exc_info.value)

    async def test_search_success(self, mock_qdrant_settings: QdrantSettings) -> None:
        """Test search returns results successfully."""
        with patch(
            "src.infrastructure.vectorstores.qdrant_vectorstore.AsyncQdrantClient"
        ) as mock_client_class:
            # Setup mock search results
            mock_point = MagicMock()
            mock_point.id = str(uuid4())
            mock_point.score = 0.95
            mock_point.payload = {
                "document_id": str(uuid4()),
                "chunk_index": 0,
                "content": "Test content",
                "start_char": 0,
                "end_char": 12,
                "metadata": {},
            }

            mock_result = MagicMock()
            mock_result.points = [mock_point]

            mock_client = AsyncMock()
            mock_client.query_points.return_value = mock_result
            mock_client_class.return_value = mock_client

            store = QdrantVectorStore(mock_qdrant_settings, embedding_dim=1536)
            query_embedding = [0.1] * 1536
            results = await store.search(query_embedding, top_k=5)

            assert len(results) == 1
            assert results[0].score == 0.95
            assert results[0].rank == 1
            assert results[0].chunk.content == "Test content"

    async def test_hybrid_search_success(
        self, mock_qdrant_settings: QdrantSettings
    ) -> None:
        """Test hybrid_search returns results using RRF fusion."""
        with patch(
            "src.infrastructure.vectorstores.qdrant_vectorstore.AsyncQdrantClient"
        ) as mock_client_class:
            # Setup mock query results
            mock_point = MagicMock()
            mock_point.id = str(uuid4())
            mock_point.score = 0.85
            mock_point.payload = {
                "document_id": str(uuid4()),
                "chunk_index": 0,
                "content": "Hybrid search content",
                "start_char": 0,
                "end_char": 20,
                "metadata": {},
            }

            mock_result = MagicMock()
            mock_result.points = [mock_point]

            mock_client = AsyncMock()
            mock_client.query_points.return_value = mock_result
            mock_client_class.return_value = mock_client

            store = QdrantVectorStore(mock_qdrant_settings, embedding_dim=1536)

            query_dense = [0.1] * 1536
            query_sparse = SparseVector(indices=(1, 2, 3), values=(0.5, 0.3, 0.2))

            results = await store.hybrid_search(
                query_text="test query",
                query_dense_embedding=query_dense,
                query_sparse_embedding=query_sparse,
                top_k=5,
                alpha=0.5,
            )

            assert len(results) == 1
            assert results[0].score == 0.85
            mock_client.query_points.assert_called_once()

    async def test_delete_by_document_id_success(
        self, mock_qdrant_settings: QdrantSettings
    ) -> None:
        """Test delete_by_document_id deletes chunks and returns count."""
        with patch(
            "src.infrastructure.vectorstores.qdrant_vectorstore.AsyncQdrantClient"
        ) as mock_client_class:
            mock_count_result = MagicMock()
            mock_count_result.count = 5

            mock_client = AsyncMock()
            mock_client.count.return_value = mock_count_result
            mock_client_class.return_value = mock_client

            store = QdrantVectorStore(mock_qdrant_settings, embedding_dim=1536)
            doc_id = uuid4()
            count = await store.delete_by_document_id(doc_id)

            assert count == 5
            mock_client.count.assert_called_once()
            mock_client.delete.assert_called_once()

    async def test_delete_by_document_id_no_chunks(
        self, mock_qdrant_settings: QdrantSettings
    ) -> None:
        """Test delete_by_document_id returns 0 when no chunks exist."""
        with patch(
            "src.infrastructure.vectorstores.qdrant_vectorstore.AsyncQdrantClient"
        ) as mock_client_class:
            mock_count_result = MagicMock()
            mock_count_result.count = 0

            mock_client = AsyncMock()
            mock_client.count.return_value = mock_count_result
            mock_client_class.return_value = mock_client

            store = QdrantVectorStore(mock_qdrant_settings, embedding_dim=1536)
            count = await store.delete_by_document_id(uuid4())

            assert count == 0
            mock_client.delete.assert_not_called()

    async def test_get_collection_stats_success(
        self, mock_qdrant_settings: QdrantSettings
    ) -> None:
        """Test get_collection_stats returns collection info."""
        with patch(
            "src.infrastructure.vectorstores.qdrant_vectorstore.AsyncQdrantClient"
        ) as mock_client_class:
            mock_info = MagicMock()
            mock_info.indexed_vectors_count = 1000
            mock_info.points_count = 1000
            mock_info.status.value = "green"
            mock_info.optimizer_status = "ok"

            mock_client = AsyncMock()
            mock_client.get_collection.return_value = mock_info
            mock_client_class.return_value = mock_client

            store = QdrantVectorStore(mock_qdrant_settings, embedding_dim=1536)
            stats = await store.get_collection_stats()

            assert stats["indexed_vectors_count"] == 1000
            assert stats["points_count"] == 1000
            assert stats["status"] == "green"

    async def test_close_closes_client(
        self, mock_qdrant_settings: QdrantSettings
    ) -> None:
        """Test close method closes the Qdrant client."""
        with patch(
            "src.infrastructure.vectorstores.qdrant_vectorstore.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            store = QdrantVectorStore(mock_qdrant_settings, embedding_dim=1536)
            await store.close()

            mock_client.close.assert_called_once()
