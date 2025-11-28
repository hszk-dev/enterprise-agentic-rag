"""Unit tests for IngestionService."""

import io
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.application.services.ingestion_service import IngestionService
from src.domain.entities import Chunk, Document
from src.domain.exceptions import DocumentNotFoundError, DocumentProcessingError
from src.domain.value_objects import ContentType, DocumentStatus


@pytest.fixture
def mock_document_repo():
    """Create mock DocumentRepository."""
    repo = AsyncMock()
    # save() returns the document that was passed to it
    repo.save = AsyncMock(side_effect=lambda doc: doc)
    repo.get_by_id = AsyncMock()
    repo.update = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def mock_blob_storage():
    """Create mock BlobStorage."""
    storage = AsyncMock()
    storage.upload = AsyncMock(return_value="uuid/filename.pdf")
    storage.download = AsyncMock(return_value=io.BytesIO(b"content"))
    storage.delete = AsyncMock(return_value=True)
    storage.exists = AsyncMock(return_value=True)
    return storage


@pytest.fixture
def mock_vector_store():
    """Create mock VectorStore."""
    store = AsyncMock()
    store.upsert_chunks = AsyncMock()
    store.delete_by_document_id = AsyncMock(return_value=5)
    store.hybrid_search = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_dense_embedding():
    """Create mock EmbeddingService."""
    service = AsyncMock()
    # Return embeddings matching the number of input texts
    service.embed_texts = AsyncMock(
        side_effect=lambda texts: [[0.1] * 1536 for _ in texts]
    )
    service.embed_text = AsyncMock(return_value=[0.1] * 1536)
    service.dimension = 1536
    return service


@pytest.fixture
def mock_sparse_embedding():
    """Create mock SparseEmbeddingService."""
    service = AsyncMock()
    # Return sparse embeddings matching the number of input texts
    service.embed_texts = AsyncMock(
        side_effect=lambda texts: [{1: 0.5, 2: 0.3} for _ in texts]
    )
    service.embed_text = AsyncMock(return_value={1: 0.5, 2: 0.3})
    return service


@pytest.fixture
def mock_parser():
    """Create mock DocumentParser."""
    parser = AsyncMock()
    parser.parse = AsyncMock(
        return_value="Parsed document content.\n\nWith multiple paragraphs."
    )
    parser.supports = MagicMock(return_value=True)
    return parser


@pytest.fixture
def mock_chunker():
    """Create mock ChunkingService."""
    chunker = MagicMock()

    def create_chunks(text, document_id, metadata=None):
        return [
            Chunk.create(
                document_id=document_id,
                content="Chunk 1 content",
                chunk_index=0,
                start_char=0,
                end_char=15,
                metadata=metadata,
            ),
            Chunk.create(
                document_id=document_id,
                content="Chunk 2 content",
                chunk_index=1,
                start_char=17,
                end_char=32,
                metadata=metadata,
            ),
        ]

    chunker.chunk = MagicMock(side_effect=create_chunks)
    chunker.chunk_size = 1000
    chunker.chunk_overlap = 200
    return chunker


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document.create(
        filename="test.pdf",
        content_type=ContentType.PDF,
        size_bytes=1024,
        metadata={"author": "Test Author"},
    )


@pytest.fixture
def ingestion_service(
    mock_document_repo,
    mock_blob_storage,
    mock_vector_store,
    mock_dense_embedding,
    mock_sparse_embedding,
    mock_parser,
    mock_chunker,
):
    """Create IngestionService with mocked dependencies."""
    return IngestionService(
        document_repo=mock_document_repo,
        blob_storage=mock_blob_storage,
        vector_store=mock_vector_store,
        dense_embedding=mock_dense_embedding,
        sparse_embedding=mock_sparse_embedding,
        parser=mock_parser,
        chunker=mock_chunker,
        batch_size=10,
    )


class TestIngestionServiceInit:
    """Tests for IngestionService initialization."""

    def test_init_with_all_dependencies(self, ingestion_service):
        """Test initialization with all dependencies."""
        assert ingestion_service._document_repo is not None
        assert ingestion_service._blob_storage is not None
        assert ingestion_service._vector_store is not None
        assert ingestion_service._dense_embedding is not None
        assert ingestion_service._sparse_embedding is not None
        assert ingestion_service._parser is not None
        assert ingestion_service._chunker is not None
        assert ingestion_service._batch_size == 10

    def test_init_without_sparse_embedding(
        self,
        mock_document_repo,
        mock_blob_storage,
        mock_vector_store,
        mock_dense_embedding,
        mock_parser,
        mock_chunker,
    ):
        """Test initialization without sparse embedding service."""
        service = IngestionService(
            document_repo=mock_document_repo,
            blob_storage=mock_blob_storage,
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=None,
            parser=mock_parser,
            chunker=mock_chunker,
        )

        assert service._sparse_embedding is None


class TestIngestionServiceIngest:
    """Tests for IngestionService.ingest_document method."""

    @pytest.mark.asyncio
    async def test_ingest_document_success(
        self,
        ingestion_service,
        sample_document,
        mock_document_repo,
        mock_blob_storage,
        mock_vector_store,
        mock_dense_embedding,
        mock_parser,
        mock_chunker,
    ):
        """Test successful document ingestion."""
        file = io.BytesIO(b"PDF content")
        mock_document_repo.update.return_value = sample_document

        result = await ingestion_service.ingest_document(sample_document, file)

        # Verify file was uploaded
        mock_blob_storage.upload.assert_called_once()

        # Verify document was parsed
        mock_parser.parse.assert_called_once()

        # Verify text was chunked
        mock_chunker.chunk.assert_called_once()

        # Verify embeddings were generated
        mock_dense_embedding.embed_texts.assert_called_once()

        # Verify chunks were stored
        mock_vector_store.upsert_chunks.assert_called_once()

        # Verify document was updated (at least twice: PROCESSING and COMPLETED)
        assert mock_document_repo.update.call_count >= 2

        # Verify final status
        assert result.status == DocumentStatus.COMPLETED
        assert result.chunk_count == 2

    @pytest.mark.asyncio
    async def test_ingest_document_sets_file_path(
        self,
        ingestion_service,
        sample_document,
        mock_blob_storage,
        mock_document_repo,
    ):
        """Test that file path is set after upload."""
        file = io.BytesIO(b"content")
        mock_blob_storage.upload.return_value = "documents/uuid/test.pdf"
        mock_document_repo.update.return_value = sample_document

        await ingestion_service.ingest_document(sample_document, file)

        assert sample_document.file_path == "documents/uuid/test.pdf"

    @pytest.mark.asyncio
    async def test_ingest_document_marks_processing(
        self,
        ingestion_service,
        sample_document,
        mock_document_repo,
    ):
        """Test that document is marked as PROCESSING."""
        file = io.BytesIO(b"content")
        mock_document_repo.update.return_value = sample_document

        # Track status changes
        statuses = []

        async def track_update(doc):
            statuses.append(doc.status)
            return doc

        mock_document_repo.update.side_effect = track_update

        await ingestion_service.ingest_document(sample_document, file)

        # First update should be PROCESSING
        assert DocumentStatus.PROCESSING in statuses

    @pytest.mark.asyncio
    async def test_ingest_document_empty_text_fails(
        self,
        ingestion_service,
        sample_document,
        mock_parser,
        mock_document_repo,
    ):
        """Test that empty parsed text causes failure."""
        file = io.BytesIO(b"content")
        mock_parser.parse.return_value = "   "  # Empty after strip
        mock_document_repo.update.return_value = sample_document

        with pytest.raises(DocumentProcessingError) as exc_info:
            await ingestion_service.ingest_document(sample_document, file)

        assert "no extractable text" in exc_info.value.reason.lower()

    @pytest.mark.asyncio
    async def test_ingest_document_no_chunks_fails(
        self,
        ingestion_service,
        sample_document,
        mock_chunker,
        mock_document_repo,
    ):
        """Test that no chunks generated causes failure."""
        file = io.BytesIO(b"content")
        # Override side_effect to return empty list
        mock_chunker.chunk.side_effect = None
        mock_chunker.chunk.return_value = []
        mock_document_repo.update.return_value = sample_document

        with pytest.raises(DocumentProcessingError) as exc_info:
            await ingestion_service.ingest_document(sample_document, file)

        assert "no chunks" in exc_info.value.reason.lower()

    @pytest.mark.asyncio
    async def test_ingest_document_upload_failure_triggers_cleanup(
        self,
        ingestion_service,
        sample_document,
        mock_blob_storage,
        mock_document_repo,
        mock_vector_store,
    ):
        """Test that upload failure triggers cleanup."""
        file = io.BytesIO(b"content")
        mock_blob_storage.upload.side_effect = Exception("Upload failed")
        mock_document_repo.update.return_value = sample_document

        with pytest.raises(DocumentProcessingError):
            await ingestion_service.ingest_document(sample_document, file)

        # Document should be marked as FAILED
        assert sample_document.status == DocumentStatus.FAILED

    @pytest.mark.asyncio
    async def test_ingest_document_parse_failure_triggers_cleanup(
        self,
        ingestion_service,
        sample_document,
        mock_parser,
        mock_document_repo,
        mock_vector_store,
    ):
        """Test that parse failure triggers cleanup."""
        file = io.BytesIO(b"content")
        mock_parser.parse.side_effect = Exception("Parse failed")
        mock_document_repo.update.return_value = sample_document

        with pytest.raises(DocumentProcessingError):
            await ingestion_service.ingest_document(sample_document, file)

        # Document should be marked as FAILED
        assert sample_document.status == DocumentStatus.FAILED
        # Cleanup should be attempted
        mock_vector_store.delete_by_document_id.assert_called_once_with(
            sample_document.id
        )

    @pytest.mark.asyncio
    async def test_ingest_document_embedding_failure_triggers_cleanup(
        self,
        ingestion_service,
        sample_document,
        mock_dense_embedding,
        mock_document_repo,
        mock_vector_store,
    ):
        """Test that embedding failure triggers cleanup."""
        file = io.BytesIO(b"content")
        mock_dense_embedding.embed_texts.side_effect = Exception("Embedding failed")
        mock_document_repo.update.return_value = sample_document

        with pytest.raises(DocumentProcessingError):
            await ingestion_service.ingest_document(sample_document, file)

        # Document should be marked as FAILED
        assert sample_document.status == DocumentStatus.FAILED

    @pytest.mark.asyncio
    async def test_ingest_document_without_sparse_embedding(
        self,
        mock_document_repo,
        mock_blob_storage,
        mock_vector_store,
        mock_dense_embedding,
        mock_parser,
        mock_chunker,
        sample_document,
    ):
        """Test ingestion without sparse embedding service."""
        service = IngestionService(
            document_repo=mock_document_repo,
            blob_storage=mock_blob_storage,
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=None,  # No sparse embedding
            parser=mock_parser,
            chunker=mock_chunker,
        )

        file = io.BytesIO(b"content")
        mock_document_repo.update.return_value = sample_document

        result = await service.ingest_document(sample_document, file)

        # Should still succeed
        assert result.status == DocumentStatus.COMPLETED


class TestIngestionServiceDelete:
    """Tests for IngestionService.delete_document method."""

    @pytest.mark.asyncio
    async def test_delete_document_success(
        self,
        ingestion_service,
        mock_document_repo,
        mock_blob_storage,
        mock_vector_store,
    ):
        """Test successful document deletion."""
        document_id = uuid4()
        document = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
        )
        document.set_file_path("uuid/test.pdf")
        mock_document_repo.get_by_id.return_value = document

        result = await ingestion_service.delete_document(document_id)

        assert result is True
        mock_vector_store.delete_by_document_id.assert_called_once_with(document_id)
        mock_blob_storage.delete.assert_called_once_with("uuid/test.pdf")
        mock_document_repo.delete.assert_called_once_with(document_id)

    @pytest.mark.asyncio
    async def test_delete_document_not_found(
        self,
        ingestion_service,
        mock_document_repo,
    ):
        """Test deleting non-existent document returns False."""
        document_id = uuid4()
        mock_document_repo.get_by_id.return_value = None

        result = await ingestion_service.delete_document(document_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_document_without_file_path(
        self,
        ingestion_service,
        mock_document_repo,
        mock_blob_storage,
    ):
        """Test deleting document without file path doesn't call blob storage."""
        document_id = uuid4()
        document = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
        )
        # No file_path set
        mock_document_repo.get_by_id.return_value = document

        await ingestion_service.delete_document(document_id)

        # Blob storage delete should not be called
        mock_blob_storage.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_document_vector_store_failure_continues(
        self,
        ingestion_service,
        mock_document_repo,
        mock_vector_store,
    ):
        """Test that vector store failure doesn't stop deletion."""
        document_id = uuid4()
        document = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
        )
        mock_document_repo.get_by_id.return_value = document
        mock_vector_store.delete_by_document_id.side_effect = Exception(
            "Vector store error"
        )

        result = await ingestion_service.delete_document(document_id)

        # Should still return True (deletion from repo succeeded)
        assert result is True
        mock_document_repo.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_blob_storage_failure_continues(
        self,
        ingestion_service,
        mock_document_repo,
        mock_blob_storage,
    ):
        """Test that blob storage failure doesn't stop deletion."""
        document_id = uuid4()
        document = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
        )
        document.set_file_path("uuid/test.pdf")
        mock_document_repo.get_by_id.return_value = document
        mock_blob_storage.delete.side_effect = Exception("Blob storage error")

        result = await ingestion_service.delete_document(document_id)

        # Should still return True
        assert result is True
        mock_document_repo.delete.assert_called_once()


class TestIngestionServiceRetry:
    """Tests for IngestionService.retry_failed_document method."""

    @pytest.mark.asyncio
    async def test_retry_failed_document_with_file(
        self,
        ingestion_service,
        mock_document_repo,
        mock_vector_store,
    ):
        """Test retrying failed document with provided file."""
        document_id = uuid4()
        document = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
        )
        document.mark_failed("Previous error")
        mock_document_repo.get_by_id.return_value = document
        mock_document_repo.update.return_value = document

        file = io.BytesIO(b"content")
        result = await ingestion_service.retry_failed_document(document_id, file)

        # Should have cleaned up existing chunks
        mock_vector_store.delete_by_document_id.assert_called()
        # Should have completed
        assert result.status == DocumentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_retry_failed_document_without_file_downloads(
        self,
        ingestion_service,
        mock_document_repo,
        mock_blob_storage,
    ):
        """Test retrying failed document downloads from blob storage."""
        document_id = uuid4()
        document = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
        )
        document.set_file_path("uuid/test.pdf")
        document.mark_failed("Previous error")
        mock_document_repo.get_by_id.return_value = document
        mock_document_repo.update.return_value = document

        result = await ingestion_service.retry_failed_document(document_id, file=None)

        # Should have downloaded file
        mock_blob_storage.download.assert_called_once_with("uuid/test.pdf")
        assert result.status == DocumentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_retry_failed_document_not_found(
        self,
        ingestion_service,
        mock_document_repo,
    ):
        """Test retrying non-existent document raises error."""
        document_id = uuid4()
        mock_document_repo.get_by_id.return_value = None

        with pytest.raises(DocumentNotFoundError):
            await ingestion_service.retry_failed_document(document_id)

    @pytest.mark.asyncio
    async def test_retry_failed_document_no_file_path_no_file(
        self,
        ingestion_service,
        mock_document_repo,
    ):
        """Test retrying without file and without file_path raises error."""
        document_id = uuid4()
        document = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
        )
        # No file_path set
        document.mark_failed("Previous error")
        mock_document_repo.get_by_id.return_value = document

        with pytest.raises(DocumentProcessingError) as exc_info:
            await ingestion_service.retry_failed_document(document_id, file=None)

        assert "no file available" in exc_info.value.reason.lower()


class TestIngestionServiceHandleFailure:
    """Tests for IngestionService._handle_failure method."""

    @pytest.mark.asyncio
    async def test_handle_failure_updates_document_status(
        self,
        ingestion_service,
        mock_document_repo,
    ):
        """Test that _handle_failure updates document status."""
        document = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
        )

        await ingestion_service._handle_failure(document, "Test error")

        assert document.status == DocumentStatus.FAILED
        assert document.error_message == "Test error"
        mock_document_repo.update.assert_called()

    @pytest.mark.asyncio
    async def test_handle_failure_cleans_up_chunks(
        self,
        ingestion_service,
        mock_vector_store,
        mock_document_repo,
    ):
        """Test that _handle_failure attempts to clean up chunks."""
        document = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
        )

        await ingestion_service._handle_failure(document, "Test error")

        mock_vector_store.delete_by_document_id.assert_called_once_with(document.id)

    @pytest.mark.asyncio
    async def test_handle_failure_continues_on_repo_error(
        self,
        ingestion_service,
        mock_document_repo,
        mock_vector_store,
    ):
        """Test that _handle_failure continues if repo update fails."""
        document = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
        )
        mock_document_repo.update.side_effect = Exception("Repo error")

        # Should not raise
        await ingestion_service._handle_failure(document, "Test error")

        # Should still try to clean up chunks
        mock_vector_store.delete_by_document_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_failure_continues_on_vector_store_error(
        self,
        ingestion_service,
        mock_document_repo,
        mock_vector_store,
    ):
        """Test that _handle_failure continues if vector store cleanup fails."""
        document = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
        )
        mock_vector_store.delete_by_document_id.side_effect = Exception(
            "Vector store error"
        )

        # Should not raise
        await ingestion_service._handle_failure(document, "Test error")

        # Document should still be updated
        mock_document_repo.update.assert_called()


class TestIngestionServiceBatching:
    """Tests for IngestionService batching behavior."""

    @pytest.mark.asyncio
    async def test_embed_and_store_batches_chunks(
        self,
        mock_document_repo,
        mock_blob_storage,
        mock_vector_store,
        mock_dense_embedding,
        mock_sparse_embedding,
        mock_parser,
    ):
        """Test that embedding is done in batches."""
        # Create chunker that returns many chunks
        mock_chunker = MagicMock()

        def create_many_chunks(text, document_id, metadata=None):
            return [
                Chunk.create(
                    document_id=document_id,
                    content=f"Chunk {i}",
                    chunk_index=i,
                    start_char=i * 10,
                    end_char=i * 10 + 7,
                )
                for i in range(25)  # 25 chunks
            ]

        mock_chunker.chunk = MagicMock(side_effect=create_many_chunks)

        service = IngestionService(
            document_repo=mock_document_repo,
            blob_storage=mock_blob_storage,
            vector_store=mock_vector_store,
            dense_embedding=mock_dense_embedding,
            sparse_embedding=mock_sparse_embedding,
            parser=mock_parser,
            chunker=mock_chunker,
            batch_size=10,  # Batch size of 10
        )

        document = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
        )
        mock_document_repo.update.return_value = document

        # Mock embed_texts to return correct number of embeddings
        def mock_embed(texts):
            return [[0.1] * 1536 for _ in texts]

        mock_dense_embedding.embed_texts.side_effect = mock_embed
        mock_sparse_embedding.embed_texts.side_effect = lambda texts: [
            {1: 0.5} for _ in texts
        ]

        file = io.BytesIO(b"content")
        await service.ingest_document(document, file)

        # Should have been called 3 times (25 chunks / 10 batch_size = 3 batches)
        assert mock_dense_embedding.embed_texts.call_count == 3
