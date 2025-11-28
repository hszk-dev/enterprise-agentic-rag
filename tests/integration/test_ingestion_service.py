"""Integration tests for IngestionService.

These tests require:
- Running MinIO instance
- Running PostgreSQL instance
- Running Qdrant instance

Run with: pytest tests/integration/test_ingestion_service.py -m integration -v
"""

import contextlib
import io
from uuid import uuid4

import pytest

from config import ChunkingSettings, DatabaseSettings, MinIOSettings, QdrantSettings
from src.application.services import IngestionService, LangChainChunkingService
from src.domain.entities import Document
from src.domain.exceptions import DocumentProcessingError
from src.domain.value_objects import ContentType, DocumentStatus
from src.infrastructure.parsers import UnstructuredParser
from src.infrastructure.repositories import PostgresDocumentRepository
from src.infrastructure.storage import MinIOStorage
from src.infrastructure.vectorstores import QdrantVectorStore
from tests.integration.conftest import MockSparseEmbedding

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ingestion_minio_settings() -> MinIOSettings:
    """Create MinIO settings for ingestion tests."""
    return MinIOSettings(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",  # pragma: allowlist secret
        bucket_name=f"test-ingestion-{uuid4().hex[:8]}",
        secure=False,
    )


@pytest.fixture
def ingestion_qdrant_settings() -> QdrantSettings:
    """Create Qdrant settings for ingestion tests."""
    return QdrantSettings(
        host="localhost",
        port=6333,
        grpc_port=6334,
        collection_name=f"test-ingestion-{uuid4().hex[:8]}",
        use_grpc=False,
        api_key=None,
    )


@pytest.fixture
def ingestion_chunking_settings() -> ChunkingSettings:
    """Chunking settings for ingestion tests."""
    return ChunkingSettings(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


@pytest.fixture
async def ingestion_minio_storage(
    ingestion_minio_settings: MinIOSettings,
) -> MinIOStorage:
    """Create and initialize MinIO storage."""
    storage = MinIOStorage(ingestion_minio_settings)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
async def ingestion_qdrant_store(
    ingestion_qdrant_settings: QdrantSettings,
) -> QdrantVectorStore:
    """Create and initialize Qdrant store."""
    store = QdrantVectorStore(ingestion_qdrant_settings, embedding_dim=1536)
    await store.initialize()
    yield store
    # Cleanup collection
    with contextlib.suppress(Exception):
        await store._client.delete_collection(ingestion_qdrant_settings.collection_name)
    await store.close()


@pytest.fixture
async def ingestion_document_repo(
    database_settings: DatabaseSettings,
) -> PostgresDocumentRepository:
    """Create and initialize document repository."""
    repo = PostgresDocumentRepository(database_settings)
    await repo.initialize()
    yield repo
    await repo.close()


@pytest.fixture
def ingestion_parser(
    ingestion_minio_storage: MinIOStorage,
) -> UnstructuredParser:
    """Create parser with blob storage."""
    return UnstructuredParser(blob_storage=ingestion_minio_storage)


@pytest.fixture
def ingestion_chunker(
    ingestion_chunking_settings: ChunkingSettings,
) -> LangChainChunkingService:
    """Create chunking service."""
    return LangChainChunkingService(settings=ingestion_chunking_settings)


@pytest.fixture
async def ingestion_service(
    ingestion_document_repo: PostgresDocumentRepository,
    ingestion_minio_storage: MinIOStorage,
    ingestion_qdrant_store: QdrantVectorStore,
    ingestion_parser: UnstructuredParser,
    ingestion_chunker: LangChainChunkingService,
    mock_dense_embedding,
    mock_sparse_embedding,
) -> IngestionService:
    """Create IngestionService with all dependencies."""
    return IngestionService(
        document_repo=ingestion_document_repo,
        blob_storage=ingestion_minio_storage,
        vector_store=ingestion_qdrant_store,
        dense_embedding=mock_dense_embedding,
        sparse_embedding=mock_sparse_embedding,
        parser=ingestion_parser,
        chunker=ingestion_chunker,
    )


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing."""
    return Document.create(
        filename="test-document.txt",
        content_type=ContentType.TXT,
        size_bytes=2048,
        metadata={"source": "ingestion-test"},
    )


@pytest.fixture
def sample_file_content() -> str:
    """Sample file content for testing."""
    return """
    Enterprise RAG Platform Documentation

    This is a comprehensive guide to building production-ready
    Retrieval-Augmented Generation systems.

    Chapter 1: Introduction

    RAG combines the power of retrieval systems with large language models
    to provide accurate, grounded responses.

    Chapter 2: Architecture

    The system uses a clean architecture with distinct layers:
    - Domain Layer: Core business logic
    - Application Layer: Use cases
    - Infrastructure Layer: External integrations

    Chapter 3: Ingestion Pipeline

    The ingestion pipeline processes documents through:
    1. Upload to object storage
    2. Text extraction
    3. Chunking
    4. Embedding generation
    5. Vector storage
    """


@pytest.fixture
def sample_file(sample_file_content: str) -> io.BytesIO:
    """Create a BytesIO file from sample content."""
    return io.BytesIO(sample_file_content.encode("utf-8"))


# =============================================================================
# Success Path Tests
# =============================================================================


@pytest.mark.integration
class TestIngestionServiceSuccess:
    """Integration tests for successful ingestion scenarios."""

    async def test_ingest_txt_document_success(
        self,
        ingestion_service: IngestionService,
        ingestion_document_repo: PostgresDocumentRepository,
        sample_document: Document,
        sample_file: io.BytesIO,
    ) -> None:
        """Test complete ingestion flow for TXT document."""
        # Save document first (required by service)
        await ingestion_document_repo.save(sample_document)

        try:
            # Ingest document
            result = await ingestion_service.ingest_document(
                sample_document, sample_file
            )

            # Verify result
            assert result.status == DocumentStatus.COMPLETED
            assert result.chunk_count > 0
            assert result.file_path is not None
            assert result.error_message is None

        finally:
            # Cleanup
            await ingestion_service.delete_document(sample_document.id)

    async def test_ingest_document_creates_chunks_in_qdrant(
        self,
        ingestion_service: IngestionService,
        ingestion_document_repo: PostgresDocumentRepository,
        ingestion_qdrant_store: QdrantVectorStore,
        sample_document: Document,
        sample_file: io.BytesIO,
    ) -> None:
        """Test that ingestion creates searchable chunks in Qdrant."""
        await ingestion_document_repo.save(sample_document)

        try:
            # Ingest document
            result = await ingestion_service.ingest_document(
                sample_document, sample_file
            )

            # Verify chunks in Qdrant
            stats = await ingestion_qdrant_store.get_collection_stats()
            assert stats["points_count"] == result.chunk_count

        finally:
            await ingestion_service.delete_document(sample_document.id)

    async def test_ingest_document_stores_file_in_minio(
        self,
        ingestion_service: IngestionService,
        ingestion_document_repo: PostgresDocumentRepository,
        ingestion_minio_storage: MinIOStorage,
        sample_document: Document,
        sample_file: io.BytesIO,
    ) -> None:
        """Test that ingestion stores file in MinIO."""
        await ingestion_document_repo.save(sample_document)

        try:
            # Ingest document
            result = await ingestion_service.ingest_document(
                sample_document, sample_file
            )

            # Verify file exists in MinIO
            assert result.file_path is not None
            assert await ingestion_minio_storage.exists(result.file_path)

        finally:
            await ingestion_service.delete_document(sample_document.id)

    async def test_ingest_markdown_document(
        self,
        ingestion_service: IngestionService,
        ingestion_document_repo: PostgresDocumentRepository,
        sample_md_content: str,
    ) -> None:
        """Test ingestion of Markdown document."""
        document = Document.create(
            filename="readme.md",
            content_type=ContentType.MD,
            size_bytes=len(sample_md_content),
            metadata={"type": "markdown"},
        )
        await ingestion_document_repo.save(document)
        file = io.BytesIO(sample_md_content.encode("utf-8"))

        try:
            result = await ingestion_service.ingest_document(document, file)

            assert result.status == DocumentStatus.COMPLETED
            assert result.chunk_count > 0

        finally:
            await ingestion_service.delete_document(document.id)


# =============================================================================
# Status Transition Tests
# =============================================================================


@pytest.mark.integration
class TestIngestionServiceStatusTransitions:
    """Integration tests for document status transitions."""

    async def test_status_pending_to_completed(
        self,
        ingestion_service: IngestionService,
        ingestion_document_repo: PostgresDocumentRepository,
        sample_document: Document,
        sample_file: io.BytesIO,
    ) -> None:
        """Test status transitions from PENDING to COMPLETED."""
        await ingestion_document_repo.save(sample_document)
        assert sample_document.status == DocumentStatus.PENDING

        try:
            result = await ingestion_service.ingest_document(
                sample_document, sample_file
            )

            assert result.status == DocumentStatus.COMPLETED

            # Verify in database
            db_doc = await ingestion_document_repo.get_by_id(sample_document.id)
            assert db_doc is not None
            assert db_doc.status == DocumentStatus.COMPLETED

        finally:
            await ingestion_service.delete_document(sample_document.id)

    async def test_status_failed_on_empty_document(
        self,
        ingestion_service: IngestionService,
        ingestion_document_repo: PostgresDocumentRepository,
    ) -> None:
        """Test that empty document results in FAILED status."""
        document = Document.create(
            filename="empty.txt",
            content_type=ContentType.TXT,
            size_bytes=0,
            metadata={},
        )
        await ingestion_document_repo.save(document)
        empty_file = io.BytesIO(b"   ")  # Only whitespace

        try:
            with pytest.raises(DocumentProcessingError):
                await ingestion_service.ingest_document(document, empty_file)

            # Verify status is FAILED
            db_doc = await ingestion_document_repo.get_by_id(document.id)
            assert db_doc is not None
            assert db_doc.status == DocumentStatus.FAILED
            assert db_doc.error_message is not None

        finally:
            # Cleanup - delete from repo only (file may not exist)
            await ingestion_document_repo.delete(document.id)


# =============================================================================
# Compensating Transaction Tests (Saga Pattern)
# =============================================================================


@pytest.mark.integration
class TestIngestionServiceCompensation:
    """Integration tests for compensating transactions (Saga Pattern)."""

    async def test_failure_marks_document_failed(
        self,
        ingestion_service: IngestionService,
        ingestion_document_repo: PostgresDocumentRepository,
    ) -> None:
        """Test that failure during processing marks document as FAILED."""
        document = Document.create(
            filename="will-fail.txt",
            content_type=ContentType.TXT,
            size_bytes=0,
            metadata={},
        )
        await ingestion_document_repo.save(document)
        empty_file = io.BytesIO(b"")

        try:
            with pytest.raises(DocumentProcessingError):
                await ingestion_service.ingest_document(document, empty_file)

            # Verify document status
            db_doc = await ingestion_document_repo.get_by_id(document.id)
            assert db_doc is not None
            assert db_doc.status == DocumentStatus.FAILED

        finally:
            await ingestion_document_repo.delete(document.id)

    async def test_failure_cleans_up_qdrant_chunks(
        self,
        ingestion_document_repo: PostgresDocumentRepository,
        ingestion_minio_storage: MinIOStorage,
        ingestion_qdrant_store: QdrantVectorStore,
        ingestion_parser: UnstructuredParser,
        ingestion_chunker: LangChainChunkingService,
    ) -> None:
        """Test that failure during embedding cleans up Qdrant chunks."""

        class FailingEmbedding:
            """Mock embedding that fails after first batch."""

            def __init__(self):
                self.call_count = 0

            @property
            def dimension(self) -> int:
                return 1536

            async def embed_text(self, text: str) -> list[float]:
                self.call_count += 1
                if self.call_count > 2:
                    raise RuntimeError("Simulated embedding failure")
                return [0.1] * 1536

            async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                # Fail after processing some texts
                if len(texts) > 1:
                    raise RuntimeError("Simulated batch embedding failure")
                return [await self.embed_text(t) for t in texts]

        failing_service = IngestionService(
            document_repo=ingestion_document_repo,
            blob_storage=ingestion_minio_storage,
            vector_store=ingestion_qdrant_store,
            dense_embedding=FailingEmbedding(),
            sparse_embedding=MockSparseEmbedding(),
            parser=ingestion_parser,
            chunker=ingestion_chunker,
            batch_size=1,  # Small batch to trigger multiple calls
        )

        # Create document with enough content to generate multiple chunks
        document = Document.create(
            filename="multi-chunk.txt",
            content_type=ContentType.TXT,
            size_bytes=5000,
            metadata={},
        )
        await ingestion_document_repo.save(document)

        large_content = "Test content paragraph. " * 200
        file = io.BytesIO(large_content.encode("utf-8"))

        try:
            with pytest.raises(DocumentProcessingError):
                await failing_service.ingest_document(document, file)

            # Verify Qdrant is cleaned up (no orphaned chunks)
            stats = await ingestion_qdrant_store.get_collection_stats()
            assert stats["points_count"] == 0

        finally:
            await ingestion_document_repo.delete(document.id)


# =============================================================================
# Delete Tests
# =============================================================================


@pytest.mark.integration
class TestIngestionServiceDelete:
    """Integration tests for document deletion."""

    async def test_delete_document_removes_all_data(
        self,
        ingestion_service: IngestionService,
        ingestion_document_repo: PostgresDocumentRepository,
        ingestion_minio_storage: MinIOStorage,
        ingestion_qdrant_store: QdrantVectorStore,
        sample_document: Document,
        sample_file: io.BytesIO,
    ) -> None:
        """Test that delete removes data from all stores."""
        await ingestion_document_repo.save(sample_document)

        # Ingest first
        result = await ingestion_service.ingest_document(sample_document, sample_file)
        file_path = result.file_path

        # Verify data exists
        assert await ingestion_minio_storage.exists(file_path)
        stats = await ingestion_qdrant_store.get_collection_stats()
        assert stats["points_count"] > 0

        # Delete
        deleted = await ingestion_service.delete_document(sample_document.id)

        assert deleted is True

        # Verify all data removed
        assert not await ingestion_minio_storage.exists(file_path)

        db_doc = await ingestion_document_repo.get_by_id(sample_document.id)
        assert db_doc is None

        stats = await ingestion_qdrant_store.get_collection_stats()
        assert stats["points_count"] == 0

    async def test_delete_nonexistent_document_returns_false(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test that deleting non-existent document returns False."""
        result = await ingestion_service.delete_document(uuid4())
        assert result is False


# =============================================================================
# Searchability Tests
# =============================================================================


@pytest.mark.integration
class TestIngestionServiceSearchability:
    """Integration tests verifying ingested documents are searchable."""

    async def test_ingested_document_is_searchable(
        self,
        ingestion_service: IngestionService,
        ingestion_document_repo: PostgresDocumentRepository,
        ingestion_qdrant_store: QdrantVectorStore,
        mock_dense_embedding,
        mock_sparse_embedding,
        sample_document: Document,
        sample_file_content: str,
    ) -> None:
        """Test that ingested document chunks are searchable via Qdrant."""
        await ingestion_document_repo.save(sample_document)
        file = io.BytesIO(sample_file_content.encode("utf-8"))

        try:
            # Ingest document
            await ingestion_service.ingest_document(sample_document, file)

            # Generate query embedding
            query_dense = await mock_dense_embedding.embed_text("RAG architecture")
            query_sparse = await mock_sparse_embedding.embed_text("RAG architecture")

            # Search
            results = await ingestion_qdrant_store.hybrid_search(
                query_text="RAG architecture",
                query_dense_embedding=query_dense,
                query_sparse_embedding=query_sparse,
                top_k=5,
                alpha=0.5,
            )

            # Verify results
            assert len(results) > 0
            assert all(r.chunk.document_id == sample_document.id for r in results)

        finally:
            await ingestion_service.delete_document(sample_document.id)

    async def test_multiple_documents_searchable(
        self,
        ingestion_service: IngestionService,
        ingestion_document_repo: PostgresDocumentRepository,
        ingestion_qdrant_store: QdrantVectorStore,
        mock_dense_embedding,
        mock_sparse_embedding,
    ) -> None:
        """Test that multiple ingested documents are all searchable."""
        docs = []
        contents = [
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand text.",
            "Vector databases store high-dimensional embeddings efficiently.",
        ]

        try:
            # Ingest multiple documents
            for i, content in enumerate(contents):
                doc = Document.create(
                    filename=f"doc-{i}.txt",
                    content_type=ContentType.TXT,
                    size_bytes=len(content),
                    metadata={"index": i},
                )
                await ingestion_document_repo.save(doc)
                file = io.BytesIO(content.encode("utf-8"))
                await ingestion_service.ingest_document(doc, file)
                docs.append(doc)

            # Verify all documents have chunks
            stats = await ingestion_qdrant_store.get_collection_stats()
            assert stats["points_count"] >= len(docs)

            # Search
            query_dense = await mock_dense_embedding.embed_text("AI and ML")
            query_sparse = await mock_sparse_embedding.embed_text("AI and ML")

            results = await ingestion_qdrant_store.hybrid_search(
                query_text="AI and ML",
                query_dense_embedding=query_dense,
                query_sparse_embedding=query_sparse,
                top_k=10,
                alpha=0.5,
            )

            # Verify results from multiple documents
            assert len(results) > 0

        finally:
            for doc in docs:
                await ingestion_service.delete_document(doc.id)
