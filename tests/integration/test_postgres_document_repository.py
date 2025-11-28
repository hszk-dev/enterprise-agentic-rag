"""Integration tests for PostgreSQL document repository.

These tests require a running PostgreSQL instance.
Run with: pytest tests/integration/test_postgres_document_repository.py -m integration
"""

from uuid import uuid4

import pytest

from config import DatabaseSettings
from src.domain.entities import Document
from src.domain.exceptions import DocumentNotFoundError
from src.domain.value_objects import ContentType, DocumentStatus
from src.infrastructure.repositories import PostgresDocumentRepository


@pytest.fixture
async def document_repository(database_settings: DatabaseSettings):
    """Create and initialize document repository for testing."""
    repo = PostgresDocumentRepository(database_settings)
    try:
        await repo.initialize()
        yield repo
    finally:
        await repo.close()


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document.create(
        filename="integration-test.pdf",
        content_type=ContentType.PDF,
        size_bytes=2048,
        metadata={"author": "Integration Test", "department": "Engineering"},
    )


@pytest.mark.integration
class TestPostgresDocumentRepositoryIntegration:
    """Integration tests for PostgresDocumentRepository."""

    async def test_save_and_get_by_id(
        self,
        document_repository: PostgresDocumentRepository,
        sample_document: Document,
    ) -> None:
        """Test saving and retrieving a document."""
        # Save document
        saved = await document_repository.save(sample_document)

        assert saved.id == sample_document.id
        assert saved.filename == sample_document.filename

        # Retrieve by ID
        retrieved = await document_repository.get_by_id(sample_document.id)

        assert retrieved is not None
        assert retrieved.id == sample_document.id
        assert retrieved.filename == "integration-test.pdf"
        assert retrieved.content_type == ContentType.PDF
        assert retrieved.size_bytes == 2048
        assert retrieved.metadata["author"] == "Integration Test"

        # Cleanup
        await document_repository.delete(sample_document.id)

    async def test_get_by_id_not_found(
        self,
        document_repository: PostgresDocumentRepository,
        sample_document: Document,
    ) -> None:
        """Test get_by_id returns None for non-existent document."""
        result = await document_repository.get_by_id(uuid4())
        assert result is None

    async def test_update_document(
        self,
        document_repository: PostgresDocumentRepository,
        sample_document: Document,
    ) -> None:
        """Test updating a document."""
        # Save document
        await document_repository.save(sample_document)

        # Update document
        sample_document.mark_processing()
        sample_document.set_file_path("documents/uuid/test.pdf")

        updated = await document_repository.update(sample_document)

        assert updated.status == DocumentStatus.PROCESSING
        assert updated.file_path == "documents/uuid/test.pdf"

        # Mark completed
        sample_document.mark_completed(chunk_count=10)
        updated = await document_repository.update(sample_document)

        assert updated.status == DocumentStatus.COMPLETED
        assert updated.chunk_count == 10

        # Cleanup
        await document_repository.delete(sample_document.id)

    async def test_update_not_found_raises_error(
        self,
        document_repository: PostgresDocumentRepository,
        sample_document: Document,
    ) -> None:
        """Test update raises DocumentNotFoundError for non-existent document."""
        with pytest.raises(DocumentNotFoundError):
            await document_repository.update(sample_document)

    async def test_delete_document(
        self,
        document_repository: PostgresDocumentRepository,
        sample_document: Document,
    ) -> None:
        """Test deleting a document."""
        # Save document
        await document_repository.save(sample_document)

        # Delete document
        result = await document_repository.delete(sample_document.id)
        assert result is True

        # Verify deletion
        retrieved = await document_repository.get_by_id(sample_document.id)
        assert retrieved is None

    async def test_delete_not_found(
        self,
        document_repository: PostgresDocumentRepository,
    ) -> None:
        """Test delete returns False for non-existent document."""
        result = await document_repository.delete(uuid4())
        assert result is False

    async def test_list_documents(
        self,
        document_repository: PostgresDocumentRepository,
    ) -> None:
        """Test listing documents."""
        # Create multiple documents
        docs = []
        for i in range(3):
            doc = Document.create(
                filename=f"list-test-{i}.pdf",
                content_type=ContentType.PDF,
                size_bytes=1000 + i * 100,
                metadata={"index": i},
            )
            await document_repository.save(doc)
            docs.append(doc)

        try:
            # List all documents
            results = await document_repository.list(limit=100)

            # Verify all test documents are present
            test_filenames = {d.filename for d in docs}
            result_filenames = {r.filename for r in results}
            assert test_filenames.issubset(result_filenames)

        finally:
            # Cleanup
            for doc in docs:
                await document_repository.delete(doc.id)

    async def test_list_with_status_filter(
        self,
        document_repository: PostgresDocumentRepository,
    ) -> None:
        """Test listing documents with status filter."""
        # Create documents with different statuses
        pending_doc = Document.create(
            filename="pending.pdf",
            content_type=ContentType.PDF,
            size_bytes=1000,
        )
        await document_repository.save(pending_doc)

        completed_doc = Document.create(
            filename="completed.pdf",
            content_type=ContentType.PDF,
            size_bytes=2000,
        )
        completed_doc.mark_completed(5)
        await document_repository.save(completed_doc)

        try:
            # List only pending
            pending_results = await document_repository.list(
                status=DocumentStatus.PENDING
            )
            pending_filenames = {r.filename for r in pending_results}
            assert "pending.pdf" in pending_filenames

            # List only completed
            completed_results = await document_repository.list(
                status=DocumentStatus.COMPLETED
            )
            completed_filenames = {r.filename for r in completed_results}
            assert "completed.pdf" in completed_filenames

        finally:
            # Cleanup
            await document_repository.delete(pending_doc.id)
            await document_repository.delete(completed_doc.id)

    async def test_count_documents(
        self,
        document_repository: PostgresDocumentRepository,
    ) -> None:
        """Test counting documents."""
        # Get initial count
        initial_count = await document_repository.count()

        # Add documents
        docs = []
        for i in range(2):
            doc = Document.create(
                filename=f"count-test-{i}.pdf",
                content_type=ContentType.PDF,
                size_bytes=1000,
            )
            await document_repository.save(doc)
            docs.append(doc)

        try:
            # Check count increased
            new_count = await document_repository.count()
            assert new_count >= initial_count + 2

        finally:
            # Cleanup
            for doc in docs:
                await document_repository.delete(doc.id)

    async def test_list_with_pagination(
        self,
        document_repository: PostgresDocumentRepository,
    ) -> None:
        """Test listing documents with pagination."""
        # Create multiple documents
        docs = []
        for i in range(5):
            doc = Document.create(
                filename=f"pagination-test-{i}.pdf",
                content_type=ContentType.PDF,
                size_bytes=1000,
            )
            await document_repository.save(doc)
            docs.append(doc)

        try:
            # Get first page
            page1 = await document_repository.list(limit=2, offset=0)
            assert len(page1) <= 2

            # Get second page
            page2 = await document_repository.list(limit=2, offset=2)
            assert len(page2) <= 2

            # Pages should be different
            if page1 and page2:
                page1_ids = {d.id for d in page1}
                page2_ids = {d.id for d in page2}
                assert page1_ids.isdisjoint(page2_ids)

        finally:
            # Cleanup
            for doc in docs:
                await document_repository.delete(doc.id)
