"""Unit tests for PostgreSQL document repository with mocked dependencies."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from config import DatabaseSettings
from src.domain.entities import Document
from src.domain.exceptions import DocumentNotFoundError
from src.domain.value_objects import ContentType, DocumentStatus
from src.infrastructure.repositories import PostgresDocumentRepository


@pytest.fixture
def mock_database_settings():
    """Create database settings for unit tests."""
    return DatabaseSettings(
        host="localhost",
        port=5432,
        user="testuser",
        password="testpass",  # pragma: allowlist secret
        database="testdb",
        pool_size=5,
        max_overflow=10,
    )


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document.create(
        filename="test.pdf",
        content_type=ContentType.PDF,
        size_bytes=1024,
        metadata={"author": "Test Author"},
    )


@pytest.mark.unit
class TestPostgresDocumentRepositoryUnit:
    """Unit tests for PostgresDocumentRepository class."""

    def test_async_url_property(self, mock_database_settings: DatabaseSettings) -> None:
        """Test async_url property returns correct connection string."""
        url = mock_database_settings.async_url
        assert url.startswith("postgresql+asyncpg://")
        assert "testuser" in url
        assert "localhost:5432" in url
        assert "testdb" in url

    def test_sync_url_property(self, mock_database_settings: DatabaseSettings) -> None:
        """Test sync_url property returns correct connection string."""
        url = mock_database_settings.sync_url
        assert url.startswith("postgresql://")
        assert "testuser" in url

    async def test_save_success(
        self,
        mock_database_settings: DatabaseSettings,
        sample_document: Document,
    ) -> None:
        """Test save returns saved document."""
        with (
            patch(
                "src.infrastructure.repositories.postgres_document_repository.create_async_engine"
            ),
            patch(
                "src.infrastructure.repositories.postgres_document_repository.async_sessionmaker"
            ) as mock_session_maker,
        ):
            # Setup mock session
            mock_session = AsyncMock()
            mock_session.add = MagicMock()
            mock_session.commit = AsyncMock()
            mock_session.refresh = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_maker.return_value = MagicMock(return_value=mock_session)

            repo = PostgresDocumentRepository(mock_database_settings)

            # Mock the refresh to set the returned model
            async def mock_refresh(model):
                # Model keeps its values
                pass

            mock_session.refresh = mock_refresh

            result = await repo.save(sample_document)

            mock_session.add.assert_called_once()
            assert result.filename == sample_document.filename

    async def test_get_by_id_found(
        self, mock_database_settings: DatabaseSettings
    ) -> None:
        """Test get_by_id returns document when found."""
        with (
            patch(
                "src.infrastructure.repositories.postgres_document_repository.create_async_engine"
            ),
            patch(
                "src.infrastructure.repositories.postgres_document_repository.async_sessionmaker"
            ) as mock_session_maker,
        ):
            doc_id = uuid4()

            # Create expected document
            expected_doc = Document(
                id=doc_id,
                filename="found.pdf",
                content_type=ContentType.PDF,
                size_bytes=2048,
                status=DocumentStatus.COMPLETED,
                metadata={},
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                error_message=None,
                chunk_count=5,
                file_path="documents/uuid/found.pdf",
            )

            # Create mock model that returns the expected document
            mock_model = MagicMock()
            mock_model.to_entity.return_value = expected_doc

            # Setup mock session
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_model

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_maker.return_value = MagicMock(return_value=mock_session)

            repo = PostgresDocumentRepository(mock_database_settings)
            result = await repo.get_by_id(doc_id)

            assert result is not None
            assert result.filename == "found.pdf"
            assert result.chunk_count == 5

    async def test_get_by_id_not_found(
        self, mock_database_settings: DatabaseSettings
    ) -> None:
        """Test get_by_id returns None when not found."""
        with (
            patch(
                "src.infrastructure.repositories.postgres_document_repository.create_async_engine"
            ),
            patch(
                "src.infrastructure.repositories.postgres_document_repository.async_sessionmaker"
            ) as mock_session_maker,
        ):
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_maker.return_value = MagicMock(return_value=mock_session)

            repo = PostgresDocumentRepository(mock_database_settings)
            result = await repo.get_by_id(uuid4())

            assert result is None

    async def test_list_returns_documents(
        self, mock_database_settings: DatabaseSettings
    ) -> None:
        """Test list returns list of documents."""
        with (
            patch(
                "src.infrastructure.repositories.postgres_document_repository.create_async_engine"
            ),
            patch(
                "src.infrastructure.repositories.postgres_document_repository.async_sessionmaker"
            ) as mock_session_maker,
        ):
            # Create expected documents
            doc1 = Document(
                id=uuid4(),
                filename="doc1.pdf",
                content_type=ContentType.PDF,
                size_bytes=1000,
                status=DocumentStatus.COMPLETED,
                metadata={},
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                error_message=None,
                chunk_count=3,
                file_path=None,
            )

            doc2 = Document(
                id=uuid4(),
                filename="doc2.txt",
                content_type=ContentType.TXT,
                size_bytes=500,
                status=DocumentStatus.PENDING,
                metadata={},
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                error_message=None,
                chunk_count=0,
                file_path=None,
            )

            # Create mock models that return the expected documents
            mock_model1 = MagicMock()
            mock_model1.to_entity.return_value = doc1

            mock_model2 = MagicMock()
            mock_model2.to_entity.return_value = doc2

            mock_scalars = MagicMock()
            mock_scalars.all.return_value = [mock_model1, mock_model2]

            mock_result = MagicMock()
            mock_result.scalars.return_value = mock_scalars

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_maker.return_value = MagicMock(return_value=mock_session)

            repo = PostgresDocumentRepository(mock_database_settings)
            results = await repo.list()

            assert len(results) == 2
            assert results[0].filename == "doc1.pdf"
            assert results[1].filename == "doc2.txt"

    async def test_update_success(
        self,
        mock_database_settings: DatabaseSettings,
        sample_document: Document,
    ) -> None:
        """Test update updates and returns document."""
        with (
            patch(
                "src.infrastructure.repositories.postgres_document_repository.create_async_engine"
            ),
            patch(
                "src.infrastructure.repositories.postgres_document_repository.async_sessionmaker"
            ) as mock_session_maker,
        ):
            # Create mock model
            mock_model = MagicMock()
            mock_model.id = sample_document.id
            mock_model.filename = sample_document.filename
            mock_model.content_type = sample_document.content_type.value
            mock_model.size_bytes = sample_document.size_bytes
            mock_model.status = sample_document.status
            mock_model.metadata_ = sample_document.metadata
            mock_model.created_at = sample_document.created_at
            mock_model.updated_at = sample_document.updated_at
            mock_model.error_message = None
            mock_model.chunk_count = 0
            mock_model.file_path = None

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_model

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.commit = AsyncMock()
            mock_session.refresh = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_maker.return_value = MagicMock(return_value=mock_session)

            repo = PostgresDocumentRepository(mock_database_settings)

            # Update document
            sample_document.mark_completed(10)
            result = await repo.update(sample_document)

            assert result is not None
            mock_session.commit.assert_called_once()

    async def test_update_not_found_raises_error(
        self,
        mock_database_settings: DatabaseSettings,
        sample_document: Document,
    ) -> None:
        """Test update raises DocumentNotFoundError when not found."""
        with (
            patch(
                "src.infrastructure.repositories.postgres_document_repository.create_async_engine"
            ),
            patch(
                "src.infrastructure.repositories.postgres_document_repository.async_sessionmaker"
            ) as mock_session_maker,
        ):
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_maker.return_value = MagicMock(return_value=mock_session)

            repo = PostgresDocumentRepository(mock_database_settings)

            with pytest.raises(DocumentNotFoundError):
                await repo.update(sample_document)

    async def test_delete_success(
        self, mock_database_settings: DatabaseSettings
    ) -> None:
        """Test delete returns True when document is deleted."""
        with (
            patch(
                "src.infrastructure.repositories.postgres_document_repository.create_async_engine"
            ),
            patch(
                "src.infrastructure.repositories.postgres_document_repository.async_sessionmaker"
            ) as mock_session_maker,
        ):
            mock_model = MagicMock()

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_model

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.delete = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_maker.return_value = MagicMock(return_value=mock_session)

            repo = PostgresDocumentRepository(mock_database_settings)
            result = await repo.delete(uuid4())

            assert result is True
            mock_session.delete.assert_called_once()

    async def test_delete_not_found(
        self, mock_database_settings: DatabaseSettings
    ) -> None:
        """Test delete returns False when document not found."""
        with (
            patch(
                "src.infrastructure.repositories.postgres_document_repository.create_async_engine"
            ),
            patch(
                "src.infrastructure.repositories.postgres_document_repository.async_sessionmaker"
            ) as mock_session_maker,
        ):
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_maker.return_value = MagicMock(return_value=mock_session)

            repo = PostgresDocumentRepository(mock_database_settings)
            result = await repo.delete(uuid4())

            assert result is False

    async def test_count_returns_count(
        self, mock_database_settings: DatabaseSettings
    ) -> None:
        """Test count returns number of documents."""
        with (
            patch(
                "src.infrastructure.repositories.postgres_document_repository.create_async_engine"
            ),
            patch(
                "src.infrastructure.repositories.postgres_document_repository.async_sessionmaker"
            ) as mock_session_maker,
        ):
            mock_result = MagicMock()
            mock_result.scalar.return_value = 42

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_maker.return_value = MagicMock(return_value=mock_session)

            repo = PostgresDocumentRepository(mock_database_settings)
            count = await repo.count()

            assert count == 42

    async def test_close_disposes_engine(
        self, mock_database_settings: DatabaseSettings
    ) -> None:
        """Test close disposes the database engine."""
        with (
            patch(
                "src.infrastructure.repositories.postgres_document_repository.create_async_engine"
            ) as mock_engine_factory,
            patch(
                "src.infrastructure.repositories.postgres_document_repository.async_sessionmaker"
            ),
        ):
            mock_engine = AsyncMock()
            mock_engine_factory.return_value = mock_engine

            repo = PostgresDocumentRepository(mock_database_settings)
            await repo.close()

            mock_engine.dispose.assert_called_once()
