"""PostgreSQL document repository implementation.

This module provides document metadata persistence using PostgreSQL with SQLAlchemy.
"""

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import DateTime, Enum, Integer, String, Text, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.domain.entities import Document
from src.domain.exceptions import DocumentNotFoundError, RepositoryError
from src.domain.value_objects import ContentType, DocumentStatus

if TYPE_CHECKING:
    from config.settings import DatabaseSettings

logger = logging.getLogger(__name__)


# =============================================================================
# SQLAlchemy Models
# =============================================================================


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


class DocumentModel(Base):
    """SQLAlchemy model for documents table.

    Maps to the domain Document entity.
    """

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum(DocumentStatus, name="document_status"),
        nullable=False,
        default=DocumentStatus.PENDING,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    def to_entity(self) -> Document:
        """Convert to domain entity.

        Returns:
            Document domain entity.
        """
        return Document(
            id=self.id,
            filename=self.filename,
            content_type=ContentType(self.content_type),
            size_bytes=self.size_bytes,
            status=DocumentStatus(self.status)
            if isinstance(self.status, str)
            else self.status,
            metadata=self.metadata_ or {},
            created_at=self.created_at,
            updated_at=self.updated_at,
            error_message=self.error_message,
            chunk_count=self.chunk_count,
            file_path=self.file_path,
        )

    @classmethod
    def from_entity(cls, entity: Document) -> "DocumentModel":
        """Create from domain entity.

        Args:
            entity: Document domain entity.

        Returns:
            DocumentModel instance.
        """
        return cls(
            id=entity.id,
            filename=entity.filename,
            content_type=entity.content_type.value,
            size_bytes=entity.size_bytes,
            status=entity.status,
            metadata_=entity.metadata,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            error_message=entity.error_message,
            chunk_count=entity.chunk_count,
            file_path=entity.file_path,
        )


# =============================================================================
# Repository Implementation
# =============================================================================


class PostgresDocumentRepository:
    """PostgreSQL document repository.

    Implements the DocumentRepository protocol using SQLAlchemy async.

    Example:
        >>> repo = PostgresDocumentRepository(settings)
        >>> await repo.initialize()
        >>> doc = await repo.save(document)
        >>> retrieved = await repo.get_by_id(doc.id)
    """

    def __init__(self, settings: "DatabaseSettings") -> None:
        """Initialize the repository.

        Args:
            settings: Database configuration settings.
        """
        self._settings = settings
        self._engine = create_async_engine(
            settings.async_url,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def initialize(self) -> None:
        """Initialize the database schema.

        Creates tables if they don't exist.

        Raises:
            RepositoryError: If initialization fails.
        """
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            msg = f"Database initialization failed: {e}"
            raise RepositoryError(msg, operation="initialize") from e

    async def save(self, document: Document) -> Document:
        """Save a new document.

        Args:
            document: Document to save.

        Returns:
            Saved document.

        Raises:
            RepositoryError: If save fails.
        """
        try:
            async with self._session_factory() as session:
                model = DocumentModel.from_entity(document)
                session.add(model)
                await session.commit()
                await session.refresh(model)
                logger.info(f"Saved document: {document.id}")
                return model.to_entity()
        except Exception as e:
            logger.error(f"Failed to save document {document.id}: {e}")
            msg = f"Failed to save document: {e}"
            raise RepositoryError(msg, operation="save") from e

    async def get_by_id(self, document_id: UUID) -> Document | None:
        """Get document by ID.

        Args:
            document_id: Document UUID.

        Returns:
            Document if found, None otherwise.

        Raises:
            RepositoryError: If database query fails.
        """
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(DocumentModel).where(DocumentModel.id == document_id)
                )
                model = result.scalar_one_or_none()
                return model.to_entity() if model else None
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            msg = f"Failed to get document: {e}"
            raise RepositoryError(msg, operation="get_by_id") from e

    async def list(
        self,
        status: DocumentStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """List documents with optional filtering.

        Args:
            status: Filter by processing status.
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.

        Returns:
            List of documents matching criteria.
        """
        try:
            async with self._session_factory() as session:
                query = select(DocumentModel)
                if status is not None:
                    query = query.where(DocumentModel.status == status)
                query = query.order_by(DocumentModel.created_at.desc())
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                models = result.scalars().all()
                return [m.to_entity() for m in models]
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    async def update(self, document: Document) -> Document:
        """Update an existing document.

        Args:
            document: Document with updated fields.

        Returns:
            Updated document.

        Raises:
            DocumentNotFoundError: If document doesn't exist.
            RepositoryError: If update fails.
        """
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(DocumentModel).where(DocumentModel.id == document.id)
                )
                model = result.scalar_one_or_none()

                if model is None:
                    raise DocumentNotFoundError(str(document.id))

                # Update fields
                model.filename = document.filename
                model.content_type = document.content_type.value
                model.size_bytes = document.size_bytes
                model.status = document.status
                model.metadata_ = document.metadata
                model.updated_at = document.updated_at
                model.error_message = document.error_message
                model.chunk_count = document.chunk_count
                model.file_path = document.file_path

                await session.commit()
                await session.refresh(model)
                logger.info(f"Updated document: {document.id}")
                return model.to_entity()

        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update document {document.id}: {e}")
            msg = f"Failed to update document: {e}"
            raise RepositoryError(msg, operation="update") from e

    async def delete(self, document_id: UUID) -> bool:
        """Delete a document by ID.

        Args:
            document_id: Document UUID.

        Returns:
            True if deleted, False if not found.
        """
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(DocumentModel).where(DocumentModel.id == document_id)
                )
                model = result.scalar_one_or_none()

                if model is None:
                    return False

                await session.delete(model)
                await session.commit()
                logger.info(f"Deleted document: {document_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def count(self, status: DocumentStatus | None = None) -> int:
        """Count documents with optional filtering.

        Args:
            status: Filter by processing status.

        Returns:
            Number of matching documents.
        """
        try:
            async with self._session_factory() as session:
                query = select(func.count()).select_from(DocumentModel)
                if status is not None:
                    query = query.where(DocumentModel.status == status)

                result = await session.execute(query)
                return result.scalar() or 0

        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0

    async def close(self) -> None:
        """Close database connections."""
        await self._engine.dispose()
        logger.info("Database connections closed")
