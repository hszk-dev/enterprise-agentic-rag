"""Document-related API schemas.

Pydantic models for document upload, retrieval, and listing endpoints.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from src.domain.value_objects import ContentType, DocumentStatus


class DocumentUploadResponse(BaseModel):
    """Response for document upload endpoint.

    Returned immediately after a document is accepted for processing.

    Attributes:
        id: Unique document identifier.
        filename: Original filename.
        content_type: MIME type of the document.
        size_bytes: File size in bytes.
        status: Current processing status (typically PENDING).
        created_at: Upload timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    content_type: ContentType
    size_bytes: int
    status: DocumentStatus
    created_at: datetime


class DocumentDetailResponse(BaseModel):
    """Detailed document information response.

    Full document details including processing status and metadata.

    Attributes:
        id: Unique document identifier.
        filename: Original filename.
        content_type: MIME type of the document.
        size_bytes: File size in bytes.
        status: Current processing status.
        metadata: User-provided metadata.
        created_at: Upload timestamp.
        updated_at: Last update timestamp.
        chunk_count: Number of chunks created (0 if not processed).
        file_path: Storage path (internal use).
        error_message: Error details if processing failed.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    content_type: ContentType
    size_bytes: int
    status: DocumentStatus
    # NOTE: Any is used here because metadata is user-provided and schema-less.
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    chunk_count: int
    file_path: str | None = None
    error_message: str | None = None


class DocumentListResponse(BaseModel):
    """Paginated document list response.

    Attributes:
        items: List of documents in current page.
        total: Total number of documents matching query.
        limit: Maximum items per page.
        offset: Number of items skipped.
    """

    items: list[DocumentDetailResponse]
    total: int
    limit: int
    offset: int
