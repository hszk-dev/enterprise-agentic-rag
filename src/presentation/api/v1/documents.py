"""Document management endpoints.

Provides endpoints for uploading, listing, retrieving, and deleting documents.
"""

import io
import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from src.domain.entities import Document
from src.domain.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    StorageError,
    UnsupportedContentTypeError,
)
from src.domain.value_objects import ContentType, DocumentStatus
from src.presentation.api.dependencies import (
    DocumentRepositoryDep,
    IngestionServiceDep,
)
from src.presentation.schemas.documents import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def _document_to_detail_response(document: Document) -> DocumentDetailResponse:
    """Convert domain Document to API response.

    Args:
        document: Domain document entity.

    Returns:
        DocumentDetailResponse for API.
    """
    return DocumentDetailResponse(
        id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        size_bytes=document.size_bytes,
        status=document.status,
        metadata=document.metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
        chunk_count=document.chunk_count,
        file_path=document.file_path,
        error_message=document.error_message,
    )


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document",
    description="Upload a document for processing. Supported formats: PDF, DOCX, TXT, MD, HTML.",
)
async def upload_document(
    file: Annotated[UploadFile, File(description="Document file to upload")],
    ingestion_service: IngestionServiceDep,
    document_repo: DocumentRepositoryDep,
) -> DocumentUploadResponse:
    """Upload and process a document.

    The document is accepted for processing and will be chunked and indexed
    asynchronously. Use GET /documents/{id} to check processing status.

    Args:
        file: The document file to upload.
        ingestion_service: Injected ingestion service.

    Returns:
        DocumentUploadResponse with document ID and initial status.

    Raises:
        HTTPException: 415 if content type is not supported.
        HTTPException: 500 if processing fails.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    # Determine content type from file extension
    try:
        extension = file.filename.rsplit(".", 1)[-1] if "." in file.filename else ""
        content_type = ContentType.from_extension(extension)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(e),
        ) from e

    # Read file content
    content = await file.read()
    file_size = len(content)

    # Reset file position for re-reading
    await file.seek(0)

    # Create document entity
    document = Document.create(
        filename=file.filename,
        content_type=content_type,
        size_bytes=file_size,
    )

    # Save document to database with PENDING status
    document = await document_repo.save(document)
    logger.info(f"Document {document.id} saved to database with PENDING status")

    try:
        # Ingest the document (wrap bytes in BytesIO for BinaryIO interface)
        document = await ingestion_service.ingest_document(
            document=document,
            file=io.BytesIO(content),
        )
    except UnsupportedContentTypeError as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=e.message,
        ) from e
    except (StorageError, DocumentProcessingError) as e:
        logger.error(f"Failed to process document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {e.message}",
        ) from e

    return DocumentUploadResponse(
        id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        size_bytes=document.size_bytes,
        status=document.status,
        created_at=document.created_at,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List documents",
    description="List all documents with optional filtering and pagination.",
)
async def list_documents(
    document_repo: DocumentRepositoryDep,
    status_filter: Annotated[
        DocumentStatus | None,
        Query(alias="status", description="Filter by processing status"),
    ] = None,
    limit: Annotated[
        int, Query(ge=1, le=100, description="Maximum number of documents to return")
    ] = 20,
    offset: Annotated[int, Query(ge=0, description="Number of documents to skip")] = 0,
) -> DocumentListResponse:
    """List documents with pagination.

    Args:
        document_repo: Injected document repository.
        status_filter: Optional status filter.
        limit: Maximum number of documents per page.
        offset: Number of documents to skip.

    Returns:
        DocumentListResponse with paginated results.
    """
    documents = await document_repo.list(
        status=status_filter,
        limit=limit,
        offset=offset,
    )

    # Get total count (for pagination)
    # Note: In a production system, this should be a separate count query
    total_count = await document_repo.count(status=status_filter)

    return DocumentListResponse(
        items=[_document_to_detail_response(doc) for doc in documents],
        total=total_count,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document details",
    description="Get detailed information about a specific document.",
)
async def get_document(
    document_id: UUID,
    document_repo: DocumentRepositoryDep,
) -> DocumentDetailResponse:
    """Get document by ID.

    Args:
        document_id: The document's unique identifier.
        document_repo: Injected document repository.

    Returns:
        DocumentDetailResponse with full document details.

    Raises:
        HTTPException: 404 if document is not found.
    """
    document = await document_repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )
    return _document_to_detail_response(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description="Delete a document and all its associated data (chunks, embeddings).",
)
async def delete_document(
    document_id: UUID,
    ingestion_service: IngestionServiceDep,
) -> None:
    """Delete a document and its associated data.

    This operation removes:
    - Document metadata from the database
    - Original file from blob storage
    - Chunks and embeddings from vector store

    Args:
        document_id: The document's unique identifier.
        ingestion_service: Injected ingestion service.

    Raises:
        HTTPException: 404 if document is not found.
        HTTPException: 500 if deletion fails.
    """
    try:
        deleted = await ingestion_service.delete_document(document_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )
    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        ) from e
    except (StorageError, DocumentProcessingError) as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {e.message}",
        ) from e
