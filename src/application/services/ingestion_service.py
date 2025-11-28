"""Document ingestion service with compensating transactions.

Implements the Saga pattern to ensure data consistency across
multiple storage systems (PostgreSQL, MinIO, Qdrant).
"""

import logging
from typing import BinaryIO
from uuid import UUID

from src.domain.entities import Chunk, Document
from src.domain.exceptions import DocumentNotFoundError, DocumentProcessingError
from src.domain.interfaces import (
    BlobStorage,
    ChunkingService,
    DocumentParser,
    DocumentRepository,
    EmbeddingService,
    SparseEmbeddingService,
    VectorStore,
)
from src.domain.value_objects import SparseVector

logger = logging.getLogger(__name__)


class IngestionService:
    """Document ingestion service with compensating transactions.

    This service orchestrates the document ingestion pipeline:
    1. Upload file to blob storage (MinIO)
    2. Update document metadata in PostgreSQL
    3. Parse document content
    4. Split into chunks
    5. Generate embeddings (dense + sparse)
    6. Store chunks in vector database (Qdrant)
    7. Update document status to COMPLETED

    Uses the Saga pattern (simplified) for error handling:
    - On failure, compensating transactions clean up partial state
    - Document status is updated to FAILED with error details
    - Orphaned chunks in Qdrant are deleted

    Note:
        This implementation handles cleanup within the same request.
        For production with high reliability requirements, consider
        implementing a Dead Letter Queue (DLQ) with async cleanup workers.
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        blob_storage: BlobStorage,
        vector_store: VectorStore,
        dense_embedding: EmbeddingService,
        sparse_embedding: SparseEmbeddingService | None,
        parser: DocumentParser,
        chunker: ChunkingService,
        batch_size: int = 10,
    ) -> None:
        """Initialize the ingestion service.

        Args:
            document_repo: Repository for document metadata.
            blob_storage: Blob storage for original files.
            vector_store: Vector store for chunk embeddings.
            dense_embedding: Dense embedding service (e.g., OpenAI).
            sparse_embedding: Sparse embedding service (e.g., SPLADE).
                            Optional - if None, only dense vectors are used.
            parser: Document parser for text extraction.
            chunker: Chunking service for text splitting.
            batch_size: Batch size for embedding generation.
        """
        self._document_repo = document_repo
        self._blob_storage = blob_storage
        self._vector_store = vector_store
        self._dense_embedding = dense_embedding
        self._sparse_embedding = sparse_embedding
        self._parser = parser
        self._chunker = chunker
        self._batch_size = batch_size

    async def ingest_document(
        self,
        document: Document,
        file: BinaryIO,
    ) -> Document:
        """Ingest a document into the system.

        This is the main entry point for document ingestion.
        The document should already be created with PENDING status.

        Flow:
            1. Upload file to MinIO
            2. Update document with file_path, status=PROCESSING
            3. Parse document to extract text
            4. Chunk text into smaller segments
            5. Generate dense + sparse embeddings
            6. Store chunks in Qdrant
            7. Update document status to COMPLETED

        Args:
            document: Document entity with PENDING status.
            file: Binary file content.

        Returns:
            Updated Document entity.

        Raises:
            DocumentProcessingError: If any step fails.
        """
        logger.info(
            f"Starting ingestion for document {document.id}: {document.filename}"
        )

        try:
            # Step 1: Upload file to blob storage
            file_path = await self._upload_to_storage(document, file)
            document.set_file_path(file_path)

            # Step 2: Update status to PROCESSING
            document.mark_processing()
            await self._document_repo.update(document)
            logger.info(f"Document {document.id} marked as PROCESSING")

            # Step 3: Parse document
            text = await self._parse_document(file_path)
            if not text.strip():
                raise DocumentProcessingError(
                    document_id=str(document.id),
                    reason="Document contains no extractable text",
                )

            # Step 4: Chunk text
            chunks = self._chunker.chunk(
                text=text,
                document_id=document.id,
                metadata={"filename": document.filename},
            )
            if not chunks:
                raise DocumentProcessingError(
                    document_id=str(document.id),
                    reason="No chunks generated from document",
                )
            logger.info(f"Document {document.id} chunked into {len(chunks)} chunks")

            # Step 5-6: Generate embeddings and store in vector DB
            await self._embed_and_store(chunks)
            logger.info(f"Document {document.id} chunks stored in vector database")

            # Step 7: Mark as completed
            document.mark_completed(len(chunks))
            await self._document_repo.update(document)
            logger.info(f"Document {document.id} ingestion completed successfully")

            return document

        except DocumentProcessingError:
            await self._handle_failure(document)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during ingestion of {document.id}: {e}")
            await self._handle_failure(document, str(e))
            raise DocumentProcessingError(
                document_id=str(document.id),
                reason=str(e),
            ) from e

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document and all associated data.

        Deletion order:
            1. Delete chunks from Qdrant
            2. Delete file from MinIO
            3. Delete metadata from PostgreSQL

        Args:
            document_id: ID of document to delete.

        Returns:
            True if document was found and deleted, False if not found.
        """
        document = await self._document_repo.get_by_id(document_id)
        if not document:
            logger.warning(f"Document {document_id} not found for deletion")
            return False

        logger.info(f"Deleting document {document_id}")

        # Step 1: Delete chunks from vector store
        try:
            deleted_count = await self._vector_store.delete_by_document_id(document_id)
            logger.info(f"Deleted {deleted_count} chunks from vector store")
        except Exception as e:
            logger.error(f"Failed to delete chunks from vector store: {e}")
            # Continue with deletion - orphaned chunks will be cleaned up later

        # Step 2: Delete file from blob storage
        if document.file_path:
            try:
                await self._blob_storage.delete(document.file_path)
                logger.info(f"Deleted file {document.file_path} from blob storage")
            except Exception as e:
                logger.error(f"Failed to delete file from blob storage: {e}")
                # Continue with deletion

        # Step 3: Delete metadata from repository
        result = await self._document_repo.delete(document_id)
        logger.info(f"Document {document_id} deletion completed")

        return result

    async def retry_failed_document(
        self,
        document_id: UUID,
        file: BinaryIO | None = None,
    ) -> Document:
        """Retry processing a failed document.

        If file is not provided, attempts to download from blob storage.

        Args:
            document_id: ID of the failed document.
            file: Optional file content. If not provided, downloads from storage.

        Returns:
            Updated Document entity.

        Raises:
            DocumentProcessingError: If retry fails.
            DocumentNotFoundError: If document doesn't exist.
        """
        document = await self._document_repo.get_by_id(document_id)
        if not document:
            raise DocumentNotFoundError(str(document_id))

        logger.info(f"Retrying failed document {document_id}")

        # Clean up any existing chunks
        try:
            await self._vector_store.delete_by_document_id(document_id)
        except Exception as e:
            logger.warning(f"Failed to clean up chunks before retry: {e}")

        # Get file content
        if file is None:
            if not document.file_path:
                raise DocumentProcessingError(
                    document_id=str(document_id),
                    reason="No file available for retry",
                )
            file = await self._blob_storage.download(document.file_path)

        # Reset document state and re-ingest
        document.status = document.status  # Keep current status for now
        return await self.ingest_document(document, file)

    async def _upload_to_storage(
        self,
        document: Document,
        file: BinaryIO,
    ) -> str:
        """Upload file to blob storage.

        Args:
            document: Document entity.
            file: Binary file content.

        Returns:
            Path in blob storage.
        """
        # Use UUID as directory to ensure uniqueness
        storage_path = f"{document.id}/{document.filename}"

        file_path = await self._blob_storage.upload(
            file=file,
            filename=storage_path,
            content_type=document.content_type.value,
        )

        logger.info(f"Uploaded {document.filename} to {file_path}")
        return file_path

    async def _parse_document(self, file_path: str) -> str:
        """Parse document and extract text.

        Args:
            file_path: Path to document in blob storage.

        Returns:
            Extracted text content.
        """
        text = await self._parser.parse(file_path)
        logger.info(f"Parsed document: {len(text)} characters extracted")
        return text

    async def _embed_and_store(self, chunks: list[Chunk]) -> None:
        """Generate embeddings and store chunks in vector database.

        Processes chunks in batches for efficient embedding generation.

        Args:
            chunks: List of chunks to embed and store.
        """
        # Process in batches
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            texts = [chunk.content for chunk in batch]

            # Generate dense embeddings
            dense_embeddings = await self._dense_embedding.embed_texts(texts)

            # Generate sparse embeddings if available
            sparse_embeddings: list[SparseVector | None]
            if self._sparse_embedding:
                sparse_embeddings = list(
                    await self._sparse_embedding.embed_texts(texts)
                )
            else:
                sparse_embeddings = [None] * len(texts)

            # Attach embeddings to chunks
            for chunk, dense, sparse in zip(
                batch, dense_embeddings, sparse_embeddings, strict=True
            ):
                chunk.set_embeddings(dense=dense, sparse=sparse)

            logger.debug(f"Generated embeddings for batch {i // self._batch_size + 1}")

        # Store all chunks in vector database
        await self._vector_store.upsert_chunks(chunks)

    async def _handle_failure(
        self,
        document: Document,
        error_message: str | None = None,
    ) -> None:
        """Handle ingestion failure with compensating transactions.

        This method:
        1. Updates document status to FAILED
        2. Attempts to clean up orphaned chunks from Qdrant

        Args:
            document: Document that failed processing.
            error_message: Optional error message (uses existing if not provided).
        """
        # Update document status
        if error_message:
            document.mark_failed(error_message)
        elif document.error_message is None:
            document.mark_failed("Unknown error during processing")

        try:
            await self._document_repo.update(document)
            logger.info(f"Document {document.id} marked as FAILED")
        except Exception as e:
            logger.error(f"Failed to update document status: {e}")

        # Clean up orphaned chunks from vector store
        try:
            deleted_count = await self._vector_store.delete_by_document_id(document.id)
            if deleted_count > 0:
                logger.info(
                    f"Cleaned up {deleted_count} orphaned chunks for document {document.id}"
                )
        except Exception as e:
            logger.warning(
                f"Failed to clean up chunks for {document.id}: {e}. "
                "Manual cleanup may be required."
            )
            # In a production system, this would push to a Dead Letter Queue
            # for async cleanup by a background worker
