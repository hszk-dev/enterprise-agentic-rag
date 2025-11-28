"""Domain interfaces (Protocols) for dependency inversion.

These protocols define the contracts that infrastructure implementations must follow.
Domain layer has NO external dependencies - only standard library and typing.

Available interfaces:
    - BlobStorage: Object storage (MinIO/S3)
    - DocumentRepository: Document metadata persistence (PostgreSQL)
    - VectorStore: Vector search operations (Qdrant)
    - EmbeddingService: Dense embedding generation (OpenAI)
    - SparseEmbeddingService: Sparse embedding generation (SPLADE)
    - Reranker: Result reranking (Cohere)
    - LLMService: LLM completion (OpenAI)
    - DocumentParser: Document parsing
    - ChunkingService: Text chunking
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from .entities import Chunk, Document, GenerationResult, SearchResult
    from .value_objects import DocumentStatus, SparseVector


# =============================================================================
# Storage Interfaces
# =============================================================================


@runtime_checkable
class BlobStorage(Protocol):
    """Object storage interface for persisting original files.

    This interface abstracts the storage backend (MinIO/S3) from the domain layer.
    Keeping original files allows re-processing when embedding models change.

    Example:
        >>> storage: BlobStorage = MinIOStorage(settings)
        >>> path = await storage.upload(file, "doc.pdf", "application/pdf")
        >>> exists = await storage.exists(path)
    """

    async def upload(
        self,
        file: BinaryIO,
        filename: str,
        content_type: str,
    ) -> str:
        """Upload a file to blob storage.

        Args:
            file: File-like object to upload.
            filename: Target filename (will be prefixed with UUID path).
            content_type: MIME type of the file.

        Returns:
            Storage path that can be used to retrieve the file later.
            Example: "documents/550e8400-e29b-41d4-a716-446655440000/report.pdf"

        Raises:
            StorageError: If upload fails.
        """
        ...

    async def download(self, path: str) -> BinaryIO:
        """Download a file from blob storage.

        Args:
            path: Storage path returned by upload().

        Returns:
            File-like object containing the file contents.

        Raises:
            StorageError: If download fails or file not found.
        """
        ...

    async def delete(self, path: str) -> bool:
        """Delete a file from blob storage.

        Args:
            path: Storage path returned by upload().

        Returns:
            True if file was deleted, False if file didn't exist.

        Raises:
            StorageError: If deletion fails for reasons other than not found.
        """
        ...

    async def exists(self, path: str) -> bool:
        """Check if a file exists in blob storage.

        Args:
            path: Storage path returned by upload().

        Returns:
            True if file exists, False otherwise.

        Raises:
            StorageError: If check fails.
        """
        ...

    async def get_presigned_url(
        self,
        path: str,
        expires_in: int = 3600,
    ) -> str:
        """Generate a presigned URL for direct file access.

        Args:
            path: Storage path returned by upload().
            expires_in: URL expiration time in seconds (default: 1 hour).

        Returns:
            Presigned URL that can be used to download the file directly.

        Raises:
            StorageError: If URL generation fails.
        """
        ...

    async def close(self) -> None:
        """Close and clean up storage resources.

        Should be called when the storage is no longer needed.
        """
        ...


@runtime_checkable
class DocumentRepository(Protocol):
    """Document metadata persistence interface.

    Abstracts the database (PostgreSQL) from the domain layer.
    Handles CRUD operations for Document entities.
    """

    async def save(self, document: "Document") -> "Document":
        """Save a new document.

        Args:
            document: Document to save.

        Returns:
            Saved document (may have updated fields).

        Raises:
            RepositoryError: If save fails.
        """
        ...

    async def get_by_id(self, document_id: UUID) -> "Document | None":
        """Get document by ID.

        Args:
            document_id: Document UUID.

        Returns:
            Document if found, None otherwise.
        """
        ...

    async def list(
        self,
        status: "DocumentStatus | None" = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list["Document"]:
        """List documents with optional filtering.

        Args:
            status: Filter by processing status.
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.

        Returns:
            List of documents matching criteria.
        """
        ...

    async def update(self, document: "Document") -> "Document":
        """Update an existing document.

        Args:
            document: Document with updated fields.

        Returns:
            Updated document.

        Raises:
            DocumentNotFoundError: If document doesn't exist.
            RepositoryError: If update fails.
        """
        ...

    async def delete(self, document_id: UUID) -> bool:
        """Delete a document by ID.

        Args:
            document_id: Document UUID.

        Returns:
            True if deleted, False if not found.
        """
        ...

    async def count(self, status: "DocumentStatus | None" = None) -> int:
        """Count documents with optional filtering.

        Args:
            status: Filter by processing status.

        Returns:
            Number of matching documents.
        """
        ...


# =============================================================================
# Vector Store Interface
# =============================================================================


@runtime_checkable
class VectorStore(Protocol):
    """Vector store interface for similarity search.

    Abstracts the vector database (Qdrant) from the domain layer.
    Supports both dense and hybrid (dense + sparse) search.
    """

    async def upsert_chunks(self, chunks: list["Chunk"]) -> None:
        """Insert or update chunks in the vector store.

        Args:
            chunks: List of chunks with embeddings set.

        Raises:
            SearchError: If upsert fails.
        """
        ...

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list["SearchResult"]:
        """Dense vector similarity search.

        Args:
            query_embedding: Query dense embedding vector.
            top_k: Number of results to return.
            filters: Metadata filters.

        Returns:
            List of search results sorted by score (descending).

        Raises:
            SearchError: If search fails.
        """
        ...

    async def hybrid_search(
        self,
        query_text: str,
        query_dense_embedding: list[float],
        query_sparse_embedding: "SparseVector",
        top_k: int = 10,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list["SearchResult"]:
        """Hybrid search combining dense and sparse vectors.

        Args:
            query_text: Original query text.
            query_dense_embedding: Query dense embedding.
            query_sparse_embedding: Query sparse embedding.
            top_k: Number of results to return.
            alpha: Weight for dense vs sparse (0=sparse only, 1=dense only).
            filters: Metadata filters.

        Returns:
            List of search results sorted by combined score.

        Raises:
            SearchError: If search fails.
        """
        ...

    async def delete_by_document_id(self, document_id: UUID) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: Parent document UUID.

        Returns:
            Number of chunks deleted.

        Raises:
            SearchError: If deletion fails.
        """
        ...

    async def get_collection_stats(self) -> dict[str, Any]:
        """Get vector store collection statistics.

        Returns:
            Dictionary with stats (vectors_count, indexed_vectors_count, etc.).
        """
        ...


# =============================================================================
# Embedding Interfaces
# =============================================================================


@runtime_checkable
class EmbeddingService(Protocol):
    """Dense embedding generation interface.

    Abstracts embedding providers (OpenAI, etc.) from the domain layer.
    """

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        ...

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text.

        Returns:
            Dense embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (batch).

        Args:
            texts: List of input texts.

        Returns:
            List of dense embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        ...


@runtime_checkable
class SparseEmbeddingService(Protocol):
    """Sparse embedding generation interface.

    Used for keyword-based search (SPLADE, BM25, etc.).
    """

    async def embed_text(self, text: str) -> "SparseVector":
        """Generate sparse embedding for a single text.

        Args:
            text: Input text.

        Returns:
            Sparse vector representation.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        ...

    async def embed_texts(self, texts: list[str]) -> list["SparseVector"]:
        """Generate sparse embeddings for multiple texts (batch).

        Args:
            texts: List of input texts.

        Returns:
            List of sparse vector representations.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        ...


# =============================================================================
# Reranking Interface
# =============================================================================


@runtime_checkable
class Reranker(Protocol):
    """Reranking service interface.

    Abstracts reranking providers (Cohere, etc.) from the domain layer.
    """

    async def rerank(
        self,
        query: str,
        results: list["SearchResult"],
        top_n: int = 5,
    ) -> list["SearchResult"]:
        """Rerank search results based on relevance to query.

        Args:
            query: Original query text.
            results: Search results to rerank.
            top_n: Number of top results to return.

        Returns:
            Reranked results with updated scores and ranks.

        Raises:
            RerankError: If reranking fails.
        """
        ...


# =============================================================================
# LLM Interface
# =============================================================================


@runtime_checkable
class LLMService(Protocol):
    """LLM service interface for text generation.

    Abstracts LLM providers (OpenAI, Bedrock, etc.) from the domain layer.
    """

    async def generate(
        self,
        prompt: str,
        context: list[str],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> "GenerationResult":
        """Generate a response using the LLM.

        Args:
            prompt: User prompt/query.
            context: List of context passages from retrieval.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.

        Returns:
            Generation result with answer and metadata.

        Raises:
            LLMError: If generation fails.
            RateLimitError: If rate limit is exceeded.
        """
        ...

    async def generate_stream(
        self,
        prompt: str,
        context: list[str],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Generate a streaming response.

        Args:
            prompt: User prompt/query.
            context: List of context passages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Yields:
            Chunks of generated text.

        Raises:
            LLMError: If generation fails.
            RateLimitError: If rate limit is exceeded.
        """
        ...


# =============================================================================
# Document Processing Interfaces
# =============================================================================


class DocumentParser(ABC):
    """Abstract base class for document parsers.

    Each content type (PDF, DOCX, etc.) requires a specific parser implementation.
    """

    @abstractmethod
    async def parse(self, file_path: str) -> str:
        """Parse a document and extract text content.

        Args:
            file_path: Path to the document (local or blob storage path).

        Returns:
            Extracted text content.

        Raises:
            DocumentProcessingError: If parsing fails.
        """
        ...

    @abstractmethod
    def supports(self, content_type: str) -> bool:
        """Check if this parser supports a content type.

        Args:
            content_type: MIME type to check.

        Returns:
            True if supported, False otherwise.
        """
        ...


class ChunkingService(ABC):
    """Abstract base class for text chunking.

    Splits text into smaller chunks for embedding and retrieval.
    """

    @abstractmethod
    def chunk(
        self,
        text: str,
        document_id: UUID,
        metadata: dict[str, Any] | None = None,
    ) -> list["Chunk"]:
        """Split text into chunks.

        Args:
            text: Full text to chunk.
            document_id: Parent document ID.
            metadata: Metadata to attach to each chunk.

        Returns:
            List of Chunk entities.
        """
        ...

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Return configured chunk size in characters."""
        ...

    @property
    @abstractmethod
    def chunk_overlap(self) -> int:
        """Return configured chunk overlap in characters."""
        ...
