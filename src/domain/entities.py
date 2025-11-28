"""Domain entities.

Entities are objects with identity that persist over time.
They are compared by their identity (ID), not by their attributes.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from .value_objects import ContentType, DocumentStatus, SparseVector, TokenUsage


def _utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC)


@dataclass
class Document:
    """Uploaded document entity.

    Represents a document in the system with its processing state.

    Invariants:
        - id is always a valid UUID
        - filename is non-empty
        - size_bytes >= 0

    Lifecycle:
        1. create() -> status=PENDING
        2. mark_processing() -> status=PROCESSING
        3. mark_completed() | mark_failed() -> status=COMPLETED | FAILED

    Attributes:
        id: Unique identifier.
        filename: Original filename.
        content_type: MIME type of the document.
        size_bytes: File size in bytes.
        status: Current processing status.
        metadata: Additional user-provided metadata.
        created_at: Creation timestamp (UTC).
        updated_at: Last update timestamp (UTC).
        error_message: Error details if status is FAILED.
        chunk_count: Number of chunks created from this document.
        file_path: Path in blob storage (MinIO/S3).
    """

    id: UUID
    filename: str
    content_type: ContentType
    size_bytes: int
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    error_message: str | None = None
    chunk_count: int = 0
    file_path: str | None = None

    @classmethod
    def create(
        cls,
        filename: str,
        content_type: ContentType,
        size_bytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> "Document":
        """Factory method to create a new document.

        Args:
            filename: Original filename.
            content_type: Document MIME type.
            size_bytes: File size in bytes.
            metadata: Optional user metadata.

        Returns:
            New Document instance with PENDING status.

        Raises:
            ValueError: If filename is empty or size_bytes is negative.
        """
        if not filename or not filename.strip():
            msg = "Filename cannot be empty"
            raise ValueError(msg)
        if size_bytes < 0:
            msg = f"Size must be non-negative, got {size_bytes}"
            raise ValueError(msg)

        return cls(
            id=uuid4(),
            filename=filename.strip(),
            content_type=content_type,
            size_bytes=size_bytes,
            metadata=metadata or {},
        )

    def mark_processing(self) -> None:
        """Transition to PROCESSING status."""
        self.status = DocumentStatus.PROCESSING
        self.updated_at = _utc_now()

    def mark_completed(self, chunk_count: int) -> None:
        """Transition to COMPLETED status.

        Args:
            chunk_count: Number of chunks created from this document.
        """
        self.status = DocumentStatus.COMPLETED
        self.chunk_count = chunk_count
        self.error_message = None
        self.updated_at = _utc_now()

    def mark_failed(self, error_message: str) -> None:
        """Transition to FAILED status.

        Args:
            error_message: Description of what went wrong.
        """
        self.status = DocumentStatus.FAILED
        self.error_message = error_message
        self.updated_at = _utc_now()

    def set_file_path(self, file_path: str) -> None:
        """Set the blob storage path.

        Args:
            file_path: Path in MinIO/S3.
        """
        self.file_path = file_path
        self.updated_at = _utc_now()


@dataclass
class Chunk:
    """Document chunk entity.

    Represents a segment of a document, used as the basic unit for retrieval.

    Attributes:
        id: Unique identifier.
        document_id: Parent document ID.
        content: Text content of the chunk.
        chunk_index: Position in the document (0-indexed).
        start_char: Start character offset in original document.
        end_char: End character offset in original document.
        metadata: Additional metadata (page number, section, etc.).
        dense_embedding: Dense vector embedding (e.g., OpenAI).
        sparse_embedding: Sparse vector embedding (e.g., SPLADE).
    """

    id: UUID
    document_id: UUID
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_embedding: list[float] | None = None
    sparse_embedding: SparseVector | None = None

    @classmethod
    def create(
        cls,
        document_id: UUID,
        content: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        metadata: dict[str, Any] | None = None,
    ) -> "Chunk":
        """Factory method to create a new chunk.

        Args:
            document_id: Parent document ID.
            content: Text content.
            chunk_index: Position in document.
            start_char: Start offset.
            end_char: End offset.
            metadata: Optional metadata.

        Returns:
            New Chunk instance.
        """
        return cls(
            id=uuid4(),
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata or {},
        )

    @property
    def char_count(self) -> int:
        """Return character count of content."""
        return len(self.content)

    def set_embeddings(
        self,
        dense: list[float] | None = None,
        sparse: SparseVector | None = None,
    ) -> None:
        """Set embeddings for this chunk.

        Args:
            dense: Dense vector embedding.
            sparse: Sparse vector embedding.
        """
        if dense is not None:
            self.dense_embedding = dense
        if sparse is not None:
            self.sparse_embedding = sparse


@dataclass
class Query:
    """Search query entity.

    Represents a user's search request with all parameters.

    Attributes:
        id: Unique identifier for tracking.
        text: Query text.
        top_k: Number of results to retrieve before reranking.
        rerank_top_n: Number of results after reranking.
        alpha: Hybrid search weight (0=sparse only, 1=dense only).
        filters: Metadata filters for search.
        include_metadata: Whether to return metadata in results.
        user_id: Optional user identifier for tracking.
        session_id: Optional session identifier for conversation context.
    """

    id: UUID
    text: str
    top_k: int = 10
    rerank_top_n: int = 5
    alpha: float = 0.5
    filters: dict[str, Any] | None = None
    include_metadata: bool = True
    user_id: str | None = None
    session_id: str | None = None

    @classmethod
    def create(
        cls,
        text: str,
        top_k: int = 10,
        rerank_top_n: int = 5,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> "Query":
        """Factory method to create a new query.

        Args:
            text: Query text.
            top_k: Number of results before reranking.
            rerank_top_n: Number of results after reranking.
            alpha: Hybrid search weight.
            filters: Metadata filters.
            user_id: User identifier.
            session_id: Session identifier.

        Returns:
            New Query instance.

        Raises:
            ValueError: If text is empty or parameters are invalid.
        """
        if not text or not text.strip():
            msg = "Query text cannot be empty"
            raise ValueError(msg)
        if top_k < 1:
            msg = f"top_k must be >= 1, got {top_k}"
            raise ValueError(msg)
        if rerank_top_n < 1:
            msg = f"rerank_top_n must be >= 1, got {rerank_top_n}"
            raise ValueError(msg)
        if not 0.0 <= alpha <= 1.0:
            msg = f"alpha must be between 0 and 1, got {alpha}"
            raise ValueError(msg)

        return cls(
            id=uuid4(),
            text=text.strip(),
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            alpha=alpha,
            filters=filters,
            user_id=user_id,
            session_id=session_id,
        )


@dataclass
class SearchResult:
    """Search result entity.

    Represents a single search result with scores.

    Attributes:
        chunk: The matched chunk.
        score: Raw search score (implementation-dependent).
        rerank_score: Score after reranking (0.0-1.0).
        rank: Position in result list (1-indexed).
    """

    chunk: Chunk
    score: float
    rerank_score: float | None = None
    rank: int = 0

    @property
    def final_score(self) -> float:
        """Get final score (rerank score if available, else raw score)."""
        return self.rerank_score if self.rerank_score is not None else self.score

    @property
    def display_score(self) -> float:
        """Get normalized score for UI display (0.0-1.0).

        Rerank scores are already normalized. Raw scores are clamped.
        """
        if self.rerank_score is not None:
            return self.rerank_score
        # Clamp raw score to 0-1 range
        return min(1.0, max(0.0, self.score))


@dataclass
class GenerationResult:
    """LLM generation result entity.

    Represents a complete RAG response with sources and metadata.

    Attributes:
        id: Unique identifier.
        query: Original query.
        answer: Generated answer text.
        sources: Search results used as context.
        usage: Token usage statistics.
        model: Model used for generation.
        latency_ms: Generation latency in milliseconds.
        created_at: Generation timestamp (UTC).
    """

    id: UUID
    query: Query
    answer: str
    sources: list[SearchResult]
    usage: TokenUsage
    model: str
    latency_ms: float
    created_at: datetime = field(default_factory=_utc_now)

    @classmethod
    def create(
        cls,
        query: Query,
        answer: str,
        sources: list[SearchResult],
        usage: TokenUsage,
        model: str,
        latency_ms: float,
    ) -> "GenerationResult":
        """Factory method to create a generation result.

        Args:
            query: Original query.
            answer: Generated answer.
            sources: Source chunks used.
            usage: Token usage.
            model: Model identifier.
            latency_ms: Generation latency.

        Returns:
            New GenerationResult instance.
        """
        return cls(
            id=uuid4(),
            query=query,
            answer=answer,
            sources=sources,
            usage=usage,
            model=model,
            latency_ms=latency_ms,
        )

    @property
    def source_count(self) -> int:
        """Return number of sources used."""
        return len(self.sources)

    @property
    def estimated_cost_usd(self) -> float:
        """Return estimated API cost."""
        return self.usage.estimated_cost_usd
