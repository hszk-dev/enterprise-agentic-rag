"""Domain layer - core business logic and interfaces.

This layer has NO external dependencies.
All interfaces are defined as Protocols for dependency inversion.
"""

from src.domain.entities import (
    Chunk,
    Document,
    GenerationResult,
    Query,
    SearchResult,
)
from src.domain.exceptions import (
    ConfigurationError,
    DocumentNotFoundError,
    DocumentProcessingError,
    DomainError,
    EmbeddingError,
    LLMError,
    RateLimitError,
    RepositoryError,
    RerankError,
    SearchError,
    StorageError,
    StorageNotFoundError,
    StorageUploadError,
    UnsupportedContentTypeError,
)
from src.domain.interfaces import (
    BlobStorage,
    ChunkingService,
    DocumentParser,
    DocumentRepository,
    EmbeddingService,
    LLMService,
    Reranker,
    SparseEmbeddingService,
    VectorStore,
)
from src.domain.value_objects import (
    ChunkMetadata,
    ContentType,
    DocumentStatus,
    SparseVector,
    TokenUsage,
)

__all__ = [
    "BlobStorage",
    "Chunk",
    "ChunkMetadata",
    "ChunkingService",
    "ConfigurationError",
    "ContentType",
    "Document",
    "DocumentNotFoundError",
    "DocumentParser",
    "DocumentProcessingError",
    "DocumentRepository",
    "DocumentStatus",
    "DomainError",
    "EmbeddingError",
    "EmbeddingService",
    "GenerationResult",
    "LLMError",
    "LLMService",
    "Query",
    "RateLimitError",
    "RepositoryError",
    "RerankError",
    "Reranker",
    "SearchError",
    "SearchResult",
    "SparseEmbeddingService",
    "SparseVector",
    "StorageError",
    "StorageNotFoundError",
    "StorageUploadError",
    "TokenUsage",
    "UnsupportedContentTypeError",
    "VectorStore",
]
