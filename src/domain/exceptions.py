"""Domain exceptions.

All domain-specific exceptions inherit from DomainError.
These exceptions are caught and converted to HTTP responses in the presentation layer.
"""


class DomainError(Exception):
    """Base exception for all domain errors.

    Attributes:
        message: Human-readable error message.
        code: Machine-readable error code for API responses.
    """

    def __init__(self, message: str, code: str | None = None) -> None:
        self.message = message
        self.code = code or self.__class__.__name__
        super().__init__(self.message)


class StorageError(DomainError):
    """Blob storage operation failed.

    Raised when MinIO/S3 operations fail.
    """

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        super().__init__(message, "STORAGE_ERROR")


class StorageNotFoundError(StorageError):
    """File not found in blob storage."""

    def __init__(self, path: str) -> None:
        super().__init__(f"File not found: {path}", path)
        self.code = "STORAGE_NOT_FOUND"


class StorageUploadError(StorageError):
    """File upload to blob storage failed."""

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"Failed to upload {filename}: {reason}")
        self.code = "STORAGE_UPLOAD_ERROR"


class RepositoryError(DomainError):
    """Database repository operation failed.

    Raised when PostgreSQL or other database operations fail.
    """

    def __init__(self, message: str, operation: str | None = None) -> None:
        self.operation = operation
        super().__init__(message, "REPOSITORY_ERROR")


class DocumentNotFoundError(DomainError):
    """Document not found in the system."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"Document not found: {document_id}", "DOCUMENT_NOT_FOUND")


class DocumentProcessingError(DomainError):
    """Document processing (parsing, chunking, embedding) failed."""

    def __init__(self, document_id: str, reason: str) -> None:
        self.document_id = document_id
        self.reason = reason
        super().__init__(
            f"Failed to process document {document_id}: {reason}",
            "DOCUMENT_PROCESSING_ERROR",
        )


class UnsupportedContentTypeError(DomainError):
    """Content type is not supported for processing."""

    def __init__(self, content_type: str) -> None:
        self.content_type = content_type
        super().__init__(
            f"Unsupported content type: {content_type}",
            "UNSUPPORTED_CONTENT_TYPE",
        )


class EmbeddingError(DomainError):
    """Embedding generation failed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "EMBEDDING_ERROR")


class SearchError(DomainError):
    """Search operation failed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "SEARCH_ERROR")


class RerankError(DomainError):
    """Reranking operation failed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "RERANK_ERROR")


class LLMError(DomainError):
    """LLM API call failed."""

    def __init__(self, message: str, model: str | None = None) -> None:
        self.model = model
        super().__init__(message, "LLM_ERROR")


class RateLimitError(LLMError):
    """API rate limit exceeded."""

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(message)
        self.code = "RATE_LIMIT_ERROR"


class ConfigurationError(DomainError):
    """Application configuration is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "CONFIGURATION_ERROR")
