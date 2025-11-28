"""Domain layer - core business logic and interfaces.

This layer has NO external dependencies.
All interfaces are defined as Protocols for dependency inversion.
"""

from src.domain.exceptions import (
    ConfigurationError,
    DocumentNotFoundError,
    DocumentProcessingError,
    DomainError,
    EmbeddingError,
    LLMError,
    RateLimitError,
    RerankError,
    SearchError,
    StorageError,
    StorageNotFoundError,
    StorageUploadError,
    UnsupportedContentTypeError,
)
from src.domain.interfaces import BlobStorage

__all__ = [
    # Interfaces
    "BlobStorage",
    # Exceptions
    "ConfigurationError",
    "DocumentNotFoundError",
    "DocumentProcessingError",
    "DomainError",
    "EmbeddingError",
    "LLMError",
    "RateLimitError",
    "RerankError",
    "SearchError",
    "StorageError",
    "StorageNotFoundError",
    "StorageUploadError",
    "UnsupportedContentTypeError",
]
