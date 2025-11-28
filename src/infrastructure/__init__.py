"""Infrastructure layer - external service implementations.

This layer contains concrete implementations of domain interfaces.
"""

from src.infrastructure.storage import MinIOStorage

__all__ = ["MinIOStorage"]
