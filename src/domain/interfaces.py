"""Domain interfaces (Protocols) for dependency inversion.

These protocols define the contracts that infrastructure implementations must follow.
Domain layer has NO external dependencies - only standard library and typing.
"""

from typing import BinaryIO, Protocol, runtime_checkable


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
