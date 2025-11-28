"""MinIO (S3 compatible) blob storage implementation."""

import asyncio
import io
import logging
from datetime import timedelta
from functools import partial
from typing import BinaryIO
from uuid import uuid4

from minio import Minio
from minio.error import S3Error

from config import MinIOSettings
from src.domain.exceptions import StorageError, StorageNotFoundError, StorageUploadError
from src.domain.interfaces import BlobStorage

logger = logging.getLogger(__name__)


class MinIOStorage(BlobStorage):
    """MinIO implementation of BlobStorage protocol.

    Uses the official MinIO Python client with asyncio wrapper for non-blocking I/O.

    Example:
        >>> settings = MinIOSettings()
        >>> storage = MinIOStorage(settings)
        >>> await storage.initialize()
        >>> path = await storage.upload(file, "doc.pdf", "application/pdf")
    """

    def __init__(self, settings: MinIOSettings) -> None:
        """Initialize MinIO storage.

        Args:
            settings: MinIO configuration settings.
        """
        self._settings = settings
        self._client = Minio(
            endpoint=settings.endpoint,
            access_key=settings.access_key,
            secret_key=settings.secret_key.get_secret_value(),
            secure=settings.secure,
        )
        self._bucket_name = settings.bucket_name
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the storage by creating bucket if it doesn't exist.

        Should be called once during application startup.
        Thread-safe: uses asyncio Lock to prevent race conditions.
        """
        async with self._init_lock:
            if self._initialized:
                return

            loop = asyncio.get_running_loop()
            try:
                exists = await loop.run_in_executor(
                    None, self._client.bucket_exists, self._bucket_name
                )
                if not exists:
                    await loop.run_in_executor(
                        None, self._client.make_bucket, self._bucket_name
                    )
                    logger.info("Created bucket: %s", self._bucket_name)
                else:
                    logger.info("Bucket already exists: %s", self._bucket_name)
                self._initialized = True
            except S3Error as e:
                logger.error("Failed to initialize MinIO bucket: %s", e)
                raise StorageError(f"Failed to initialize MinIO bucket: {e}") from e

    async def upload(
        self,
        file: BinaryIO,
        filename: str,
        content_type: str,
    ) -> str:
        """Upload a file to MinIO.

        Files are stored with a UUID prefix to avoid collisions:
        documents/{uuid}/{filename}

        Args:
            file: File-like object to upload.
            filename: Target filename.
            content_type: MIME type of the file.

        Returns:
            Storage path: "documents/{uuid}/{filename}"

        Raises:
            StorageUploadError: If upload fails.
        """
        object_name = f"documents/{uuid4()}/{filename}"
        loop = asyncio.get_running_loop()

        try:
            # Get file size
            file.seek(0, io.SEEK_END)
            file_size = file.tell()
            file.seek(0)

            # Upload file
            await loop.run_in_executor(
                None,
                partial(
                    self._client.put_object,
                    bucket_name=self._bucket_name,
                    object_name=object_name,
                    data=file,
                    length=file_size,
                    content_type=content_type,
                ),
            )
            logger.info("Uploaded file: %s (%d bytes)", object_name, file_size)
            return object_name

        except S3Error as e:
            msg = f"S3 error: {e}"
            logger.error("Failed to upload %s: %s", filename, msg)
            raise StorageUploadError(filename, msg) from e
        except Exception as e:
            msg = str(e)
            logger.error("Failed to upload %s: %s", filename, msg)
            raise StorageUploadError(filename, msg) from e

    async def download(self, path: str) -> BinaryIO:
        """Download a file from MinIO.

        Args:
            path: Storage path returned by upload().

        Returns:
            BytesIO object containing the file contents.

        Raises:
            StorageNotFoundError: If file doesn't exist.
            StorageError: If download fails.
        """
        loop = asyncio.get_running_loop()

        try:
            response = await loop.run_in_executor(
                None,
                partial(
                    self._client.get_object,
                    bucket_name=self._bucket_name,
                    object_name=path,
                ),
            )
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()

            logger.info(f"Downloaded file: {path} ({len(data)} bytes)")
            return io.BytesIO(data)

        except S3Error as e:
            if e.code == "NoSuchKey":
                raise StorageNotFoundError(path) from e
            msg = f"Failed to download {path}: {e}"
            logger.error(msg)
            raise StorageError(msg, path) from e

    async def delete(self, path: str) -> bool:
        """Delete a file from MinIO.

        Args:
            path: Storage path returned by upload().

        Returns:
            True if file was deleted, False if file didn't exist.

        Raises:
            StorageError: If deletion fails.
        """
        loop = asyncio.get_running_loop()

        try:
            # Check if object exists first
            exists = await self.exists(path)
            if not exists:
                logger.info(f"File not found, nothing to delete: {path}")
                return False

            await loop.run_in_executor(
                None,
                partial(
                    self._client.remove_object,
                    bucket_name=self._bucket_name,
                    object_name=path,
                ),
            )
            logger.info(f"Deleted file: {path}")
            return True

        except S3Error as e:
            msg = f"Failed to delete {path}: {e}"
            logger.error(msg)
            raise StorageError(msg, path) from e

    async def exists(self, path: str) -> bool:
        """Check if a file exists in MinIO.

        Args:
            path: Storage path returned by upload().

        Returns:
            True if file exists, False otherwise.

        Raises:
            StorageError: If check fails.
        """
        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(
                None,
                partial(
                    self._client.stat_object,
                    bucket_name=self._bucket_name,
                    object_name=path,
                ),
            )
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            msg = f"Failed to check existence of {path}: {e}"
            logger.error(msg)
            raise StorageError(msg, path) from e

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
            Presigned URL for direct download.

        Raises:
            StorageError: If URL generation fails.
        """
        loop = asyncio.get_running_loop()

        try:
            url = await loop.run_in_executor(
                None,
                partial(
                    self._client.presigned_get_object,
                    bucket_name=self._bucket_name,
                    object_name=path,
                    expires=timedelta(seconds=expires_in),
                ),
            )
            logger.debug(f"Generated presigned URL for: {path}")
            return url
        except S3Error as e:
            msg = f"Failed to generate presigned URL for {path}: {e}"
            logger.error(msg)
            raise StorageError(msg, path) from e

    async def close(self) -> None:
        """Close the MinIO client connections.

        Note: The MinIO client doesn't require explicit cleanup,
        but this method is provided for consistency with other services.
        """
        logger.info("MinIO storage closed")
