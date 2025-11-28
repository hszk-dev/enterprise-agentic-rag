"""Integration tests for MinIO storage.

These tests require a running MinIO instance.
Run with: pytest tests/integration/ -m integration
"""

import io

import pytest

from config import MinIOSettings
from src.domain.exceptions import StorageNotFoundError
from src.infrastructure.storage import MinIOStorage


@pytest.fixture
def minio_test_settings():
    """Create MinIO settings for integration tests."""
    return MinIOSettings(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",  # pragma: allowlist secret
        bucket_name="test-integration",
        secure=False,
    )


@pytest.fixture
async def minio_storage(minio_test_settings: MinIOSettings):
    """Create and initialize MinIO storage for tests."""
    storage = MinIOStorage(minio_test_settings)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.mark.integration
class TestMinIOStorageIntegration:
    """Integration tests for MinIO storage operations."""

    async def test_initialize_creates_bucket(
        self, minio_test_settings: MinIOSettings
    ) -> None:
        """Test that initialize() creates the bucket if it doesn't exist."""
        storage = MinIOStorage(minio_test_settings)
        await storage.initialize()

        # Second initialization should not raise
        await storage.initialize()

    async def test_upload_and_download(self, minio_storage: MinIOStorage) -> None:
        """Test uploading and downloading a file."""
        # Prepare test file
        content = b"Hello, MinIO! This is a test file."
        file = io.BytesIO(content)
        filename = "test-document.txt"
        content_type = "text/plain"

        # Upload
        path = await minio_storage.upload(file, filename, content_type)

        # Verify path format
        assert path.startswith("documents/")
        assert path.endswith(f"/{filename}")

        # Download and verify content
        downloaded = await minio_storage.download(path)
        downloaded_content = downloaded.read()
        assert downloaded_content == content

        # Cleanup
        await minio_storage.delete(path)

    async def test_upload_pdf_file(self, minio_storage: MinIOStorage) -> None:
        """Test uploading a PDF-like file."""
        # Simulate PDF content (just bytes for testing)
        content = b"%PDF-1.4 fake pdf content"
        file = io.BytesIO(content)
        filename = "report.pdf"
        content_type = "application/pdf"

        path = await minio_storage.upload(file, filename, content_type)
        assert path.endswith("/report.pdf")

        # Verify exists
        assert await minio_storage.exists(path)

        # Cleanup
        await minio_storage.delete(path)

    async def test_exists_returns_true_for_existing_file(
        self, minio_storage: MinIOStorage
    ) -> None:
        """Test that exists() returns True for uploaded files."""
        content = b"test content"
        file = io.BytesIO(content)
        path = await minio_storage.upload(file, "exists-test.txt", "text/plain")

        assert await minio_storage.exists(path)

        # Cleanup
        await minio_storage.delete(path)

    async def test_exists_returns_false_for_nonexistent_file(
        self, minio_storage: MinIOStorage
    ) -> None:
        """Test that exists() returns False for non-existent files."""
        assert not await minio_storage.exists("documents/nonexistent/file.txt")

    async def test_delete_existing_file(self, minio_storage: MinIOStorage) -> None:
        """Test deleting an existing file."""
        content = b"to be deleted"
        file = io.BytesIO(content)
        path = await minio_storage.upload(file, "delete-test.txt", "text/plain")

        # Verify exists before delete
        assert await minio_storage.exists(path)

        # Delete
        result = await minio_storage.delete(path)
        assert result is True

        # Verify no longer exists
        assert not await minio_storage.exists(path)

    async def test_delete_nonexistent_file(self, minio_storage: MinIOStorage) -> None:
        """Test deleting a non-existent file returns False."""
        result = await minio_storage.delete("documents/nonexistent/file.txt")
        assert result is False

    async def test_download_nonexistent_file_raises_error(
        self, minio_storage: MinIOStorage
    ) -> None:
        """Test that downloading a non-existent file raises StorageNotFoundError."""
        with pytest.raises(StorageNotFoundError) as exc_info:
            await minio_storage.download("documents/nonexistent/file.txt")

        assert "documents/nonexistent/file.txt" in str(exc_info.value)

    async def test_get_presigned_url(self, minio_storage: MinIOStorage) -> None:
        """Test generating a presigned URL."""
        content = b"presigned url test"
        file = io.BytesIO(content)
        path = await minio_storage.upload(file, "presigned-test.txt", "text/plain")

        # Generate presigned URL
        url = await minio_storage.get_presigned_url(path, expires_in=300)

        # Verify URL format
        assert "localhost:9000" in url
        assert "test-integration" in url
        assert "X-Amz-Signature" in url

        # Cleanup
        await minio_storage.delete(path)

    async def test_upload_large_file(self, minio_storage: MinIOStorage) -> None:
        """Test uploading a larger file (1MB)."""
        # Create 1MB of data
        content = b"x" * (1024 * 1024)
        file = io.BytesIO(content)

        path = await minio_storage.upload(
            file, "large-file.bin", "application/octet-stream"
        )

        # Download and verify size
        downloaded = await minio_storage.download(path)
        downloaded_content = downloaded.read()
        assert len(downloaded_content) == len(content)

        # Cleanup
        await minio_storage.delete(path)
