"""Unit tests for MinIO storage with mocked dependencies."""

import io
from unittest.mock import MagicMock, patch

import pytest
from minio.error import S3Error

from config import MinIOSettings
from src.domain.exceptions import StorageNotFoundError, StorageUploadError
from src.infrastructure.storage import MinIOStorage


@pytest.fixture
def mock_minio_settings():
    """Create MinIO settings for unit tests."""
    return MinIOSettings(
        endpoint="localhost:9000",
        access_key="testuser",
        secret_key="testpass",  # pragma: allowlist secret
        bucket_name="test-bucket",
        secure=False,
    )


@pytest.mark.unit
class TestMinIOStorageUnit:
    """Unit tests for MinIOStorage class."""

    def test_init_creates_client(self, mock_minio_settings: MinIOSettings) -> None:
        """Test that __init__ creates MinIO client with correct settings."""
        with patch("src.infrastructure.storage.minio_storage.Minio") as mock_minio:
            MinIOStorage(mock_minio_settings)

            mock_minio.assert_called_once_with(
                endpoint="localhost:9000",
                access_key="testuser",
                secret_key="testpass",  # pragma: allowlist secret
                secure=False,
            )

    def test_endpoint_url_property(self, mock_minio_settings: MinIOSettings) -> None:
        """Test endpoint_url property returns correct URL."""
        assert mock_minio_settings.endpoint_url == "http://localhost:9000"

        # Test with secure=True
        secure_settings = MinIOSettings(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",  # pragma: allowlist secret
            bucket_name="test",
            secure=True,
        )
        assert secure_settings.endpoint_url == "https://localhost:9000"

    async def test_initialize_creates_bucket_if_not_exists(
        self, mock_minio_settings: MinIOSettings
    ) -> None:
        """Test initialize creates bucket when it doesn't exist."""
        with patch(
            "src.infrastructure.storage.minio_storage.Minio"
        ) as mock_minio_class:
            mock_client = MagicMock()
            mock_client.bucket_exists.return_value = False
            mock_minio_class.return_value = mock_client

            storage = MinIOStorage(mock_minio_settings)
            await storage.initialize()

            mock_client.bucket_exists.assert_called_once_with("test-bucket")
            mock_client.make_bucket.assert_called_once_with("test-bucket")

    async def test_initialize_skips_creation_if_bucket_exists(
        self, mock_minio_settings: MinIOSettings
    ) -> None:
        """Test initialize doesn't create bucket when it already exists."""
        with patch(
            "src.infrastructure.storage.minio_storage.Minio"
        ) as mock_minio_class:
            mock_client = MagicMock()
            mock_client.bucket_exists.return_value = True
            mock_minio_class.return_value = mock_client

            storage = MinIOStorage(mock_minio_settings)
            await storage.initialize()

            mock_client.bucket_exists.assert_called_once()
            mock_client.make_bucket.assert_not_called()

    async def test_upload_returns_path_with_uuid(
        self, mock_minio_settings: MinIOSettings
    ) -> None:
        """Test upload returns path with UUID prefix."""
        with patch(
            "src.infrastructure.storage.minio_storage.Minio"
        ) as mock_minio_class:
            mock_client = MagicMock()
            mock_minio_class.return_value = mock_client

            storage = MinIOStorage(mock_minio_settings)
            file = io.BytesIO(b"test content")

            path = await storage.upload(file, "test.txt", "text/plain")

            assert path.startswith("documents/")
            assert path.endswith("/test.txt")
            # Verify UUID format in path
            parts = path.split("/")
            assert len(parts) == 3
            assert len(parts[1]) == 36  # UUID length

    async def test_upload_calls_put_object(
        self, mock_minio_settings: MinIOSettings
    ) -> None:
        """Test upload calls MinIO put_object with correct parameters."""
        with patch(
            "src.infrastructure.storage.minio_storage.Minio"
        ) as mock_minio_class:
            mock_client = MagicMock()
            mock_minio_class.return_value = mock_client

            storage = MinIOStorage(mock_minio_settings)
            content = b"test content"
            file = io.BytesIO(content)

            await storage.upload(file, "doc.pdf", "application/pdf")

            mock_client.put_object.assert_called_once()
            call_kwargs = mock_client.put_object.call_args
            assert call_kwargs.kwargs["bucket_name"] == "test-bucket"
            assert call_kwargs.kwargs["content_type"] == "application/pdf"
            assert call_kwargs.kwargs["length"] == len(content)

    async def test_upload_raises_error_on_failure(
        self, mock_minio_settings: MinIOSettings
    ) -> None:
        """Test upload raises StorageUploadError on failure."""
        with patch(
            "src.infrastructure.storage.minio_storage.Minio"
        ) as mock_minio_class:
            mock_client = MagicMock()
            mock_client.put_object.side_effect = S3Error(
                code="AccessDenied",
                message="Access Denied",
                resource="/test-bucket/test.txt",
                request_id="123",
                host_id="456",
                response=MagicMock(status=403),
            )
            mock_minio_class.return_value = mock_client

            storage = MinIOStorage(mock_minio_settings)
            file = io.BytesIO(b"test")

            with pytest.raises(StorageUploadError) as exc_info:
                await storage.upload(file, "test.txt", "text/plain")

            assert "test.txt" in str(exc_info.value)

    async def test_download_returns_bytesio(
        self, mock_minio_settings: MinIOSettings
    ) -> None:
        """Test download returns BytesIO with file content."""
        with patch(
            "src.infrastructure.storage.minio_storage.Minio"
        ) as mock_minio_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.read.return_value = b"downloaded content"
            mock_client.get_object.return_value = mock_response
            mock_minio_class.return_value = mock_client

            storage = MinIOStorage(mock_minio_settings)
            result = await storage.download("documents/uuid/file.txt")

            assert result.read() == b"downloaded content"
            mock_response.close.assert_called_once()
            mock_response.release_conn.assert_called_once()

    async def test_download_raises_not_found_error(
        self, mock_minio_settings: MinIOSettings
    ) -> None:
        """Test download raises StorageNotFoundError when file doesn't exist."""
        with patch(
            "src.infrastructure.storage.minio_storage.Minio"
        ) as mock_minio_class:
            mock_client = MagicMock()
            mock_client.get_object.side_effect = S3Error(
                code="NoSuchKey",
                message="Not Found",
                resource="/test-bucket/missing.txt",
                request_id="123",
                host_id="456",
                response=MagicMock(status=404),
            )
            mock_minio_class.return_value = mock_client

            storage = MinIOStorage(mock_minio_settings)

            with pytest.raises(StorageNotFoundError):
                await storage.download("documents/uuid/missing.txt")

    async def test_exists_returns_true_when_file_exists(
        self, mock_minio_settings: MinIOSettings
    ) -> None:
        """Test exists returns True when stat_object succeeds."""
        with patch(
            "src.infrastructure.storage.minio_storage.Minio"
        ) as mock_minio_class:
            mock_client = MagicMock()
            mock_minio_class.return_value = mock_client

            storage = MinIOStorage(mock_minio_settings)
            result = await storage.exists("documents/uuid/file.txt")

            assert result is True
            mock_client.stat_object.assert_called_once()

    async def test_exists_returns_false_when_file_not_found(
        self, mock_minio_settings: MinIOSettings
    ) -> None:
        """Test exists returns False when file doesn't exist."""
        with patch(
            "src.infrastructure.storage.minio_storage.Minio"
        ) as mock_minio_class:
            mock_client = MagicMock()
            mock_client.stat_object.side_effect = S3Error(
                code="NoSuchKey",
                message="Not Found",
                resource="/test-bucket/missing.txt",
                request_id="123",
                host_id="456",
                response=MagicMock(status=404),
            )
            mock_minio_class.return_value = mock_client

            storage = MinIOStorage(mock_minio_settings)
            result = await storage.exists("documents/uuid/missing.txt")

            assert result is False

    async def test_delete_returns_true_on_success(
        self, mock_minio_settings: MinIOSettings
    ) -> None:
        """Test delete returns True when file is deleted."""
        with patch(
            "src.infrastructure.storage.minio_storage.Minio"
        ) as mock_minio_class:
            mock_client = MagicMock()
            mock_minio_class.return_value = mock_client

            storage = MinIOStorage(mock_minio_settings)
            result = await storage.delete("documents/uuid/file.txt")

            assert result is True
            mock_client.remove_object.assert_called_once()

    async def test_delete_returns_false_when_not_found(
        self, mock_minio_settings: MinIOSettings
    ) -> None:
        """Test delete returns False when file doesn't exist."""
        with patch(
            "src.infrastructure.storage.minio_storage.Minio"
        ) as mock_minio_class:
            mock_client = MagicMock()
            mock_client.stat_object.side_effect = S3Error(
                code="NoSuchKey",
                message="Not Found",
                resource="/test-bucket/missing.txt",
                request_id="123",
                host_id="456",
                response=MagicMock(status=404),
            )
            mock_minio_class.return_value = mock_client

            storage = MinIOStorage(mock_minio_settings)
            result = await storage.delete("documents/uuid/missing.txt")

            assert result is False
            mock_client.remove_object.assert_not_called()
