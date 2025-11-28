"""Pytest configuration and fixtures."""

import pytest

from config import MinIOSettings


@pytest.fixture
def minio_settings():
    """Create MinIO settings for testing."""
    return MinIOSettings(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",  # pragma: allowlist secret
        bucket_name="test-documents",
        secure=False,
    )
