"""Pytest configuration and fixtures."""

import os

import pytest

from config import (
    CohereSettings,
    DatabaseSettings,
    MinIOSettings,
    OpenAISettings,
    QdrantSettings,
)


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


@pytest.fixture
def qdrant_settings():
    """Create Qdrant settings for testing."""
    return QdrantSettings(
        host="localhost",
        port=6333,
        grpc_port=6334,
        collection_name="test-documents",
        use_grpc=False,
        api_key=None,
    )


@pytest.fixture
def database_settings():
    """Create database settings for testing.

    Uses DATABASE_URL environment variable if set (for CI),
    otherwise uses default local development settings.
    """
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        # Parse DATABASE_URL format  # pragma: allowlist secret
        # Remove the driver prefix for parsing
        url = database_url.replace("postgresql+asyncpg://", "")
        user_pass, host_db = url.split("@")
        user, password = user_pass.split(":")
        host_port, database = host_db.split("/")
        host, port = host_port.split(":") if ":" in host_port else (host_port, "5432")
        return DatabaseSettings(
            host=host,
            port=int(port),
            user=user,
            password=password,  # pragma: allowlist secret
            database=database,
            pool_size=5,
            max_overflow=10,
        )
    return DatabaseSettings(
        host="localhost",
        port=5432,
        user="rag_user",
        password="rag_password",  # pragma: allowlist secret
        database="rag_db",
        pool_size=5,
        max_overflow=10,
    )


@pytest.fixture
def openai_settings():
    """Create OpenAI settings for testing.

    Uses environment variable for API key, or a placeholder for unit tests.
    """
    api_key = os.environ.get(
        "OPENAI_API_KEY", "test-api-key"
    )  # pragma: allowlist secret
    return OpenAISettings(
        api_key=api_key,  # pragma: allowlist secret
        model="gpt-4o",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        max_retries=3,
        timeout=30.0,
    )


@pytest.fixture
def cohere_settings():
    """Create Cohere settings for testing.

    Uses environment variable for API key, or a placeholder for unit tests.
    """
    api_key = os.environ.get(
        "COHERE_API_KEY", "test-api-key"
    )  # pragma: allowlist secret
    return CohereSettings(
        api_key=api_key,  # pragma: allowlist secret
        rerank_model="rerank-v3.5",
        max_retries=3,
        timeout=30.0,
    )
