"""Integration test fixtures.

These fixtures require running external services (MinIO, PostgreSQL, Qdrant).
Start services with: make up
"""

import contextlib
import io
from pathlib import Path
from uuid import uuid4

import pytest

from config import ChunkingSettings, MinIOSettings, QdrantSettings
from src.domain.entities import Document
from src.domain.value_objects import ContentType, SparseVector
from src.infrastructure.storage import MinIOStorage
from src.infrastructure.vectorstores import QdrantVectorStore

# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_txt_path(fixtures_dir: Path) -> Path:
    """Return path to sample.txt fixture file."""
    return fixtures_dir / "sample.txt"


@pytest.fixture
def sample_md_path(fixtures_dir: Path) -> Path:
    """Return path to sample.md fixture file."""
    return fixtures_dir / "sample.md"


@pytest.fixture
def sample_pdf_path(fixtures_dir: Path) -> Path:
    """Return path to sample.pdf fixture file."""
    return fixtures_dir / "sample.pdf"


@pytest.fixture
def sample_docx_path(fixtures_dir: Path) -> Path:
    """Return path to sample.docx fixture file."""
    return fixtures_dir / "sample.docx"


# =============================================================================
# Content Fixtures
# =============================================================================


@pytest.fixture
def sample_txt_content(sample_txt_path: Path) -> str:
    """Read sample.txt content."""
    return sample_txt_path.read_text()


@pytest.fixture
def sample_md_content(sample_md_path: Path) -> str:
    """Read sample.md content."""
    return sample_md_path.read_text()


@pytest.fixture
def sample_txt_file(sample_txt_content: str) -> io.BytesIO:
    """Create a BytesIO object from sample.txt content."""
    return io.BytesIO(sample_txt_content.encode("utf-8"))


@pytest.fixture
def sample_md_file(sample_md_content: str) -> io.BytesIO:
    """Create a BytesIO object from sample.md content."""
    return io.BytesIO(sample_md_content.encode("utf-8"))


# =============================================================================
# Document Fixtures
# =============================================================================


@pytest.fixture
def sample_txt_document() -> Document:
    """Create a sample TXT document for testing."""
    return Document.create(
        filename="integration-test.txt",
        content_type=ContentType.TXT,
        size_bytes=2048,
        metadata={"source": "integration-test", "type": "text"},
    )


@pytest.fixture
def sample_md_document() -> Document:
    """Create a sample Markdown document for testing."""
    return Document.create(
        filename="integration-test.md",
        content_type=ContentType.MD,
        size_bytes=1024,
        metadata={"source": "integration-test", "type": "markdown"},
    )


# =============================================================================
# Settings Fixtures (Integration-specific)
# =============================================================================


@pytest.fixture
def integration_minio_settings() -> MinIOSettings:
    """Create MinIO settings for integration tests with unique bucket."""
    return MinIOSettings(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",  # pragma: allowlist secret
        bucket_name=f"test-integration-{uuid4().hex[:8]}",
        secure=False,
    )


@pytest.fixture
def integration_qdrant_settings() -> QdrantSettings:
    """Create Qdrant settings for integration tests with unique collection."""
    return QdrantSettings(
        host="localhost",
        port=6333,
        grpc_port=6334,
        collection_name=f"test-integration-{uuid4().hex[:8]}",
        use_grpc=False,
        api_key=None,
    )


@pytest.fixture
def integration_chunking_settings() -> ChunkingSettings:
    """Chunking settings optimized for integration tests."""
    return ChunkingSettings(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# =============================================================================
# Service Fixtures
# =============================================================================


@pytest.fixture
async def integration_minio_storage(
    integration_minio_settings: MinIOSettings,
) -> MinIOStorage:
    """Create and initialize MinIO storage for integration tests."""
    storage = MinIOStorage(integration_minio_settings)
    await storage.initialize()
    yield storage
    # Cleanup: Note that bucket contents should be cleaned by tests
    await storage.close()


@pytest.fixture
async def integration_qdrant_store(
    integration_qdrant_settings: QdrantSettings,
) -> QdrantVectorStore:
    """Create and initialize Qdrant store for integration tests."""
    store = QdrantVectorStore(integration_qdrant_settings, embedding_dim=1536)
    await store.initialize()
    yield store
    # Cleanup: delete collection
    with contextlib.suppress(Exception):
        await store._client.delete_collection(
            integration_qdrant_settings.collection_name
        )
    await store.close()


# =============================================================================
# Mock Fixtures for External APIs
# =============================================================================


class MockSparseEmbedding:
    """Mock sparse embedding service for integration tests.

    Generates deterministic sparse vectors without requiring FastEmbed.
    """

    async def embed_text(self, text: str) -> SparseVector:
        """Generate a mock sparse embedding."""
        # Create deterministic indices based on text hash
        text_hash = hash(text) % 10000
        indices = tuple(sorted([text_hash % 1000, (text_hash + 100) % 1000, 500]))
        values = (0.5, 0.3, 0.2)
        return SparseVector(indices=indices, values=values)

    async def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        """Generate mock sparse embeddings for multiple texts."""
        return [await self.embed_text(text) for text in texts]


@pytest.fixture
def mock_sparse_embedding() -> MockSparseEmbedding:
    """Create a mock sparse embedding service."""
    return MockSparseEmbedding()


class MockDenseEmbedding:
    """Mock dense embedding service for integration tests.

    Generates deterministic dense vectors without requiring OpenAI API.
    """

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        """Generate a mock dense embedding."""
        # Create deterministic embedding based on text hash
        text_hash = hash(text)
        embedding = []
        for i in range(self._dimension):
            # Generate pseudo-random but deterministic values
            val = ((text_hash + i * 31) % 1000) / 1000.0
            embedding.append(val)
        return embedding

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate mock dense embeddings for multiple texts."""
        return [await self.embed_text(text) for text in texts]


@pytest.fixture
def mock_dense_embedding() -> MockDenseEmbedding:
    """Create a mock dense embedding service."""
    return MockDenseEmbedding(dimension=1536)
