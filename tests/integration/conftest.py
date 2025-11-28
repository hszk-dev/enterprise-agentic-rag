"""Integration test fixtures.

These fixtures require running external services (MinIO, PostgreSQL, Qdrant).
Start services with: make up
"""

import contextlib
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import pytest

from config import ChunkingSettings, MinIOSettings, QdrantSettings
from src.domain.entities import Document
from src.domain.value_objects import ContentType, SparseVector, TokenUsage
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


# =============================================================================
# Cost Tracking for LLM Quality Tests
# =============================================================================

logger = logging.getLogger(__name__)


@dataclass
class TokenUsageTracker:
    """Track token usage across LLM quality tests.

    This class accumulates token usage from generation and evaluation calls,
    then calculates the estimated cost using TokenUsage.estimated_cost_usd.
    """

    # Generation (gpt-4o)
    generation_input_tokens: int = 0
    generation_output_tokens: int = 0

    # Evaluation/Judge (gpt-4o-mini)
    evaluation_input_tokens: int = 0
    evaluation_output_tokens: int = 0

    # Tracking metadata
    test_count: int = 0
    _test_names: list[str] = field(default_factory=list)

    def add_generation_usage(
        self,
        usage: TokenUsage | None = None,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Add token usage from generation (gpt-4o).

        Args:
            usage: TokenUsage object from GenerationResult.usage (preferred).
            prompt_tokens: Prompt tokens (fallback if usage not provided).
            completion_tokens: Completion tokens (fallback if usage not provided).
        """
        if usage is not None:
            self.generation_input_tokens += usage.prompt_tokens
            self.generation_output_tokens += usage.completion_tokens
        else:
            self.generation_input_tokens += prompt_tokens
            self.generation_output_tokens += completion_tokens

    def add_evaluation_usage(
        self,
        usage: TokenUsage | None = None,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Add token usage from evaluation/judge (gpt-4o-mini).

        Args:
            usage: TokenUsage object (if available).
            prompt_tokens: Prompt tokens (fallback if usage not provided).
            completion_tokens: Completion tokens (fallback if usage not provided).
        """
        if usage is not None:
            self.evaluation_input_tokens += usage.prompt_tokens
            self.evaluation_output_tokens += usage.completion_tokens
        else:
            self.evaluation_input_tokens += prompt_tokens
            self.evaluation_output_tokens += completion_tokens

    def record_test(self, test_name: str) -> None:
        """Record that a test was executed."""
        self.test_count += 1
        self._test_names.append(test_name)

    def calculate_cost(self) -> dict[str, float]:
        """Calculate estimated cost breakdown using TokenUsage.estimated_cost_usd."""
        generation_usage = TokenUsage(
            prompt_tokens=self.generation_input_tokens,
            completion_tokens=self.generation_output_tokens,
            total_tokens=self.generation_input_tokens + self.generation_output_tokens,
            model="gpt-4o",
        )
        evaluation_usage = TokenUsage(
            prompt_tokens=self.evaluation_input_tokens,
            completion_tokens=self.evaluation_output_tokens,
            total_tokens=self.evaluation_input_tokens + self.evaluation_output_tokens,
            model="gpt-4o-mini",
        )
        return {
            "generation_cost": generation_usage.estimated_cost_usd,
            "evaluation_cost": evaluation_usage.estimated_cost_usd,
            "total_cost": generation_usage.estimated_cost_usd
            + evaluation_usage.estimated_cost_usd,
        }

    def get_summary(self) -> str:
        """Generate a human-readable cost summary."""
        costs = self.calculate_cost()
        total_tokens = (
            self.generation_input_tokens
            + self.generation_output_tokens
            + self.evaluation_input_tokens
            + self.evaluation_output_tokens
        )

        return f"""
=== LLM Quality Test Cost Summary ===

Tests executed: {self.test_count}

Token Usage:
  Generation (gpt-4o):
    Input:  {self.generation_input_tokens:,} tokens
    Output: {self.generation_output_tokens:,} tokens
  Evaluation (gpt-4o-mini):
    Input:  {self.evaluation_input_tokens:,} tokens
    Output: {self.evaluation_output_tokens:,} tokens
  Total: {total_tokens:,} tokens

Estimated Cost:
  Generation:  ${costs["generation_cost"]:.4f}
  Evaluation:  ${costs["evaluation_cost"]:.4f}
  Total:       ${costs["total_cost"]:.4f}

=====================================
"""


@pytest.fixture(scope="session")
def token_usage_tracker(request: pytest.FixtureRequest) -> TokenUsageTracker:
    """Session-scoped token usage tracker.

    This fixture provides a shared tracker across all LLM quality tests
    in a session. Use it to record token usage from generation and
    evaluation calls.

    Example:
        async def test_something(token_usage_tracker, generation_service):
            result, metrics = await generation_service.generate(query, results)
            token_usage_tracker.add_generation_usage(
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
            )
    """
    tracker = TokenUsageTracker()
    # Store tracker in pytest config for access in hooks
    request.config._token_tracker = tracker  # type: ignore[attr-defined]
    return tracker


@pytest.fixture(autouse=True)
def track_llm_test(
    request: pytest.FixtureRequest,
    token_usage_tracker: TokenUsageTracker,
) -> None:
    """Auto-use fixture to track LLM quality test execution.

    This fixture automatically records each test that runs with the
    'llm_quality' marker.
    """
    yield

    # Only track tests with llm_quality marker
    if request.node.get_closest_marker("llm_quality") is not None:
        token_usage_tracker.record_test(request.node.name)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Print cost summary at end of test session."""
    tracker: TokenUsageTracker | None = getattr(session.config, "_token_tracker", None)

    if tracker is None or tracker.test_count == 0:
        return

    # Only print if there was actual token usage
    total_tokens = (
        tracker.generation_input_tokens
        + tracker.generation_output_tokens
        + tracker.evaluation_input_tokens
        + tracker.evaluation_output_tokens
    )

    if total_tokens > 0:
        print(tracker.get_summary())
        logger.info(tracker.get_summary())
