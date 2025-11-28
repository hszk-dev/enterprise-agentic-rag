"""Unit tests for domain entities."""

from datetime import UTC
from uuid import UUID, uuid4

import pytest

from src.domain.entities import (
    Chunk,
    Document,
    GenerationResult,
    Query,
    SearchResult,
)
from src.domain.value_objects import (
    ContentType,
    DocumentStatus,
    SparseVector,
    TokenUsage,
)


class TestDocument:
    """Tests for Document entity."""

    def test_create_factory(self) -> None:
        """Test Document.create factory method."""
        doc = Document.create(
            filename="report.pdf",
            content_type=ContentType.PDF,
            size_bytes=1024,
            metadata={"author": "Test"},
        )

        assert isinstance(doc.id, UUID)
        assert doc.filename == "report.pdf"
        assert doc.content_type == ContentType.PDF
        assert doc.size_bytes == 1024
        assert doc.status == DocumentStatus.PENDING
        assert doc.metadata == {"author": "Test"}
        assert doc.error_message is None
        assert doc.chunk_count == 0
        assert doc.file_path is None

    def test_create_strips_filename(self) -> None:
        """Test filename whitespace is stripped."""
        doc = Document.create(
            filename="  report.pdf  ",
            content_type=ContentType.PDF,
            size_bytes=100,
        )
        assert doc.filename == "report.pdf"

    def test_create_empty_filename_raises(self) -> None:
        """Test empty filename raises ValueError."""
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            Document.create(
                filename="",
                content_type=ContentType.PDF,
                size_bytes=100,
            )

    def test_create_whitespace_filename_raises(self) -> None:
        """Test whitespace-only filename raises ValueError."""
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            Document.create(
                filename="   ",
                content_type=ContentType.PDF,
                size_bytes=100,
            )

    def test_create_negative_size_raises(self) -> None:
        """Test negative size_bytes raises ValueError."""
        with pytest.raises(ValueError, match="Size must be non-negative"):
            Document.create(
                filename="test.pdf",
                content_type=ContentType.PDF,
                size_bytes=-1,
            )

    def test_mark_processing(self) -> None:
        """Test transition to PROCESSING status."""
        doc = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=100,
        )
        original_updated = doc.updated_at

        doc.mark_processing()

        assert doc.status == DocumentStatus.PROCESSING
        assert doc.updated_at >= original_updated

    def test_mark_completed(self) -> None:
        """Test transition to COMPLETED status."""
        doc = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=100,
        )
        doc.mark_processing()
        doc.mark_completed(chunk_count=10)

        assert doc.status == DocumentStatus.COMPLETED
        assert doc.chunk_count == 10
        assert doc.error_message is None

    def test_mark_failed(self) -> None:
        """Test transition to FAILED status."""
        doc = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=100,
        )
        doc.mark_processing()
        doc.mark_failed("Parse error: invalid PDF")

        assert doc.status == DocumentStatus.FAILED
        assert doc.error_message == "Parse error: invalid PDF"

    def test_set_file_path(self) -> None:
        """Test setting blob storage path."""
        doc = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=100,
        )
        doc.set_file_path("documents/abc123/test.pdf")

        assert doc.file_path == "documents/abc123/test.pdf"

    def test_created_at_is_utc(self) -> None:
        """Test created_at timestamp is UTC."""
        doc = Document.create(
            filename="test.pdf",
            content_type=ContentType.PDF,
            size_bytes=100,
        )
        assert doc.created_at.tzinfo == UTC


class TestChunk:
    """Tests for Chunk entity."""

    def test_create_factory(self) -> None:
        """Test Chunk.create factory method."""
        doc_id = uuid4()
        chunk = Chunk.create(
            document_id=doc_id,
            content="This is the chunk content.",
            chunk_index=0,
            start_char=0,
            end_char=26,
            metadata={"page": 1},
        )

        assert isinstance(chunk.id, UUID)
        assert chunk.document_id == doc_id
        assert chunk.content == "This is the chunk content."
        assert chunk.chunk_index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 26
        assert chunk.metadata == {"page": 1}
        assert chunk.dense_embedding is None
        assert chunk.sparse_embedding is None

    def test_char_count_property(self) -> None:
        """Test char_count returns content length."""
        chunk = Chunk.create(
            document_id=uuid4(),
            content="Hello",
            chunk_index=0,
            start_char=0,
            end_char=5,
        )
        assert chunk.char_count == 5

    def test_set_embeddings(self) -> None:
        """Test setting embeddings."""
        chunk = Chunk.create(
            document_id=uuid4(),
            content="Test",
            chunk_index=0,
            start_char=0,
            end_char=4,
        )
        dense = [0.1, 0.2, 0.3]
        sparse = SparseVector(indices=(1, 2), values=(0.5, 0.5))

        chunk.set_embeddings(dense=dense, sparse=sparse)

        assert chunk.dense_embedding == dense
        assert chunk.sparse_embedding == sparse

    def test_set_embeddings_partial(self) -> None:
        """Test setting only dense embedding."""
        chunk = Chunk.create(
            document_id=uuid4(),
            content="Test",
            chunk_index=0,
            start_char=0,
            end_char=4,
        )
        dense = [0.1, 0.2, 0.3]

        chunk.set_embeddings(dense=dense)

        assert chunk.dense_embedding == dense
        assert chunk.sparse_embedding is None

    def test_create_empty_content_raises(self) -> None:
        """Test empty content raises ValueError."""
        with pytest.raises(ValueError, match="Chunk content cannot be empty"):
            Chunk.create(
                document_id=uuid4(),
                content="",
                chunk_index=0,
                start_char=0,
                end_char=0,
            )

    def test_create_negative_chunk_index_raises(self) -> None:
        """Test negative chunk_index raises ValueError."""
        with pytest.raises(ValueError, match="chunk_index must be >= 0"):
            Chunk.create(
                document_id=uuid4(),
                content="Test content",
                chunk_index=-1,
                start_char=0,
                end_char=12,
            )

    def test_create_invalid_char_range_raises(self) -> None:
        """Test end_char <= start_char raises ValueError."""
        with pytest.raises(ValueError, match=r"end_char .* must be > start_char"):
            Chunk.create(
                document_id=uuid4(),
                content="Test",
                chunk_index=0,
                start_char=10,
                end_char=5,
            )

    def test_create_equal_char_range_raises(self) -> None:
        """Test end_char == start_char raises ValueError."""
        with pytest.raises(ValueError, match=r"end_char .* must be > start_char"):
            Chunk.create(
                document_id=uuid4(),
                content="Test",
                chunk_index=0,
                start_char=5,
                end_char=5,
            )

    def test_create_with_default_metadata(self) -> None:
        """Test creation with metadata=None uses empty dict."""
        chunk = Chunk.create(
            document_id=uuid4(),
            content="Test content",
            chunk_index=0,
            start_char=0,
            end_char=12,
            metadata=None,
        )
        assert chunk.metadata == {}


class TestQuery:
    """Tests for Query entity."""

    def test_create_factory(self) -> None:
        """Test Query.create factory method."""
        query = Query.create(
            text="What is RAG?",
            top_k=5,
            rerank_top_n=3,
            alpha=0.7,
        )

        assert isinstance(query.id, UUID)
        assert query.text == "What is RAG?"
        assert query.top_k == 5
        assert query.rerank_top_n == 3
        assert query.alpha == 0.7

    def test_create_strips_text(self) -> None:
        """Test query text whitespace is stripped."""
        query = Query.create(text="  What is RAG?  ")
        assert query.text == "What is RAG?"

    def test_create_empty_text_raises(self) -> None:
        """Test empty text raises ValueError."""
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            Query.create(text="")

    def test_create_invalid_top_k_raises(self) -> None:
        """Test top_k < 1 raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            Query.create(text="test", top_k=0)

    def test_create_invalid_rerank_top_n_raises(self) -> None:
        """Test rerank_top_n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="rerank_top_n must be >= 1"):
            Query.create(text="test", rerank_top_n=0)

    def test_create_invalid_alpha_raises(self) -> None:
        """Test alpha out of range raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            Query.create(text="test", alpha=1.5)

        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            Query.create(text="test", alpha=-0.1)

    def test_create_rerank_top_n_exceeds_top_k_raises(self) -> None:
        """Test rerank_top_n > top_k raises ValueError."""
        with pytest.raises(ValueError, match=r"rerank_top_n .* cannot exceed top_k"):
            Query.create(text="test", top_k=5, rerank_top_n=10)

    def test_create_with_user_context(self) -> None:
        """Test creation with user and session IDs."""
        query = Query.create(
            text="test query",
            user_id="user-123",
            session_id="session-456",
        )
        assert query.user_id == "user-123"
        assert query.session_id == "session-456"


class TestSearchResult:
    """Tests for SearchResult entity."""

    @pytest.fixture
    def sample_chunk(self) -> Chunk:
        """Create a sample chunk for tests."""
        return Chunk.create(
            document_id=uuid4(),
            content="Sample content",
            chunk_index=0,
            start_char=0,
            end_char=14,
        )

    def test_final_score_with_rerank(self, sample_chunk: Chunk) -> None:
        """Test final_score returns rerank_score when available."""
        result = SearchResult(
            chunk=sample_chunk,
            score=0.8,
            rerank_score=0.95,
            rank=1,
        )
        assert result.final_score == 0.95

    def test_final_score_without_rerank(self, sample_chunk: Chunk) -> None:
        """Test final_score returns raw score when no rerank."""
        result = SearchResult(
            chunk=sample_chunk,
            score=0.8,
            rerank_score=None,
            rank=1,
        )
        assert result.final_score == 0.8

    def test_display_score_with_rerank(self, sample_chunk: Chunk) -> None:
        """Test display_score returns rerank_score directly."""
        result = SearchResult(
            chunk=sample_chunk,
            score=0.8,
            rerank_score=0.95,
        )
        assert result.display_score == 0.95

    def test_display_score_clamps_high(self, sample_chunk: Chunk) -> None:
        """Test display_score clamps values > 1.0."""
        result = SearchResult(
            chunk=sample_chunk,
            score=1.5,
            rerank_score=None,
        )
        assert result.display_score == 1.0

    def test_display_score_clamps_low(self, sample_chunk: Chunk) -> None:
        """Test display_score clamps values < 0.0."""
        result = SearchResult(
            chunk=sample_chunk,
            score=-0.5,
            rerank_score=None,
        )
        assert result.display_score == 0.0

    def test_display_score_at_boundary_zero(self, sample_chunk: Chunk) -> None:
        """Test display_score handles exactly 0.0."""
        result = SearchResult(
            chunk=sample_chunk,
            score=0.0,
            rerank_score=None,
        )
        assert result.display_score == 0.0

    def test_display_score_at_boundary_one(self, sample_chunk: Chunk) -> None:
        """Test display_score handles exactly 1.0."""
        result = SearchResult(
            chunk=sample_chunk,
            score=1.0,
            rerank_score=None,
        )
        assert result.display_score == 1.0


class TestGenerationResult:
    """Tests for GenerationResult entity."""

    @pytest.fixture
    def sample_query(self) -> Query:
        """Create a sample query for tests."""
        return Query.create(text="What is RAG?")

    @pytest.fixture
    def sample_sources(self) -> list[SearchResult]:
        """Create sample search results for tests."""
        chunk = Chunk.create(
            document_id=uuid4(),
            content="RAG stands for Retrieval-Augmented Generation.",
            chunk_index=0,
            start_char=0,
            end_char=46,
        )
        return [SearchResult(chunk=chunk, score=0.9, rank=1)]

    def test_create_factory(
        self,
        sample_query: Query,
        sample_sources: list[SearchResult],
    ) -> None:
        """Test GenerationResult.create factory method."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        result = GenerationResult.create(
            query=sample_query,
            answer="RAG combines retrieval and generation.",
            sources=sample_sources,
            usage=usage,
            model="gpt-4o",
            latency_ms=500.5,
        )

        assert isinstance(result.id, UUID)
        assert result.query == sample_query
        assert result.answer == "RAG combines retrieval and generation."
        assert result.sources == sample_sources
        assert result.usage == usage
        assert result.model == "gpt-4o"
        assert result.latency_ms == 500.5

    def test_source_count_property(
        self,
        sample_query: Query,
        sample_sources: list[SearchResult],
    ) -> None:
        """Test source_count returns number of sources."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        result = GenerationResult.create(
            query=sample_query,
            answer="Answer",
            sources=sample_sources,
            usage=usage,
            model="gpt-4o",
            latency_ms=100,
        )
        assert result.source_count == 1

    def test_estimated_cost_property(
        self,
        sample_query: Query,
        sample_sources: list[SearchResult],
    ) -> None:
        """Test estimated_cost_usd delegates to usage."""
        usage = TokenUsage(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
            model="gpt-5",
        )
        result = GenerationResult.create(
            query=sample_query,
            answer="Answer",
            sources=sample_sources,
            usage=usage,
            model="gpt-5",
            latency_ms=100,
        )
        # GPT-5: $1.25/1M input + $10.00/1M output = $11.25
        assert result.estimated_cost_usd == pytest.approx(11.25)

    def test_created_at_is_utc(
        self,
        sample_query: Query,
        sample_sources: list[SearchResult],
    ) -> None:
        """Test created_at timestamp is UTC."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        result = GenerationResult.create(
            query=sample_query,
            answer="Answer",
            sources=sample_sources,
            usage=usage,
            model="gpt-4o",
            latency_ms=100,
        )
        assert result.created_at.tzinfo == UTC
