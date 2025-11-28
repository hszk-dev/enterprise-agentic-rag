"""Unit tests for Generation service with mocked dependencies."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.application.services.generation_service import (
    GenerationMetrics,
    GenerationService,
)
from src.domain.entities import Chunk, GenerationResult, Query, SearchResult
from src.domain.exceptions import LLMError
from src.domain.value_objects import TokenUsage


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    mock = MagicMock()
    mock.generate = AsyncMock()
    mock.generate_stream = AsyncMock()
    return mock


@pytest.fixture
def sample_query():
    """Create a sample query for testing."""
    return Query.create(
        text="What is Python?",
        top_k=10,
        rerank_top_n=5,
    )


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing."""
    chunks = [
        Chunk.create(
            document_id=uuid4(),
            content="Python is a high-level programming language.",
            chunk_index=0,
            start_char=0,
            end_char=45,
            metadata={"filename": "python_intro.pdf"},
        ),
        Chunk.create(
            document_id=uuid4(),
            content="Python was created by Guido van Rossum in 1991.",
            chunk_index=1,
            start_char=46,
            end_char=93,
            metadata={"filename": "python_history.pdf"},
        ),
        Chunk.create(
            document_id=uuid4(),
            content="Python is widely used for web development, data science, and AI.",
            chunk_index=2,
            start_char=94,
            end_char=158,
            metadata={},  # No filename
        ),
    ]

    return [
        SearchResult(chunk=chunks[0], score=0.95, rerank_score=0.98, rank=1),
        SearchResult(chunk=chunks[1], score=0.90, rerank_score=0.92, rank=2),
        SearchResult(chunk=chunks[2], score=0.85, rerank_score=0.88, rank=3),
    ]


def create_mock_generation_result(
    query: Query,
    answer: str = "Python is a programming language.",
    model: str = "gpt-4o",
) -> GenerationResult:
    """Create a mock generation result."""
    return GenerationResult.create(
        query=query,
        answer=answer,
        sources=[],
        usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model=model,
        ),
        model=model,
        latency_ms=500.0,
    )


@pytest.mark.unit
class TestGenerationServiceUnit:
    """Unit tests for GenerationService class."""

    def test_init_default_values(self, mock_llm_service: MagicMock) -> None:
        """Test initialization with default values."""
        service = GenerationService(llm_service=mock_llm_service)

        assert service._max_context_length == 4000
        assert service._context_separator == "\n\n---\n\n"

    def test_init_custom_values(self, mock_llm_service: MagicMock) -> None:
        """Test initialization with custom values."""
        service = GenerationService(
            llm_service=mock_llm_service,
            max_context_length=2000,
            context_separator="\n---\n",
        )

        assert service._max_context_length == 2000
        assert service._context_separator == "\n---\n"

    def test_format_context_with_results(
        self,
        mock_llm_service: MagicMock,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test _format_context formats search results correctly."""
        service = GenerationService(llm_service=mock_llm_service)
        passages = service._format_context(sample_search_results)

        assert len(passages) == 3
        assert "[Source: python_intro.pdf]" in passages[0]
        assert "Python is a high-level programming language" in passages[0]
        assert "[Source: python_history.pdf]" in passages[1]
        # Third chunk has no filename, so no source prefix
        assert "[Source:" not in passages[2]

    def test_format_context_empty_results(
        self,
        mock_llm_service: MagicMock,
    ) -> None:
        """Test _format_context with empty results."""
        service = GenerationService(llm_service=mock_llm_service)
        passages = service._format_context([])

        assert passages == []

    def test_format_context_truncates_long_content(
        self,
        mock_llm_service: MagicMock,
    ) -> None:
        """Test _format_context truncates when exceeding max length."""
        service = GenerationService(
            llm_service=mock_llm_service,
            max_context_length=500,  # Enough for first chunk, not second
        )

        chunks = [
            Chunk.create(
                document_id=uuid4(),
                content="A" * 200,
                chunk_index=0,
                start_char=0,
                end_char=200,
            ),
            Chunk.create(
                document_id=uuid4(),
                content="B" * 400,  # This would exceed limit
                chunk_index=1,
                start_char=200,
                end_char=600,
            ),
        ]
        results = [
            SearchResult(chunk=chunks[0], score=0.9),
            SearchResult(chunk=chunks[1], score=0.8),
        ]

        passages = service._format_context(results)

        # First passage should be included, second truncated or excluded
        assert len(passages) >= 1
        assert len(passages) <= 2
        total_length = sum(len(p) for p in passages)
        assert total_length <= 500

    def test_estimate_tokens(self, mock_llm_service: MagicMock) -> None:
        """Test token estimation."""
        service = GenerationService(llm_service=mock_llm_service)

        # ~4 characters per token
        assert service._estimate_tokens("") == 0
        assert service._estimate_tokens("Hello") == 1  # 5 chars / 4 = 1
        assert service._estimate_tokens("Hello World!") == 3  # 12 chars / 4 = 3

    async def test_generate_success(
        self,
        mock_llm_service: MagicMock,
        sample_query: Query,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test generate returns GenerationResult on success."""
        mock_result = create_mock_generation_result(
            query=sample_query,
            answer="Python is a versatile programming language.",
        )
        mock_llm_service.generate.return_value = mock_result

        service = GenerationService(llm_service=mock_llm_service)
        result, metrics = await service.generate(
            query=sample_query,
            search_results=sample_search_results,
        )

        # Verify result
        assert result.answer == "Python is a versatile programming language."
        assert result.query.text == sample_query.text
        assert len(result.sources) == 3
        assert result.model == "gpt-4o"

        # Verify metrics
        assert isinstance(metrics, GenerationMetrics)
        assert metrics.sources_count == 3
        assert metrics.total_latency_ms > 0
        assert metrics.llm_latency_ms > 0

        # Verify LLM service was called with context
        mock_llm_service.generate.assert_called_once()
        call_kwargs = mock_llm_service.generate.call_args[1]
        assert call_kwargs["prompt"] == sample_query.text
        assert len(call_kwargs["context"]) == 3

    async def test_generate_with_empty_results(
        self,
        mock_llm_service: MagicMock,
        sample_query: Query,
    ) -> None:
        """Test generate with no search results."""
        mock_result = create_mock_generation_result(
            query=sample_query,
            answer="I don't have enough context to answer.",
        )
        mock_llm_service.generate.return_value = mock_result

        service = GenerationService(llm_service=mock_llm_service)
        result, metrics = await service.generate(
            query=sample_query,
            search_results=[],
        )

        assert result.answer == "I don't have enough context to answer."
        assert len(result.sources) == 0
        assert metrics.sources_count == 0

        # Verify empty context was passed
        call_kwargs = mock_llm_service.generate.call_args[1]
        assert call_kwargs["context"] == []

    async def test_generate_with_custom_params(
        self,
        mock_llm_service: MagicMock,
        sample_query: Query,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test generate respects temperature and max_tokens."""
        mock_result = create_mock_generation_result(query=sample_query)
        mock_llm_service.generate.return_value = mock_result

        service = GenerationService(llm_service=mock_llm_service)
        await service.generate(
            query=sample_query,
            search_results=sample_search_results,
            temperature=0.7,
            max_tokens=500,
        )

        call_kwargs = mock_llm_service.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 500

    async def test_generate_llm_error_propagates(
        self,
        mock_llm_service: MagicMock,
        sample_query: Query,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test generate propagates LLMError."""
        mock_llm_service.generate.side_effect = LLMError("LLM API failed")

        service = GenerationService(llm_service=mock_llm_service)

        with pytest.raises(LLMError) as exc_info:
            await service.generate(
                query=sample_query,
                search_results=sample_search_results,
            )

        assert "LLM API failed" in str(exc_info.value)

    async def test_generate_generic_error_wrapped(
        self,
        mock_llm_service: MagicMock,
        sample_query: Query,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test generate wraps generic exceptions in LLMError."""
        mock_llm_service.generate.side_effect = RuntimeError("Unexpected error")

        service = GenerationService(llm_service=mock_llm_service)

        with pytest.raises(LLMError) as exc_info:
            await service.generate(
                query=sample_query,
                search_results=sample_search_results,
            )

        assert "Generation failed" in str(exc_info.value)

    async def test_generate_stream_success(
        self,
        mock_llm_service: MagicMock,
        sample_query: Query,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test generate_stream yields text chunks."""

        async def mock_stream(*args, **kwargs):
            for chunk in ["Hello", " ", "World", "!"]:
                yield chunk

        # Return the async generator directly (not a coroutine)
        mock_llm_service.generate_stream = mock_stream

        service = GenerationService(llm_service=mock_llm_service)
        chunks = []
        async for chunk in service.generate_stream(
            query=sample_query,
            search_results=sample_search_results,
        ):
            chunks.append(chunk)

        assert chunks == ["Hello", " ", "World", "!"]
        assert "".join(chunks) == "Hello World!"

    async def test_generate_stream_llm_error(
        self,
        mock_llm_service: MagicMock,
        sample_query: Query,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test generate_stream propagates LLMError."""

        async def mock_stream_error(*args, **kwargs):
            raise LLMError("Stream failed")
            yield  # Make it a generator (never reached)

        mock_llm_service.generate_stream = mock_stream_error

        service = GenerationService(llm_service=mock_llm_service)

        with pytest.raises(LLMError) as exc_info:
            async for _ in service.generate_stream(
                query=sample_query,
                search_results=sample_search_results,
            ):
                pass

        assert "Stream failed" in str(exc_info.value)

    async def test_generate_stream_generic_error(
        self,
        mock_llm_service: MagicMock,
        sample_query: Query,
        sample_search_results: list[SearchResult],
    ) -> None:
        """Test generate_stream wraps generic errors."""

        async def mock_stream_error(*args, **kwargs):
            raise RuntimeError("Unexpected")
            yield  # Make it a generator (never reached)

        mock_llm_service.generate_stream = mock_stream_error

        service = GenerationService(llm_service=mock_llm_service)

        with pytest.raises(LLMError) as exc_info:
            async for _ in service.generate_stream(
                query=sample_query,
                search_results=sample_search_results,
            ):
                pass

        assert "Streaming generation failed" in str(exc_info.value)

    async def test_generate_with_no_context(
        self,
        mock_llm_service: MagicMock,
        sample_query: Query,
    ) -> None:
        """Test generate_with_no_context helper method."""
        mock_result = create_mock_generation_result(
            query=sample_query,
            answer="Without context answer.",
        )
        mock_llm_service.generate.return_value = mock_result

        service = GenerationService(llm_service=mock_llm_service)
        result, metrics = await service.generate_with_no_context(
            query=sample_query,
            temperature=0.5,
            max_tokens=256,
        )

        assert result.answer == "Without context answer."
        assert len(result.sources) == 0
        assert metrics.sources_count == 0

        call_kwargs = mock_llm_service.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["context"] == []
