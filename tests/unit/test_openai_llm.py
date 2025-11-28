"""Unit tests for OpenAI LLM service with mocked dependencies."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import RateLimitError as OpenAIRateLimitError

from config import OpenAISettings
from src.domain.exceptions import LLMError, RateLimitError
from src.infrastructure.llm import OpenAILLMService


@pytest.fixture
def mock_openai_settings():
    """Create OpenAI settings for unit tests."""
    return OpenAISettings(
        api_key="test-api-key",  # pragma: allowlist secret
        model="gpt-4o",
        fallback_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        max_retries=3,
        timeout=30.0,
    )


def create_mock_completion_response(
    content: str = "Test response",
    model: str = "gpt-4o",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> MagicMock:
    """Create a mock chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    mock_response.usage.total_tokens = prompt_tokens + completion_tokens
    mock_response.model = model
    return mock_response


@pytest.mark.unit
class TestOpenAILLMServiceUnit:
    """Unit tests for OpenAILLMService class."""

    def test_init_with_default_system_prompt(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test initialization uses default system prompt."""
        with patch("src.infrastructure.llm.openai_llm.AsyncOpenAI"):
            service = OpenAILLMService(mock_openai_settings)
            assert "helpful assistant" in service._system_prompt.lower()

    def test_init_with_custom_system_prompt(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test initialization with custom system prompt."""
        with patch("src.infrastructure.llm.openai_llm.AsyncOpenAI"):
            custom_prompt = "You are a code reviewer."
            service = OpenAILLMService(
                mock_openai_settings, system_prompt=custom_prompt
            )
            assert service._system_prompt == custom_prompt

    def test_build_messages_with_context(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test _build_messages formats context correctly."""
        with patch("src.infrastructure.llm.openai_llm.AsyncOpenAI"):
            service = OpenAILLMService(mock_openai_settings)
            messages = service._build_messages(
                prompt="What is Python?",
                context=["Python is a language.", "It was created in 1991."],
            )

            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert "[1]" in messages[1]["content"]
            assert "[2]" in messages[1]["content"]
            assert "Python is a language." in messages[1]["content"]
            assert "Question: What is Python?" in messages[1]["content"]

    def test_build_messages_without_context(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test _build_messages without context."""
        with patch("src.infrastructure.llm.openai_llm.AsyncOpenAI"):
            service = OpenAILLMService(mock_openai_settings)
            messages = service._build_messages(prompt="Hello", context=[])

            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "Hello"

    async def test_generate_success(self, mock_openai_settings: OpenAISettings) -> None:
        """Test generate returns GenerationResult on success."""
        with patch(
            "src.infrastructure.llm.openai_llm.AsyncOpenAI"
        ) as mock_client_class:
            mock_response = create_mock_completion_response(
                content="Python is a programming language.",
                prompt_tokens=150,
                completion_tokens=20,
            )
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            service = OpenAILLMService(mock_openai_settings)
            result = await service.generate(
                prompt="What is Python?",
                context=["Python is a high-level language."],
            )

            assert result.answer == "Python is a programming language."
            assert result.model == "gpt-4o"
            assert result.usage.prompt_tokens == 150
            assert result.usage.completion_tokens == 20
            assert result.latency_ms > 0
            assert result.query.text == "What is Python?"
            mock_client.chat.completions.create.assert_called_once()

    async def test_generate_with_custom_temperature(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test generate respects temperature parameter."""
        with patch(
            "src.infrastructure.llm.openai_llm.AsyncOpenAI"
        ) as mock_client_class:
            mock_response = create_mock_completion_response()
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            service = OpenAILLMService(mock_openai_settings)
            await service.generate(
                prompt="Test",
                context=[],
                temperature=0.7,
                max_tokens=500,
            )

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 500

    async def test_generate_api_error_raises_llm_error(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test generate raises LLMError on API failure (both models fail)."""
        with patch(
            "src.infrastructure.llm.openai_llm.AsyncOpenAI"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            mock_client_class.return_value = mock_client

            service = OpenAILLMService(mock_openai_settings)

            with pytest.raises(LLMError) as exc_info:
                await service.generate(prompt="Test", context=[])

            assert "API Error" in str(exc_info.value)

    async def test_generate_rate_limit_raises_rate_limit_error(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test generate raises RateLimitError on rate limit."""
        with patch(
            "src.infrastructure.llm.openai_llm.AsyncOpenAI"
        ) as mock_client_class:
            # Create a proper RateLimitError mock
            mock_response = MagicMock()
            mock_response.headers = {"retry-after": "30"}
            rate_limit_error = OpenAIRateLimitError(
                message="Rate limit exceeded",
                response=mock_response,
                body=None,
            )

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=rate_limit_error
            )
            mock_client_class.return_value = mock_client

            service = OpenAILLMService(mock_openai_settings)

            with pytest.raises(RateLimitError) as exc_info:
                await service.generate(prompt="Test", context=[])

            assert exc_info.value.retry_after == 30

    async def test_generate_fallback_on_primary_failure(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test generate falls back to fallback model when primary fails."""
        with patch(
            "src.infrastructure.llm.openai_llm.AsyncOpenAI"
        ) as mock_client_class:
            mock_response = create_mock_completion_response(
                content="Fallback response",
                model="gpt-4o-mini",
            )

            # Primary fails, fallback succeeds
            call_count = 0

            async def mock_create(**kwargs):
                nonlocal call_count
                call_count += 1
                if kwargs.get("model") == "gpt-4o":
                    raise Exception("Primary model unavailable")
                return mock_response

            mock_client = AsyncMock()
            mock_client.chat.completions.create = mock_create
            mock_client_class.return_value = mock_client

            service = OpenAILLMService(mock_openai_settings)
            result = await service.generate(prompt="Test", context=[])

            assert result.answer == "Fallback response"
            assert result.model == "gpt-4o-mini"
            assert call_count == 2  # Primary + Fallback

    async def test_generate_both_models_fail(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test generate raises error when both models fail."""
        with patch(
            "src.infrastructure.llm.openai_llm.AsyncOpenAI"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("All models failed")
            )
            mock_client_class.return_value = mock_client

            service = OpenAILLMService(mock_openai_settings)

            with pytest.raises(LLMError):
                await service.generate(prompt="Test", context=[])

    async def test_generate_stream_success(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test generate_stream yields text chunks."""
        with patch(
            "src.infrastructure.llm.openai_llm.AsyncOpenAI"
        ) as mock_client_class:
            # Create async iterator for streaming
            async def mock_stream():
                chunks = ["Hello", " ", "World", "!"]
                for chunk_text in chunks:
                    chunk = MagicMock()
                    chunk.choices = [MagicMock()]
                    chunk.choices[0].delta.content = chunk_text
                    yield chunk

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
            mock_client_class.return_value = mock_client

            service = OpenAILLMService(mock_openai_settings)
            result_chunks = []
            async for chunk in service.generate_stream(prompt="Test", context=[]):
                result_chunks.append(chunk)

            assert result_chunks == ["Hello", " ", "World", "!"]
            assert "".join(result_chunks) == "Hello World!"

    async def test_generate_stream_api_error(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test generate_stream raises LLMError on API failure."""
        with patch(
            "src.infrastructure.llm.openai_llm.AsyncOpenAI"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Streaming failed")
            )
            mock_client_class.return_value = mock_client

            service = OpenAILLMService(mock_openai_settings)

            with pytest.raises(LLMError) as exc_info:
                async for _ in service.generate_stream(prompt="Test", context=[]):
                    pass

            assert "Streaming failed" in str(exc_info.value)

    async def test_generate_stream_rate_limit(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test generate_stream raises RateLimitError on rate limit."""
        with patch(
            "src.infrastructure.llm.openai_llm.AsyncOpenAI"
        ) as mock_client_class:
            mock_response = MagicMock()
            mock_response.headers = {"retry-after": "60"}
            rate_limit_error = OpenAIRateLimitError(
                message="Rate limit exceeded",
                response=mock_response,
                body=None,
            )

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=rate_limit_error
            )
            mock_client_class.return_value = mock_client

            service = OpenAILLMService(mock_openai_settings)

            with pytest.raises(RateLimitError) as exc_info:
                async for _ in service.generate_stream(prompt="Test", context=[]):
                    pass

            assert exc_info.value.retry_after == 60

    async def test_close_closes_client(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test close method closes the OpenAI client."""
        with patch(
            "src.infrastructure.llm.openai_llm.AsyncOpenAI"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            service = OpenAILLMService(mock_openai_settings)
            await service.close()

            mock_client.close.assert_called_once()

    async def test_generate_null_response_content(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test generate handles null response content gracefully."""
        with patch(
            "src.infrastructure.llm.openai_llm.AsyncOpenAI"
        ) as mock_client_class:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = None
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 0
            mock_response.usage.total_tokens = 50

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            service = OpenAILLMService(mock_openai_settings)
            result = await service.generate(prompt="Test", context=[])

            assert result.answer == ""

    async def test_generate_null_usage(
        self, mock_openai_settings: OpenAISettings
    ) -> None:
        """Test generate handles null usage gracefully."""
        with patch(
            "src.infrastructure.llm.openai_llm.AsyncOpenAI"
        ) as mock_client_class:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Response"
            mock_response.usage = None

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            service = OpenAILLMService(mock_openai_settings)
            result = await service.generate(prompt="Test", context=[])

            assert result.answer == "Response"
            assert result.usage.prompt_tokens == 0
            assert result.usage.completion_tokens == 0
            assert result.usage.total_tokens == 0
