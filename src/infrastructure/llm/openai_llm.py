"""OpenAI LLM service implementation.

Provides LLM functionality using OpenAI's chat completion API.
Supports both synchronous and streaming generation with automatic
retry and fallback mechanisms.
"""

import logging
import time
from collections.abc import AsyncIterator

from openai import (
    APIConnectionError,
    AsyncOpenAI,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import OpenAISettings
from src.domain.entities import GenerationResult, Query
from src.domain.exceptions import LLMError, RateLimitError
from src.domain.value_objects import TokenUsage

logger = logging.getLogger(__name__)

# Exceptions that trigger automatic retry
RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    OpenAIRateLimitError,
    TimeoutError,
)

# Default system prompt for RAG
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Answer the question based ONLY on the provided context
- If the context doesn't contain enough information to answer, say so clearly
- Be concise and direct in your answers
- Cite relevant parts of the context when appropriate
- If the question is unclear, ask for clarification
- Use the same language as the question for your response"""


class OpenAILLMService:
    """OpenAI LLM service for text generation.

    Implements the LLMService protocol using OpenAI's chat completion API.
    Supports automatic retry with exponential backoff and fallback to
    alternative models on failure.

    Example:
        >>> service = OpenAILLMService(settings)
        >>> result = await service.generate(
        ...     prompt="What is Python?",
        ...     context=["Python is a programming language..."]
        ... )
        >>> print(result.answer)
    """

    def __init__(
        self,
        settings: OpenAISettings,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the OpenAI LLM service.

        Args:
            settings: OpenAI configuration settings.
            system_prompt: Custom system prompt for RAG. If None, uses default.
        """
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(),
            timeout=settings.timeout,
            max_retries=0,  # We handle retries with tenacity
        )
        self._model = settings.model
        self._fallback_model = settings.fallback_model
        self._max_retries = settings.max_retries
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def _build_messages(
        self,
        prompt: str,
        context: list[str],
    ) -> list[dict[str, str]]:
        """Build chat messages from prompt and context.

        Args:
            prompt: User question/prompt.
            context: List of context passages from retrieval.

        Returns:
            List of message dictionaries for OpenAI API.
        """
        # Format context as numbered passages
        if context:
            context_text = "\n\n".join(
                f"[{i + 1}] {passage}" for i, passage in enumerate(context)
            )
            user_content = f"""Context:
{context_text}

Question: {prompt}"""
        else:
            user_content = prompt

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, TokenUsage, float]:
        """Call OpenAI API with retry logic.

        Args:
            messages: Chat messages.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            Tuple of (answer text, token usage, latency in ms).

        Raises:
            LLMError: If API call fails after retries.
            RateLimitError: If rate limit is exceeded.
        """
        start_time = time.perf_counter()

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            answer = response.choices[0].message.content or ""
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=(
                    response.usage.completion_tokens if response.usage else 0
                ),
                total_tokens=response.usage.total_tokens if response.usage else 0,
                model=model,
            )

            logger.debug(
                f"LLM response generated: model={model}, "
                f"tokens={usage.total_tokens}, latency={latency_ms:.0f}ms"
            )

            return answer, usage, latency_ms

        except OpenAIRateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            # Extract retry-after if available
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after_header = e.response.headers.get("retry-after")
                if retry_after_header:
                    retry_after = int(retry_after_header)
            raise RateLimitError(str(e), retry_after=retry_after) from e

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise LLMError(str(e), model=model) from e

    async def generate(
        self,
        prompt: str,
        context: list[str],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """Generate a response using the LLM.

        Attempts generation with the primary model first, then falls back
        to the fallback model if the primary fails.

        Args:
            prompt: User prompt/query.
            context: List of context passages from retrieval.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.

        Returns:
            Generation result with answer and metadata.

        Raises:
            LLMError: If generation fails on both models.
            RateLimitError: If rate limit is exceeded.
        """
        messages = self._build_messages(prompt, context)

        # Try primary model first
        try:
            answer, usage, latency_ms = await self._call_api(
                messages=messages,
                model=self._model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            used_model = self._model

        except (LLMError, RateLimitError) as primary_error:
            # If primary fails and we have a different fallback, try it
            if self._fallback_model and self._fallback_model != self._model:
                logger.warning(
                    f"Primary model {self._model} failed, "
                    f"trying fallback {self._fallback_model}"
                )
                try:
                    answer, usage, latency_ms = await self._call_api(
                        messages=messages,
                        model=self._fallback_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    used_model = self._fallback_model

                except Exception as fallback_error:
                    logger.error(
                        f"Fallback model {self._fallback_model} also failed: "
                        f"{fallback_error}"
                    )
                    # Re-raise the original primary error
                    raise primary_error from fallback_error
            else:
                raise

        # Create a minimal Query object for the result
        query = Query.create(text=prompt)

        return GenerationResult.create(
            query=query,
            answer=answer,
            sources=[],  # Sources are populated by GenerationService
            usage=usage,
            model=used_model,
            latency_ms=latency_ms,
        )

    async def generate_stream(
        self,
        prompt: str,
        context: list[str],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Generate a streaming response.

        Yields text chunks as they are generated. Does not include
        fallback logic - uses primary model only for streaming.

        Args:
            prompt: User prompt/query.
            context: List of context passages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Yields:
            Chunks of generated text.

        Raises:
            LLMError: If generation fails.
            RateLimitError: If rate limit is exceeded.
        """
        messages = self._build_messages(prompt, context)

        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:  # type: ignore[union-attr]
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except OpenAIRateLimitError as e:
            logger.warning(f"Rate limit exceeded during streaming: {e}")
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after_header = e.response.headers.get("retry-after")
                if retry_after_header:
                    retry_after = int(retry_after_header)
            raise RateLimitError(str(e), retry_after=retry_after) from e

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise LLMError(str(e), model=self._model) from e

    async def close(self) -> None:
        """Close the OpenAI client.

        Releases any resources held by the client.
        """
        await self._client.close()
