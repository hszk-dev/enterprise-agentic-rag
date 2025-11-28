"""Generation service for RAG answer generation.

Orchestrates the complete RAG pipeline: search -> context formatting -> LLM generation.
"""

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from src.domain.entities import GenerationResult, Query, SearchResult
from src.domain.exceptions import LLMError
from src.domain.interfaces import LLMService

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Metrics for generation pipeline.

    Attributes:
        total_latency_ms: Total generation latency.
        search_latency_ms: Time spent on search.
        llm_latency_ms: Time spent on LLM generation.
        context_tokens_estimate: Estimated context tokens.
        sources_count: Number of sources used.
    """

    total_latency_ms: float
    search_latency_ms: float
    llm_latency_ms: float
    context_tokens_estimate: int
    sources_count: int


class GenerationService:
    """Generation service for RAG answer generation.

    Orchestrates the RAG pipeline:
    1. Use pre-searched results or search via SearchService
    2. Format context from search results
    3. Generate answer using LLM
    4. Return GenerationResult with sources

    Example:
        >>> service = GenerationService(llm_service=openai_llm)
        >>> query = Query.create("What is RAG?")
        >>> search_results = [...]  # From SearchService
        >>> result, metrics = await service.generate(query, search_results)
        >>> print(result.answer)
    """

    def __init__(
        self,
        llm_service: LLMService,
        max_context_length: int = 4000,
        context_separator: str = "\n\n---\n\n",
    ) -> None:
        """Initialize the generation service.

        Args:
            llm_service: LLM service for text generation.
            max_context_length: Maximum context length in characters.
            context_separator: Separator between context passages.
        """
        self._llm_service = llm_service
        self._max_context_length = max_context_length
        self._context_separator = context_separator

    def _format_context(
        self,
        search_results: list[SearchResult],
    ) -> list[str]:
        """Format search results into context passages.

        Includes source metadata for citation and truncates if needed.

        Args:
            search_results: Search results with chunks.

        Returns:
            List of formatted context strings.
        """
        context_passages: list[str] = []
        total_length = 0

        for result in search_results:
            chunk = result.chunk
            # Format with source info for better citation
            passage = chunk.content

            # Add metadata if available
            if chunk.metadata:
                filename = chunk.metadata.get("filename", "")
                if filename:
                    passage = f"[Source: {filename}]\n{passage}"

            # Check if adding this passage would exceed limit
            passage_length = len(passage) + len(self._context_separator)
            if total_length + passage_length > self._max_context_length:
                # Truncate last passage if needed
                remaining = self._max_context_length - total_length
                if remaining > 100:  # Only add if meaningful content remains
                    truncated = passage[: remaining - 3] + "..."
                    context_passages.append(truncated)
                break

            context_passages.append(passage)
            total_length += passage_length

        return context_passages

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple heuristic of ~4 characters per token.
        For more accurate counts, use tiktoken.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count.
        """
        # Rough estimate: ~4 characters per token for English
        # This is a simplification; actual tokenization varies by model
        return len(text) // 4

    async def generate(
        self,
        query: Query,
        search_results: list[SearchResult],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[GenerationResult, GenerationMetrics]:
        """Generate an answer using RAG.

        Args:
            query: User query.
            search_results: Pre-searched results from SearchService.
            temperature: LLM sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            Tuple of (GenerationResult with answer and sources, metrics).

        Raises:
            LLMError: If generation fails.
        """
        start_time = time.perf_counter()

        try:
            # Format context from search results
            context_passages = self._format_context(search_results)
            context_text = self._context_separator.join(context_passages)
            context_tokens_estimate = self._estimate_tokens(context_text)

            logger.debug(
                f"Formatted {len(context_passages)} passages, "
                f"~{context_tokens_estimate} tokens"
            )

            # Generate answer
            llm_start = time.perf_counter()

            llm_result = await self._llm_service.generate(
                prompt=query.text,
                context=context_passages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            llm_latency_ms = (time.perf_counter() - llm_start) * 1000
            total_latency_ms = (time.perf_counter() - start_time) * 1000

            # Create final result with actual query and sources
            result = GenerationResult.create(
                query=query,
                answer=llm_result.answer,
                sources=search_results,
                usage=llm_result.usage,
                model=llm_result.model,
                latency_ms=total_latency_ms,
            )

            metrics = GenerationMetrics(
                total_latency_ms=total_latency_ms,
                search_latency_ms=0.0,  # Search was done externally
                llm_latency_ms=llm_latency_ms,
                context_tokens_estimate=context_tokens_estimate,
                sources_count=len(search_results),
            )

            logger.info(
                f"Generation completed in {total_latency_ms:.1f}ms "
                f"(LLM: {llm_latency_ms:.1f}ms), "
                f"sources: {len(search_results)}, "
                f"model: {llm_result.model}"
            )

            return result, metrics

        except LLMError:
            # Re-raise LLM errors as-is
            raise

        except Exception as e:
            logger.error(f"Generation failed for query '{query.text[:50]}...': {e}")
            msg = f"Generation failed: {e}"
            raise LLMError(msg) from e

    async def generate_stream(
        self,
        query: Query,
        search_results: list[SearchResult],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Generate a streaming answer using RAG.

        Yields answer chunks as they are generated.
        Note: Does not return metrics or complete GenerationResult.

        Args:
            query: User query.
            search_results: Pre-searched results from SearchService.
            temperature: LLM sampling temperature.
            max_tokens: Maximum response tokens.

        Yields:
            Chunks of generated answer text.

        Raises:
            LLMError: If generation fails.
        """
        # Format context from search results
        context_passages = self._format_context(search_results)

        logger.debug(
            f"Starting streaming generation with {len(context_passages)} passages"
        )

        try:
            async for chunk in self._llm_service.generate_stream(  # type: ignore[attr-defined]
                prompt=query.text,
                context=context_passages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield chunk

        except LLMError:
            raise

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            msg = f"Streaming generation failed: {e}"
            raise LLMError(msg) from e

    async def generate_with_no_context(
        self,
        query: Query,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[GenerationResult, GenerationMetrics]:
        """Generate an answer without RAG context.

        Useful for comparison or when no relevant documents are found.

        Args:
            query: User query.
            temperature: LLM sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            Tuple of (GenerationResult, metrics).

        Raises:
            LLMError: If generation fails.
        """
        return await self.generate(
            query=query,
            search_results=[],
            temperature=temperature,
            max_tokens=max_tokens,
        )
