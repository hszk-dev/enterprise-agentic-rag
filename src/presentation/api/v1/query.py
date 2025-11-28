"""Query and RAG generation endpoints.

Provides endpoints for searching documents and generating answers.
"""

import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from src.domain.entities import Query, SearchResult
from src.domain.exceptions import EmbeddingError, LLMError, RateLimitError, SearchError
from src.presentation.api.dependencies import (
    GenerationServiceDep,
    SearchServiceDep,
)
from src.presentation.schemas.query import (
    QueryRequest,
    QueryResponse,
    SourceResponse,
    StreamChunkResponse,
    TokenUsageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


def _create_source_response(result: SearchResult) -> SourceResponse:
    """Convert SearchResult to SourceResponse.

    Args:
        result: Domain SearchResult entity.

    Returns:
        SourceResponse for API.
    """

    return SourceResponse(
        chunk_id=result.chunk.id,
        document_id=result.chunk.document_id,
        content=result.chunk.content,
        score=result.score,
        rerank_score=result.rerank_score,
        display_score=result.display_score,
        rank=result.rank,
        metadata=result.chunk.metadata,
    )


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute RAG query",
    description="Search for relevant documents and generate an answer using LLM.",
)
async def execute_query(
    request: QueryRequest,
    search_service: SearchServiceDep,
    generation_service: GenerationServiceDep,
) -> QueryResponse:
    """Execute a RAG query.

    This endpoint:
    1. Searches for relevant document chunks using hybrid search
    2. Reranks results for better relevance
    3. Generates an answer using the LLM with retrieved context

    Args:
        request: Query parameters.
        search_service: Injected search service.
        generation_service: Injected generation service.

    Returns:
        QueryResponse with answer and sources.

    Raises:
        HTTPException: 429 if rate limited.
        HTTPException: 500 if search or generation fails.
    """
    # Create domain Query object
    query = Query.create(
        text=request.query,
        top_k=request.top_k,
        rerank_top_n=request.rerank_top_n,
        alpha=request.alpha,
        filters=request.filters,
    )

    try:
        # Step 1: Search for relevant chunks
        search_results, search_metrics = await search_service.search(query)

        if not search_results:
            # No results found - generate answer without context
            logger.info(f"No search results for query: {request.query[:50]}...")
            generation_result, _ = await generation_service.generate_with_no_context(
                query=query,
            )
            return QueryResponse(
                id=generation_result.id,
                query=request.query,
                answer=generation_result.answer,
                sources=[],
                model=generation_result.model,
                usage=TokenUsageResponse(
                    prompt_tokens=generation_result.usage.prompt_tokens,
                    completion_tokens=generation_result.usage.completion_tokens,
                    total_tokens=generation_result.usage.total_tokens,
                    estimated_cost_usd=generation_result.usage.estimated_cost_usd,
                ),
                latency_ms=generation_result.latency_ms
                + search_metrics.total_latency_ms,
                created_at=generation_result.created_at,
            )

        # Step 2: Generate answer with context
        generation_result, generation_metrics = await generation_service.generate(
            query=query,
            search_results=search_results,
        )

        # Calculate total latency
        total_latency = (
            search_metrics.total_latency_ms + generation_metrics.total_latency_ms
        )

        # Build sources if requested
        sources = []
        if request.include_sources:
            sources = [_create_source_response(r) for r in search_results]

        return QueryResponse(
            id=generation_result.id,
            query=request.query,
            answer=generation_result.answer,
            sources=sources,
            model=generation_result.model,
            usage=TokenUsageResponse(
                prompt_tokens=generation_result.usage.prompt_tokens,
                completion_tokens=generation_result.usage.completion_tokens,
                total_tokens=generation_result.usage.total_tokens,
                estimated_cost_usd=generation_result.usage.estimated_cost_usd,
            ),
            latency_ms=total_latency,
            created_at=generation_result.created_at,
        )

    except RateLimitError as e:
        logger.warning(f"Rate limit exceeded: {e}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
        ) from e

    except (SearchError, EmbeddingError) as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e.message}",
        ) from e

    except LLMError as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {e.message}",
        ) from e


@router.post(
    "/stream",
    status_code=status.HTTP_200_OK,
    summary="Execute streaming RAG query",
    description="Search for documents and stream the generated answer using SSE.",
)
async def execute_stream_query(
    request: QueryRequest,
    search_service: SearchServiceDep,
    generation_service: GenerationServiceDep,
) -> StreamingResponse:
    """Execute a streaming RAG query.

    This endpoint streams the generated answer as Server-Sent Events (SSE).
    Each event contains a JSON object with the chunk text and metadata.

    Args:
        request: Query parameters.
        search_service: Injected search service.
        generation_service: Injected generation service.

    Returns:
        StreamingResponse with SSE events.

    Raises:
        HTTPException: 429 if rate limited.
        HTTPException: 500 if search fails.
    """
    # Create domain Query object
    query = Query.create(
        text=request.query,
        top_k=request.top_k,
        rerank_top_n=request.rerank_top_n,
        alpha=request.alpha,
        filters=request.filters,
    )

    async def generate_stream_events() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        sources: list[SourceResponse] = []

        try:
            # Step 1: Search for relevant chunks
            search_results, _ = await search_service.search(query)

            if request.include_sources:
                sources = [_create_source_response(r) for r in search_results]

            if not search_results:
                # No results - send empty response
                chunk_data = StreamChunkResponse(
                    chunk="I couldn't find relevant information to answer your question.",
                    done=True,
                    sources=[],
                )
                yield f"data: {chunk_data.model_dump_json()}\n\n"
                return

            # Step 2: Stream the generated answer
            async for text_chunk in generation_service.generate_stream(
                query=query,
                search_results=search_results,
            ):
                chunk_data = StreamChunkResponse(chunk=text_chunk, done=False)
                yield f"data: {chunk_data.model_dump_json()}\n\n"

            # Final event with sources
            final_data = StreamChunkResponse(
                chunk="",
                done=True,
                sources=sources if request.include_sources else None,
            )
            yield f"data: {final_data.model_dump_json()}\n\n"

        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded during stream: {e}")
            error_data = StreamChunkResponse(
                done=True,
                error="Rate limit exceeded. Please try again later.",
            )
            yield f"data: {error_data.model_dump_json()}\n\n"

        except (SearchError, EmbeddingError, LLMError) as e:
            logger.error(f"Stream error: {e}")
            error_data = StreamChunkResponse(
                done=True,
                error=f"Error: {e.message}",
            )
            yield f"data: {error_data.model_dump_json()}\n\n"

        except Exception as e:
            logger.error(f"Unexpected stream error: {e}")
            error_data = StreamChunkResponse(
                done=True,
                error="An unexpected error occurred.",
            )
            yield f"data: {error_data.model_dump_json()}\n\n"

    return StreamingResponse(
        generate_stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
