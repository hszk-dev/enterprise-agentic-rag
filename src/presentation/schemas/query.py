"""Query-related API schemas.

Pydantic models for RAG query and generation endpoints.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request for RAG query endpoint.

    Attributes:
        query: The search query text.
        top_k: Number of chunks to retrieve before reranking.
        rerank_top_n: Number of chunks to keep after reranking.
        alpha: Hybrid search weight (0=sparse only, 1=dense only).
        filters: Optional metadata filters for search.
        include_sources: Whether to include source chunks in response.
        stream: Whether to use streaming response (SSE).
    """

    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Chunks to retrieve")
    rerank_top_n: int = Field(default=5, ge=1, le=20, description="Chunks after rerank")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Hybrid weight")
    # NOTE: Any is used here because filters can have various types (str, int, list).
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filters")
    include_sources: bool = Field(default=True, description="Include sources")
    stream: bool = Field(default=False, description="Use streaming")


class TokenUsageResponse(BaseModel):
    """Token usage statistics response.

    Attributes:
        prompt_tokens: Tokens in the prompt.
        completion_tokens: Tokens in the completion.
        total_tokens: Total tokens used.
        estimated_cost_usd: Estimated API cost.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class SourceResponse(BaseModel):
    """Source chunk information in query response.

    Represents a single source chunk used for answer generation.

    Attributes:
        chunk_id: Unique chunk identifier.
        document_id: Parent document identifier.
        content: Chunk text content.
        score: Raw search score.
        rerank_score: Score after reranking (if applied).
        display_score: Normalized score for UI (0.0-1.0).
        rank: Position in result list (1-indexed).
        metadata: Chunk metadata (page number, section, etc.).
    """

    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    rerank_score: float | None = None
    display_score: float
    rank: int
    # NOTE: Any is used here because metadata can contain various types.
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    """Response for RAG query endpoint.

    Contains the generated answer and optional source information.

    Attributes:
        id: Unique response identifier.
        query: Original query text.
        answer: Generated answer text.
        sources: Source chunks used (if include_sources=True).
        model: LLM model used for generation.
        usage: Token usage statistics.
        latency_ms: Total processing latency in milliseconds.
        created_at: Response generation timestamp.
    """

    id: UUID
    query: str
    answer: str
    sources: list[SourceResponse]
    model: str
    usage: TokenUsageResponse
    latency_ms: float
    created_at: datetime


class StreamChunkResponse(BaseModel):
    """Streaming response chunk for SSE.

    Attributes:
        chunk: Text chunk from the stream.
        done: Whether this is the final chunk.
        sources: Source chunks (only in final chunk).
        usage: Token usage (only in final chunk).
        error: Error message if generation failed.
    """

    chunk: str = ""
    done: bool = False
    sources: list[SourceResponse] | None = None
    usage: TokenUsageResponse | None = None
    error: str | None = None
