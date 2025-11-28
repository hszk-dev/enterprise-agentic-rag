"""Domain value objects.

Value objects are immutable objects that represent concepts with no identity.
They are compared by their attribute values, not by reference.
"""

from dataclasses import dataclass
from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status.

    Lifecycle: PENDING -> PROCESSING -> COMPLETED | FAILED
    """

    PENDING = "pending"  # Uploaded, waiting for processing
    PROCESSING = "processing"  # Currently being processed
    COMPLETED = "completed"  # Successfully processed
    FAILED = "failed"  # Processing failed


class ContentType(str, Enum):
    """Supported document content types.

    Each type requires a specific parser implementation.
    """

    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    TXT = "text/plain"
    MD = "text/markdown"
    HTML = "text/html"

    @classmethod
    def from_extension(cls, extension: str) -> "ContentType":
        """Get content type from file extension.

        Args:
            extension: File extension (with or without leading dot).

        Returns:
            Corresponding ContentType.

        Raises:
            ValueError: If extension is not supported.
        """
        ext = extension.lower().lstrip(".")
        mapping = {
            "pdf": cls.PDF,
            "docx": cls.DOCX,
            "txt": cls.TXT,
            "md": cls.MD,
            "markdown": cls.MD,
            "html": cls.HTML,
            "htm": cls.HTML,
        }
        if ext not in mapping:
            supported = ", ".join(mapping.keys())
            msg = f"Unsupported file extension: {ext}. Supported: {supported}"
            raise ValueError(msg)
        return mapping[ext]

    @classmethod
    def from_mime_type(cls, mime_type: str) -> "ContentType":
        """Get content type from MIME type string.

        Args:
            mime_type: MIME type string.

        Returns:
            Corresponding ContentType.

        Raises:
            ValueError: If MIME type is not supported.
        """
        # Normalize mime type (remove charset, etc.)
        normalized = mime_type.split(";")[0].strip().lower()
        for content_type in cls:
            if content_type.value == normalized:
                return content_type
        supported = ", ".join(ct.value for ct in cls)
        msg = f"Unsupported MIME type: {mime_type}. Supported: {supported}"
        raise ValueError(msg)


@dataclass(frozen=True)
class TokenUsage:
    """LLM token usage tracking.

    Immutable value object for tracking API consumption and cost estimation.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated response.
        total_tokens: Total tokens used (prompt + completion).
        model: Model identifier for cost calculation.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str = "gpt-5"

    @property
    def estimated_cost_usd(self) -> float:
        """Estimate API cost in USD.

        Pricing based on OpenAI rates (August 2025):

        GPT-4 series :
        - GPT-4o: $2.50/1M input, $10.00/1M output
        - GPT-4o-mini: $0.15/1M input, $0.60/1M output

        GPT-5 series:
        - GPT-5: $1.25/1M input, $10.00/1M output
        - GPT-5 mini: $0.25/1M input, $2.00/1M output
        - GPT-5 nano: $0.05/1M input, $0.40/1M output

        Returns:
            Estimated cost in USD.
        """
        model_lower = self.model.lower()

        # GPT-4 series
        if "gpt-4o-mini" in model_lower:
            input_cost = (self.prompt_tokens / 1_000_000) * 0.15
            output_cost = (self.completion_tokens / 1_000_000) * 0.60
        elif "gpt-4o" in model_lower:
            input_cost = (self.prompt_tokens / 1_000_000) * 2.50
            output_cost = (self.completion_tokens / 1_000_000) * 10.00
        # GPT-5 series
        elif "nano" in model_lower:
            # GPT-5 nano pricing
            input_cost = (self.prompt_tokens / 1_000_000) * 0.05
            output_cost = (self.completion_tokens / 1_000_000) * 0.40
        elif "mini" in model_lower:
            # GPT-5 mini pricing
            input_cost = (self.prompt_tokens / 1_000_000) * 0.25
            output_cost = (self.completion_tokens / 1_000_000) * 2.00
        else:
            # GPT-5 pricing (default)
            input_cost = (self.prompt_tokens / 1_000_000) * 1.25
            output_cost = (self.completion_tokens / 1_000_000) * 10.00

        return input_cost + output_cost


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata for a document chunk.

    Optional metadata that provides context about chunk origin.

    Attributes:
        page_number: Page number in original document (1-indexed).
        section_title: Section or heading the chunk belongs to.
        source_url: URL if document was fetched from web.
    """

    page_number: int | None = None
    section_title: str | None = None
    source_url: str | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        """Convert to dictionary for storage.

        Returns:
            Dictionary with non-None values only.
        """
        result: dict[str, int | str | None] = {}
        if self.page_number is not None:
            result["page_number"] = self.page_number
        if self.section_title is not None:
            result["section_title"] = self.section_title
        if self.source_url is not None:
            result["source_url"] = self.source_url
        return result


@dataclass(frozen=True)
class SparseVector:
    """Sparse vector representation for keyword search.

    Used for SPLADE or BM25-style sparse embeddings.

    Attributes:
        indices: Token indices with non-zero weights.
        values: Corresponding weight values.
    """

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate indices and values have same length."""
        if len(self.indices) != len(self.values):
            msg = f"Indices length ({len(self.indices)}) != values length ({len(self.values)})"
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, sparse_dict: dict[int, float]) -> "SparseVector":
        """Create from dictionary representation.

        Args:
            sparse_dict: Dictionary mapping indices to values.

        Returns:
            SparseVector instance.
        """
        if not sparse_dict:
            return cls(indices=(), values=())
        indices = tuple(sparse_dict.keys())
        values = tuple(sparse_dict.values())
        return cls(indices=indices, values=values)

    def to_dict(self) -> dict[int, float]:
        """Convert to dictionary representation.

        Returns:
            Dictionary mapping indices to values.
        """
        return dict(zip(self.indices, self.values, strict=True))

    def __len__(self) -> int:
        """Return number of non-zero elements."""
        return len(self.indices)
