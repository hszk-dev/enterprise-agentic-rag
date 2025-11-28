"""Unit tests for domain value objects."""

import pytest

from src.domain.value_objects import (
    ChunkMetadata,
    ContentType,
    DocumentStatus,
    SparseVector,
    TokenUsage,
)


class TestDocumentStatus:
    """Tests for DocumentStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Verify all expected statuses are defined."""
        assert DocumentStatus.PENDING == "pending"
        assert DocumentStatus.PROCESSING == "processing"
        assert DocumentStatus.COMPLETED == "completed"
        assert DocumentStatus.FAILED == "failed"

    def test_status_is_string(self) -> None:
        """Verify status values can be used as strings."""
        status = DocumentStatus.PENDING
        assert isinstance(status.value, str)
        assert str(status.value) == "pending"


class TestContentType:
    """Tests for ContentType enum."""

    def test_from_extension_pdf(self) -> None:
        """Test PDF extension mapping."""
        assert ContentType.from_extension("pdf") == ContentType.PDF
        assert ContentType.from_extension(".pdf") == ContentType.PDF
        assert ContentType.from_extension("PDF") == ContentType.PDF

    def test_from_extension_docx(self) -> None:
        """Test DOCX extension mapping."""
        assert ContentType.from_extension("docx") == ContentType.DOCX

    def test_from_extension_txt(self) -> None:
        """Test TXT extension mapping."""
        assert ContentType.from_extension("txt") == ContentType.TXT

    def test_from_extension_markdown(self) -> None:
        """Test Markdown extension mapping."""
        assert ContentType.from_extension("md") == ContentType.MD
        assert ContentType.from_extension("markdown") == ContentType.MD

    def test_from_extension_html(self) -> None:
        """Test HTML extension mapping."""
        assert ContentType.from_extension("html") == ContentType.HTML
        assert ContentType.from_extension("htm") == ContentType.HTML

    def test_from_extension_unsupported(self) -> None:
        """Test unsupported extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            ContentType.from_extension("xyz")

    def test_from_mime_type_pdf(self) -> None:
        """Test PDF MIME type mapping."""
        assert ContentType.from_mime_type("application/pdf") == ContentType.PDF

    def test_from_mime_type_with_charset(self) -> None:
        """Test MIME type with charset is handled."""
        assert (
            ContentType.from_mime_type("text/plain; charset=utf-8") == ContentType.TXT
        )

    def test_from_mime_type_unsupported(self) -> None:
        """Test unsupported MIME type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported MIME type"):
            ContentType.from_mime_type("application/octet-stream")


class TestTokenUsage:
    """Tests for TokenUsage value object."""

    def test_creation(self) -> None:
        """Test TokenUsage creation."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4o",
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.model == "gpt-4o"

    def test_immutability(self) -> None:
        """Test TokenUsage is immutable (frozen dataclass)."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        with pytest.raises(AttributeError):
            usage.prompt_tokens = 200  # type: ignore[misc]

    def test_estimated_cost_gpt5(self) -> None:
        """Test cost estimation for GPT-5."""
        usage = TokenUsage(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
            model="gpt-5",
        )
        # GPT-5: $1.25/1M input + $10.00/1M output = $11.25
        assert usage.estimated_cost_usd == pytest.approx(11.25)

    def test_estimated_cost_gpt5_mini(self) -> None:
        """Test cost estimation for GPT-5 mini."""
        usage = TokenUsage(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
            model="gpt-5-mini",
        )
        # GPT-5 mini: $0.25/1M input + $2.00/1M output = $2.25
        assert usage.estimated_cost_usd == pytest.approx(2.25)

    def test_estimated_cost_gpt5_nano(self) -> None:
        """Test cost estimation for GPT-5 nano."""
        usage = TokenUsage(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
            model="gpt-5-nano",
        )
        # GPT-5 nano: $0.05/1M input + $0.40/1M output = $0.45
        assert usage.estimated_cost_usd == pytest.approx(0.45)

    def test_default_model(self) -> None:
        """Test default model is gpt-5."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert usage.model == "gpt-5"


class TestChunkMetadata:
    """Tests for ChunkMetadata value object."""

    def test_creation_with_all_fields(self) -> None:
        """Test creation with all fields."""
        metadata = ChunkMetadata(
            page_number=5,
            section_title="Introduction",
            source_url="https://example.com/doc.pdf",
        )
        assert metadata.page_number == 5
        assert metadata.section_title == "Introduction"
        assert metadata.source_url == "https://example.com/doc.pdf"

    def test_creation_with_defaults(self) -> None:
        """Test creation with default None values."""
        metadata = ChunkMetadata()
        assert metadata.page_number is None
        assert metadata.section_title is None
        assert metadata.source_url is None

    def test_to_dict_with_all_fields(self) -> None:
        """Test to_dict returns all non-None fields."""
        metadata = ChunkMetadata(
            page_number=5,
            section_title="Introduction",
            source_url="https://example.com/doc.pdf",
        )
        result = metadata.to_dict()
        assert result == {
            "page_number": 5,
            "section_title": "Introduction",
            "source_url": "https://example.com/doc.pdf",
        }

    def test_to_dict_with_partial_fields(self) -> None:
        """Test to_dict excludes None fields."""
        metadata = ChunkMetadata(page_number=3)
        result = metadata.to_dict()
        assert result == {"page_number": 3}

    def test_to_dict_empty(self) -> None:
        """Test to_dict returns empty dict when all None."""
        metadata = ChunkMetadata()
        result = metadata.to_dict()
        assert result == {}

    def test_immutability(self) -> None:
        """Test ChunkMetadata is immutable."""
        metadata = ChunkMetadata(page_number=1)
        with pytest.raises(AttributeError):
            metadata.page_number = 2  # type: ignore[misc]


class TestSparseVector:
    """Tests for SparseVector value object."""

    def test_creation(self) -> None:
        """Test SparseVector creation."""
        vector = SparseVector(
            indices=(1, 5, 10),
            values=(0.5, 0.3, 0.2),
        )
        assert vector.indices == (1, 5, 10)
        assert vector.values == (0.5, 0.3, 0.2)

    def test_length_mismatch_raises_error(self) -> None:
        """Test that mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="Indices length"):
            SparseVector(
                indices=(1, 2, 3),
                values=(0.5, 0.3),
            )

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        sparse_dict = {1: 0.5, 5: 0.3, 10: 0.2}
        vector = SparseVector.from_dict(sparse_dict)
        assert set(vector.indices) == {1, 5, 10}
        assert len(vector.values) == 3

    def test_from_dict_empty(self) -> None:
        """Test creation from empty dictionary."""
        vector = SparseVector.from_dict({})
        assert vector.indices == ()
        assert vector.values == ()

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        vector = SparseVector(
            indices=(1, 5, 10),
            values=(0.5, 0.3, 0.2),
        )
        result = vector.to_dict()
        assert result == {1: 0.5, 5: 0.3, 10: 0.2}

    def test_len(self) -> None:
        """Test __len__ returns number of non-zero elements."""
        vector = SparseVector(
            indices=(1, 5, 10),
            values=(0.5, 0.3, 0.2),
        )
        assert len(vector) == 3

    def test_immutability(self) -> None:
        """Test SparseVector is immutable."""
        vector = SparseVector(indices=(1,), values=(0.5,))
        with pytest.raises(AttributeError):
            vector.indices = (2,)  # type: ignore[misc]

    def test_roundtrip_dict_conversion(self) -> None:
        """Test dict -> SparseVector -> dict roundtrip."""
        original = {100: 0.9, 200: 0.8, 300: 0.7}
        vector = SparseVector.from_dict(original)
        result = vector.to_dict()
        assert result == original
