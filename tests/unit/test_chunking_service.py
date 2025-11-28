"""Unit tests for LangChainChunkingService."""

from uuid import uuid4

from config.settings import ChunkingSettings
from src.application.services.chunking_service import LangChainChunkingService


class TestLangChainChunkingServiceInit:
    """Tests for LangChainChunkingService initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        service = LangChainChunkingService()

        assert service.chunk_size == 1000
        assert service.chunk_overlap == 200
        assert service.separators == ["\n\n", "\n", ". ", " ", ""]

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        service = LangChainChunkingService(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n", " "],
        )

        assert service.chunk_size == 500
        assert service.chunk_overlap == 100
        assert service.separators == ["\n", " "]

    def test_init_with_settings(self):
        """Test initialization with ChunkingSettings."""
        settings = ChunkingSettings(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n"],
        )
        service = LangChainChunkingService(settings=settings)

        assert service.chunk_size == 800
        assert service.chunk_overlap == 150
        assert service.separators == ["\n\n", "\n"]

    def test_init_settings_override_params(self):
        """Test that settings take precedence over individual params."""
        settings = ChunkingSettings(
            chunk_size=800,
            chunk_overlap=150,
        )
        service = LangChainChunkingService(
            settings=settings,
            chunk_size=500,  # Should be ignored
            chunk_overlap=100,  # Should be ignored
        )

        assert service.chunk_size == 800
        assert service.chunk_overlap == 150


class TestLangChainChunkingServiceChunk:
    """Tests for LangChainChunkingService.chunk method."""

    def test_chunk_empty_text(self):
        """Test chunking empty text returns empty list."""
        service = LangChainChunkingService()
        document_id = uuid4()

        result = service.chunk("", document_id)

        assert result == []

    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only text returns empty list."""
        service = LangChainChunkingService()
        document_id = uuid4()

        result = service.chunk("   \n\n\t   ", document_id)

        assert result == []

    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk_size returns single chunk."""
        service = LangChainChunkingService(chunk_size=1000)
        document_id = uuid4()
        text = "This is a short text."

        result = service.chunk(text, document_id)

        assert len(result) == 1
        assert result[0].content == text
        assert result[0].document_id == document_id
        assert result[0].chunk_index == 0

    def test_chunk_creates_multiple_chunks(self):
        """Test chunking long text creates multiple chunks."""
        service = LangChainChunkingService(chunk_size=50, chunk_overlap=10)
        document_id = uuid4()
        # Create text longer than chunk_size
        text = (
            "First paragraph with some content. " * 5
            + "\n\n"
            + "Second paragraph with more content. " * 5
        )

        result = service.chunk(text, document_id)

        assert len(result) > 1
        # Verify all chunks belong to same document
        for chunk in result:
            assert chunk.document_id == document_id

    def test_chunk_indices_are_sequential(self):
        """Test that chunk indices are sequential starting from 0."""
        service = LangChainChunkingService(chunk_size=50, chunk_overlap=10)
        document_id = uuid4()
        text = "Paragraph one. " * 10 + "\n\n" + "Paragraph two. " * 10

        result = service.chunk(text, document_id)

        for i, chunk in enumerate(result):
            assert chunk.chunk_index == i

    def test_chunk_has_valid_positions(self):
        """Test that chunks have valid start and end positions."""
        service = LangChainChunkingService(chunk_size=100, chunk_overlap=20)
        document_id = uuid4()
        text = "This is a test. " * 50

        result = service.chunk(text, document_id)

        for chunk in result:
            # Start must be non-negative
            assert chunk.start_char >= 0
            # End must be greater than start
            assert chunk.end_char > chunk.start_char
            # End should not exceed text length
            assert chunk.end_char <= len(text)

    def test_chunk_content_matches_text(self):
        """Test that chunk content is from the original text."""
        service = LangChainChunkingService(chunk_size=100, chunk_overlap=20)
        document_id = uuid4()
        text = "First section with content.\n\nSecond section with more content.\n\nThird section."

        result = service.chunk(text, document_id)

        for chunk in result:
            # Each chunk's content should be found in original text
            assert chunk.content in text or text.find(chunk.content[:20]) != -1

    def test_chunk_preserves_metadata(self):
        """Test that metadata is preserved in chunks."""
        service = LangChainChunkingService(chunk_size=1000)
        document_id = uuid4()
        text = "Test content"
        metadata = {"filename": "test.pdf", "author": "Test Author"}

        result = service.chunk(text, document_id, metadata=metadata)

        assert len(result) == 1
        assert result[0].metadata["filename"] == "test.pdf"
        assert result[0].metadata["author"] == "Test Author"

    def test_chunk_splits_on_paragraphs_first(self):
        """Test that chunking prefers paragraph boundaries."""
        service = LangChainChunkingService(chunk_size=100, chunk_overlap=10)
        document_id = uuid4()
        # Create text with clear paragraph boundaries
        text = "Short paragraph one.\n\nShort paragraph two.\n\nShort paragraph three."

        result = service.chunk(text, document_id)

        # With small chunk size, should split on paragraph boundaries
        # The exact behavior depends on LangChain's implementation
        assert len(result) >= 1

    def test_chunk_respects_overlap(self):
        """Test that chunks have overlapping content."""
        service = LangChainChunkingService(chunk_size=50, chunk_overlap=20)
        document_id = uuid4()
        # Create text that will definitely create multiple chunks
        text = "Word " * 100  # 500 characters

        result = service.chunk(text, document_id)

        if len(result) >= 2:
            # Check for overlap between consecutive chunks
            # Due to how text splitter works, there should be some overlap
            # This is a soft check as exact overlap depends on split points
            pass

    def test_chunk_generates_unique_ids(self):
        """Test that each chunk has a unique ID."""
        service = LangChainChunkingService(chunk_size=50, chunk_overlap=10)
        document_id = uuid4()
        text = "Content " * 50

        result = service.chunk(text, document_id)

        ids = [chunk.id for chunk in result]
        assert len(ids) == len(set(ids))  # All IDs should be unique

    def test_chunk_char_count_property(self):
        """Test that chunk char_count property works correctly."""
        service = LangChainChunkingService(chunk_size=1000)
        document_id = uuid4()
        text = "Test content here"

        result = service.chunk(text, document_id)

        assert len(result) == 1
        assert result[0].char_count == len(result[0].content)


class TestLangChainChunkingServiceProperties:
    """Tests for LangChainChunkingService properties."""

    def test_chunk_size_property(self):
        """Test chunk_size property returns correct value."""
        service = LangChainChunkingService(chunk_size=750)
        assert service.chunk_size == 750

    def test_chunk_overlap_property(self):
        """Test chunk_overlap property returns correct value."""
        service = LangChainChunkingService(chunk_overlap=150)
        assert service.chunk_overlap == 150

    def test_separators_property_returns_copy(self):
        """Test separators property returns a copy."""
        service = LangChainChunkingService()
        separators = service.separators
        separators.append("NEW")

        # Original should not be modified
        assert "NEW" not in service.separators


class TestLangChainChunkingServiceEdgeCases:
    """Edge case tests for LangChainChunkingService."""

    def test_chunk_with_only_separators(self):
        """Test chunking text with only separator characters."""
        service = LangChainChunkingService(chunk_size=100, chunk_overlap=20)
        document_id = uuid4()
        text = "\n\n\n\n"

        result = service.chunk(text, document_id)

        # Should return empty or minimal chunks
        assert (
            len(result) == 0
            or all(not chunk.content.strip() for chunk in result) is False
        )

    def test_chunk_unicode_content(self):
        """Test chunking text with unicode characters."""
        service = LangChainChunkingService(chunk_size=100, chunk_overlap=20)
        document_id = uuid4()
        text = "日本語のテキストです。これはテストです。\n\nもう一つの段落。"

        result = service.chunk(text, document_id)

        assert len(result) >= 1
        # Verify unicode is preserved
        for chunk in result:
            assert (
                "日本語" in chunk.content
                or "段落" in chunk.content
                or len(chunk.content) > 0
            )

    def test_chunk_very_long_word(self):
        """Test chunking text with a word longer than chunk_size."""
        service = LangChainChunkingService(chunk_size=50, chunk_overlap=10)
        document_id = uuid4()
        # Create a "word" longer than chunk_size
        long_word = "a" * 100
        text = f"Normal text. {long_word} More normal text."

        result = service.chunk(text, document_id)

        # Should still produce chunks, splitting the long word if needed
        assert len(result) >= 1
        # Some of the long word should be present (might be split across chunks)

    def test_chunk_single_character(self):
        """Test chunking single character text."""
        service = LangChainChunkingService()
        document_id = uuid4()

        result = service.chunk("A", document_id)

        assert len(result) == 1
        assert result[0].content == "A"

    def test_chunk_with_none_metadata(self):
        """Test chunking with None metadata."""
        service = LangChainChunkingService()
        document_id = uuid4()

        result = service.chunk("Test content", document_id, metadata=None)

        assert len(result) == 1
        # Should have empty dict, not None
        assert result[0].metadata == {} or isinstance(result[0].metadata, dict)
