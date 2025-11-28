"""Unit tests for UnstructuredParser."""

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.exceptions import DocumentProcessingError, UnsupportedContentTypeError
from src.domain.value_objects import ContentType
from src.infrastructure.parsers.unstructured_parser import (
    SUPPORTED_CONTENT_TYPES,
    UnstructuredParser,
)


class TestUnstructuredParser:
    """Tests for UnstructuredParser."""

    def test_init_without_blob_storage(self):
        """Test initialization without blob storage."""
        parser = UnstructuredParser()
        assert parser._blob_storage is None

    def test_init_with_blob_storage(self):
        """Test initialization with blob storage."""
        mock_storage = MagicMock()
        parser = UnstructuredParser(blob_storage=mock_storage)
        assert parser._blob_storage == mock_storage

    def test_supports_pdf(self):
        """Test that PDF content type is supported."""
        parser = UnstructuredParser()
        assert parser.supports(ContentType.PDF.value) is True

    def test_supports_docx(self):
        """Test that DOCX content type is supported."""
        parser = UnstructuredParser()
        assert parser.supports(ContentType.DOCX.value) is True

    def test_supports_txt(self):
        """Test that TXT content type is supported."""
        parser = UnstructuredParser()
        assert parser.supports(ContentType.TXT.value) is True

    def test_supports_markdown(self):
        """Test that Markdown content type is supported."""
        parser = UnstructuredParser()
        assert parser.supports(ContentType.MD.value) is True

    def test_supports_html(self):
        """Test that HTML content type is supported."""
        parser = UnstructuredParser()
        assert parser.supports(ContentType.HTML.value) is True

    def test_supports_unsupported_type(self):
        """Test that unsupported content type returns False."""
        parser = UnstructuredParser()
        assert parser.supports("application/octet-stream") is False
        assert parser.supports("image/png") is False
        assert parser.supports("video/mp4") is False

    def test_supported_content_types_constant(self):
        """Test that SUPPORTED_CONTENT_TYPES contains all expected types."""
        assert ContentType.PDF.value in SUPPORTED_CONTENT_TYPES
        assert ContentType.DOCX.value in SUPPORTED_CONTENT_TYPES
        assert ContentType.TXT.value in SUPPORTED_CONTENT_TYPES
        assert ContentType.MD.value in SUPPORTED_CONTENT_TYPES
        assert ContentType.HTML.value in SUPPORTED_CONTENT_TYPES

    def test_detect_content_type_pdf(self):
        """Test content type detection for PDF files."""
        parser = UnstructuredParser()
        assert (
            parser._detect_content_type(Path("document.pdf")) == ContentType.PDF.value
        )

    def test_detect_content_type_docx(self):
        """Test content type detection for DOCX files."""
        parser = UnstructuredParser()
        assert (
            parser._detect_content_type(Path("document.docx")) == ContentType.DOCX.value
        )

    def test_detect_content_type_txt(self):
        """Test content type detection for TXT files."""
        parser = UnstructuredParser()
        assert (
            parser._detect_content_type(Path("document.txt")) == ContentType.TXT.value
        )

    def test_detect_content_type_markdown(self):
        """Test content type detection for Markdown files."""
        parser = UnstructuredParser()
        assert parser._detect_content_type(Path("document.md")) == ContentType.MD.value
        assert (
            parser._detect_content_type(Path("document.markdown"))
            == ContentType.MD.value
        )

    def test_detect_content_type_html(self):
        """Test content type detection for HTML files."""
        parser = UnstructuredParser()
        assert (
            parser._detect_content_type(Path("document.html")) == ContentType.HTML.value
        )
        assert (
            parser._detect_content_type(Path("document.htm")) == ContentType.HTML.value
        )

    def test_detect_content_type_unknown(self):
        """Test content type detection for unknown extensions."""
        parser = UnstructuredParser()
        assert (
            parser._detect_content_type(Path("document.xyz"))
            == "application/octet-stream"
        )

    def test_detect_content_type_case_insensitive(self):
        """Test that content type detection is case insensitive."""
        parser = UnstructuredParser()
        assert (
            parser._detect_content_type(Path("document.PDF")) == ContentType.PDF.value
        )
        assert (
            parser._detect_content_type(Path("document.TXT")) == ContentType.TXT.value
        )

    def test_is_blob_path_uuid_format(self):
        """Test blob path detection for UUID-based paths."""
        parser = UnstructuredParser()
        # Valid UUID format path
        assert (
            parser._is_blob_path("550e8400-e29b-41d4-a716-446655440000/document.pdf")
            is True
        )

    def test_is_blob_path_local_path(self):
        """Test blob path detection for local paths."""
        parser = UnstructuredParser()
        # Absolute local path
        assert parser._is_blob_path("/home/user/documents/file.pdf") is False
        # Relative local path
        assert parser._is_blob_path("documents/file.pdf") is False
        # Simple filename
        assert parser._is_blob_path("file.pdf") is False

    def test_elements_to_text_single_element(self):
        """Test converting single element to text."""
        parser = UnstructuredParser()
        mock_element = MagicMock()
        mock_element.__str__ = MagicMock(return_value="Hello World")

        result = parser._elements_to_text([mock_element])
        assert result == "Hello World"

    def test_elements_to_text_multiple_elements(self):
        """Test converting multiple elements to text."""
        parser = UnstructuredParser()
        element1 = MagicMock()
        element1.__str__ = MagicMock(return_value="First paragraph")
        element2 = MagicMock()
        element2.__str__ = MagicMock(return_value="Second paragraph")

        result = parser._elements_to_text([element1, element2])
        assert result == "First paragraph\n\nSecond paragraph"

    def test_elements_to_text_empty_elements(self):
        """Test converting empty elements."""
        parser = UnstructuredParser()
        result = parser._elements_to_text([])
        assert result == ""

    def test_elements_to_text_strips_whitespace(self):
        """Test that whitespace is stripped from elements."""
        parser = UnstructuredParser()
        element = MagicMock()
        element.__str__ = MagicMock(return_value="  Hello World  \n")

        result = parser._elements_to_text([element])
        assert result == "Hello World"

    def test_elements_to_text_skips_empty_elements(self):
        """Test that empty elements are skipped."""
        parser = UnstructuredParser()
        element1 = MagicMock()
        element1.__str__ = MagicMock(return_value="Content")
        element2 = MagicMock()
        element2.__str__ = MagicMock(return_value="   ")  # Empty after strip

        result = parser._elements_to_text([element1, element2])
        assert result == "Content"


class TestUnstructuredParserAsync:
    """Async tests for UnstructuredParser."""

    @pytest.mark.asyncio
    async def test_parse_local_file_not_found(self):
        """Test parsing non-existent local file raises error."""
        parser = UnstructuredParser()

        with pytest.raises(DocumentProcessingError) as exc_info:
            await parser.parse("/nonexistent/path/document.pdf")

        assert "File not found" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_parse_unsupported_content_type(self, tmp_path):
        """Test parsing unsupported file type raises error."""
        parser = UnstructuredParser()

        # Create a file with unsupported extension
        test_file = tmp_path / "document.xyz"
        test_file.write_text("Some content")

        with pytest.raises(UnsupportedContentTypeError) as exc_info:
            await parser.parse(str(test_file))

        assert "application/octet-stream" in exc_info.value.content_type

    @pytest.mark.asyncio
    async def test_parse_bytes_unsupported_content_type(self):
        """Test parse_bytes with unsupported content type raises error."""
        parser = UnstructuredParser()
        content = io.BytesIO(b"Some content")

        with pytest.raises(UnsupportedContentTypeError):
            await parser.parse_bytes(
                content=content,
                filename="document.xyz",
                content_type="application/octet-stream",
            )

    @pytest.mark.asyncio
    async def test_parse_local_txt_file(self, tmp_path):
        """Test parsing a local TXT file."""
        parser = UnstructuredParser()

        # Create a test TXT file
        test_file = tmp_path / "test.txt"
        test_file.write_text(
            "Hello, this is a test document.\n\nWith multiple paragraphs."
        )

        with patch(
            "src.infrastructure.parsers.unstructured_parser.partition"
        ) as mock_partition:
            # Mock the partition function
            mock_element1 = MagicMock()
            mock_element1.__str__ = MagicMock(
                return_value="Hello, this is a test document."
            )
            mock_element2 = MagicMock()
            mock_element2.__str__ = MagicMock(return_value="With multiple paragraphs.")
            mock_partition.return_value = [mock_element1, mock_element2]

            result = await parser.parse(str(test_file))

            assert "Hello, this is a test document." in result
            assert "With multiple paragraphs." in result
            mock_partition.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_bytes_success(self):
        """Test parse_bytes successfully parses content."""
        parser = UnstructuredParser()
        content = io.BytesIO(b"Hello World")

        with patch(
            "src.infrastructure.parsers.unstructured_parser.partition"
        ) as mock_partition:
            mock_element = MagicMock()
            mock_element.__str__ = MagicMock(return_value="Hello World")
            mock_partition.return_value = [mock_element]

            result = await parser.parse_bytes(
                content=content,
                filename="test.txt",
                content_type=ContentType.TXT.value,
            )

            assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_parse_from_blob_storage(self):
        """Test parsing file from blob storage."""
        mock_storage = AsyncMock()
        mock_file = io.BytesIO(b"Blob content")
        mock_storage.download.return_value = mock_file

        parser = UnstructuredParser(blob_storage=mock_storage)

        with patch(
            "src.infrastructure.parsers.unstructured_parser.partition"
        ) as mock_partition:
            mock_element = MagicMock()
            mock_element.__str__ = MagicMock(return_value="Blob content")
            mock_partition.return_value = [mock_element]

            result = await parser.parse(
                "550e8400-e29b-41d4-a716-446655440000/document.txt"
            )

            assert result == "Blob content"
            mock_storage.download.assert_called_once_with(
                "550e8400-e29b-41d4-a716-446655440000/document.txt"
            )

    @pytest.mark.asyncio
    async def test_parse_from_blob_without_storage_configured(self):
        """Test parsing blob path without storage configured raises error."""
        parser = UnstructuredParser()  # No blob storage

        # This should fall through to local file parsing and fail
        with pytest.raises(DocumentProcessingError):
            await parser.parse("550e8400-e29b-41d4-a716-446655440000/document.txt")

    @pytest.mark.asyncio
    async def test_parse_wraps_unexpected_errors(self, tmp_path):
        """Test that unexpected errors are wrapped in DocumentProcessingError."""
        parser = UnstructuredParser()

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        with patch(
            "src.infrastructure.parsers.unstructured_parser.partition"
        ) as mock_partition:
            mock_partition.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(DocumentProcessingError) as exc_info:
                await parser.parse(str(test_file))

            assert "Unexpected error" in exc_info.value.reason
