"""Integration tests for UnstructuredParser.

These tests require:
- Running MinIO instance for blob storage tests
- Sample files in tests/integration/fixtures/

Run with: pytest tests/integration/test_unstructured_parser.py -m integration -v
"""

import io
from pathlib import Path

import pytest

from src.domain.exceptions import (
    DocumentProcessingError,
    StorageNotFoundError,
    UnsupportedContentTypeError,
)
from src.infrastructure.parsers import UnstructuredParser
from src.infrastructure.storage import MinIOStorage


@pytest.fixture
def parser_without_storage() -> UnstructuredParser:
    """Create parser without blob storage for local file tests."""
    return UnstructuredParser(blob_storage=None)


@pytest.fixture
def parser_with_storage(integration_minio_storage: MinIOStorage) -> UnstructuredParser:
    """Create parser with blob storage for MinIO tests."""
    return UnstructuredParser(blob_storage=integration_minio_storage)


@pytest.mark.integration
class TestUnstructuredParserLocalFiles:
    """Integration tests for parsing local files."""

    async def test_parse_local_txt_file(
        self,
        parser_without_storage: UnstructuredParser,
        sample_txt_path: Path,
    ) -> None:
        """Test parsing a local TXT file."""
        result = await parser_without_storage.parse(str(sample_txt_path))

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Enterprise RAG Platform" in result
        assert "Chapter 1" in result

    async def test_parse_local_md_file(
        self,
        parser_without_storage: UnstructuredParser,
        sample_md_path: Path,
    ) -> None:
        """Test parsing a local Markdown file."""
        result = await parser_without_storage.parse(str(sample_md_path))

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Enterprise Agentic RAG Platform" in result
        assert "Hybrid Search" in result

    async def test_parse_local_pdf_file(
        self,
        parser_without_storage: UnstructuredParser,
        sample_pdf_path: Path,
    ) -> None:
        """Test parsing a local PDF file."""
        result = await parser_without_storage.parse(str(sample_pdf_path))

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Enterprise RAG Platform" in result
        assert "Architecture" in result

    async def test_parse_local_docx_file(
        self,
        parser_without_storage: UnstructuredParser,
        sample_docx_path: Path,
    ) -> None:
        """Test parsing a local DOCX file."""
        result = await parser_without_storage.parse(str(sample_docx_path))

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Enterprise RAG Platform" in result
        assert "Clean Architecture" in result

    async def test_parse_extracts_chapters(
        self,
        parser_without_storage: UnstructuredParser,
        sample_txt_path: Path,
    ) -> None:
        """Test that parser extracts all chapters from document."""
        result = await parser_without_storage.parse(str(sample_txt_path))

        assert "Chapter 1: Introduction" in result
        assert "Chapter 2: Architecture" in result
        assert "Chapter 3: Ingestion Pipeline" in result
        assert "Chapter 4: Search and Retrieval" in result
        assert "Chapter 5: Generation" in result

    async def test_parse_preserves_text_structure(
        self,
        parser_without_storage: UnstructuredParser,
        sample_txt_path: Path,
    ) -> None:
        """Test that parser preserves text structure (paragraphs, lists)."""
        result = await parser_without_storage.parse(str(sample_txt_path))

        # Check bullet points content is preserved
        assert "Reduced hallucinations" in result
        assert "Access to proprietary" in result

    async def test_parse_nonexistent_file_raises_error(
        self,
        parser_without_storage: UnstructuredParser,
    ) -> None:
        """Test that parsing a non-existent file raises DocumentProcessingError."""
        with pytest.raises(DocumentProcessingError) as exc_info:
            await parser_without_storage.parse("/nonexistent/path/file.txt")

        assert "nonexistent" in str(exc_info.value).lower()


@pytest.mark.integration
class TestUnstructuredParserBytes:
    """Integration tests for parsing bytes content."""

    async def test_parse_bytes_txt(
        self,
        parser_without_storage: UnstructuredParser,
        sample_txt_content: str,
    ) -> None:
        """Test parsing TXT content from bytes."""
        content = io.BytesIO(sample_txt_content.encode("utf-8"))

        result = await parser_without_storage.parse_bytes(
            content=content,
            filename="test.txt",
            content_type="text/plain",
        )

        assert isinstance(result, str)
        assert "Enterprise RAG Platform" in result

    async def test_parse_bytes_md(
        self,
        parser_without_storage: UnstructuredParser,
        sample_md_content: str,
    ) -> None:
        """Test parsing Markdown content from bytes."""
        content = io.BytesIO(sample_md_content.encode("utf-8"))

        result = await parser_without_storage.parse_bytes(
            content=content,
            filename="test.md",
            content_type="text/markdown",
        )

        assert isinstance(result, str)
        assert "Enterprise Agentic RAG Platform" in result

    async def test_parse_bytes_unsupported_type_raises(
        self,
        parser_without_storage: UnstructuredParser,
    ) -> None:
        """Test that parsing unsupported content type raises error."""
        content = io.BytesIO(b"some binary data")

        with pytest.raises(UnsupportedContentTypeError):
            await parser_without_storage.parse_bytes(
                content=content,
                filename="test.xyz",
                content_type="application/xyz-unsupported",
            )

    async def test_parse_bytes_empty_content(
        self,
        parser_without_storage: UnstructuredParser,
    ) -> None:
        """Test parsing empty content returns empty string."""
        content = io.BytesIO(b"")

        result = await parser_without_storage.parse_bytes(
            content=content,
            filename="empty.txt",
            content_type="text/plain",
        )

        # Empty file should return empty or whitespace-only string
        assert result.strip() == ""


@pytest.mark.integration
class TestUnstructuredParserWithMinIO:
    """Integration tests for parsing files from MinIO blob storage."""

    async def test_parse_from_minio(
        self,
        parser_with_storage: UnstructuredParser,
        integration_minio_storage: MinIOStorage,
        sample_txt_content: str,
    ) -> None:
        """Test parsing a file stored in MinIO."""
        # Upload file to MinIO
        content = io.BytesIO(sample_txt_content.encode("utf-8"))
        file_path = await integration_minio_storage.upload(
            file=content,
            filename="test-document.txt",
            content_type="text/plain",
        )

        try:
            # Parse from MinIO path
            result = await parser_with_storage.parse(file_path)

            assert isinstance(result, str)
            assert "Enterprise RAG Platform" in result
            assert "Chapter 1" in result

        finally:
            # Cleanup
            await integration_minio_storage.delete(file_path)

    async def test_parse_markdown_from_minio(
        self,
        parser_with_storage: UnstructuredParser,
        integration_minio_storage: MinIOStorage,
        sample_md_content: str,
    ) -> None:
        """Test parsing a Markdown file stored in MinIO."""
        # Upload file to MinIO
        content = io.BytesIO(sample_md_content.encode("utf-8"))
        file_path = await integration_minio_storage.upload(
            file=content,
            filename="readme.md",
            content_type="text/markdown",
        )

        try:
            # Parse from MinIO path
            result = await parser_with_storage.parse(file_path)

            assert isinstance(result, str)
            assert "Enterprise Agentic RAG Platform" in result

        finally:
            # Cleanup
            await integration_minio_storage.delete(file_path)

    async def test_parse_pdf_from_minio(
        self,
        parser_with_storage: UnstructuredParser,
        integration_minio_storage: MinIOStorage,
        sample_pdf_path: Path,
    ) -> None:
        """Test parsing a PDF file stored in MinIO."""
        # Upload file to MinIO
        with sample_pdf_path.open("rb") as f:
            content = io.BytesIO(f.read())
        file_path = await integration_minio_storage.upload(
            file=content,
            filename="document.pdf",
            content_type="application/pdf",
        )

        try:
            # Parse from MinIO path
            result = await parser_with_storage.parse(file_path)

            assert isinstance(result, str)
            assert "Enterprise RAG Platform" in result

        finally:
            # Cleanup
            await integration_minio_storage.delete(file_path)

    async def test_parse_docx_from_minio(
        self,
        parser_with_storage: UnstructuredParser,
        integration_minio_storage: MinIOStorage,
        sample_docx_path: Path,
    ) -> None:
        """Test parsing a DOCX file stored in MinIO."""
        # Upload file to MinIO
        with sample_docx_path.open("rb") as f:
            content = io.BytesIO(f.read())
        file_path = await integration_minio_storage.upload(
            file=content,
            filename="document.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        try:
            # Parse from MinIO path
            result = await parser_with_storage.parse(file_path)

            assert isinstance(result, str)
            assert "Enterprise RAG Platform" in result

        finally:
            # Cleanup
            await integration_minio_storage.delete(file_path)

    async def test_parse_nonexistent_minio_path_raises_error(
        self,
        parser_with_storage: UnstructuredParser,
    ) -> None:
        """Test that parsing a non-existent MinIO path raises error."""
        # Use a blob-like path that doesn't exist
        fake_path = "documents/nonexistent-uuid/fake-file.txt"

        with pytest.raises((DocumentProcessingError, StorageNotFoundError)):
            await parser_with_storage.parse(fake_path)


@pytest.mark.integration
class TestUnstructuredParserContentTypes:
    """Integration tests for various content type support."""

    async def test_supports_txt(
        self,
        parser_without_storage: UnstructuredParser,
    ) -> None:
        """Test that parser supports text/plain."""
        assert parser_without_storage.supports("text/plain")

    async def test_supports_markdown(
        self,
        parser_without_storage: UnstructuredParser,
    ) -> None:
        """Test that parser supports text/markdown."""
        assert parser_without_storage.supports("text/markdown")

    async def test_supports_html(
        self,
        parser_without_storage: UnstructuredParser,
    ) -> None:
        """Test that parser supports text/html."""
        assert parser_without_storage.supports("text/html")

    async def test_supports_pdf(
        self,
        parser_without_storage: UnstructuredParser,
    ) -> None:
        """Test that parser supports application/pdf."""
        assert parser_without_storage.supports("application/pdf")

    async def test_supports_docx(
        self,
        parser_without_storage: UnstructuredParser,
    ) -> None:
        """Test that parser supports DOCX."""
        docx_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert parser_without_storage.supports(docx_type)

    async def test_not_supports_unknown_type(
        self,
        parser_without_storage: UnstructuredParser,
    ) -> None:
        """Test that parser does not support unknown types."""
        assert not parser_without_storage.supports("application/octet-stream")
        assert not parser_without_storage.supports("image/png")


@pytest.mark.integration
class TestUnstructuredParserPerformance:
    """Performance-related integration tests."""

    async def test_parse_large_text_content(
        self,
        parser_without_storage: UnstructuredParser,
    ) -> None:
        """Test parsing larger text content (simulated)."""
        # Create a larger document by repeating content
        base_content = (
            """
        This is a test paragraph with some content that will be repeated.
        It contains multiple sentences to simulate real document text.
        The content should be processed efficiently by the parser.
        """
            * 100
        )  # ~15KB of text

        content = io.BytesIO(base_content.encode("utf-8"))

        result = await parser_without_storage.parse_bytes(
            content=content,
            filename="large.txt",
            content_type="text/plain",
        )

        assert len(result) > 1000
        assert "test paragraph" in result

    async def test_parse_html_content(
        self,
        parser_without_storage: UnstructuredParser,
    ) -> None:
        """Test parsing HTML content."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Document</title></head>
        <body>
            <h1>Enterprise RAG System</h1>
            <p>This is a paragraph with important information.</p>
            <ul>
                <li>First item</li>
                <li>Second item</li>
            </ul>
            <h2>Architecture</h2>
            <p>The system uses clean architecture principles.</p>
        </body>
        </html>
        """
        content = io.BytesIO(html_content.encode("utf-8"))

        result = await parser_without_storage.parse_bytes(
            content=content,
            filename="test.html",
            content_type="text/html",
        )

        assert "Enterprise RAG System" in result
        assert "First item" in result
        assert "clean architecture" in result
