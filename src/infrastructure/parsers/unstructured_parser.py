"""Document parser using Unstructured library.

Supports PDF, DOCX, TXT, Markdown, and HTML documents.
"""

from __future__ import annotations

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from unstructured.partition.auto import partition

if TYPE_CHECKING:
    from src.domain.interfaces import BlobStorage

from src.domain.exceptions import DocumentProcessingError, UnsupportedContentTypeError
from src.domain.interfaces import DocumentParser
from src.domain.value_objects import ContentType

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound parsing operations
_executor = ThreadPoolExecutor(max_workers=4)

# Supported content types for this parser
SUPPORTED_CONTENT_TYPES: set[str] = {
    ContentType.PDF.value,
    ContentType.DOCX.value,
    ContentType.TXT.value,
    ContentType.MD.value,
    ContentType.HTML.value,
}


class UnstructuredParser(DocumentParser):
    """Document parser using Unstructured library.

    This parser uses the unstructured library to extract text from various
    document formats. It runs parsing in a thread pool to avoid blocking
    the async event loop.

    Supported formats:
        - PDF (application/pdf)
        - DOCX (application/vnd.openxmlformats-officedocument.wordprocessingml.document)
        - TXT (text/plain)
        - Markdown (text/markdown)
        - HTML (text/html)
    """

    def __init__(self, blob_storage: BlobStorage | None = None) -> None:
        """Initialize the parser.

        Args:
            blob_storage: Optional blob storage for downloading remote files.
                         If None, only local files are supported.
        """
        self._blob_storage = blob_storage

    async def parse(self, file_path: str) -> str:
        """Parse a document and extract text content.

        This method handles both local files and blob storage paths.
        For blob storage paths, it downloads the file to a temporary location,
        parses it, and cleans up afterward.

        Args:
            file_path: Path to the document. Can be a local path or blob storage path.

        Returns:
            Extracted text content with elements separated by double newlines.

        Raises:
            DocumentProcessingError: If parsing fails.
            UnsupportedContentTypeError: If the file type is not supported.
        """
        logger.info(f"Parsing document: {file_path}")

        try:
            # Check if this is a blob storage path that needs downloading
            if self._blob_storage and self._is_blob_path(file_path):
                return await self._parse_from_blob(file_path)

            # Parse local file
            return await self._parse_local_file(file_path)

        except UnsupportedContentTypeError:
            raise
        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse document {file_path}: {e}")
            raise DocumentProcessingError(
                document_id=file_path,
                reason=str(e),
            ) from e

    async def parse_bytes(
        self,
        content: BinaryIO,
        filename: str,
        content_type: str,
    ) -> str:
        """Parse document content from bytes.

        Args:
            content: Binary content of the document.
            filename: Original filename (used for format detection).
            content_type: MIME type of the document.

        Returns:
            Extracted text content.

        Raises:
            DocumentProcessingError: If parsing fails.
            UnsupportedContentTypeError: If the content type is not supported.
        """
        if not self.supports(content_type):
            raise UnsupportedContentTypeError(content_type)

        try:
            # Read content
            data = content.read()

            # Run parsing in thread pool
            # Note: partition() requires file=BytesIO for in-memory content
            loop = asyncio.get_event_loop()
            elements = await loop.run_in_executor(
                _executor,
                lambda: partition(file=io.BytesIO(data), metadata_filename=filename),
            )

            # Combine elements into text
            text = self._elements_to_text(elements)
            logger.info(f"Parsed {filename}: {len(text)} characters")
            return text

        except UnsupportedContentTypeError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse bytes for {filename}: {e}")
            raise DocumentProcessingError(
                document_id=filename,
                reason=str(e),
            ) from e

    def supports(self, content_type: str) -> bool:
        """Check if this parser supports a content type.

        Args:
            content_type: MIME type to check.

        Returns:
            True if supported, False otherwise.
        """
        return content_type in SUPPORTED_CONTENT_TYPES

    async def _parse_local_file(self, file_path: str) -> str:
        """Parse a local file.

        Args:
            file_path: Path to local file.

        Returns:
            Extracted text content.
        """
        path = Path(file_path)

        if not path.exists():
            raise DocumentProcessingError(
                document_id=file_path,
                reason=f"File not found: {file_path}",
            )

        # Detect content type from extension
        content_type = self._detect_content_type(path)
        if not self.supports(content_type):
            raise UnsupportedContentTypeError(content_type)

        # Run parsing in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        elements = await loop.run_in_executor(
            _executor,
            lambda: partition(filename=str(path)),
        )

        text = self._elements_to_text(elements)
        logger.info(f"Parsed {file_path}: {len(text)} characters")
        return text

    async def _parse_from_blob(self, blob_path: str) -> str:
        """Parse a file from blob storage.

        Downloads the file, parses it, and returns the text.

        Args:
            blob_path: Path in blob storage.

        Returns:
            Extracted text content.
        """
        if not self._blob_storage:
            raise DocumentProcessingError(
                document_id=blob_path,
                reason="Blob storage not configured",
            )

        # Download file from blob storage
        file_obj = await self._blob_storage.download(blob_path)

        # Detect content type from path
        content_type = self._detect_content_type(Path(blob_path))
        if not self.supports(content_type):
            raise UnsupportedContentTypeError(content_type)

        # Parse the downloaded content
        return await self.parse_bytes(
            content=file_obj,
            filename=Path(blob_path).name,
            content_type=content_type,
        )

    def _is_blob_path(self, path: str) -> bool:
        """Check if a path is a blob storage path.

        Blob paths are in format: documents/{uuid}/{filename}
        Local paths are absolute or relative file paths.

        Args:
            path: Path to check.

        Returns:
            True if this looks like a blob storage path.
        """
        # Blob paths start with "documents/" prefix and contain a UUID segment
        parts = path.split("/")
        if len(parts) >= 3 and parts[0] == "documents":
            # Check if second part looks like a UUID (36 chars with hyphens)
            uuid_part = parts[1]
            if len(uuid_part) == 36 and uuid_part.count("-") == 4:
                return True
        return False

    def _detect_content_type(self, path: Path) -> str:
        """Detect content type from file extension.

        Args:
            path: File path.

        Returns:
            MIME type string.
        """
        extension_map = {
            ".pdf": ContentType.PDF.value,
            ".docx": ContentType.DOCX.value,
            ".txt": ContentType.TXT.value,
            ".md": ContentType.MD.value,
            ".markdown": ContentType.MD.value,
            ".html": ContentType.HTML.value,
            ".htm": ContentType.HTML.value,
        }

        suffix = path.suffix.lower()
        return extension_map.get(suffix, "application/octet-stream")

    def _elements_to_text(self, elements: list[Any]) -> str:
        """Convert unstructured elements to plain text.

        Args:
            elements: List of unstructured Element objects.

        Returns:
            Combined text with elements separated by double newlines.
        """
        texts = []
        for element in elements:
            text = str(element).strip()
            if text:
                texts.append(text)

        return "\n\n".join(texts)
