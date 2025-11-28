"""Chunking service using LangChain's RecursiveCharacterTextSplitter.

Splits documents into smaller chunks suitable for embedding and retrieval.
"""

import logging
from typing import Any
from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import ChunkingSettings
from src.domain.entities import Chunk
from src.domain.interfaces import ChunkingService

logger = logging.getLogger(__name__)


class LangChainChunkingService(ChunkingService):
    """Chunking service using LangChain's RecursiveCharacterTextSplitter.

    This service splits text into chunks using a hierarchical approach:
    1. Try to split on double newlines (paragraphs)
    2. Fall back to single newlines
    3. Fall back to sentence boundaries (". ")
    4. Fall back to word boundaries (" ")
    5. Finally, split at character level if needed

    This ensures that chunks maintain semantic coherence while staying
    within the configured size limits.
    """

    def __init__(
        self,
        settings: ChunkingSettings | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize the chunking service.

        Args:
            settings: ChunkingSettings instance. If provided, other params are ignored.
            chunk_size: Maximum chunk size in characters (default: 1000).
            chunk_overlap: Overlap between consecutive chunks (default: 200).
            separators: List of separators to try in order.
        """
        if settings:
            self._chunk_size = settings.chunk_size
            self._chunk_overlap = settings.chunk_overlap
            self._separators = settings.separators
        else:
            self._chunk_size = chunk_size or 1000
            self._chunk_overlap = chunk_overlap or 200
            self._separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=self._separators,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"Initialized chunking service: size={self._chunk_size}, "
            f"overlap={self._chunk_overlap}"
        )

    def chunk(
        self,
        text: str,
        document_id: UUID,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Full text to chunk.
            document_id: Parent document ID.
            metadata: Metadata to attach to each chunk.

        Returns:
            List of Chunk entities with position information.
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for document {document_id}")
            return []

        # Split text into chunks with metadata
        documents = self._splitter.create_documents(
            texts=[text],
            metadatas=[metadata or {}],
        )

        chunks: list[Chunk] = []
        current_position = 0

        for index, doc in enumerate(documents):
            chunk_text = doc.page_content

            # Find the start position of this chunk in the original text
            # This handles the case where chunks may overlap
            start_char = text.find(chunk_text, current_position)
            if start_char == -1:
                # Fallback: search from beginning if not found after current position
                start_char = text.find(chunk_text)
            if start_char == -1:
                # If still not found, use current position as approximation
                start_char = current_position

            end_char = start_char + len(chunk_text)

            # Create chunk entity
            chunk = Chunk.create(
                document_id=document_id,
                content=chunk_text,
                chunk_index=index,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    **(metadata or {}),
                    **doc.metadata,
                },
            )
            chunks.append(chunk)

            # Update position for next search, accounting for overlap
            current_position = max(
                current_position,
                end_char - self._chunk_overlap,
            )

        logger.info(
            f"Chunked document {document_id}: {len(chunks)} chunks "
            f"from {len(text)} characters"
        )

        return chunks

    @property
    def chunk_size(self) -> int:
        """Return configured chunk size in characters."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Return configured chunk overlap in characters."""
        return self._chunk_overlap

    @property
    def separators(self) -> list[str]:
        """Return configured separators."""
        return self._separators.copy()
