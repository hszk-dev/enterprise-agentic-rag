"""Dependency injection configuration for FastAPI.

This module provides factory functions for creating service instances
using FastAPI's dependency injection system. All dependencies are
scoped per-request by default.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from config.settings import Settings, get_settings
from src.application.services.chunking_service import LangChainChunkingService
from src.application.services.generation_service import GenerationService
from src.application.services.ingestion_service import IngestionService
from src.application.services.search_service import SearchService
from src.domain.interfaces import (
    BlobStorage,
    ChunkingService,
    DocumentParser,
    DocumentRepository,
    EmbeddingService,
    LLMService,
    Reranker,
    SparseEmbeddingService,
    VectorStore,
)
from src.infrastructure.embeddings.fastembed_sparse import (
    FastEmbedSparseEmbeddingService,
)
from src.infrastructure.embeddings.openai_embedding import OpenAIEmbeddingService
from src.infrastructure.llm.openai_llm import OpenAILLMService
from src.infrastructure.parsers.unstructured_parser import UnstructuredParser
from src.infrastructure.repositories.postgres_document_repository import (
    PostgresDocumentRepository,
)
from src.infrastructure.rerankers.cohere_reranker import CohereReranker
from src.infrastructure.storage.minio_storage import MinIOStorage
from src.infrastructure.vectorstores.qdrant_vectorstore import QdrantVectorStore

# Type alias for cleaner dependency annotations
SettingsDep = Annotated[Settings, Depends(get_settings)]


# =============================================================================
# Infrastructure Layer Dependencies
# =============================================================================


@lru_cache
def get_blob_storage(settings: Settings = Depends(get_settings)) -> BlobStorage:
    """Get the blob storage instance (MinIO).

    Returns:
        BlobStorage instance for file operations.
    """
    return MinIOStorage(settings.minio)


@lru_cache
def get_document_repository(
    settings: Settings = Depends(get_settings),
) -> DocumentRepository:
    """Get the document repository instance (PostgreSQL).

    Returns:
        DocumentRepository instance for document metadata.
    """
    return PostgresDocumentRepository(settings.database)


@lru_cache
def get_vector_store(settings: Settings = Depends(get_settings)) -> VectorStore:
    """Get the vector store instance (Qdrant).

    Returns:
        VectorStore instance for vector operations.
    """
    return QdrantVectorStore(
        settings.qdrant,
        embedding_dim=settings.openai.embedding_dimensions,
    )


@lru_cache
def get_embedding_service(
    settings: Settings = Depends(get_settings),
) -> EmbeddingService:
    """Get the dense embedding service instance (OpenAI).

    Returns:
        EmbeddingService instance for dense embeddings.
    """
    return OpenAIEmbeddingService(settings.openai)


@lru_cache
def get_sparse_embedding_service() -> SparseEmbeddingService:
    """Get the sparse embedding service instance (FastEmbed SPLADE).

    Returns:
        SparseEmbeddingService instance for sparse embeddings.
    """
    return FastEmbedSparseEmbeddingService()


@lru_cache
def get_reranker(settings: Settings = Depends(get_settings)) -> Reranker:
    """Get the reranker instance (Cohere).

    Returns:
        Reranker instance for result reranking.
    """
    return CohereReranker(settings.cohere)


@lru_cache
def get_llm_service(settings: Settings = Depends(get_settings)) -> LLMService:
    """Get the LLM service instance (OpenAI).

    Returns:
        LLMService instance for text generation.
    """
    return OpenAILLMService(settings.openai)


def get_document_parser(
    blob_storage: BlobStorage = Depends(get_blob_storage),
) -> DocumentParser:
    """Get the document parser instance (Unstructured).

    Args:
        blob_storage: Blob storage for downloading remote files.

    Returns:
        DocumentParser instance for text extraction.
    """
    return UnstructuredParser(blob_storage)


def get_chunking_service(
    settings: Settings = Depends(get_settings),
) -> ChunkingService:
    """Get the chunking service instance (LangChain).

    Args:
        settings: Application settings.

    Returns:
        ChunkingService instance for text chunking.
    """
    return LangChainChunkingService(settings.chunking)


# =============================================================================
# Application Layer Dependencies
# =============================================================================


def get_ingestion_service(
    document_repo: DocumentRepository = Depends(get_document_repository),
    blob_storage: BlobStorage = Depends(get_blob_storage),
    vector_store: VectorStore = Depends(get_vector_store),
    dense_embedding: EmbeddingService = Depends(get_embedding_service),
    sparse_embedding: SparseEmbeddingService = Depends(get_sparse_embedding_service),
    parser: DocumentParser = Depends(get_document_parser),
    chunker: ChunkingService = Depends(get_chunking_service),
) -> IngestionService:
    """Get the ingestion service instance.

    Combines all required dependencies for document ingestion.

    Returns:
        IngestionService instance.
    """
    return IngestionService(
        document_repo=document_repo,
        blob_storage=blob_storage,
        vector_store=vector_store,
        dense_embedding=dense_embedding,
        sparse_embedding=sparse_embedding,
        parser=parser,
        chunker=chunker,
    )


def get_search_service(
    vector_store: VectorStore = Depends(get_vector_store),
    dense_embedding: EmbeddingService = Depends(get_embedding_service),
    sparse_embedding: SparseEmbeddingService = Depends(get_sparse_embedding_service),
    reranker: Reranker = Depends(get_reranker),
) -> SearchService:
    """Get the search service instance.

    Combines all required dependencies for hybrid search.

    Returns:
        SearchService instance.
    """
    return SearchService(
        vector_store=vector_store,
        dense_embedding=dense_embedding,
        sparse_embedding=sparse_embedding,
        reranker=reranker,
    )


def get_generation_service(
    llm_service: LLMService = Depends(get_llm_service),
) -> GenerationService:
    """Get the generation service instance.

    Returns:
        GenerationService instance.
    """
    return GenerationService(llm_service=llm_service)


# =============================================================================
# Type Aliases for Dependency Injection
# =============================================================================

BlobStorageDep = Annotated[BlobStorage, Depends(get_blob_storage)]
DocumentRepositoryDep = Annotated[DocumentRepository, Depends(get_document_repository)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
SparseEmbeddingServiceDep = Annotated[
    SparseEmbeddingService, Depends(get_sparse_embedding_service)
]
RerankerDep = Annotated[Reranker, Depends(get_reranker)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
DocumentParserDep = Annotated[DocumentParser, Depends(get_document_parser)]
ChunkingServiceDep = Annotated[ChunkingService, Depends(get_chunking_service)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
GenerationServiceDep = Annotated[GenerationService, Depends(get_generation_service)]
