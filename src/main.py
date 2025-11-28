"""FastAPI application entry point.

This module creates and configures the FastAPI application instance.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from src.domain.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    DomainError,
    EmbeddingError,
    LLMError,
    RateLimitError,
    SearchError,
    StorageError,
    UnsupportedContentTypeError,
)
from src.presentation.api.dependencies import (
    get_document_repository,
    get_vector_store,
)
from src.presentation.api.v1 import documents, health, query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Enterprise Agentic RAG Platform...")
    settings = get_settings()
    logger.info(f"Environment: debug={settings.debug}, log_level={settings.log_level}")

    # Initialize database schema
    repo = get_document_repository()
    await repo.initialize()
    logger.info("Database initialized successfully")

    # Initialize vector store collection
    vector_store = get_vector_store()
    await vector_store.initialize()
    logger.info("Vector store initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Enterprise Agentic RAG Platform...")
    await vector_store.close()
    await repo.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Enterprise Agentic RAG Platform - Advanced RAG with Hybrid Search",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
    )

    # Add CORS middleware (configured via environment variables)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allowed_origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allowed_methods,
        allow_headers=settings.cors.allowed_headers,
    )

    # Register exception handlers
    _register_exception_handlers(app)

    # Include routers
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        """Root endpoint."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "running",
        }

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers.

    Maps domain exceptions to appropriate HTTP responses.
    """

    @app.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(
        request: Request, exc: DocumentNotFoundError
    ) -> JSONResponse:
        """Handle document not found errors."""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": exc.message, "code": exc.code},
        )

    @app.exception_handler(UnsupportedContentTypeError)
    async def unsupported_content_type_handler(
        request: Request, exc: UnsupportedContentTypeError
    ) -> JSONResponse:
        """Handle unsupported content type errors."""
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={"detail": exc.message, "code": exc.code},
        )

    @app.exception_handler(RateLimitError)
    async def rate_limit_handler(request: Request, exc: RateLimitError) -> JSONResponse:
        """Handle rate limit errors."""
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": exc.message, "code": exc.code},
            headers={"Retry-After": "60"},  # Suggest retry after 60 seconds
        )

    @app.exception_handler(StorageError)
    async def storage_error_handler(
        request: Request, exc: StorageError
    ) -> JSONResponse:
        """Handle storage errors."""
        logger.error(f"Storage error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Storage operation failed", "code": exc.code},
        )

    @app.exception_handler(EmbeddingError)
    async def embedding_error_handler(
        request: Request, exc: EmbeddingError
    ) -> JSONResponse:
        """Handle embedding errors."""
        logger.error(f"Embedding error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Embedding generation failed", "code": exc.code},
        )

    @app.exception_handler(SearchError)
    async def search_error_handler(request: Request, exc: SearchError) -> JSONResponse:
        """Handle search errors."""
        logger.error(f"Search error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Search operation failed", "code": exc.code},
        )

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        """Handle LLM errors."""
        logger.error(f"LLM error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "LLM generation failed", "code": exc.code},
        )

    @app.exception_handler(DocumentProcessingError)
    async def document_processing_error_handler(
        request: Request, exc: DocumentProcessingError
    ) -> JSONResponse:
        """Handle document processing errors."""
        logger.error(f"Document processing error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": exc.message, "code": exc.code},
        )

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        """Handle generic domain errors."""
        logger.error(f"Domain error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": exc.message, "code": exc.code},
        )


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
