"""Health check endpoints.

Provides endpoints for monitoring application health and readiness.
"""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, status
from pydantic import BaseModel

from src.presentation.api.dependencies import (
    DocumentRepositoryDep,
    VectorStoreDep,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Health status ("healthy" or "unhealthy").
        timestamp: Current server timestamp.
        version: Application version.
    """

    status: str
    timestamp: datetime
    version: str


class ReadinessResponse(BaseModel):
    """Readiness check response.

    Attributes:
        ready: Whether the application is ready to serve requests.
        checks: Individual component check results.
    """

    ready: bool
    checks: dict[str, bool]


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if the application is running.",
)
async def health_check() -> HealthResponse:
    """Check application health.

    Returns:
        HealthResponse with current status.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC),
        version="0.1.0",
    )


@router.get(
    "/live",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Kubernetes liveness probe endpoint.",
)
async def liveness_probe() -> HealthResponse:
    """Kubernetes liveness probe.

    Returns:
        HealthResponse indicating the application is alive.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC),
        version="0.1.0",
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Kubernetes readiness probe endpoint.",
)
async def readiness_probe(
    document_repo: DocumentRepositoryDep,
    vector_store: VectorStoreDep,
) -> ReadinessResponse:
    """Kubernetes readiness probe.

    Checks if all dependencies are available by performing actual
    connectivity tests to the database and vector store.

    Args:
        document_repo: Injected document repository.
        vector_store: Injected vector store.

    Returns:
        ReadinessResponse with individual component statuses.
    """
    checks: dict[str, bool] = {}

    # Check database connectivity
    try:
        await document_repo.count()
        checks["database"] = True
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        checks["database"] = False

    # Check vector store connectivity
    try:
        await vector_store.get_collection_stats()
        checks["vector_store"] = True
    except Exception as e:
        logger.warning(f"Vector store health check failed: {e}")
        checks["vector_store"] = False

    return ReadinessResponse(
        ready=all(checks.values()),
        checks=checks,
    )
