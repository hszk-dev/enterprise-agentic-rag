"""Health check endpoints.

Provides endpoints for monitoring application health and readiness.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, status
from pydantic import BaseModel

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
async def readiness_probe() -> ReadinessResponse:
    """Kubernetes readiness probe.

    Checks if all dependencies are available.

    Returns:
        ReadinessResponse with individual component statuses.

    Note:
        In a production setup, this would check:
        - Database connectivity
        - Vector store connectivity
        - Redis connectivity
        - External API availability
    """
    # TODO: Add actual dependency checks
    checks = {
        "database": True,
        "vector_store": True,
        "cache": True,
    }
    return ReadinessResponse(
        ready=all(checks.values()),
        checks=checks,
    )
