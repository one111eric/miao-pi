import logging

from fastapi import APIRouter

from app.health.models import HealthCheckResponse

router = APIRouter(prefix="/health", tags=["health"])


logger = logging.getLogger(__name__)


@router.get(
    "",
    response_model=HealthCheckResponse,
    description="Check the health of the server",
    operation_id="healthCheck",
    responses={
        200: {
            "description": "OK",
            "model": HealthCheckResponse,
        },
        500: {
            "description": "Internal Server Error",
        },
    },
)
async def health_check() -> HealthCheckResponse:
    logger.info("received health check request")
    return HealthCheckResponse(status="ok")
