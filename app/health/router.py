import logging

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


logger = logging.getLogger(__name__)


@router.get("")
async def health_check() -> dict[str, str]:
    logger.info("received health check request")
    return {"status": "ok"}
