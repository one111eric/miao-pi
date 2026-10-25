import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

logger = logging.getLogger(__name__)

LifespanCallback = Callable[[FastAPI], Awaitable[None]]


def lifespan_function(
    on_startup: LifespanCallback | None = None,
    on_shutdown: LifespanCallback | None = None,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Create a FastAPI lifespan context with optional startup and shutdown hooks."""

    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Entering lifespan startup")
        if on_startup:
            await on_startup(app)

        logger.info("Lifespan startup complete")
        try:
            yield
        finally:
            logger.info("Entering lifespan shutdown")
            if on_shutdown:
                await on_shutdown(app)
            logger.info("Lifespan shutdown complete")

    return lifespan
