import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI
from starlette.types import StatelessLifespan

logger = logging.getLogger(__name__)

type LifespanCallback = Callable[[FastAPI], Awaitable[None]]


def lifespan_function(
    on_startup: LifespanCallback | None = None,
    on_shutdown: LifespanCallback | None = None,
) -> StatelessLifespan[FastAPI]:
    """Create a FastAPI lifespan context with optional startup and shutdown hooks."""

    @asynccontextmanager
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

    # The contextlib return type is compatible with Starlette's lifespan protocol,
    # but current type stubs do not preserve that relationship through the decorator.
    return cast(StatelessLifespan[FastAPI], lifespan)
