from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.service import create_app


@pytest.fixture
def app() -> FastAPI:
    """Create an isolated application instance for each test."""
    return create_app()


@pytest.fixture
def anyio_backend() -> str:
    """Run asynchronous tests with asyncio."""
    return "asyncio"


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Provide an HTTP client with the application lifespan active."""
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as async_client:
            yield async_client
