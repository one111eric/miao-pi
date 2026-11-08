from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.agent.workspace import Workspace
from app.service import create_app
from tests.fakes import EchoChatModel


@pytest.fixture(autouse=True)
def no_sandbox_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the sandbox self-test at app startup unless a test asks for it."""
    monkeypatch.setenv("SANDBOX_BACKEND", "off")


@pytest.fixture
def echo_model() -> EchoChatModel:
    """The fake chat model the app uses; its ``calls`` show what the model saw."""
    return EchoChatModel()


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Provide an isolated workspace with no project context by default."""
    return Workspace(tmp_path)


@pytest.fixture
def app(echo_model: EchoChatModel, workspace: Workspace) -> FastAPI:
    """Create an isolated application instance for each test."""
    return create_app(llm=echo_model, workspace=workspace)


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
