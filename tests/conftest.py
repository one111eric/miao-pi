from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.service import create_app


class EchoChatModel(BaseChatModel):
    """Chat model that echoes the last message, so tests never call a real LLM."""

    @property
    def _llm_type(self) -> str:
        return "echo"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=f"Received: {messages[-1].text}")
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def app() -> FastAPI:
    """Create an isolated application instance for each test."""
    return create_app(llm=EchoChatModel())


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
