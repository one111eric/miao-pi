from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from httpx2 import Request
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from openai import APIError, APITimeoutError

from app.agent.graph import build_agent_graph


class TimeoutChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "timeout"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise APITimeoutError(request=Request("POST", "https://provider.invalid"))


class FailingChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise APIError(
            "sensitive provider details",
            request=Request("POST", "https://provider.invalid"),
            body=None,
        )


@pytest.mark.anyio
async def test_provider_timeout_returns_504(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    app.state.agent_graph = build_agent_graph(TimeoutChatModel())

    response = await client.post("/invoke", json={"message": "Hello"})

    assert response.status_code == 504
    assert response.json() == {"detail": "LLM provider request timed out"}


@pytest.mark.anyio
async def test_provider_error_returns_502_without_details(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    app.state.agent_graph = build_agent_graph(FailingChatModel())

    response = await client.post("/invoke", json={"message": "Hello"})

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM provider request failed"}
    assert "sensitive provider details" not in response.text
