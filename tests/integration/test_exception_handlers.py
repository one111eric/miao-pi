import logging
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
from common.logging_helper import RequestFormatter


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
    app.state.agent_graph = build_agent_graph(
        TimeoutChatModel(), checkpointer=app.state.thread_store.checkpointer
    )

    response = await client.post("/invoke", json={"message": "Hello"})

    assert response.status_code == 504
    assert response.json() == {"detail": "LLM provider request timed out"}


@pytest.mark.anyio
async def test_provider_error_returns_502_without_details(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    app.state.agent_graph = build_agent_graph(
        FailingChatModel(), checkpointer=app.state.thread_store.checkpointer
    )

    response = await client.post("/invoke", json={"message": "Hello"})

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM provider request failed"}
    assert "sensitive provider details" not in response.text


@pytest.mark.anyio
async def test_failed_model_call_is_logged_with_its_thread_and_duration(
    app: FastAPI,
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app.state.agent_graph = build_agent_graph(
        FailingChatModel(), checkpointer=app.state.thread_store.checkpointer
    )

    original_formatter = caplog.handler.formatter
    caplog.handler.setFormatter(RequestFormatter("%(context)s%(message)s"))
    try:
        with caplog.at_level(logging.WARNING, logger="app.agent.nodes"):
            await client.post(
                "/invoke",
                json={"message": "Hello", "thread_id": "thread-1"},
            )
        rendered_logs = caplog.text
    finally:
        caplog.handler.setFormatter(original_formatter)

    failed_log = next(
        record for record in caplog.records if record.message == "LLM call failed"
    )
    assert failed_log.__dict__["error_type"] == "APIError"
    assert failed_log.__dict__["model_duration_ms"] >= 0
    failed_line = next(
        line for line in rendered_logs.splitlines() if line.endswith("LLM call failed")
    )
    assert "thread_id=thread-1" in failed_line
    assert "sensitive provider details" not in rendered_logs
