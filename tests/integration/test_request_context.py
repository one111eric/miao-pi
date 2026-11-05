import logging
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from app.agent.graph import build_agent_graph
from common.logging_helper import RequestFormatter


class BrokenChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "broken"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("unexpected bug")


@pytest.mark.anyio
async def test_generated_request_id_is_logged_and_returned(
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    original_formatter = caplog.handler.formatter
    caplog.handler.setFormatter(RequestFormatter("%(context)s%(message)s"))
    try:
        with caplog.at_level(logging.INFO, logger="app.health.router"):
            response = await client.get("/health")
        rendered_logs = caplog.text
    finally:
        caplog.handler.setFormatter(original_formatter)

    request_id = response.headers["X-Request-ID"]
    UUID(request_id)
    assert f"request_id={request_id}" in rendered_logs


@pytest.mark.anyio
async def test_supplied_request_id_is_preserved(client: AsyncClient) -> None:
    response = await client.get(
        "/health",
        headers={"X-Request-ID": "request-from-client"},
    )

    assert response.headers["X-Request-ID"] == "request-from-client"


@pytest.mark.parametrize(
    "supplied_request_id",
    ["abc level=ERROR", "x" * 129],
)
@pytest.mark.anyio
async def test_invalid_supplied_request_id_is_replaced(
    client: AsyncClient,
    supplied_request_id: str,
) -> None:
    response = await client.get(
        "/health",
        headers={"X-Request-ID": supplied_request_id},
    )

    UUID(response.headers["X-Request-ID"])


@pytest.mark.anyio
async def test_unhandled_error_keeps_the_request_id(
    app: FastAPI,
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app.state.agent_graph = build_agent_graph(
        BrokenChatModel(), checkpointer=app.state.thread_store.checkpointer
    )

    original_formatter = caplog.handler.formatter
    caplog.handler.setFormatter(RequestFormatter("%(context)s%(message)s"))
    try:
        with caplog.at_level(logging.ERROR, logger="app.service.middlewares"):
            response = await client.post(
                "/invoke",
                json={"message": "Hello"},
                headers={"X-Request-ID": "failing-request"},
            )
        rendered_logs = caplog.text
    finally:
        caplog.handler.setFormatter(original_formatter)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert response.headers["X-Request-ID"] == "failing-request"
    assert "request_id=failing-request" in rendered_logs
    assert "unexpected bug" not in response.text


@pytest.mark.anyio
async def test_each_request_logs_its_completion(
    app: FastAPI,
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app.state.agent_graph = build_agent_graph(
        BrokenChatModel(), checkpointer=app.state.thread_store.checkpointer
    )

    with caplog.at_level(logging.INFO, logger="app.service.middlewares"):
        await client.get("/health?token=not-for-logs")
        await client.post("/invoke", json={"message": ""})
        await client.post("/invoke", json={"message": "Hello"})

    completed = [
        record for record in caplog.records if record.message == "Request completed"
    ]
    # The path excludes the query string.
    assert [
        (
            record.__dict__["method"],
            record.__dict__["path"],
            record.__dict__["status_code"],
        )
        for record in completed
    ] == [("GET", "/health", 200), ("POST", "/invoke", 422), ("POST", "/invoke", 500)]
    assert all(record.__dict__["duration_ms"] >= 0 for record in completed)
