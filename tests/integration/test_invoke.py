import logging

import pytest
from httpx import AsyncClient

from common.logging_helper import RequestFormatter


@pytest.mark.anyio
async def test_invoke_agent(
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    original_formatter = caplog.handler.formatter
    caplog.handler.setFormatter(RequestFormatter("%(context)s%(message)s"))
    try:
        with caplog.at_level(logging.INFO, logger="app.agent"):
            response = await client.post(
                "/invoke",
                json={"message": "Hello", "thread_id": "thread-1"},
            )
        rendered_logs = caplog.text
    finally:
        caplog.handler.setFormatter(original_formatter)

    assert response.status_code == 200
    assert response.json() == {
        "response": "Received: Hello",
        "thread_id": "thread-1",
    }
    model_log = next(
        record for record in caplog.records if record.message == "LLM call completed"
    )
    assert model_log.__dict__["model_duration_ms"] >= 0
    assert model_log.__dict__["input_tokens"] == 2
    assert model_log.__dict__["output_tokens"] == 3
    assert model_log.__dict__["total_tokens"] == 5
    completed_log = next(
        record
        for record in caplog.records
        if record.message == "Agent invocation completed"
    )
    assert completed_log.__dict__["conversation_message_count"] == 2
    assert "request_id=" in rendered_logs
    assert "thread_id=thread-1" in rendered_logs


@pytest.mark.anyio
async def test_invoke_rejects_an_empty_message(client: AsyncClient) -> None:
    response = await client.post("/invoke", json={"message": ""})

    assert response.status_code == 422


@pytest.mark.anyio
@pytest.mark.parametrize(("thread_id", "supplied"), [(None, False), ("thread-1", True)])
async def test_invocation_log_says_whether_a_thread_was_supplied(
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
    thread_id: str | None,
    supplied: bool,
) -> None:
    with caplog.at_level(logging.INFO, logger="app.agent"):
        await client.post("/invoke", json={"message": "Hello", "thread_id": thread_id})

    started_log = next(
        record
        for record in caplog.records
        if record.message == "Agent invocation started"
    )
    assert started_log.__dict__["thread_supplied"] is supplied
