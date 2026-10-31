import logging

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_invoke_agent(
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="app.agent.nodes"):
        response = await client.post(
            "/invoke",
            json={"message": "Hello", "thread_id": "thread-1"},
        )

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


@pytest.mark.anyio
async def test_invoke_rejects_an_empty_message(client: AsyncClient) -> None:
    response = await client.post("/invoke", json={"message": ""})

    assert response.status_code == 422
