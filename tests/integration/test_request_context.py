import logging
from uuid import UUID

import pytest
from httpx import AsyncClient

from common.logging_helper import RequestFormatter


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
