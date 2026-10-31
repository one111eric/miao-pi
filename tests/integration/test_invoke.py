import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_invoke_agent(client: AsyncClient) -> None:
    response = await client.post(
        "/invoke",
        json={"message": "Hello", "thread_id": "thread-1"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": "Received: Hello",
        "thread_id": "thread-1",
    }


@pytest.mark.anyio
async def test_invoke_rejects_an_empty_message(client: AsyncClient) -> None:
    response = await client.post("/invoke", json={"message": ""})

    assert response.status_code == 422
