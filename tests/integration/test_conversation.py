import asyncio
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.agent.graph import build_agent_graph
from app.agent.nodes import SYSTEM_PROMPT
from app.agent.threads import ThreadStore
from tests.fakes import EchoChatModel


class GatedChatModel(BaseChatModel):
    """Holds every call until ``release`` is set, so a run stays in flight."""

    started: asyncio.Event = Field(default_factory=asyncio.Event)
    release: asyncio.Event = Field(default_factory=asyncio.Event)

    @property
    def _llm_type(self) -> str:
        return "gated"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("only the async path is used")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.started.set()
        await self.release.wait()
        message = AIMessage(content=f"Received: {messages[-1].text}")
        return ChatResult(generations=[ChatGeneration(message=message)])


def summarize(messages: list[BaseMessage]) -> list[tuple[str, str]]:
    """Reduce messages to (type, text) pairs so they compare without ids."""
    return [(message.type, message.text) for message in messages]


async def send(client: AsyncClient, message: str, thread_id: str | None) -> str:
    body = {"message": message}
    if thread_id is not None:
        body["thread_id"] = thread_id
    response = await client.post("/invoke", json=body)
    assert response.status_code == 200
    returned_thread_id: str = response.json()["thread_id"]
    return returned_thread_id


@pytest.mark.anyio
async def test_same_thread_continues_the_conversation(
    client: AsyncClient,
    echo_model: EchoChatModel,
) -> None:
    await send(client, "Hello", "thread-1")
    await send(client, "Again", "thread-1")

    assert summarize(echo_model.calls[1]) == [
        ("system", SYSTEM_PROMPT),
        ("human", "Hello"),
        ("ai", "Received: Hello"),
        ("human", "Again"),
    ]


@pytest.mark.anyio
async def test_threads_do_not_share_history(
    client: AsyncClient,
    echo_model: EchoChatModel,
) -> None:
    await send(client, "Hello", "thread-1")
    await send(client, "Other", "thread-2")

    assert summarize(echo_model.calls[1]) == [
        ("system", SYSTEM_PROMPT),
        ("human", "Other"),
    ]


@pytest.mark.anyio
async def test_a_request_without_thread_id_starts_a_new_thread(
    client: AsyncClient,
    echo_model: EchoChatModel,
) -> None:
    first_thread_id = await send(client, "Hello", None)
    second_thread_id = await send(client, "Hello", None)
    continued_thread_id = await send(client, "Again", first_thread_id)

    UUID(first_thread_id)
    assert second_thread_id != first_thread_id
    assert continued_thread_id == first_thread_id
    assert summarize(echo_model.calls[2])[1:] == [
        ("human", "Hello"),
        ("ai", "Received: Hello"),
        ("human", "Again"),
    ]


@pytest.mark.anyio
async def test_system_prompt_is_sent_but_not_stored(
    app: FastAPI,
    client: AsyncClient,
    echo_model: EchoChatModel,
) -> None:
    await send(client, "Hello", "thread-1")

    first_message = echo_model.calls[0][0]
    assert isinstance(first_message, SystemMessage)
    assert first_message.text == SYSTEM_PROMPT

    snapshot = await app.state.agent_graph.aget_state(
        {"configurable": {"thread_id": "thread-1"}}
    )
    stored = snapshot.values["messages"]
    assert [type(message) for message in stored] == [HumanMessage, AIMessage]


@pytest.mark.parametrize(
    "thread_id",
    ["", "has space", "line\nbreak", "x" * 129],
)
@pytest.mark.anyio
async def test_invalid_thread_id_is_rejected(
    client: AsyncClient,
    thread_id: str,
) -> None:
    response = await client.post(
        "/invoke",
        json={"message": "Hello", "thread_id": thread_id},
    )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_a_second_request_while_the_thread_is_running_gets_409(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    model = GatedChatModel()
    app.state.agent_graph = build_agent_graph(
        model,
        checkpointer=app.state.thread_store.checkpointer,
    )

    first = asyncio.create_task(
        client.post("/invoke", json={"message": "one", "thread_id": "thread-1"})
    )
    try:
        await asyncio.wait_for(model.started.wait(), timeout=2)
        # Without the busy check this request would wait on the gated model,
        # so the timeout turns that into a failure instead of a hang.
        second = await asyncio.wait_for(
            client.post("/invoke", json={"message": "two", "thread_id": "thread-1"}),
            timeout=2,
        )
    finally:
        model.release.set()
    first_response = await asyncio.wait_for(first, timeout=2)

    assert second.status_code == 409
    assert second.json() == {"detail": "Thread already has a run in progress"}
    assert first_response.status_code == 200
    snapshot = await app.state.agent_graph.aget_state(
        {"configurable": {"thread_id": "thread-1"}}
    )
    assert summarize(snapshot.values["messages"]) == [
        ("human", "one"),
        ("ai", "Received: one"),
    ]


@pytest.mark.anyio
async def test_an_evicted_thread_starts_fresh(
    app: FastAPI,
    client: AsyncClient,
    echo_model: EchoChatModel,
) -> None:
    threads = ThreadStore(ttl=3600, max_threads=1)
    app.state.thread_store = threads
    app.state.agent_graph = build_agent_graph(
        echo_model,
        checkpointer=threads.checkpointer,
    )

    await send(client, "Hello", "thread-1")
    await send(client, "Other", "thread-2")
    await send(client, "Again", "thread-1")

    assert summarize(echo_model.calls[2]) == [
        ("system", SYSTEM_PROMPT),
        ("human", "Again"),
    ]
