import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import StructuredTool
from pydantic import Field
from starlette.types import Message, Scope

from app.agent.approval import REQUIRES_APPROVAL
from app.agent.graph import build_agent_graph
from app.agent.nodes import TIME_LIMIT_MESSAGE
from app.agent.service import AgentLimits
from tests.apps import serve
from tests.fakes import (
    EchoChatModel,
    GatedChatModel,
    ScriptedChatModel,
    recording_tool,
    tool_call,
)
from tests.integration.test_exception_handlers import FailingChatModel, TimeoutChatModel


async def read_events(
    client: AsyncClient, path: str, body: dict[str, object]
) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    async with client.stream("POST", path, json=body) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "no-store" in response.headers["cache-control"]
        assert response.headers["X-Thread-ID"]
        name = "message"
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                assert data["thread_id"] == response.headers["X-Thread-ID"]
                events.append((name, data))
    return events


@pytest.mark.anyio
async def test_stream_finishes_and_preserves_conversation(
    client: AsyncClient, echo_model: EchoChatModel
) -> None:
    events = await read_events(
        client, "/stream", {"message": "Hello", "thread_id": "stream-1"}
    )

    assert events[-1] == (
        "done",
        {
            "status": "completed",
            "response": "Received: Hello",
            "thread_id": "stream-1",
            "actions": [],
        },
    )
    assert all(data["thread_id"] == "stream-1" for _, data in events)
    await client.post("/invoke", json={"message": "Again", "thread_id": "stream-1"})
    assert [message.text for message in echo_model.calls[-1][1:]] == [
        "Hello",
        "Received: Hello",
        "Again",
    ]


@pytest.mark.anyio
async def test_stream_emits_tool_call_before_result(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")
    model = ScriptedChatModel(
        responses=[
            tool_call("read", "read-1", path="notes.txt"),
            AIMessage(content="Read it."),
        ]
    )
    async with serve(model, tmp_path) as (_, client):
        events = await read_events(client, "/stream", {"message": "Read it"})

    names = [name for name, _ in events]
    assert names.index("tool_call") < names.index("tool_result") < names.index("done")
    call = next(data for name, data in events if name == "tool_call")
    result = next(data for name, data in events if name == "tool_result")
    assert call["id"] == "read-1"
    assert call["args"] == {"path": "notes.txt"}
    assert result["tool_call_id"] == "read-1"
    assert result["text"] == "notes"


@pytest.mark.anyio
@pytest.mark.parametrize("approved", [True, False])
async def test_stream_pauses_then_resumes_with_a_decision(
    tmp_path: Path, approved: bool
) -> None:
    effects: list[str] = []
    model = ScriptedChatModel(
        responses=[
            tool_call("record", "call-1", text="hello"),
            AIMessage(content="Done."),
        ]
    )
    async with serve(model, tmp_path, [recording_tool(effects)]) as (_, client):
        events = await read_events(
            client, "/stream", {"message": "Record", "thread_id": "approval-stream"}
        )
        assert effects == []
        assert [name for name, _ in events] == ["tool_call", "awaiting_approval"]
        pending = events[-1][1]
        assert pending["actions"][0]["tool_call_id"] == "call-1"
        conflict = await client.post(
            "/stream", json={"message": "Again", "thread_id": "approval-stream"}
        )
        assert conflict.status_code == 409
        resumed = await read_events(
            client, "/threads/approval-stream/resume/stream", {"approved": approved}
        )

    assert effects == (["hello"] if approved else [])
    assert resumed[0][0] == "tool_result"
    assert resumed[0][1]["status"] == ("success" if approved else "error")
    assert resumed[-1][0] == "done"
    assert resumed[-1][1]["response"] == "Done."


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("model", "code"),
    [(FailingChatModel(), "provider_error"), (TimeoutChatModel(), "provider_timeout")],
)
async def test_stream_provider_errors_are_events_without_private_details(
    app: FastAPI, client: AsyncClient, model: BaseChatModel, code: str
) -> None:
    app.state.agent_graph = build_agent_graph(
        model, checkpointer=app.state.thread_store.checkpointer
    )
    events = await read_events(client, "/stream", {"message": "Hello"})
    assert [name for name, _ in events] == ["error"]
    assert events[-1][1]["code"] == code
    assert "sensitive" not in str(events)


@pytest.mark.anyio
async def test_stream_backstop_times_out_and_frees_the_thread(
    app: FastAPI, client: AsyncClient
) -> None:
    model = GatedChatModel()
    app.state.agent_graph = build_agent_graph(
        model, checkpointer=app.state.thread_store.checkpointer
    )
    app.state.agent_limits = AgentLimits(
        max_steps=25, run_timeout=0.01, backstop_timeout=0.02
    )
    events = await read_events(
        client, "/stream", {"message": "Hello", "thread_id": "timeout-stream"}
    )
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "run_timeout"
    async with app.state.thread_store.use("timeout-stream"):
        pass


@pytest.mark.anyio
async def test_stream_resume_validation_is_http_before_headers(
    client: AsyncClient,
) -> None:
    missing = await client.post(
        "/threads/missing/resume/stream", json={"approved": True}
    )
    assert missing.status_code == 404
    await client.post("/invoke", json={"message": "Hello", "thread_id": "not-paused"})
    not_paused = await client.post(
        "/threads/not-paused/resume/stream", json={"approved": True}
    )
    assert not_paused.status_code == 409
    invalid = await client.post("/stream", json={"message": ""})
    assert invalid.status_code == 422


class TokenGatedChatModel(EchoChatModel):
    release: asyncio.Event = Field(default_factory=asyncio.Event)
    started: asyncio.Event = Field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = Field(default_factory=asyncio.Event)
    call_tool: bool = False
    fail_after_token: bool = False

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        try:
            yield ChatGenerationChunk(message=AIMessageChunk(content="Hello "))
            self.started.set()
            await self.release.wait()
            if self.fail_after_token:
                raise RuntimeError("private model details")
            if self.call_tool:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {"name": "ls", "args": "{}", "id": "late-call", "index": 0}
                        ],
                    )
                )
            else:
                yield ChatGenerationChunk(message=AIMessageChunk(content="world"))
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@pytest.mark.anyio
async def test_stream_uses_the_canonical_limit_response_after_provisional_tokens(
    tmp_path: Path,
) -> None:
    model = TokenGatedChatModel(call_tool=True)
    async with serve(model, tmp_path) as (app, client):
        app.state.agent_limits = AgentLimits(
            max_steps=25, run_timeout=0.01, backstop_timeout=1
        )
        request = asyncio.create_task(read_events(client, "/stream", {"message": "Hi"}))
        try:
            await asyncio.wait_for(model.started.wait(), timeout=2)
            await asyncio.sleep(0.02)
            model.release.set()
            events = await asyncio.wait_for(request, timeout=2)
        finally:
            model.release.set()
            await asyncio.gather(request, return_exceptions=True)
    assert events[0][0] == "token"
    assert not any(name == "tool_call" for name, _ in events)
    assert events[-1][0] == "done"
    assert events[-1][1]["response"] == TIME_LIMIT_MESSAGE


@pytest.mark.anyio
async def test_stream_failure_after_a_token_is_a_safe_terminal_event(
    tmp_path: Path,
) -> None:
    model = TokenGatedChatModel(fail_after_token=True)
    model.release.set()
    async with serve(model, tmp_path) as (_, client):
        events = await read_events(client, "/stream", {"message": "Hi"})
    assert [name for name, _ in events] == ["token", "error"]
    assert events[-1][1]["code"] == "internal_error"
    assert "private model details" not in str(events)


@pytest.mark.anyio
@pytest.mark.parametrize("disconnect", [False, True])
async def test_tokens_arrive_while_model_is_running_and_disconnect_cleans_up(
    tmp_path: Path, disconnect: bool
) -> None:
    # ASGITransport buffers its response, so drive ASGI directly to observe
    # a token before the model finishes and to simulate an actual disconnect.
    model = TokenGatedChatModel()
    token_seen, disconnected = asyncio.Event(), asyncio.Event()
    body = json.dumps({"message": "Hi", "thread_id": "live-stream"}).encode()
    sent_body = False
    chunks: list[bytes] = []

    async def receive() -> Message:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.body":
            chunk = message.get("body", b"")
            chunks.append(chunk)
            if b"event: token" in chunk:
                token_seen.set()

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "method": "POST",
        "scheme": "http",
        "path": "/stream",
        "raw_path": b"/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "http_version": "1.1",
    }
    async with serve(model, tmp_path) as (app, client):
        request = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(token_seen.wait(), timeout=2)
            assert not model.release.is_set()
            assert not request.done()
            busy = await client.post(
                "/stream", json={"message": "Again", "thread_id": "live-stream"}
            )
            assert busy.status_code == 409
            if disconnect:
                disconnected.set()
            else:
                model.release.set()
            await asyncio.wait_for(request, timeout=2)
            if disconnect:
                assert model.cancelled.is_set()
                assert b"event: done" not in b"".join(chunks)
            else:
                assert b"event: done" in b"".join(chunks)
                snapshot = await app.state.agent_graph.aget_state(
                    {"configurable": {"thread_id": "live-stream"}}
                )
                assert snapshot.values["messages"][-1].text == "Hello world"
            async with app.state.thread_store.use("live-stream"):
                pass
        finally:
            model.release.set()
            disconnected.set()
            if not request.done():
                request.cancel()
            await asyncio.gather(request, return_exceptions=True)


@pytest.mark.anyio
async def test_disconnect_holds_thread_until_approved_tool_cleanup_finishes(
    tmp_path: Path,
) -> None:
    started, cleanup_started, allow_cleanup, cleaned = (
        asyncio.Event() for _ in range(4)
    )
    disconnected = asyncio.Event()

    async def slow_tool() -> str:
        try:
            started.set()
            await asyncio.Event().wait()
            return "unreachable"
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()
            cleaned.set()

    tool = StructuredTool.from_function(
        coroutine=slow_tool,
        name="slow",
        description="A slow approved tool.",
        metadata={REQUIRES_APPROVAL: True},
    )
    model = ScriptedChatModel(responses=[tool_call("slow", "slow-1")])
    sent_body = False

    async def receive() -> Message:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {
                "type": "http.request",
                "body": b'{"approved":true}',
                "more_body": False,
            }
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        pass

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "method": "POST",
        "scheme": "http",
        "path": "/threads/tool-disconnect/resume/stream",
        "query_string": b"",
        "root_path": "",
        "http_version": "1.1",
        "headers": [(b"content-type", b"application/json")],
    }
    async with serve(model, tmp_path, [tool]) as (app, client):
        await client.post(
            "/invoke", json={"message": "Run", "thread_id": "tool-disconnect"}
        )
        request = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            disconnected.set()
            await asyncio.wait_for(cleanup_started.wait(), timeout=2)
            busy = await client.post(
                "/stream", json={"message": "Again", "thread_id": "tool-disconnect"}
            )
            assert busy.status_code == 409
            allow_cleanup.set()
            await asyncio.wait_for(request, timeout=2)
            assert cleaned.is_set()
            async with app.state.thread_store.use("tool-disconnect"):
                pass
        finally:
            disconnected.set()
            allow_cleanup.set()
            await asyncio.gather(request, return_exceptions=True)
