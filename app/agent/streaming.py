"""Map live graph output to the agent's public SSE event vocabulary."""

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, cast

from anyio import CancelScope
from fastapi.sse import ServerSentEvent
from langchain_core.messages import AIMessage, ToolMessage
from openai import APIError, APITimeoutError

from app.agent.graph import AgentGraph
from app.agent.service import AgentLimits, AgentRun, _response
from common.logging_helper import logging_context

logger = logging.getLogger(__name__)

STREAM_DESCRIPTION = (
    "Stream a run as text/event-stream with JSON data. Each event includes thread_id. "
    "token: {message_id, text} is provisional model text; "
    "tool_call: {message_id, text, id, name, args} is a complete model call; "
    "tool_result: {tool_call_id, name, text, status} is a tool outcome. "
    "awaiting_approval has the same payload as /invoke and ends a paused stream; "
    "done has the completed /invoke payload, including the authoritative final response. "
    "error: {code, detail} ends a failed stream without exposing internal error text. "
    "Tokens may precede a tool call or a final time/step-limit response. "
    "Use the final response to reconcile provisional text. "
    "Validation and thread conflicts return HTTP errors before streaming starts. "
    "Disconnecting cancels the run; check saved history before retrying an action. "
    "Streams do not support event replay or automatic resume."
)


def _event(event_type: str, thread_id: str, **data: Any) -> ServerSentEvent:
    return ServerSentEvent(event=event_type, data={"thread_id": thread_id, **data})


def _updates(thread_id: str, update: dict[str, Any]) -> list[ServerSentEvent]:
    events: list[ServerSentEvent] = []
    for node in ("react", "tools"):
        values = update.get(node)
        if not isinstance(values, dict):
            continue
        for message in values.get("messages", []):
            if isinstance(message, AIMessage):
                events.extend(
                    _event(
                        "tool_call",
                        thread_id,
                        message_id=message.id,
                        text=message.text,
                        id=call["id"],
                        name=call["name"],
                        args=call["args"],
                    )
                    for call in message.tool_calls
                )
            elif isinstance(message, ToolMessage):
                events.append(
                    _event(
                        "tool_result",
                        thread_id,
                        tool_call_id=message.tool_call_id,
                        name=message.name,
                        text=message.text,
                        status=message.status,
                    )
                )
    return events


async def _produce(
    run: AgentRun,
    graph: AgentGraph,
    limits: AgentLimits,
    queue: asyncio.Queue[ServerSentEvent | None],
) -> None:
    """Own graph execution and its timeout in one task, including cleanup."""
    with logging_context(thread_id=run.thread_id):
        deadline = asyncio.timeout(limits.run_timeout + limits.backstop_timeout)
        try:
            async with deadline:
                # A list of modes yields (mode, payload) pairs. LangGraph's
                # return annotation also includes its single-mode dictionary form.
                output = cast(
                    AsyncGenerator[tuple[str, Any], None],
                    graph.astream(
                        run.graph_input,
                        config=run.config,
                        stream_mode=["messages", "updates"],
                    ),
                )
                async with aclosing(output) as stream:
                    async for mode, payload in stream:
                        if mode == "messages":
                            message, metadata = payload
                            if (
                                metadata.get("langgraph_node") == "react"
                                and isinstance(message, AIMessage)
                                and message.text
                            ):
                                await queue.put(
                                    _event(
                                        "token",
                                        run.thread_id,
                                        message_id=message.id,
                                        text=message.text,
                                    )
                                )
                        elif mode == "updates":
                            for event in _updates(run.thread_id, payload):
                                await queue.put(event)
                # The stream is exhausted only after its checkpoint writes finish.
                snapshot = await graph.aget_state(run.config)
                result = _response(
                    run.thread_id,
                    {
                        **snapshot.values,
                        "__interrupt__": snapshot.interrupts,
                    },
                )
                terminal = ServerSentEvent(
                    event="awaiting_approval" if result.actions else "done",
                    data=result.model_dump(mode="json"),
                )
        except asyncio.CancelledError:
            logger.warning("Agent stream cancelled")
            raise
        except Exception as error:
            code, detail = (
                "internal_error",
                "The agent encountered an unexpected error.",
            )
            if isinstance(error, APITimeoutError):
                code, detail = "provider_timeout", "LLM provider request timed out"
            elif isinstance(error, APIError):
                code, detail = "provider_error", "LLM provider request failed"
            elif isinstance(error, TimeoutError) and deadline.expired():
                code, detail = "run_timeout", "Agent run timed out"
            logger.warning(
                "Agent stream failed",
                extra={"error_type": type(error).__name__, "reason": code},
            )
            terminal = _event("error", run.thread_id, code=code, detail=detail)
        await queue.put(terminal)
        await queue.put(None)


@dataclass(frozen=True)
class AgentStream:
    thread_id: str
    events: AsyncIterator[ServerSentEvent]


@asynccontextmanager
async def open_agent_stream(
    run: AgentRun, graph: AgentGraph, limits: AgentLimits
) -> AsyncIterator[AgentStream]:
    """Keep the producer alive until response teardown, then cancel and join it.

    The request dependency owns this context, so disconnects close the graph
    before releasing the thread even when the SSE generator is suspended at yield.
    """
    queue: asyncio.Queue[ServerSentEvent | None] = asyncio.Queue(maxsize=1)
    producer = asyncio.create_task(_produce(run, graph, limits, queue))

    async def events() -> AsyncIterator[ServerSentEvent]:
        while (event := await queue.get()) is not None:
            yield event

    try:
        yield AgentStream(run.thread_id, events())
    finally:
        with CancelScope(shield=True):
            if not producer.done():
                producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer
