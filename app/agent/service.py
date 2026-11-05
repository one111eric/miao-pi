import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Interrupt

from app.agent.graph import AgentGraph
from app.agent.models import (
    AgentRequest,
    AgentResponse,
    PendingAction,
    ResumeRequest,
    ThreadHistory,
    ThreadMessage,
    ToolCallView,
)
from app.agent.state import AgentState, ApprovalDecision
from app.agent.threads import ThreadStore
from common.logging_helper import logging_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentLimits:
    """Bounds on one run.

    ``max_steps`` is LangGraph's recursion limit: every node run is a step, so
    each tool round takes two. ``run_timeout`` is in seconds.
    """

    max_steps: int
    run_timeout: float


class AgentRunTimeoutError(Exception):
    """Raised when a run takes longer than its time limit."""

    def __init__(self, timeout: float) -> None:
        super().__init__(f"Agent run exceeded {timeout} seconds")
        self.timeout = timeout


class ThreadAwaitingApprovalError(Exception):
    """Raised when a new message arrives while the thread waits for approval."""

    def __init__(self, thread_id: str) -> None:
        super().__init__(f"Thread {thread_id} is waiting for approval")
        self.thread_id = thread_id


class ThreadNotFoundError(Exception):
    """Raised when a thread has no saved state."""

    def __init__(self, thread_id: str) -> None:
        super().__init__(f"Thread {thread_id} not found")
        self.thread_id = thread_id


class NothingToApproveError(Exception):
    """Raised when a resume names a thread with no pending approval."""

    def __init__(self, thread_id: str) -> None:
        super().__init__(f"Thread {thread_id} has nothing waiting for approval")
        self.thread_id = thread_id


def _config(thread_id: str, limits: AgentLimits) -> RunnableConfig:
    return {
        # The checkpointer finds the conversation by this config value.
        "configurable": {"thread_id": thread_id},
        "recursion_limit": limits.max_steps,
    }


def _pending_actions(interrupts: Sequence[Interrupt]) -> list[PendingAction]:
    return [
        PendingAction.model_validate(action)
        for pending in interrupts
        if isinstance(pending.value, dict)
        for action in pending.value.get("actions", [])
    ]


async def _run(
    graph: AgentGraph,
    graph_input: AgentState | Command[Any],
    config: RunnableConfig,
    limits: AgentLimits,
) -> dict[str, Any]:
    deadline = asyncio.timeout(limits.run_timeout)
    try:
        async with deadline:
            # The graph's state is a TypedDict, but ainvoke returns a plain dict
            # that can also carry LangGraph's "__interrupt__" entry.
            result: dict[str, Any] = await graph.ainvoke(graph_input, config=config)
    except TimeoutError as error:
        # Only our deadline counts; a TimeoutError from inside the run is some
        # other failure.
        if not deadline.expired():
            raise
        logger.warning(
            "Agent run timed out",
            extra={"agent_run_timeout": limits.run_timeout},
        )
        raise AgentRunTimeoutError(limits.run_timeout) from error
    return result


def _response(thread_id: str, result: dict[str, Any]) -> AgentResponse:
    actions = _pending_actions(result.get("__interrupt__", []))
    if actions:
        logger.info(
            "Agent paused for approval",
            extra={"pending_tools": [action.tool for action in actions]},
        )
        return AgentResponse(
            status="awaiting_approval",
            response=None,
            thread_id=thread_id,
            actions=actions,
        )

    final_message = result["messages"][-1]
    if not isinstance(final_message, AIMessage):
        raise RuntimeError("Agent graph did not produce an AI response")
    logger.info(
        "Agent invocation completed",
        extra={"conversation_message_count": len(result["messages"])},
    )
    return AgentResponse(
        status="completed",
        response=final_message.text,
        thread_id=thread_id,
    )


async def invoke_agent(
    request: AgentRequest,
    graph: AgentGraph,
    threads: ThreadStore,
    limits: AgentLimits,
) -> AgentResponse:
    """Invoke the agent graph for one user message in a conversation thread.

    The run either completes or pauses on tool calls that need approval.
    """
    thread_id = request.thread_id or str(uuid4())
    config = _config(thread_id, limits)

    with logging_context(thread_id=thread_id):
        # A supplied thread that ends with conversation_message_count=2 had no
        # history: the caller chose a new id, or the thread had expired.
        logger.info(
            "Agent invocation started",
            extra={"thread_supplied": request.thread_id is not None},
        )
        async with threads.use(thread_id):
            # A new message would start a fresh run and silently drop the
            # pending one, so the decision has to come first.
            if request.thread_id is not None:
                snapshot = await graph.aget_state(config)
                if _pending_actions(snapshot.interrupts):
                    raise ThreadAwaitingApprovalError(thread_id)
            result = await _run(
                graph,
                {"messages": [HumanMessage(content=request.message)]},
                config,
                limits,
            )
        return _response(thread_id, result)


async def resume_agent(
    thread_id: str,
    request: ResumeRequest,
    graph: AgentGraph,
    threads: ThreadStore,
    limits: AgentLimits,
) -> AgentResponse:
    """Apply a human's decision to a paused run and continue it."""
    config = _config(thread_id, limits)

    with logging_context(thread_id=thread_id):
        async with threads.use(thread_id):
            snapshot = await graph.aget_state(config)
            if not snapshot.values.get("messages"):
                raise ThreadNotFoundError(thread_id)
            actions = _pending_actions(snapshot.interrupts)
            if not actions:
                raise NothingToApproveError(thread_id)
            # The audit record of the decision; the request id ties it to the call.
            logger.info(
                "Approval decision received",
                extra={
                    "approved": request.approved,
                    "pending_tools": [action.tool for action in actions],
                },
            )
            decision: ApprovalDecision = {
                "approved": request.approved,
                "note": request.note,
            }
            result = await _run(graph, Command(resume=decision), config, limits)
        return _response(thread_id, result)


def _thread_message(message: AnyMessage) -> ThreadMessage | None:
    if isinstance(message, HumanMessage):
        return ThreadMessage(type="human", text=message.text)
    if isinstance(message, AIMessage):
        return ThreadMessage(
            type="ai",
            text=message.text,
            tool_calls=[
                ToolCallView(id=call["id"], name=call["name"], args=call["args"])
                for call in message.tool_calls
            ],
        )
    if isinstance(message, ToolMessage):
        return ThreadMessage(
            type="tool",
            text=message.text,
            tool_call_id=message.tool_call_id,
            name=message.name,
            status=message.status,
        )
    return None


async def read_thread_history(
    graph: AgentGraph,
    thread_id: str,
) -> ThreadHistory | None:
    """Return a thread's saved messages, or ``None`` if it doesn't exist.

    This reads the last checkpoint without ``ThreadStore.use()``, so it works
    while a run is in progress and doesn't count as using the thread.
    """
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    messages: list[AnyMessage] = snapshot.values.get("messages", [])
    if not messages:
        return None
    views = [_thread_message(message) for message in messages]
    return ThreadHistory(
        thread_id=thread_id,
        messages=[view for view in views if view is not None],
        pending_actions=_pending_actions(snapshot.interrupts),
    )
