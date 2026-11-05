"""Human approval for tools with side effects.

A tool opts in with ``metadata={"requires_approval": True}``. When the model
asks for one, ``react`` routes to the ``approve`` node instead of ``tools``.
That node pauses the run with ``interrupt``, listing every such call in the
message, and the run resumes with the human's decision; ``tools`` runs next.
The gate in front of each tool call refuses an approval-requiring call
without an approved decision, so a routing mistake fails closed.
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command, interrupt

from app.agent.state import AgentNode, AgentState, AgentUpdate, ApprovalDecision

REQUIRES_APPROVAL = "requires_approval"

# ToolNode results are a ToolMessage or a Command whose goto can name any node.
type ToolResult = ToolMessage | Command[Any]
type ToolExecutor = Callable[[ToolCallRequest], Awaitable[ToolResult]]
type ToolCallWrapper = Callable[[ToolCallRequest, ToolExecutor], Awaitable[ToolResult]]
type NextNode = Literal["approve", "tools", "finalize"]


class PendingActionPayload(TypedDict):
    """One tool call waiting for a decision, as the interrupt carries it."""

    tool_call_id: str
    tool: str
    # Tool arguments are whatever JSON the model produced.
    args: dict[str, Any]


def requires_approval(tool: BaseTool) -> bool:
    return bool((tool.metadata or {}).get(REQUIRES_APPROVAL))


def approval_tool_names(tools: Sequence[BaseTool]) -> frozenset[str]:
    return frozenset(tool.name for tool in tools if requires_approval(tool))


def pending_actions(
    message: AIMessage,
    approval_tools: frozenset[str],
) -> list[PendingActionPayload]:
    return [
        {"tool_call_id": call["id"], "tool": call["name"], "args": call["args"]}
        for call in message.tool_calls
        if call["name"] in approval_tools and call["id"] is not None
    ]


def rejection_text(decision: ApprovalDecision | None) -> str:
    if decision is None:
        return "Not run: this action needs a human's approval, and none was given."
    text = "Not run: the user rejected this action."
    if decision["note"]:
        text += f" Their note: {decision['note']}"
    return text


def _parse_decision(value: object) -> ApprovalDecision:
    if isinstance(value, dict) and isinstance(value.get("approved"), bool):
        note = value.get("note")
        return {
            "approved": value["approved"],
            "note": note if isinstance(note, str) else None,
        }
    # Anything unexpected counts as a rejection, never as consent.
    return {"approved": False, "note": None}


def make_route_after_react(
    approval_tools: frozenset[str],
) -> Callable[[AgentState], NextNode]:
    """Send tool calls to ``approve`` if any needs a decision, else to ``tools``."""

    def route_after_react(state: AgentState) -> NextNode:
        message = state["messages"][-1]
        if not isinstance(message, AIMessage) or not message.tool_calls:
            return "finalize"
        if any(call["name"] in approval_tools for call in message.tool_calls):
            return "approve"
        return "tools"

    return route_after_react


def make_approve(approval_tools: frozenset[str]) -> AgentNode:
    """Create the node that pauses the run until a human decides."""

    async def approve(state: AgentState) -> AgentUpdate:
        message = state["messages"][-1]
        if not isinstance(message, AIMessage):
            return {}
        actions = pending_actions(message, approval_tools)
        # On resume this node runs again from the top, and interrupt returns
        # the decision instead of pausing. One decision covers every action.
        decision = _parse_decision(interrupt({"actions": actions}))
        return {"approvals": {action["tool_call_id"]: decision for action in actions}}

    return approve


def make_approval_gate(approval_tools: frozenset[str]) -> ToolCallWrapper:
    """Create a ToolNode wrapper that runs an approval-requiring call only if
    the recorded decision for that call approved it."""

    async def gate(request: ToolCallRequest, execute: ToolExecutor) -> ToolResult:
        name = request.tool_call["name"]
        if name not in approval_tools:
            return await execute(request)
        call_id = request.tool_call["id"]
        state = request.state if isinstance(request.state, dict) else {}
        decision = state.get("approvals", {}).get(call_id)
        if decision is not None and decision["approved"]:
            return await execute(request)
        return ToolMessage(
            content=rejection_text(decision),
            tool_call_id=call_id or "",
            name=name,
            status="error",
        )

    return gate
