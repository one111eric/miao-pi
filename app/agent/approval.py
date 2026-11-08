"""Human approval for tools with side effects.

A tool opts in with ``metadata={"requires_approval": True}``. When the model
asks for one, ``react`` routes to the ``approve`` node instead of ``tools``.
That node pauses the run with ``interrupt``, listing every such call in the
message, and the run resumes with the human's decision; ``tools`` runs next.
The gate in front of each tool call refuses an approval-requiring call
without an approved decision, so a routing mistake fails closed.
"""

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command, interrupt

from app.agent.state import AgentNode, AgentState, AgentUpdate, ApprovalDecision

logger = logging.getLogger(__name__)

REQUIRES_APPROVAL = "requires_approval"
APPROVAL_PREFLIGHT = "approval_preflight"

# ToolNode results are a ToolMessage or a Command whose goto can name any node.
type ToolResult = ToolMessage | Command[Any]
type ToolExecutor = Callable[[ToolCallRequest], Awaitable[ToolResult]]
type ToolCallWrapper = Callable[[ToolCallRequest, ToolExecutor], Awaitable[ToolResult]]
type NextNode = Literal["approve", "tools", "finalize"]


@dataclass(frozen=True)
class PreflightFailure:
    """A safe validation failure reported before asking for approval."""

    message: str
    path: str | None = None
    reason: Literal["outside", "denied"] | None = None


type ApprovalPreflight = Callable[[dict[str, Any]], PreflightFailure | None]
type ApprovalPreflights = dict[str, ApprovalPreflight]


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


def approval_preflights(tools: Sequence[BaseTool]) -> ApprovalPreflights:
    """Return safe, side-effect-free validation callbacks supplied by tools."""
    preflights: ApprovalPreflights = {}
    for tool in tools:
        callback = (tool.metadata or {}).get(APPROVAL_PREFLIGHT)
        if requires_approval(tool) and callable(callback):
            preflight = cast(ApprovalPreflight, callback)
            cached: dict[str, PreflightFailure | None] = {}

            def check(
                args: dict[str, Any],
                *,
                callback: ApprovalPreflight = preflight,
                cache: dict[str, PreflightFailure | None] = cached,
            ) -> PreflightFailure | None:
                key = json.dumps(args, sort_keys=True, separators=(",", ":"))
                if key not in cache:
                    cache[key] = callback(args)
                return cache[key]

            preflights[tool.name] = check
    return preflights


def _preflight_error(
    call: ToolCall, preflights: ApprovalPreflights
) -> PreflightFailure | None:
    callback = preflights.get(call["name"])
    return callback(call["args"]) if callback is not None else None


def pending_actions(
    message: AIMessage,
    approval_tools: frozenset[str],
    preflights: ApprovalPreflights | None = None,
) -> list[PendingActionPayload]:
    checks = preflights or {}
    return [
        {"tool_call_id": call["id"], "tool": call["name"], "args": call["args"]}
        for call in message.tool_calls
        if (
            call["name"] in approval_tools
            and call["id"] is not None
            and _preflight_error(call, checks) is None
        )
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
    preflights: ApprovalPreflights | None = None,
) -> Callable[[AgentState], NextNode]:
    """Send tool calls to ``approve`` if any needs a decision, else to ``tools``."""

    def route_after_react(state: AgentState) -> NextNode:
        checks = preflights or {}
        message = state["messages"][-1]
        if not isinstance(message, AIMessage) or not message.tool_calls:
            return "finalize"
        if any(
            call["name"] in approval_tools and _preflight_error(call, checks) is None
            for call in message.tool_calls
        ):
            return "approve"
        return "tools"

    return route_after_react


def make_approve(
    approval_tools: frozenset[str], preflights: ApprovalPreflights | None = None
) -> AgentNode:
    """Create the node that pauses the run until a human decides."""

    async def approve(state: AgentState) -> AgentUpdate:
        message = state["messages"][-1]
        if not isinstance(message, AIMessage):
            return {}
        actions = pending_actions(message, approval_tools, preflights)
        if not actions:
            # Nothing to show a human (react gives every call an id, so this
            # shouldn't happen); the gate in front of the tools refuses them.
            return {}
        # On resume this node runs again from the top, and interrupt returns
        # the decision instead of pausing. One decision covers every action.
        decision = _parse_decision(interrupt({"actions": actions}))
        return {"approvals": {action["tool_call_id"]: decision for action in actions}}

    return approve


def make_approval_gate(
    approval_tools: frozenset[str], preflights: ApprovalPreflights | None = None
) -> ToolCallWrapper:
    """Create a ToolNode wrapper that runs an approval-requiring call only if
    the recorded decision for that call approved it."""

    async def gate(request: ToolCallRequest, execute: ToolExecutor) -> ToolResult:
        checks = preflights or {}
        name = request.tool_call["name"]
        if name not in approval_tools:
            return await execute(request)
        call_id = request.tool_call["id"]
        if failure := _preflight_error(request.tool_call, checks):
            logger.warning(
                "Tool call refused",
                extra={
                    "tool": name,
                    "reason": (
                        "path_rejected" if failure.reason else "invalid_arguments"
                    ),
                },
            )
            if failure.reason is not None:
                logger.warning(
                    "Tool path rejected",
                    extra={
                        "tool": name,
                        "tool_path": failure.path or "<patch>",
                        "reason": failure.reason,
                    },
                )
            return ToolMessage(
                content=f"Not run: {failure.message}",
                tool_call_id=call_id or "",
                name=name,
                status="error",
            )
        state = request.state if isinstance(request.state, dict) else {}
        decision = state.get("approvals", {}).get(call_id)
        if decision is not None and decision["approved"]:
            return await execute(request)
        logger.warning(
            "Tool call refused",
            extra={
                "tool": name,
                "reason": (
                    "approval_missing" if decision is None else "approval_rejected"
                ),
            },
        )
        return ToolMessage(
            content=rejection_text(decision),
            tool_call_id=call_id or "",
            name=name,
            status="error",
        )

    return gate
