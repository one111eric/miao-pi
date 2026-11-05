import asyncio
import logging
from collections.abc import Sequence
from time import perf_counter
from uuid import uuid4

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolCallRequest

from app.agent.approval import ToolExecutor, ToolResult
from app.agent.state import AgentNode, AgentState, AgentUpdate
from app.agent.workspace import (
    REQUIRED_DENIED_PATTERNS,
    Workspace,
    WorkspaceAccessError,
    WorkspaceError,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are Miao Pi, a concise and helpful coding assistant."

STEP_LIMIT_MESSAGE = (
    "I stopped because this request reached its step limit before I finished. "
    "Ask me to continue and I'll pick up from here."
)

# A timed-out run can't stop a tool already running in a worker thread, so
# the call may have finished after all; saying "not run" could get it repeated.
INTERRUPTED_TOOL_RESULT = (
    "No result: the previous request ended before this tool call finished, so "
    "it may or may not have run. Check before repeating anything with side "
    "effects."
)

# Another tool round needs three more steps: tools, react again, and finalize.
# A round through approve needs none extra: the run pauses there, and the
# resumed run gets a fresh step budget.
_STEPS_FOR_TOOL_ROUND = 3


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


def _load_project_context(workspace: Workspace, max_chars: int) -> str:
    for filename in ("AGENTS.md", "README.md"):
        try:
            path = workspace.resolve(filename)
            if not path.is_file():
                continue
            content = workspace.read_text(filename, max_chars=max_chars)
        except (WorkspaceError, OSError) as exc:
            skip_fields: dict[str, object] = {
                "context_file": filename,
                "error_type": type(exc).__name__,
            }
            if isinstance(exc, WorkspaceAccessError):
                skip_fields["reason"] = exc.reason
            logger.warning("Skipping inaccessible project context", extra=skip_fields)
            continue

        logger.info(
            "Project context loaded",
            extra={
                "context_file": filename,
                "context_chars": len(content.text),
                "context_truncated": content.truncated,
            },
        )
        return content.text

    logger.info("No project context found")
    return ""


# Preplanning Node
def make_preplanning(
    workspace: Workspace | None,
    context_max_chars: int,
) -> AgentNode:
    """Create a node that reloads project instructions for every turn."""

    async def preplanning(state: AgentState) -> AgentUpdate:
        del state
        if workspace is None:
            return {"context": ""}
        context = await asyncio.to_thread(
            _load_project_context,
            workspace,
            context_max_chars,
        )
        return {"context": context}

    return preplanning


def _blocked_paths_note(workspace: Workspace | None) -> str:
    """Tell the model which paths it can't read, so it doesn't try or offer them.

    Only the required patterns are named. Configured ones were chosen because
    their names are sensitive, and a hosted provider would see the prompt.
    """
    if workspace is None:
        return ""
    note = (
        f" Paths matching {', '.join(REQUIRED_DENIED_PATTERNS)}, at any depth, "
        "are blocked and can't be read or listed."
    )
    if workspace.has_configured_patterns:
        note += " Other paths may be blocked too."
    return note


def _system_prompt(
    context: str | None,
    tool_names: Sequence[str],
    blocked_paths_note: str,
) -> str:
    parts = [SYSTEM_PROMPT]
    if tool_names:
        parts.append(
            f"Inspect the project with these tools: {', '.join(tool_names)}. "
            f"Paths are relative to the project root.{blocked_paths_note}"
        )
    if context:
        parts.append(f"Project context:\n{context}")
    return "\n\n".join(parts)


def _close_dangling_tool_calls(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Answer tool calls that a stopped run left without results.

    A run that times out after the model asks for tools saves that request
    with no answers, and providers reject such a conversation. The
    placeholders go to the model only; the saved history is unchanged.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    closed: list[AnyMessage] = []
    for message in messages:
        closed.append(message)
        if isinstance(message, AIMessage):
            closed.extend(
                ToolMessage(
                    content=INTERRUPTED_TOOL_RESULT,
                    tool_call_id=call["id"],
                    status="error",
                )
                for call in message.tool_calls
                if call["id"] is not None and call["id"] not in answered
            )
    return closed


# React Node
def _with_tool_call_ids(response: AIMessage) -> AIMessage:
    """Give every tool call an id; some providers leave it out.

    Approval, the tool gate, and tool results all match calls by id, so a
    call without one would slip through all three.
    """
    if all(call["id"] is not None for call in response.tool_calls):
        return response
    return response.model_copy(
        update={
            "tool_calls": [
                tool_call(
                    name=call["name"],
                    args=call["args"],
                    id=call["id"] or f"call_{uuid4().hex}",
                )
                for call in response.tool_calls
            ]
        }
    )


def make_react(
    llm: BaseChatModel,
    tools: Sequence[BaseTool] = (),
    workspace: Workspace | None = None,
) -> AgentNode:
    """Create the node that invokes the chat model with the given tools bound.

    With a ``workspace``, the prompt says which of its paths are blocked.
    """
    model: Runnable[LanguageModelInput, AIMessage] = (
        llm.bind_tools(tools) if tools else llm
    )
    tool_names = [tool.name for tool in tools]
    blocked_paths_note = _blocked_paths_note(workspace)

    async def react(state: AgentState) -> AgentUpdate:
        started_at = perf_counter()
        # Added per call rather than stored, so threads never save the prompt.
        system_prompt = _system_prompt(
            state.get("context"), tool_names, blocked_paths_note
        )
        try:
            response = await model.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    *_close_dangling_tool_calls(state["messages"]),
                ]
            )
        except Exception as exc:
            # Logged here so the failure keeps the thread id and its duration;
            # the HTTP error mapping happens later, outside the thread context.
            logger.warning(
                "LLM call failed",
                extra={
                    "model_duration_ms": _elapsed_ms(started_at),
                    "error_type": type(exc).__name__,
                },
            )
            raise

        log_fields: dict[str, object] = {
            "model_duration_ms": _elapsed_ms(started_at),
        }
        if response.usage_metadata is not None:
            log_fields.update(response.usage_metadata)
        logger.info("LLM call completed", extra=log_fields)

        response = _with_tool_call_ids(response)

        # Ending here keeps the history valid: a GraphRecursionError would leave
        # the requested tool calls unanswered.
        if response.tool_calls and state["remaining_steps"] <= _STEPS_FOR_TOOL_ROUND:
            logger.warning(
                "Agent step limit reached",
                extra={"requested_tool_calls": len(response.tool_calls)},
            )
            response = AIMessage(content=STEP_LIMIT_MESSAGE)

        return {"messages": [response]}

    return react


async def log_tool_call(
    request: ToolCallRequest,
    execute: ToolExecutor,
) -> ToolResult:
    """Log each tool call's name, outcome, and duration, never its output."""
    started_at = perf_counter()
    tool_name = request.tool_call["name"]
    try:
        result = await execute(request)
    except Exception as exc:
        logger.warning(
            "Tool call failed",
            extra={
                "tool": tool_name,
                "tool_duration_ms": _elapsed_ms(started_at),
                "error_type": type(exc).__name__,
            },
        )
        raise
    logger.info(
        "Tool call completed",
        extra={
            "tool": tool_name,
            "tool_status": (
                result.status if isinstance(result, ToolMessage) else "command"
            ),
            "tool_duration_ms": _elapsed_ms(started_at),
        },
    )
    return result


# Finalize Node
async def finalize(state: AgentState) -> AgentUpdate:
    """Leave the completed state as is; later post-processing can live here."""
    return {}
