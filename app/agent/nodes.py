import asyncio
import logging
from collections.abc import Awaitable
from time import perf_counter
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage

from app.agent.state import AgentState, AgentUpdate
from app.agent.workspace import Workspace, WorkspaceAccessError, WorkspaceError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are Miao Pi, a concise and helpful coding assistant."


class AgentNode(Protocol):
    # LangGraph matches nodes by the parameter name, so a Callable alias won't do.
    def __call__(self, state: AgentState) -> Awaitable[AgentUpdate]: ...


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


def make_react(llm: BaseChatModel) -> AgentNode:
    """Create the node that invokes the given chat model."""

    async def react(state: AgentState) -> AgentUpdate:
        started_at = perf_counter()
        # Added per call rather than stored, so threads never save the prompt.
        system_prompt = SYSTEM_PROMPT
        if context := state.get("context"):
            system_prompt = f"{SYSTEM_PROMPT}\n\nProject context:\n{context}"
        try:
            response = await llm.ainvoke(
                [SystemMessage(content=system_prompt), *state["messages"]]
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
        if isinstance(response, AIMessage) and response.usage_metadata is not None:
            log_fields.update(response.usage_metadata)
        logger.info("LLM call completed", extra=log_fields)

        return {"messages": [response]}

    return react


async def finalize(state: AgentState) -> AgentUpdate:
    """Leave the completed state as is; later post-processing can live here."""
    return {}
