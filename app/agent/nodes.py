import logging
from collections.abc import Awaitable
from time import perf_counter
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage

from app.agent.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are Miao Pi, a concise and helpful coding assistant."


class AgentNode(Protocol):
    # LangGraph matches nodes by the parameter name, so a Callable alias won't do.
    def __call__(self, state: AgentState) -> Awaitable[AgentState]: ...


async def preplanning(state: AgentState) -> AgentState:
    """
    Add the agent's basic instruction before model invocation.
    Will add contextize capabilities to this node too.
    """
    if state["messages"] and isinstance(state["messages"][0], SystemMessage):
        return state

    return {
        **state,
        "messages": [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]],
    }


def make_react(llm: BaseChatModel) -> AgentNode:
    """Create the node that invokes the given chat model."""

    async def react(state: AgentState) -> AgentState:
        started_at = perf_counter()
        response = await llm.ainvoke(state["messages"])

        log_fields: dict[str, object] = {
            "model_duration_ms": round((perf_counter() - started_at) * 1000, 2),
        }
        if isinstance(response, AIMessage) and response.usage_metadata is not None:
            log_fields.update(response.usage_metadata)
        logger.info("LLM call completed", extra=log_fields)

        return {**state, "messages": [*state["messages"], response]}

    return react


async def finalize(state: AgentState) -> AgentState:
    """Return the completed state; later post-processing can live here."""
    return state
