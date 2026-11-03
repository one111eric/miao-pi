import logging
from collections.abc import Awaitable
from time import perf_counter
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage

from app.agent.state import AgentState, AgentUpdate

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are Miao Pi, a concise and helpful coding assistant."


class AgentNode(Protocol):
    # LangGraph matches nodes by the parameter name, so a Callable alias won't do.
    def __call__(self, state: AgentState) -> Awaitable[AgentUpdate]: ...


async def preplanning(state: AgentState) -> AgentUpdate:
    """
    Prepare the run before model invocation.
    Will add contextize capabilities to this node too.
    """
    return {}


def make_react(llm: BaseChatModel) -> AgentNode:
    """Create the node that invokes the given chat model."""

    async def react(state: AgentState) -> AgentUpdate:
        started_at = perf_counter()
        # Added per call rather than stored, so threads never save the prompt.
        response = await llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )

        log_fields: dict[str, object] = {
            "model_duration_ms": round((perf_counter() - started_at) * 1000, 2),
        }
        if isinstance(response, AIMessage) and response.usage_metadata is not None:
            log_fields.update(response.usage_metadata)
        logger.info("LLM call completed", extra=log_fields)

        return {"messages": [response]}

    return react


async def finalize(state: AgentState) -> AgentUpdate:
    """Leave the completed state as is; later post-processing can live here."""
    return {}
