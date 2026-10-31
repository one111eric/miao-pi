from collections.abc import Awaitable
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from app.agent.state import AgentState

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
        response = await llm.ainvoke(state["messages"])
        return {**state, "messages": [*state["messages"], response]}

    return react


async def finalize(state: AgentState) -> AgentState:
    """Return the completed state; later post-processing can live here."""
    return state
