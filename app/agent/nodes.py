from langchain_core.messages import SystemMessage

from app.agent.state import AgentState
from common.llm import LLM

SYSTEM_PROMPT = "You are Miao Pi, a concise and helpful coding assistant."


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


async def react(state: AgentState) -> AgentState:
    """Invoke the configured chat model."""
    response = await LLM().ainvoke(state["messages"])
    return {**state, "messages": [*state["messages"], response]}


async def finalize(state: AgentState) -> AgentState:
    """Return the completed state; later post-processing can live here."""
    return state
