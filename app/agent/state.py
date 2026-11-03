from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class AgentUpdate(TypedDict, total=False):
    """The part of the state a node changes; LangGraph merges it into the state."""

    messages: list[AnyMessage]
