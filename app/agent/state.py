from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    context: NotRequired[str]
    # Filled in by LangGraph from the run's recursion limit; never saved.
    remaining_steps: NotRequired[RemainingSteps]


class AgentUpdate(TypedDict, total=False):
    """The part of the state a node changes; LangGraph merges it into the state."""

    messages: list[AnyMessage]
    context: str
