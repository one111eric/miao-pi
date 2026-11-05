from collections.abc import Awaitable
from typing import Annotated, NotRequired, Protocol, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps


class ApprovalDecision(TypedDict):
    approved: bool
    note: str | None


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    context: NotRequired[str]
    # The human's decision for each pending tool call, keyed by tool call id.
    approvals: NotRequired[dict[str, ApprovalDecision]]
    # Filled in by LangGraph from the run's recursion limit; never saved.
    remaining_steps: NotRequired[RemainingSteps]


class AgentUpdate(TypedDict, total=False):
    """The part of the state a node changes; LangGraph merges it into the state."""

    messages: list[AnyMessage]
    context: str
    approvals: dict[str, ApprovalDecision]


class AgentNode(Protocol):
    # LangGraph matches nodes by the parameter name, so a Callable alias won't do.
    def __call__(self, state: AgentState) -> Awaitable[AgentUpdate]: ...
