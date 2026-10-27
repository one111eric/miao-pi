from typing import NotRequired, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: list[BaseMessage]
    thread_id: NotRequired[str]
