from typing import Literal

from pydantic import BaseModel, Field, JsonValue

# Thread ids end up in log lines, so they are limited to a safe character set.
THREAD_ID_PATTERN = r"^[A-Za-z0-9._-]{1,128}$"


class AgentResponse(BaseModel):
    response: str
    thread_id: str = Field(
        description="Send this back as thread_id to continue the conversation",
    )


class AgentRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = Field(
        default=None,
        pattern=THREAD_ID_PATTERN,
        description="Continue an existing conversation; omit to start a new one",
    )


class ToolCallView(BaseModel):
    id: str | None
    name: str
    args: dict[str, JsonValue]


class ThreadMessage(BaseModel):
    """One saved message. Only AI messages have ``tool_calls``; only tool
    results have ``tool_call_id``, ``name``, and ``status``."""

    type: Literal["human", "ai", "tool"]
    text: str
    tool_calls: list[ToolCallView] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    status: Literal["success", "error"] | None = None


class ThreadHistory(BaseModel):
    thread_id: str
    messages: list[ThreadMessage]
