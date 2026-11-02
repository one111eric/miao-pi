from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    response: str
    thread_id: str = Field(
        description="Send this back as thread_id to continue the conversation",
    )


class AgentRequest(BaseModel):
    message: str = Field(min_length=1)
    # Thread ids end up in log lines, so they are limited to a safe character set.
    thread_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9._-]{1,128}$",
        description="Continue an existing conversation; omit to start a new one",
    )
