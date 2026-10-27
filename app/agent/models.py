from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    response: str
    thread_id: str | None = None


class AgentRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None
