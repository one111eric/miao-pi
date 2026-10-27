from fastapi import APIRouter

from app.agent.models import AgentRequest, AgentResponse
from app.agent.service import invoke_agent

router = APIRouter(tags=["agent"])


@router.post(
    "/invoke",
    response_model=AgentResponse,
    description="Invoke the coding agent with one message",
)
async def invoke(request: AgentRequest) -> AgentResponse:
    return await invoke_agent(request)
