from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.agent.graph import AgentGraph
from app.agent.models import AgentRequest, AgentResponse
from app.agent.service import invoke_agent

router = APIRouter(tags=["agent"])


def get_agent_graph(request: Request) -> AgentGraph:
    graph: object | None = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent graph is not initialized",
        )
    return cast(AgentGraph, graph)


AgentGraphDependency = Annotated[AgentGraph, Depends(get_agent_graph)]


@router.post(
    "/invoke",
    response_model=AgentResponse,
    description="Invoke the coding agent with one message",
)
async def invoke(
    request: AgentRequest,
    graph: AgentGraphDependency,
) -> AgentResponse:
    return await invoke_agent(request, graph)
