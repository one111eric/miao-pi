from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.agent.graph import AgentGraph
from app.agent.models import AgentRequest, AgentResponse
from app.agent.service import invoke_agent
from app.agent.threads import ThreadBusyError, ThreadStore

router = APIRouter(tags=["agent"])


def get_agent_graph(request: Request) -> AgentGraph:
    graph: object | None = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent graph is not initialized",
        )
    return cast(AgentGraph, graph)


def get_thread_store(request: Request) -> ThreadStore:
    threads: object | None = getattr(request.app.state, "thread_store", None)
    if threads is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Thread store is not initialized",
        )
    return cast(ThreadStore, threads)


AgentGraphDependency = Annotated[AgentGraph, Depends(get_agent_graph)]
ThreadStoreDependency = Annotated[ThreadStore, Depends(get_thread_store)]


@router.post(
    "/invoke",
    response_model=AgentResponse,
    description=(
        "Send one message to the coding agent. Pass the returned thread_id "
        "to continue the conversation."
    ),
    responses={
        409: {"description": "The thread already has a run in progress"},
    },
)
async def invoke(
    request: AgentRequest,
    graph: AgentGraphDependency,
    threads: ThreadStoreDependency,
) -> AgentResponse:
    try:
        return await invoke_agent(request, graph, threads)
    except ThreadBusyError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Thread already has a run in progress",
        ) from error
