import logging
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.agent.graph import AgentGraph
from app.agent.models import AgentRequest, AgentResponse
from app.agent.service import AgentLimits, AgentRunTimeoutError, invoke_agent
from app.agent.threads import ThreadBusyError, ThreadStore

logger = logging.getLogger(__name__)

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


def get_agent_limits(request: Request) -> AgentLimits:
    limits: object | None = getattr(request.app.state, "agent_limits", None)
    if limits is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent limits are not initialized",
        )
    return cast(AgentLimits, limits)


AgentGraphDependency = Annotated[AgentGraph, Depends(get_agent_graph)]
ThreadStoreDependency = Annotated[ThreadStore, Depends(get_thread_store)]
AgentLimitsDependency = Annotated[AgentLimits, Depends(get_agent_limits)]


@router.post(
    "/invoke",
    response_model=AgentResponse,
    description=(
        "Send one message to the coding agent. Pass the returned thread_id "
        "to continue the conversation."
    ),
    responses={
        409: {"description": "The thread already has a run in progress"},
        504: {"description": "The run or the LLM provider timed out"},
    },
)
async def invoke(
    request: AgentRequest,
    graph: AgentGraphDependency,
    threads: ThreadStoreDependency,
    limits: AgentLimitsDependency,
) -> AgentResponse:
    try:
        return await invoke_agent(request, graph, threads, limits)
    except ThreadBusyError as error:
        logger.info(
            "Agent invocation rejected",
            extra={"thread_id": error.thread_id, "reason": "thread_busy"},
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Thread already has a run in progress",
        ) from error
    except AgentRunTimeoutError as error:
        # Already logged with the thread id where the deadline expired.
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Agent run timed out",
        ) from error
