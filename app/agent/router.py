import logging
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response, status

from app.agent.graph import AgentGraph
from app.agent.models import (
    THREAD_ID_PATTERN,
    AgentRequest,
    AgentResponse,
    ThreadHistory,
)
from app.agent.service import (
    AgentLimits,
    AgentRunTimeoutError,
    invoke_agent,
    read_thread_history,
)
from app.agent.threads import ThreadBusyError, ThreadStore
from app.ui.gate import require_dev_ui

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


@router.get(
    "/threads/{thread_id}/messages",
    response_model=ThreadHistory,
    dependencies=[Depends(require_dev_ui)],
    description=(
        "Read a conversation's saved messages, including tool calls and their "
        "results. Only available when the developer test UI is enabled."
    ),
    responses={404: {"description": "No such thread, or the test UI is off"}},
)
async def thread_messages(
    thread_id: Annotated[str, Path(pattern=THREAD_ID_PATTERN)],
    graph: AgentGraphDependency,
    response: Response,
) -> ThreadHistory:
    # Tool results include file contents; no browser or proxy should keep them.
    response.headers["Cache-Control"] = "no-store"
    history = await read_thread_history(graph, thread_id)
    if history is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        )
    return history
