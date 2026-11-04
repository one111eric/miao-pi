import asyncio
import logging
from dataclasses import dataclass
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.agent.graph import AgentGraph
from app.agent.models import AgentRequest, AgentResponse
from app.agent.threads import ThreadStore
from common.logging_helper import logging_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentLimits:
    """Bounds on one run.

    ``max_steps`` is LangGraph's recursion limit: every node run is a step, so
    each tool round takes two. ``run_timeout`` is in seconds.
    """

    max_steps: int
    run_timeout: float


class AgentRunTimeoutError(Exception):
    """Raised when a run takes longer than its time limit."""

    def __init__(self, timeout: float) -> None:
        super().__init__(f"Agent run exceeded {timeout} seconds")
        self.timeout = timeout


async def invoke_agent(
    request: AgentRequest,
    graph: AgentGraph,
    threads: ThreadStore,
    limits: AgentLimits,
) -> AgentResponse:
    """Invoke the agent graph for one user message in a conversation thread."""
    thread_id = request.thread_id or str(uuid4())
    config: RunnableConfig = {
        # The checkpointer finds the conversation by this config value.
        "configurable": {"thread_id": thread_id},
        "recursion_limit": limits.max_steps,
    }

    with logging_context(thread_id=thread_id):
        # A supplied thread that ends with conversation_message_count=2 had no
        # history: the caller chose a new id, or the thread had expired.
        logger.info(
            "Agent invocation started",
            extra={"thread_supplied": request.thread_id is not None},
        )
        async with threads.use(thread_id):
            deadline = asyncio.timeout(limits.run_timeout)
            try:
                async with deadline:
                    result = await graph.ainvoke(
                        {"messages": [HumanMessage(content=request.message)]},
                        config=config,
                    )
            except TimeoutError as error:
                # Only our deadline counts; a TimeoutError from inside the run
                # is some other failure.
                if not deadline.expired():
                    raise
                logger.warning(
                    "Agent run timed out",
                    extra={"agent_run_timeout": limits.run_timeout},
                )
                raise AgentRunTimeoutError(limits.run_timeout) from error

        final_message = result["messages"][-1]
        if not isinstance(final_message, AIMessage):
            raise RuntimeError("Agent graph did not produce an AI response")

        logger.info(
            "Agent invocation completed",
            extra={"conversation_message_count": len(result["messages"])},
        )
        return AgentResponse(response=final_message.text, thread_id=thread_id)
