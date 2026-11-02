from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.agent.graph import AgentGraph
from app.agent.models import AgentRequest, AgentResponse
from app.agent.threads import ThreadStore
from common.logging_helper import logging_context


async def invoke_agent(
    request: AgentRequest,
    graph: AgentGraph,
    threads: ThreadStore,
) -> AgentResponse:
    """Invoke the agent graph for one user message in a conversation thread."""
    thread_id = request.thread_id or str(uuid4())
    # The checkpointer finds the conversation by this config value.
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    with logging_context(thread_id=thread_id):
        async with threads.use(thread_id):
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
            )

    final_message = result["messages"][-1]
    if not isinstance(final_message, AIMessage):
        raise RuntimeError("Agent graph did not produce an AI response")

    return AgentResponse(response=final_message.text, thread_id=thread_id)
