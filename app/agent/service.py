from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import AgentGraph
from app.agent.models import AgentRequest, AgentResponse
from app.agent.state import AgentState


async def invoke_agent(request: AgentRequest, graph: AgentGraph) -> AgentResponse:
    """Invoke the agent graph for one user message."""
    state: AgentState = {"messages": [HumanMessage(content=request.message)]}
    if request.thread_id is not None:
        state["thread_id"] = request.thread_id

    result = await graph.ainvoke(state)
    final_message = result["messages"][-1]
    if not isinstance(final_message, AIMessage):
        raise RuntimeError("Agent graph did not produce an AI response")

    return AgentResponse(
        response=final_message.text,
        thread_id=result.get("thread_id"),
    )
