from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import finalize, preplanning, react
from app.agent.state import AgentState

type AgentGraph = CompiledStateGraph[
    AgentState,
    None,
    AgentState,
    AgentState,
]


def build_agent_graph() -> AgentGraph:
    """Build and compile the agent workflow."""
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("preplanning", preplanning)
    graph_builder.add_node("react", react)
    graph_builder.add_node("finalize", finalize)

    graph_builder.add_edge(START, "preplanning")
    graph_builder.add_edge("preplanning", "react")
    graph_builder.add_edge("react", "finalize")
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile()
