from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.nodes import finalize, log_tool_call, make_preplanning, make_react
from app.agent.state import AgentState
from app.agent.workspace import Workspace

type AgentGraph = CompiledStateGraph[
    AgentState,
    None,
    AgentState,
    AgentState,
]


def build_agent_graph(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    workspace: Workspace | None = None,
    context_max_chars: int = 20_000,
    tools: Sequence[BaseTool] = (),
) -> AgentGraph:
    """Build and compile the agent workflow around the given chat model.

    With a checkpointer, runs that share a ``thread_id`` in their config
    continue the same conversation. With tools, ``react`` loops through a
    ``tools`` node until the model answers without calling one.
    """
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node(
        "preplanning",
        make_preplanning(workspace, context_max_chars),
    )
    graph_builder.add_node("react", make_react(llm, tools))
    graph_builder.add_node("finalize", finalize)

    graph_builder.add_edge(START, "preplanning")
    graph_builder.add_edge("preplanning", "react")
    if tools:
        graph_builder.add_node("tools", ToolNode(tools, awrap_tool_call=log_tool_call))
        graph_builder.add_conditional_edges(
            "react",
            tools_condition,
            {"tools": "tools", END: "finalize"},
        )
        graph_builder.add_edge("tools", "react")
    else:
        graph_builder.add_edge("react", "finalize")
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile(checkpointer=checkpointer)
