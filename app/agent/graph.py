from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest

from app.agent.approval import (
    ToolExecutor,
    ToolResult,
    approval_preflights,
    approval_tool_names,
    make_approval_gate,
    make_approve,
    make_route_after_react,
)
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
    ``tools`` node until the model answers without calling one. Calls to a
    tool that requires approval go through ``approve`` first, which pauses
    the run until a human decides (this needs a checkpointer).
    """
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node(
        "preplanning",
        make_preplanning(workspace, context_max_chars),
    )
    graph_builder.add_node("react", make_react(llm, tools, workspace))
    graph_builder.add_node("finalize", finalize)

    graph_builder.add_edge(START, "preplanning")
    graph_builder.add_edge("preplanning", "react")
    if tools:
        approval_tools = approval_tool_names(tools)
        preflights = approval_preflights(tools)
        gate = make_approval_gate(approval_tools, preflights)

        async def wrap_tool_call(
            request: ToolCallRequest, execute: ToolExecutor
        ) -> ToolResult:
            return await log_tool_call(request, lambda r: gate(r, execute))

        graph_builder.add_node("tools", ToolNode(tools, awrap_tool_call=wrap_tool_call))
        next_nodes = ["tools", "finalize"]
        if approval_tools:
            graph_builder.add_node("approve", make_approve(approval_tools, preflights))
            graph_builder.add_edge("approve", "tools")
            next_nodes.append("approve")
        graph_builder.add_conditional_edges(
            "react", make_route_after_react(approval_tools, preflights), next_nodes
        )
        graph_builder.add_edge("tools", "react")
    else:
        graph_builder.add_edge("react", "finalize")
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile(checkpointer=checkpointer)
