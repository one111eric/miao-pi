from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from app.agent.graph import build_agent_graph
from app.agent.tools import make_read_tools
from app.agent.workspace import Workspace
from app.service import create_app


@asynccontextmanager
async def serve(
    model: BaseChatModel,
    root: Path,
    extra_tools: Sequence[BaseTool] = (),
) -> AsyncIterator[tuple[FastAPI, AsyncClient]]:
    """Run the app with its lifespan, a given model, and a workspace at ``root``.

    ``extra_tools`` are added to the read tools, for tests that need a tool
    the app doesn't ship, such as one with a side effect.
    """
    workspace = Workspace(root)
    app = create_app(llm=model, workspace=workspace)
    async with app.router.lifespan_context(app):
        if extra_tools:
            app.state.agent_graph = build_agent_graph(
                model,
                checkpointer=app.state.thread_store.checkpointer,
                workspace=workspace,
                tools=[*make_read_tools(workspace), *extra_tools],
            )
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield app, client
