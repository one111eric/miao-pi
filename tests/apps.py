from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models import BaseChatModel

from app.agent.workspace import Workspace
from app.service import create_app


@asynccontextmanager
async def serve(
    model: BaseChatModel,
    root: Path,
) -> AsyncIterator[tuple[FastAPI, AsyncClient]]:
    """Run the app with its lifespan, a given model, and a workspace at ``root``."""
    app = create_app(llm=model, workspace=Workspace(root))
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            yield app, client
