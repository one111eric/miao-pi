import logging

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel

from app.agent import router as agent_router
from app.agent.graph import build_agent_graph
from app.health.router import router as health_router
from app.service.lifespan import LifespanCallback, lifespan_function
from common.llm import LLM

logger = logging.getLogger(__name__)

service_name = "miao-pi-agent-svc"
description = "Web server for the miao-pi agent"


def make_on_startup(llm: BaseChatModel | None = None) -> LifespanCallback:
    """Create the startup hook; the configured ``LLM`` is used unless one is given."""

    async def on_startup(app: FastAPI) -> None:
        logger.info("Starting up...")
        # Built once here so a misconfigured provider fails at startup.
        app.state.agent_graph = build_agent_graph(llm or LLM())
        logger.info("Startup complete")

    return on_startup


async def on_shutdown(app: FastAPI) -> None:
    logger.info("Shutting down...")
    del app.state.agent_graph
    logger.info("Shutdown complete")


def create_app(llm: BaseChatModel | None = None) -> FastAPI:
    app = FastAPI(
        title=service_name,
        description=description,
        lifespan=lifespan_function(make_on_startup(llm), on_shutdown),
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        contact={
            "name": "Miao Pi Agent",
            "url": "https://github.com/one111eric/miao-pi",
            "email": "one1eric111@gmail.com",
        },
    )

    app.include_router(agent_router)
    app.include_router(health_router)

    return app
