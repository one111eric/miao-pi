import logging

from fastapi import FastAPI

from app.agent import router as agent_router
from app.agent.graph import build_agent_graph
from app.health.router import router as health_router
from app.service.lifespan import lifespan_function

logger = logging.getLogger(__name__)

service_name = "miao-pi-agent-svc"
description = "Web server for the miao-pi agent"


async def on_startup(app: FastAPI) -> None:
    logger.info("Starting up...")
    app.state.agent_graph = build_agent_graph()
    logger.info("Startup complete")


async def on_shutdown(app: FastAPI) -> None:
    logger.info("Shutting down...")
    del app.state.agent_graph
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=service_name,
        description=description,
        lifespan=lifespan_function(on_startup, on_shutdown),
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        contact={
            "name": "Miao Pi Agent",
            "url": "https://github.com/miao-pi-agent",
            "email": "one1eric111@gmail.com",
        },
    )

    app.include_router(agent_router)
    app.include_router(health_router)

    return app
