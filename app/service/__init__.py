import logging

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel

from app.agent import router as agent_router
from app.agent.graph import build_agent_graph
from app.agent.sandbox.startup import make_sandbox
from app.agent.service import AgentLimits
from app.agent.threads import ThreadStore
from app.agent.tools import make_read_tools
from app.agent.workspace import Workspace
from app.health.router import router as health_router
from app.service.error_handling import register_exception_handlers
from app.service.lifespan import LifespanCallback, lifespan_function
from app.service.middlewares import RequestContextMiddleware
from app.ui.router import router as ui_router
from common.config import SvcSettings
from common.llm import LLM

logger = logging.getLogger(__name__)

service_name = "miao-pi-agent-svc"
description = "Web server for the miao-pi agent"


def make_on_startup(
    llm: BaseChatModel | None = None,
    thread_store: ThreadStore | None = None,
    workspace: Workspace | None = None,
) -> LifespanCallback:
    """Create the startup hook; configured defaults are used unless given."""

    async def on_startup(app: FastAPI) -> None:
        logger.info("Starting up...")
        settings = SvcSettings()
        threads = thread_store or ThreadStore(
            ttl=settings.thread_ttl,
            max_threads=settings.max_threads,
        )
        active_workspace = workspace or Workspace(
            root=settings.workspace_root,
            denied_paths=settings.workspace_denied_paths,
        )
        tools = make_read_tools(active_workspace)
        # Self-tested before use; step 6 offers command tools only if it works.
        sandbox_setup = await make_sandbox(settings, active_workspace)
        app.state.sandbox = sandbox_setup.sandbox
        app.state.thread_store = threads
        app.state.workspace = active_workspace
        app.state.dev_ui_enabled = settings.dev_ui_enabled
        app.state.agent_limits = AgentLimits(
            max_steps=settings.agent_max_steps,
            run_timeout=settings.agent_run_timeout,
        )
        # Built once here so a misconfigured provider fails at startup.
        app.state.agent_graph = build_agent_graph(
            llm or LLM(settings=settings),
            checkpointer=threads.checkpointer,
            workspace=active_workspace,
            context_max_chars=settings.workspace_context_max_chars,
            tools=tools,
        )
        # The root defaults to the working directory, so make the choice visible.
        logger.info(
            "Startup complete",
            extra={
                "workspace_root": str(active_workspace.root),
                "workspace_denied_patterns": list(active_workspace.denied_patterns),
                "workspace_context_max_chars": settings.workspace_context_max_chars,
                "agent_tools": [tool.name for tool in tools],
                "agent_max_steps": settings.agent_max_steps,
                "agent_run_timeout": settings.agent_run_timeout,
                "dev_ui_enabled": settings.dev_ui_enabled,
                "sandbox": sandbox_setup.status,
                "sandbox_read_paths": [str(path) for path in sandbox_setup.read_paths],
            },
        )

    return on_startup


async def on_shutdown(app: FastAPI) -> None:
    logger.info("Shutting down...")
    del app.state.agent_graph
    del app.state.thread_store
    del app.state.workspace
    del app.state.agent_limits
    del app.state.dev_ui_enabled
    del app.state.sandbox
    logger.info("Shutdown complete")


def create_app(
    llm: BaseChatModel | None = None,
    thread_store: ThreadStore | None = None,
    workspace: Workspace | None = None,
) -> FastAPI:
    app = FastAPI(
        title=service_name,
        description=description,
        lifespan=lifespan_function(
            make_on_startup(llm, thread_store, workspace),
            on_shutdown,
        ),
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        contact={
            "name": "Miao Pi Agent",
            "url": "https://github.com/one111eric/miao-pi",
            "email": "one1eric111@gmail.com",
        },
    )

    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(agent_router)
    app.include_router(health_router)
    app.include_router(ui_router)

    return app
