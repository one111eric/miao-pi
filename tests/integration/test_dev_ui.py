import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from langchain_core.messages import AIMessage

from app.agent.graph import build_agent_graph
from app.agent.workspace import Workspace
from app.service import create_app
from app.ui.router import CONTENT_SECURITY_POLICY
from tests.apps import serve
from tests.fakes import EchoChatModel, GatedChatModel, ScriptedChatModel, tool_call


def message(type: str, text: str, **fields: object) -> dict[str, object]:
    """A history entry, with the fields a message type doesn't use left empty."""
    return {
        "type": type,
        "text": text,
        "tool_calls": [],
        "tool_call_id": None,
        "name": None,
        "status": None,
        **fields,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "path",
    [
        "/ui",
        "/ui/",
        "/ui/app.js",
        "/ui/sse.js",
        "/ui/app.css",
        "/threads/thread-1/messages",
    ],
)
async def test_dev_routes_are_hidden_when_the_ui_is_off(
    app: FastAPI,
    client: AsyncClient,
    path: str,
) -> None:
    # Set explicitly, so a developer's .env can't turn it on for this test.
    app.state.dev_ui_enabled = False

    response = await client.get(path)

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


@pytest.mark.anyio
@pytest.mark.parametrize(("value", "enabled"), [("true", True), ("false", False)])
async def test_startup_reads_the_dev_ui_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    enabled: bool,
) -> None:
    monkeypatch.setenv("DEV_UI_ENABLED", value)
    app = create_app(llm=EchoChatModel(), workspace=Workspace(tmp_path))

    async with app.router.lifespan_context(app):
        assert app.state.dev_ui_enabled is enabled


@pytest.mark.anyio
async def test_the_page_and_its_assets_are_served(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    app.state.dev_ui_enabled = True

    redirect = await client.get("/ui")
    page = await client.get("/ui/")
    script = await client.get("/ui/app.js")
    parser = await client.get("/ui/sse.js")
    stylesheet = await client.get("/ui/app.css")

    assert redirect.status_code == 307
    assert redirect.headers["location"] == "/ui/"
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert page.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert '<script src="app.js" defer></script>' in page.text
    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert '<script src="sse.js" defer></script>' in page.text
    assert parser.status_code == 200
    assert parser.headers["content-type"].startswith("text/javascript")
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")


@pytest.mark.anyio
async def test_history_shows_tool_calls_and_their_results(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("The answer is 42.", encoding="utf-8")
    model = ScriptedChatModel(
        responses=[
            tool_call("read", "call-1", path="notes.txt"),
            AIMessage(content="It says 42."),
        ]
    )

    async with serve(model, tmp_path) as (app, client):
        app.state.dev_ui_enabled = True
        await client.post(
            "/invoke", json={"message": "Read notes.txt", "thread_id": "thread-1"}
        )
        response = await client.get("/threads/thread-1/messages")

    assert response.status_code == 200
    # The body holds file contents, which no cache should keep.
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "thread_id": "thread-1",
        "messages": [
            message("human", "Read notes.txt"),
            message(
                "ai",
                "",
                tool_calls=[
                    {"id": "call-1", "name": "read", "args": {"path": "notes.txt"}}
                ],
            ),
            message(
                "tool",
                "The answer is 42.",
                tool_call_id="call-1",
                name="read",
                status="success",
            ),
            message("ai", "It says 42."),
        ],
        "pending_actions": [],
    }


@pytest.mark.anyio
async def test_history_of_an_unknown_or_invalid_thread_is_refused(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    app.state.dev_ui_enabled = True

    unknown = await client.get("/threads/no-such-thread/messages")
    invalid = await client.get("/threads/has%20space/messages")

    assert unknown.status_code == 404
    assert unknown.json() == {"detail": "Thread not found"}
    assert invalid.status_code == 422


@pytest.mark.anyio
async def test_history_can_be_read_while_a_run_is_in_progress(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    app.state.dev_ui_enabled = True
    model = GatedChatModel()
    app.state.agent_graph = build_agent_graph(
        model,
        checkpointer=app.state.thread_store.checkpointer,
    )

    run = asyncio.create_task(
        client.post("/invoke", json={"message": "Hello", "thread_id": "busy"})
    )
    try:
        await asyncio.wait_for(model.started.wait(), timeout=2)
        # Going through ThreadStore.use() would 409 here instead.
        history = await asyncio.wait_for(
            client.get("/threads/busy/messages"), timeout=2
        )
    finally:
        model.release.set()
    finished = await asyncio.wait_for(run, timeout=2)

    assert history.status_code == 200
    assert history.json()["messages"] == [message("human", "Hello")]
    assert finished.status_code == 200
