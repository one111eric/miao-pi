import logging
from pathlib import Path

import pytest
from httpx import AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_agent_graph
from app.agent.nodes import SYSTEM_PROMPT
from app.agent.workspace import Workspace, WorkspaceAccessError, WorkspaceText
from app.service import create_app
from tests.fakes import EchoChatModel


def log_record(caplog: pytest.LogCaptureFixture, message: str) -> logging.LogRecord:
    return next(record for record in caplog.records if record.message == message)


@pytest.mark.anyio
async def test_agents_file_reaches_the_model_but_not_message_history(
    tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text("Use project conventions.", encoding="utf-8")
    (tmp_path / "README.md").write_text("Fallback context.", encoding="utf-8")
    model = EchoChatModel()
    graph = build_agent_graph(model, workspace=Workspace(tmp_path))

    result = await graph.ainvoke({"messages": [HumanMessage(content="Hello")]})

    system_prompt = model.calls[0][0].text
    assert system_prompt.startswith(SYSTEM_PROMPT)
    assert "Use project conventions." in system_prompt
    assert "Fallback context." not in system_prompt
    assert [type(message) for message in result["messages"]] == [
        HumanMessage,
        AIMessage,
    ]


@pytest.mark.anyio
async def test_readme_is_used_when_agents_file_is_missing(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("README context.", encoding="utf-8")
    model = EchoChatModel()
    graph = build_agent_graph(model, workspace=Workspace(tmp_path))

    await graph.ainvoke({"messages": [HumanMessage(content="Hello")]})

    assert "README context." in model.calls[0][0].text


@pytest.mark.anyio
async def test_missing_context_file_uses_only_the_base_prompt(tmp_path: Path) -> None:
    model = EchoChatModel()
    graph = build_agent_graph(model, workspace=Workspace(tmp_path))

    await graph.ainvoke({"messages": [HumanMessage(content="Hello")]})

    assert model.calls[0][0].text == SYSTEM_PROMPT


@pytest.mark.anyio
async def test_context_is_capped_and_reloaded_each_turn(tmp_path: Path) -> None:
    context_file = tmp_path / "AGENTS.md"
    context_file.write_text("first-context-that-is-long", encoding="utf-8")
    model = EchoChatModel()
    graph = build_agent_graph(
        model,
        workspace=Workspace(tmp_path),
        context_max_chars=13,
    )

    await graph.ainvoke({"messages": [HumanMessage(content="First")]})
    context_file.write_text("second-context-is-new", encoding="utf-8")
    await graph.ainvoke({"messages": [HumanMessage(content="Second")]})

    first_prompt = model.calls[0][0].text
    second_prompt = model.calls[1][0].text
    assert "first-context" in first_prompt
    assert "second-contex" in second_prompt
    assert "first-context" not in second_prompt
    assert "[Content truncated]" in first_prompt
    assert "[Content truncated]" in second_prompt


@pytest.mark.anyio
async def test_removed_context_stops_applying_within_a_thread(
    client: AsyncClient,
    echo_model: EchoChatModel,
    workspace: Workspace,
) -> None:
    context_file = workspace.root / "AGENTS.md"
    context_file.write_text("Old instructions.", encoding="utf-8")

    first = await client.post("/invoke", json={"message": "First"})
    context_file.unlink()
    second = await client.post(
        "/invoke",
        json={"message": "Second", "thread_id": first.json()["thread_id"]},
    )

    assert second.status_code == 200
    assert "Old instructions." in echo_model.calls[0][0].text
    # Same thread: the earlier turn is in history, but its context is not.
    assert echo_model.calls[1][1].text == "First"
    assert echo_model.calls[1][0].text == SYSTEM_PROMPT


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("failure", "error_type", "reason"),
    [
        ("escape", "WorkspaceAccessError", "outside"),
        ("denied", "WorkspaceAccessError", "denied"),
        ("unreadable", "PermissionError", None),
    ],
)
async def test_inaccessible_agents_file_falls_back_to_readme(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
    error_type: str,
    reason: str | None,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside-agents.md"
    outside.write_text("Outside context.", encoding="utf-8")
    agents_file = root / "AGENTS.md"
    if failure == "escape":
        agents_file.symlink_to(outside)
        workspace = Workspace(root)
    else:
        agents_file.write_text("Unavailable context.", encoding="utf-8")
        workspace = Workspace(
            root,
            denied_paths=("AGENTS.md",) if failure == "denied" else (),
        )
    (root / "README.md").write_text("Fallback context.", encoding="utf-8")

    if failure == "unreadable":
        original_read_text = workspace.read_text

        def fail_for_agents(path: str | Path, *, max_chars: int) -> WorkspaceText:
            if Path(path).name == "AGENTS.md":
                raise PermissionError("unreadable")
            return original_read_text(path, max_chars=max_chars)

        monkeypatch.setattr(workspace, "read_text", fail_for_agents)

    model = EchoChatModel()
    graph = build_agent_graph(model, workspace=workspace)

    with caplog.at_level(logging.INFO, logger="app.agent.nodes"):
        await graph.ainvoke({"messages": [HumanMessage(content="Hello")]})

    prompt = model.calls[0][0].text
    assert "Fallback context." in prompt
    assert "Unavailable context." not in prompt
    assert "Outside context." not in prompt
    skipped = log_record(caplog, "Skipping inaccessible project context")
    assert skipped.__dict__["context_file"] == "AGENTS.md"
    assert skipped.__dict__["error_type"] == error_type
    assert skipped.__dict__.get("reason") == reason
    assert log_record(caplog, "Project context loaded").__dict__["context_file"] == (
        "README.md"
    )


@pytest.mark.anyio
async def test_startup_builds_workspace_from_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("WORKSPACE_DENIED_PATHS", '["private"]')
    app = create_app(llm=EchoChatModel())

    with caplog.at_level(logging.INFO, logger="app.service"):
        async with app.router.lifespan_context(app):
            pass
    startup = log_record(caplog, "Startup complete")
    assert startup.__dict__["workspace_root"] == str(tmp_path.resolve())
    assert startup.__dict__["workspace_denied_patterns"] == [".env*", ".git", "private"]
    assert startup.__dict__["workspace_context_max_chars"] == 20_000

    async with app.router.lifespan_context(app):
        assert app.state.workspace.root == tmp_path.resolve()
        with pytest.raises(WorkspaceAccessError):
            app.state.workspace.resolve("private/file.txt")
        with pytest.raises(WorkspaceAccessError):
            app.state.workspace.resolve(".env")


@pytest.mark.anyio
async def test_loaded_context_is_logged_with_its_size(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / "AGENTS.md").write_text("x" * 20, encoding="utf-8")
    graph = build_agent_graph(
        EchoChatModel(),
        workspace=Workspace(tmp_path),
        context_max_chars=10,
    )

    with caplog.at_level(logging.INFO, logger="app.agent.nodes"):
        await graph.ainvoke({"messages": [HumanMessage(content="Hello")]})

    loaded = log_record(caplog, "Project context loaded")
    assert loaded.__dict__["context_file"] == "AGENTS.md"
    assert loaded.__dict__["context_chars"] == len("x" * 10 + "\n\n[Content truncated]")
    assert loaded.__dict__["context_truncated"] is True


@pytest.mark.anyio
async def test_missing_context_is_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    graph = build_agent_graph(EchoChatModel(), workspace=Workspace(tmp_path))

    with caplog.at_level(logging.INFO, logger="app.agent.nodes"):
        await graph.ainvoke({"messages": [HumanMessage(content="Hello")]})

    assert log_record(caplog, "No project context found")
