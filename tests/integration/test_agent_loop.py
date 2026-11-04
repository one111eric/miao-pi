import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agent.graph import build_agent_graph
from app.agent.nodes import INTERRUPTED_TOOL_RESULT, STEP_LIMIT_MESSAGE
from app.agent.service import AgentLimits
from app.agent.tools import make_read_tools
from app.agent.workspace import Workspace
from tests.apps import serve
from tests.fakes import EchoChatModel, ScriptedChatModel, tool_call


def log_record(caplog: pytest.LogCaptureFixture, message: str) -> logging.LogRecord:
    return next(record for record in caplog.records if record.message == message)


class AlwaysCallsTools(BaseChatModel):
    """Asks to list the root on every call, so only the step limit stops it."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "always-calls-tools"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "AlwaysCallsTools":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        message = tool_call("ls", f"call-{self.calls}")
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.mark.anyio
async def test_startup_binds_the_read_only_tools(
    echo_model: EchoChatModel, client: AsyncClient
) -> None:
    del client
    assert echo_model.bound_tools == ["read", "ls", "find", "grep"]


@pytest.mark.anyio
async def test_the_model_reads_a_file_then_answers(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / "notes.txt").write_text("The answer is 42.", encoding="utf-8")
    model = ScriptedChatModel(
        responses=[
            tool_call("read", "call-1", path="notes.txt"),
            AIMessage(content="It says 42."),
        ]
    )

    async with serve(model, tmp_path) as (_, client):
        with caplog.at_level(logging.INFO, logger="app.agent"):
            response = await client.post(
                "/invoke", json={"message": "What's in notes.txt?"}
            )

    assert response.status_code == 200
    assert response.json()["response"] == "It says 42."
    tool_result = model.calls[1][-1]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.tool_call_id == "call-1"
    assert tool_result.content == "The answer is 42."
    completed = log_record(caplog, "Tool call completed")
    assert completed.__dict__["tool"] == "read"
    assert completed.__dict__["tool_status"] == "success"
    assert completed.__dict__["tool_duration_ms"] >= 0
    # The prompt names the blocked paths, so the model doesn't ask for them.
    prompt = model.calls[0][0].text
    assert "Paths matching .env*, .git, .claude, at any depth, are blocked" in prompt
    assert "Other paths may be blocked too." not in prompt


@pytest.mark.anyio
async def test_the_prompt_never_names_configured_deny_patterns(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path, denied_paths=("acme-merger-*", "payroll.xlsx"))
    model = ScriptedChatModel(responses=[AIMessage(content="Okay.")])
    graph = build_agent_graph(
        model, workspace=workspace, tools=make_read_tools(workspace)
    )

    await graph.ainvoke({"messages": [HumanMessage(content="Hello")]})

    # Operators deny these for being sensitive; a hosted provider sees the prompt.
    prompt = model.calls[0][0].text
    assert "acme-merger" not in prompt
    assert "payroll" not in prompt
    assert "Other paths may be blocked too." in prompt


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"path": ".env"}, "Access denied: .env is blocked by workspace policy"),
        ({"path": "notes.txt", "offset": 0}, "offset"),
    ],
)
async def test_tool_errors_reach_the_model_as_results(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    args: dict[str, object],
    expected: str,
) -> None:
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")
    model = ScriptedChatModel(
        responses=[tool_call("read", "call-1", **args), AIMessage(content="Okay.")]
    )

    async with serve(model, tmp_path) as (_, client):
        with caplog.at_level(logging.INFO, logger="app.agent"):
            response = await client.post("/invoke", json={"message": "Read it"})

    assert response.status_code == 200
    tool_result = model.calls[1][-1]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.status == "error"
    assert expected in str(tool_result.content)
    assert "SECRET" not in str(tool_result.content)
    assert log_record(caplog, "Tool call completed").__dict__["tool_status"] == "error"


@pytest.mark.anyio
@pytest.mark.parametrize("max_steps", [5, 6, 7, 25])
async def test_the_step_limit_ends_the_run_with_a_valid_history(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    max_steps: int,
) -> None:
    model = AlwaysCallsTools()

    async with serve(model, tmp_path) as (app, client):
        app.state.agent_limits = AgentLimits(max_steps=max_steps, run_timeout=30)
        with caplog.at_level(logging.WARNING, logger="app.agent"):
            response = await client.post(
                "/invoke", json={"message": "Loop", "thread_id": "looping"}
            )
        config: RunnableConfig = {"configurable": {"thread_id": "looping"}}
        state = await app.state.agent_graph.aget_state(config)

    assert response.status_code == 200
    assert response.json()["response"] == STEP_LIMIT_MESSAGE
    assert log_record(caplog, "Agent step limit reached")
    messages: list[BaseMessage] = state.values["messages"]
    requested = {
        call["id"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    assert requested == answered


@tool
async def slow() -> str:
    """Takes longer than the run is allowed."""
    await asyncio.sleep(10)
    return "done"


@pytest.mark.anyio
async def test_a_timed_out_run_returns_504_and_its_thread_recovers(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = ScriptedChatModel(
        responses=[tool_call("slow", "call-1"), AIMessage(content="Recovered.")]
    )

    async with serve(model, tmp_path) as (app, client):
        app.state.agent_graph = build_agent_graph(
            model,
            checkpointer=app.state.thread_store.checkpointer,
            tools=[slow],
        )
        app.state.agent_limits = AgentLimits(max_steps=25, run_timeout=0.2)
        with caplog.at_level(logging.WARNING, logger="app.agent"):
            timed_out = await client.post(
                "/invoke", json={"message": "Be slow", "thread_id": "slow"}
            )
        app.state.agent_limits = AgentLimits(max_steps=25, run_timeout=30)
        recovered = await client.post(
            "/invoke", json={"message": "Try again", "thread_id": "slow"}
        )

    assert timed_out.status_code == 504
    assert timed_out.json() == {"detail": "Agent run timed out"}
    assert (
        log_record(caplog, "Agent run timed out").__dict__["agent_run_timeout"] == 0.2
    )
    assert recovered.status_code == 200
    assert recovered.json()["response"] == "Recovered."
    # The unanswered call gets a placeholder result, so providers accept the history.
    sent = [(type(m).__name__, m.text) for m in model.calls[1][1:]]
    assert sent == [
        ("HumanMessage", "Be slow"),
        ("AIMessage", ""),
        ("ToolMessage", INTERRUPTED_TOOL_RESULT),
        ("HumanMessage", "Try again"),
    ]
