import logging
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.agent.graph import build_agent_graph
from app.agent.nodes import STEP_LIMIT_MESSAGE
from app.agent.service import AgentLimits
from app.agent.threads import ThreadStore
from common.logging_helper import RequestFormatter
from tests.apps import serve
from tests.fakes import EchoChatModel, ScriptedChatModel, recording_tool, tool_call

PENDING_RECORD = {
    "tool_call_id": "call-1",
    "tool": "record",
    "args": {"text": "hello"},
}


def log_record(caplog: pytest.LogCaptureFixture, message: str) -> logging.LogRecord:
    return next(record for record in caplog.records if record.message == message)


def last_tool_result(messages: list[BaseMessage]) -> ToolMessage:
    result = messages[-1]
    assert isinstance(result, ToolMessage)
    return result


def record_then_answer() -> ScriptedChatModel:
    return ScriptedChatModel(
        responses=[
            tool_call("record", "call-1", text="hello"),
            AIMessage(content="Done."),
        ]
    )


@pytest.mark.anyio
async def test_a_tool_that_needs_approval_pauses_the_run(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    effects: list[str] = []
    model = record_then_answer()

    async with serve(model, tmp_path, [recording_tool(effects)]) as (_, client):
        with caplog.at_level(logging.INFO, logger="app.agent"):
            response = await client.post(
                "/invoke", json={"message": "Note hello", "thread_id": "t1"}
            )

    assert response.status_code == 200
    assert response.json() == {
        "status": "awaiting_approval",
        "response": None,
        "thread_id": "t1",
        "actions": [PENDING_RECORD],
    }
    assert effects == []
    assert len(model.calls) == 1
    paused = log_record(caplog, "Agent paused for approval")
    assert paused.__dict__["pending_tools"] == ["record"]


@pytest.mark.anyio
async def test_approving_runs_the_tool_and_finishes_the_run(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effects: list[str] = []
    model = record_then_answer()
    monkeypatch.setattr(
        caplog.handler, "formatter", RequestFormatter("%(context)s%(message)s")
    )

    async with serve(model, tmp_path, [recording_tool(effects)]) as (_, client):
        await client.post("/invoke", json={"message": "Note hello", "thread_id": "t1"})
        with caplog.at_level(logging.INFO, logger="app.agent"):
            response = await client.post("/threads/t1/resume", json={"approved": True})

    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "response": "Done.",
        "thread_id": "t1",
        "actions": [],
    }
    assert effects == ["hello"]
    result = last_tool_result(model.calls[1])
    assert result.status == "success"
    assert result.content == "Recorded hello"
    decision = log_record(caplog, "Approval decision received")
    assert decision.__dict__["approved"] is True
    assert decision.__dict__["pending_tools"] == ["record"]
    assert decision.__dict__["pending_tool_call_ids"] == ["call-1"]
    completed = log_record(caplog, "Tool call completed")
    assert "tool_call_id=call-1" in completed.__dict__["context"]


@pytest.mark.anyio
async def test_rejecting_skips_the_tool_and_tells_the_model(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    effects: list[str] = []
    model = record_then_answer()

    async with serve(model, tmp_path, [recording_tool(effects)]) as (_, client):
        await client.post("/invoke", json={"message": "Note hello", "thread_id": "t1"})
        response = await client.post(
            "/threads/t1/resume", json={"approved": False, "note": "Not today."}
        )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert effects == []
    result = last_tool_result(model.calls[1])
    assert result.status == "error"
    assert result.content == (
        "Not run: the user rejected this action. Their note: Not today."
    )
    refused = log_record(caplog, "Tool call refused")
    assert refused.__dict__["reason"] == "approval_rejected"
    assert "Not today" not in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize(("approved", "effects_after"), [(True, ["x"]), (False, [])])
async def test_only_the_calls_that_need_approval_wait_for_it(
    tmp_path: Path,
    approved: bool,
    effects_after: list[str],
) -> None:
    (tmp_path / "notes.txt").write_text("Notes.", encoding="utf-8")
    effects: list[str] = []
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "read", "args": {"path": "notes.txt"}, "id": "read-1"},
            {"name": "record", "args": {"text": "x"}, "id": "record-1"},
        ],
    )
    model = ScriptedChatModel(responses=[both, AIMessage(content="Done.")])

    async with serve(model, tmp_path, [recording_tool(effects)]) as (_, client):
        paused = await client.post(
            "/invoke", json={"message": "Read, then note", "thread_id": "t1"}
        )
        await client.post("/threads/t1/resume", json={"approved": approved})

    assert [action["tool"] for action in paused.json()["actions"]] == ["record"]
    assert effects == effects_after
    results = {
        message.tool_call_id: message.status
        for message in model.calls[1]
        if isinstance(message, ToolMessage)
    }
    assert results == {
        "read-1": "success",
        "record-1": "success" if approved else "error",
    }


@pytest.mark.anyio
async def test_a_new_message_waits_for_the_decision(tmp_path: Path) -> None:
    effects: list[str] = []
    model = record_then_answer()

    async with serve(model, tmp_path, [recording_tool(effects)]) as (_, client):
        await client.post("/invoke", json={"message": "Note hello", "thread_id": "t1"})
        refused = await client.post(
            "/invoke", json={"message": "Something else", "thread_id": "t1"}
        )
        approved = await client.post("/threads/t1/resume", json={"approved": True})

    assert refused.status_code == 409
    assert refused.json() == {
        "detail": "Thread is waiting for an approval; resume it first"
    }
    assert approved.status_code == 200
    assert effects == ["hello"]


@pytest.mark.anyio
async def test_resume_refuses_threads_with_nothing_to_approve(tmp_path: Path) -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="Hi.")])

    async with serve(model, tmp_path) as (_, client):
        await client.post("/invoke", json={"message": "Hi", "thread_id": "done"})
        nothing = await client.post("/threads/done/resume", json={"approved": True})
        unknown = await client.post("/threads/nope/resume", json={"approved": True})
        invalid = await client.post(
            "/threads/has%20space/resume", json={"approved": True}
        )
        no_decision = await client.post("/threads/done/resume", json={})

    assert nothing.status_code == 409
    assert nothing.json() == {"detail": "Thread has nothing waiting for approval"}
    assert unknown.status_code == 404
    assert unknown.json() == {"detail": "Thread not found"}
    assert invalid.status_code == 422
    assert no_decision.status_code == 422


@pytest.mark.anyio
async def test_history_lists_the_pending_actions(tmp_path: Path) -> None:
    model = record_then_answer()

    async with serve(model, tmp_path, [recording_tool([])]) as (app, client):
        app.state.dev_ui_enabled = True
        await client.post("/invoke", json={"message": "Note hello", "thread_id": "t1"})
        paused = await client.get("/threads/t1/messages")
        await client.post("/threads/t1/resume", json={"approved": True})
        finished = await client.get("/threads/t1/messages")

    assert paused.json()["pending_actions"] == [PENDING_RECORD]
    assert finished.json()["pending_actions"] == []


@pytest.mark.anyio
async def test_the_tools_node_refuses_a_call_with_no_approval(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    effects: list[str] = []
    model = record_then_answer()

    async with serve(model, tmp_path, [recording_tool(effects)]) as (app, client):
        await client.post("/invoke", json={"message": "Note hello", "thread_id": "t1"})
        # Act as if approve had finished without recording a decision, then go
        # on: the gate in front of the tool must still refuse it.
        graph = app.state.agent_graph
        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
        await graph.aupdate_state(config, {"approvals": {}}, as_node="approve")
        await graph.ainvoke(None, config)

    assert effects == []
    result = last_tool_result(model.calls[1])
    assert result.status == "error"
    assert result.content == (
        "Not run: this action needs a human's approval, and none was given."
    )
    refused = log_record(caplog, "Tool call refused")
    assert refused.__dict__["reason"] == "approval_missing"


@pytest.mark.anyio
@pytest.mark.parametrize("max_steps", [5, 6, 7, 8, 25])
async def test_the_step_limit_leaves_room_for_an_approval(
    tmp_path: Path,
    max_steps: int,
) -> None:
    effects: list[str] = []
    model = record_then_answer()

    async with serve(model, tmp_path, [recording_tool(effects)]) as (app, client):
        app.state.agent_limits = AgentLimits(max_steps=max_steps, run_timeout=30)
        response = await client.post(
            "/invoke", json={"message": "Note hello", "thread_id": "t1"}
        )
        if response.json()["status"] == "awaiting_approval":
            response = await client.post("/threads/t1/resume", json={"approved": True})

    # Either the approval round fits, or the run stops cleanly before asking;
    # it never ends in a GraphRecursionError.
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert (body["response"], effects) in [
        ("Done.", ["hello"]),
        (STEP_LIMIT_MESSAGE, []),
    ]


@pytest.mark.anyio
async def test_resuming_an_unknown_thread_leaves_real_ones_alone(
    tmp_path: Path,
) -> None:
    model = EchoChatModel()

    async with serve(model, tmp_path) as (app, client):
        threads = ThreadStore(ttl=3600, max_threads=1)
        app.state.thread_store = threads
        app.state.agent_graph = build_agent_graph(
            model, checkpointer=threads.checkpointer
        )
        await client.post("/invoke", json={"message": "Keep me", "thread_id": "real"})
        unknown = await client.post("/threads/fake/resume", json={"approved": True})
        await client.post(
            "/invoke", json={"message": "Still here?", "thread_id": "real"}
        )

    assert unknown.status_code == 404
    # "real" kept its history: an unknown id never counted toward the cap.
    assert [message.text for message in model.calls[-1][1:]] == [
        "Keep me",
        "Received: Keep me",
        "Still here?",
    ]


@pytest.mark.anyio
async def test_a_tool_call_without_an_id_still_waits_for_approval(
    tmp_path: Path,
) -> None:
    effects: list[str] = []
    no_id = AIMessage(
        content="",
        tool_calls=[
            {"name": "record", "args": {"text": "x"}, "id": None, "type": "tool_call"}
        ],
    )
    model = ScriptedChatModel(responses=[no_id, AIMessage(content="Done.")])

    async with serve(model, tmp_path, [recording_tool(effects)]) as (_, client):
        paused = await client.post("/invoke", json={"message": "Go", "thread_id": "t1"})
        done = await client.post("/threads/t1/resume", json={"approved": True})

    assert paused.json()["status"] == "awaiting_approval"
    (action,) = paused.json()["actions"]
    assert action["tool"] == "record"
    assert action["tool_call_id"].startswith("call_")
    assert done.json()["response"] == "Done."
    assert effects == ["x"]
    assert last_tool_result(model.calls[1]).tool_call_id == action["tool_call_id"]
