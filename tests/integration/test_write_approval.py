import asyncio
import logging
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import StructuredTool

from app.agent.approval import REQUIRES_APPROVAL
from app.agent.nodes import INTERRUPTED_TOOL_RESULT
from app.agent.service import AgentLimits
from tests.apps import serve
from tests.fakes import ScriptedChatModel, tool_call


def last_tool_result(messages: list[BaseMessage]) -> ToolMessage:
    result = messages[-1]
    assert isinstance(result, ToolMessage)
    return result


@pytest.mark.anyio
async def test_write_file_waits_for_approval_then_writes(tmp_path: Path) -> None:
    model = ScriptedChatModel(
        responses=[
            tool_call("write_file", "write-1", path="notes.txt", content="hello\n"),
            AIMessage(content="Done."),
        ]
    )

    async with serve(model, tmp_path) as (_, client):
        pending = await client.post("/invoke", json={"message": "Write a note"})
        assert not (tmp_path / "notes.txt").exists()
        completed = await client.post(
            f"/threads/{pending.json()['thread_id']}/resume", json={"approved": True}
        )

    assert pending.json()["actions"] == [
        {
            "tool_call_id": "write-1",
            "tool": "write_file",
            "args": {"path": "notes.txt", "content": "hello\n"},
        }
    ]
    assert completed.json()["response"] == "Done."
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello\n"
    assert last_tool_result(model.calls[1]).content == "Wrote notes.txt."


@pytest.mark.anyio
async def test_denied_write_is_rejected_without_asking_for_approval(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    model = ScriptedChatModel(
        responses=[
            tool_call("write_file", "write-1", path=".env", content="changed"),
            AIMessage(content="Cannot change it."),
        ]
    )

    async with serve(model, tmp_path) as (_, client):
        response = await client.post("/invoke", json={"message": "Change the secret"})

    assert response.json()["status"] == "completed"
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "SECRET=1"
    result = last_tool_result(model.calls[1])
    assert result.status == "error"
    assert (
        result.content == "Not run: Access denied: .env is blocked by workspace policy"
    )


@pytest.mark.anyio
async def test_denied_write_logs_one_rejection_with_its_tool_name(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    model = ScriptedChatModel(
        responses=[
            tool_call("write_file", "write-1", path="../outside.txt", content="no"),
            AIMessage(content="Cannot change it."),
        ]
    )

    async with serve(model, tmp_path) as (_, client):
        with caplog.at_level(logging.WARNING, logger="app.agent"):
            await client.post("/invoke", json={"message": "Write outside"})

    rejected = [
        record for record in caplog.records if record.message == "Tool path rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0].__dict__["tool"] == "write_file"
    assert rejected[0].__dict__["tool_path"] == "../outside.txt"
    assert rejected[0].__dict__["reason"] == "outside"


@pytest.mark.anyio
async def test_oversized_write_is_rejected_without_asking_for_approval(
    tmp_path: Path,
) -> None:
    content = "x" * 1_000_001
    model = ScriptedChatModel(
        responses=[
            tool_call("write_file", "write-1", path="notes.txt", content=content),
            AIMessage(content="The requested file is too large."),
        ]
    )

    async with serve(model, tmp_path) as (_, client):
        response = await client.post("/invoke", json={"message": "Write it"})

    assert response.json()["status"] == "completed"
    assert response.json()["response"] == "The requested file is too large."
    assert not (tmp_path / "notes.txt").exists()
    result = last_tool_result(model.calls[1])
    assert (
        result.content
        == "Not run: Invalid write arguments: content exceeds 1000000 characters."
    )


@pytest.mark.anyio
async def test_missing_patch_source_is_rejected_without_asking_for_approval(
    tmp_path: Path,
) -> None:
    patch = """--- a/gone.txt
+++ /dev/null
@@ -1 +0,0 @@
-gone
"""
    model = ScriptedChatModel(
        responses=[
            tool_call("apply_patch", "patch-1", patch=patch),
            AIMessage(content="It is already absent."),
        ]
    )

    async with serve(model, tmp_path) as (_, client):
        response = await client.post("/invoke", json={"message": "Delete it"})

    assert response.json()["status"] == "completed"
    result = last_tool_result(model.calls[1])
    assert result.content == "Not run: Patch does not apply: gone.txt does not exist"


@pytest.mark.anyio
async def test_approval_resume_refreshes_the_between_steps_deadline(
    tmp_path: Path,
) -> None:
    model = ScriptedChatModel(
        responses=[
            tool_call("write_file", "write-1", path="notes.txt", content="hello\n"),
            AIMessage(content="Done."),
        ]
    )

    async with serve(model, tmp_path) as (app, client):
        app.state.agent_limits = AgentLimits(
            max_steps=25, run_timeout=0.05, backstop_timeout=1
        )
        pending = await client.post("/invoke", json={"message": "Write a note"})
        await asyncio.sleep(0.06)
        completed = await client.post(
            f"/threads/{pending.json()['thread_id']}/resume", json={"approved": True}
        )

    assert completed.json()["response"] == "Done."
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello\n"


@pytest.mark.anyio
async def test_backstop_preserves_an_ambiguous_approved_tool_call_in_history(
    tmp_path: Path,
) -> None:
    async def slow_write() -> str:
        await asyncio.sleep(1)
        return "done"

    slow_tool = StructuredTool.from_function(
        coroutine=slow_write,
        name="slow_write",
        description="Deliberately slow approved test tool.",
        metadata={REQUIRES_APPROVAL: True},
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("slow_write", "slow-1"),
            AIMessage(content="Recovered."),
        ]
    )

    async with serve(model, tmp_path, [slow_tool]) as (app, client):
        app.state.agent_limits = AgentLimits(
            max_steps=25, run_timeout=0.01, backstop_timeout=0.05
        )
        pending = await client.post("/invoke", json={"message": "Be slow"})
        timed_out = await client.post(
            f"/threads/{pending.json()['thread_id']}/resume", json={"approved": True}
        )
        app.state.agent_limits = AgentLimits(max_steps=25, run_timeout=30)
        recovered = await client.post(
            "/invoke",
            json={"message": "Continue", "thread_id": pending.json()["thread_id"]},
        )

    assert timed_out.status_code == 504
    assert recovered.json()["response"] == "Recovered."
    sent = [(type(message).__name__, message.text) for message in model.calls[1][1:]]
    assert ("ToolMessage", INTERRUPTED_TOOL_RESULT) in sent


@pytest.mark.anyio
async def test_apply_patch_waits_for_approval_then_checks_its_base(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.txt").write_text("old\n", encoding="utf-8")
    patch = """--- a/notes.txt
+++ b/notes.txt
@@ -1 +1 @@
-old
+new
"""
    model = ScriptedChatModel(
        responses=[
            tool_call("apply_patch", "patch-1", patch=patch),
            AIMessage(content="Patched."),
        ]
    )

    async with serve(model, tmp_path) as (_, client):
        pending = await client.post("/invoke", json={"message": "Patch the note"})
        assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "old\n"
        completed = await client.post(
            f"/threads/{pending.json()['thread_id']}/resume", json={"approved": True}
        )

    assert pending.json()["actions"][0]["tool"] == "apply_patch"
    assert completed.json()["response"] == "Patched."
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "new\n"
