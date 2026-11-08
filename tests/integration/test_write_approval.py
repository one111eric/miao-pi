import asyncio
import logging
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import StructuredTool

from app.agent.approval import REQUIRES_APPROVAL
from app.agent.nodes import INTERRUPTED_TOOL_RESULT
from app.agent.sandbox import CommandResult
from app.agent.service import AgentLimits
from app.agent.tools.command import make_command_tool
from app.agent.workspace import Workspace
from tests.apps import serve
from tests.fakes import FakeSandbox, ScriptedChatModel, tool_call


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


@pytest.mark.anyio
async def test_run_command_waits_for_approval_then_uses_the_sandbox(
    tmp_path: Path,
) -> None:
    sandbox = FakeSandbox(CommandResult(0, "1 passed\n", False, 9, False, 12.5))
    command = make_command_tool(Workspace(tmp_path), sandbox)
    model = ScriptedChatModel(
        responses=[
            tool_call("run_command", "command-1", argv=["pytest", "-q"]),
            AIMessage(content="Tests pass."),
        ]
    )

    async with serve(model, tmp_path, [command]) as (_, client):
        pending = await client.post("/invoke", json={"message": "Run tests"})
        completed = await client.post(
            f"/threads/{pending.json()['thread_id']}/resume", json={"approved": True}
        )

    assert pending.json()["actions"] == [
        {
            "tool_call_id": "command-1",
            "tool": "run_command",
            "args": {"argv": ["pytest", "-q"]},
        }
    ]
    assert completed.json()["response"] == "Tests pass."
    assert sandbox.calls == [(["pytest", "-q"], tmp_path.resolve(), None)]
    assert (
        last_tool_result(model.calls[1]).content
        == "Command exited with code 0.\n\n1 passed\n"
    )


@pytest.mark.anyio
async def test_rejected_command_does_not_reach_the_sandbox(tmp_path: Path) -> None:
    sandbox = FakeSandbox(CommandResult(0, "", False, 0, False, 0))
    command = make_command_tool(Workspace(tmp_path), sandbox)
    model = ScriptedChatModel(
        responses=[
            tool_call("run_command", "command-1", argv=["pytest", "-q"]),
            AIMessage(content="I did not run it."),
        ]
    )

    async with serve(model, tmp_path, [command]) as (_, client):
        pending = await client.post("/invoke", json={"message": "Run tests"})
        completed = await client.post(
            f"/threads/{pending.json()['thread_id']}/resume", json={"approved": False}
        )

    assert completed.json()["response"] == "I did not run it."
    assert sandbox.calls == []


@pytest.mark.anyio
async def test_invalid_command_is_rejected_without_asking_for_approval(
    tmp_path: Path,
) -> None:
    sandbox = FakeSandbox(CommandResult(0, "", False, 0, False, 0))
    command = make_command_tool(Workspace(tmp_path), sandbox)
    model = ScriptedChatModel(
        responses=[
            tool_call("run_command", "command-1", argv=[]),
            AIMessage(content="That command is invalid."),
        ]
    )

    async with serve(model, tmp_path, [command]) as (_, client):
        response = await client.post("/invoke", json={"message": "Run nothing"})

    assert response.json()["status"] == "completed"
    assert response.json()["response"] == "That command is invalid."
    assert sandbox.calls == []
    assert (
        last_tool_result(model.calls[1]).content
        == "Not run: Invalid command arguments: argv must be a non-empty array."
    )


@pytest.mark.anyio
async def test_read_write_and_command_each_continue_through_approval(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.txt").write_text("old\n", encoding="utf-8")
    sandbox = FakeSandbox(CommandResult(0, "1 passed\n", False, 9, False, 12.5))
    command = make_command_tool(Workspace(tmp_path), sandbox)
    model = ScriptedChatModel(
        responses=[
            tool_call("read", "read-1", path="notes.txt"),
            tool_call("write_file", "write-1", path="notes.txt", content="new\n"),
            tool_call("run_command", "command-1", argv=["pytest", "-q"]),
            AIMessage(content="Updated the note and the tests pass."),
        ]
    )

    async with serve(model, tmp_path, [command]) as (_, client):
        write_pending = await client.post("/invoke", json={"message": "Fix it"})
        command_pending = await client.post(
            f"/threads/{write_pending.json()['thread_id']}/resume",
            json={"approved": True},
        )
        completed = await client.post(
            f"/threads/{write_pending.json()['thread_id']}/resume",
            json={"approved": True},
        )

    assert write_pending.json()["actions"][0]["tool"] == "write_file"
    assert command_pending.json()["actions"][0]["tool"] == "run_command"
    assert completed.json()["response"] == "Updated the note and the tests pass."
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "new\n"
    assert sandbox.calls == [(["pytest", "-q"], tmp_path.resolve(), None)]
