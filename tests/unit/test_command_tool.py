from collections.abc import Sequence
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, ToolException

from app.agent.approval import approval_tool_names
from app.agent.sandbox import CommandResult, ContainmentError
from app.agent.tools import make_agent_tools
from app.agent.tools.command import make_command_tool, run_command
from app.agent.workspace import Workspace, WorkspaceError
from tests.fakes import FakeSandbox


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path)


async def call(tool: BaseTool, **args: object) -> ToolMessage:
    result = await tool.ainvoke(
        {"type": "tool_call", "name": tool.name, "args": args, "id": "call-1"}
    )
    assert isinstance(result, ToolMessage)
    return result


@pytest.mark.anyio
async def test_run_command_uses_the_sandbox_and_formats_its_result(
    workspace: Workspace,
) -> None:
    sandbox = FakeSandbox(
        CommandResult(
            exit_code=0,
            output="all tests passed\n",
            truncated=False,
            output_bytes=17,
            timed_out=False,
            duration_ms=12.5,
        )
    )
    tool = make_command_tool(workspace, sandbox)

    result = await call(tool, argv=["pytest", "-q"])

    assert approval_tool_names([tool]) == {"run_command"}
    assert result.status == "success"
    assert result.content == "Command exited with code 0.\n\nall tests passed\n"
    assert sandbox.calls == [(["pytest", "-q"], workspace.root, None)]


@pytest.mark.anyio
async def test_run_command_reports_timeout_and_truncation(
    workspace: Workspace,
) -> None:
    sandbox = FakeSandbox(
        CommandResult(
            exit_code=None,
            output="tail",
            truncated=True,
            output_bytes=60_000,
            timed_out=True,
            duration_ms=120_000,
        )
    )

    result = await run_command(workspace, sandbox, ["make", "test"])

    assert result == (
        "Command timed out and was stopped.\n\ntail\n\n"
        "[Output truncated; showing the final captured bytes.]"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("argv", [[], [""], ["-bad"], ["echo", "bad\0arg"]])
async def test_run_command_refuses_invalid_argv_before_sandbox_execution(
    workspace: Workspace, argv: list[str]
) -> None:
    sandbox = FakeSandbox(CommandResult(0, "", False, 0, False, 0))

    with pytest.raises(ToolException, match="Invalid command arguments"):
        await run_command(workspace, sandbox, argv)

    assert sandbox.calls == []


def test_command_tool_is_off_without_a_sandbox(workspace: Workspace) -> None:
    assert "run_command" not in {tool.name for tool in make_agent_tools(workspace)}
    sandbox = FakeSandbox(CommandResult(0, "", False, 0, False, 0))
    assert "run_command" in {tool.name for tool in make_agent_tools(workspace, sandbox)}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (WorkspaceError("/private/project changed"), "the workspace is unavailable"),
        (
            ValueError("not a directory: /private/project"),
            "invalid sandbox configuration or workspace",
        ),
        (
            ContainmentError("/private/project: sweep failed"),
            "disabled until the service restarts",
        ),
    ],
)
async def test_sandbox_failures_are_safe_tool_errors(
    workspace: Workspace, failure: Exception, expected: str
) -> None:
    class FailingSandbox:
        name = "fake"

        async def run(
            self, argv: Sequence[str], *, cwd: Path, timeout: float | None = None
        ) -> CommandResult:
            raise failure

    result = await call(make_command_tool(workspace, FailingSandbox()), argv=["true"])

    assert result.status == "error"
    assert expected in str(result.content)
    assert "/private/project" not in str(result.content)
    if isinstance(failure, ContainmentError):
        assert "processes may still be running" in str(result.content)
