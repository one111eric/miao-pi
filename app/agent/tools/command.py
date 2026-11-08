"""Approval-gated, sandboxed command execution for the agent."""

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field

from app.agent.approval import (
    APPROVAL_PREFLIGHT,
    REQUIRES_APPROVAL,
    PreflightFailure,
)
from app.agent.sandbox import ContainmentError, Sandbox
from app.agent.sandbox.runner import CommandResult
from app.agent.workspace import Workspace, WorkspaceError

DESCRIPTION = (
    "Run a command as an argv array in the project root. The command runs only "
    "inside the configured OS sandbox: it has no network and can access only "
    "the allowed workspace files. No shell is invoked automatically; an "
    "explicit shell in argv can interpret shell syntax. Human approval is required."
)
MAX_ARGV = 128
MAX_ARGUMENT_CHARS = 10_000
MAX_ARGV_CHARS = 50_000


class RunCommandArgs(BaseModel):
    argv: list[str] = Field(
        min_length=1,
        max_length=MAX_ARGV,
        description=(
            "Program and arguments as separate strings; no shell is invoked automatically. "
            f"At most {MAX_ARGV} arguments."
        ),
    )


type ApprovalPreflight = Callable[[dict[str, Any]], PreflightFailure | None]


def _validate_argv(argv: object) -> PreflightFailure | None:
    if not isinstance(argv, list) or not argv:
        return PreflightFailure(
            "Invalid command arguments: argv must be a non-empty array."
        )
    if len(argv) > MAX_ARGV:
        return PreflightFailure(
            f"Invalid command arguments: argv has more than {MAX_ARGV} entries."
        )
    values: list[str] = []
    total = 0
    for index, value in enumerate(argv):
        if not isinstance(value, str):
            return PreflightFailure(
                f"Invalid command arguments: argv[{index}] must be a string."
            )
        if "\0" in value:
            return PreflightFailure(
                f"Invalid command arguments: argv[{index}] contains a NUL byte."
            )
        if len(value) > MAX_ARGUMENT_CHARS:
            return PreflightFailure(
                "Invalid command arguments: an argv entry exceeds "
                f"{MAX_ARGUMENT_CHARS} characters."
            )
        total += len(value)
        values.append(value)
    if total > MAX_ARGV_CHARS:
        return PreflightFailure(
            "Invalid command arguments: argv exceeds " f"{MAX_ARGV_CHARS} characters."
        )
    if not values[0]:
        return PreflightFailure("Invalid command arguments: program name is empty.")
    if values[0].startswith("-"):
        return PreflightFailure(
            "Invalid command arguments: program name cannot start with '-'."
        )
    return None


def _command_preflight() -> ApprovalPreflight:
    def check(args: dict[str, Any]) -> PreflightFailure | None:
        return _validate_argv(args.get("argv"))

    return check


def _format_result(result: CommandResult) -> str:
    if result.timed_out:
        heading = "Command timed out and was stopped."
    else:
        assert result.exit_code is not None
        heading = f"Command exited with code {result.exit_code}."
    output = result.output or "(no output)"
    notice = (
        "\n\n[Output truncated; showing the final captured bytes.]"
        if result.truncated
        else ""
    )
    return f"{heading}\n\n{output}{notice}"


async def run_command(workspace: Workspace, sandbox: Sandbox, argv: list[str]) -> str:
    """Run ``argv`` in the workspace root through the configured sandbox."""
    if failure := _validate_argv(argv):
        raise ToolException(failure.message)
    try:
        result = await sandbox.run(argv, cwd=workspace.root)
    except ContainmentError as error:
        raise ToolException(
            "Command containment failed; processes may still be running in the "
            "sandbox. Command execution is disabled until the service restarts."
        ) from error
    except WorkspaceError as error:
        raise ToolException(
            "Command could not run: the workspace is unavailable."
        ) from error
    except ValueError as error:
        raise ToolException(
            "Command could not run: invalid sandbox configuration or workspace."
        ) from error
    except OSError as error:
        raise ToolException(
            f"Command could not be started: {type(error).__name__}"
        ) from error
    return _format_result(result)


def make_command_tool(workspace: Workspace, sandbox: Sandbox) -> BaseTool:
    """Create the approval-gated command tool for one sandboxed workspace."""

    async def command(argv: list[str]) -> str:
        return await run_command(workspace, sandbox, argv)

    return StructuredTool.from_function(
        coroutine=command,
        name="run_command",
        description=DESCRIPTION,
        args_schema=RunCommandArgs,
        metadata={
            REQUIRES_APPROVAL: True,
            APPROVAL_PREFLIGHT: _command_preflight(),
        },
        handle_tool_error=True,
    )
