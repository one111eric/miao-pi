"""Approval-gated tools that change regular files in the workspace."""

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field

from app.agent.approval import (
    APPROVAL_PREFLIGHT,
    REQUIRES_APPROVAL,
    PreflightFailure,
)
from app.agent.workspace import (
    Workspace,
    WorkspaceAccessError,
    WorkspaceError,
    WorkspaceTransactionError,
)

logger = logging.getLogger(__name__)

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
MAX_WRITE_CHARS = 1_000_000


class WriteFileArgs(BaseModel):
    path: str = Field(min_length=1, description="Path relative to the project root")
    content: str = Field(
        max_length=MAX_WRITE_CHARS,
        description="Complete replacement text for the file (at most 1,000,000 characters)",
    )


class ApplyPatchArgs(BaseModel):
    patch: str = Field(
        min_length=1,
        max_length=MAX_WRITE_CHARS,
        description=(
            "Unified diff using --- and +++ file headers and @@ hunk headers. "
            "Paths must be relative to the project root (at most 1,000,000 characters)."
        ),
    )


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class FilePatch:
    old_path: str | None
    new_path: str | None
    hunks: tuple[Hunk, ...]


class PatchError(ValueError):
    """The supplied patch is malformed or doesn't match its base file."""


type ApprovalPreflight = Callable[[dict[str, Any]], PreflightFailure | None]


def _access_error(
    path: str,
    error: WorkspaceAccessError,
    *,
    tool: str,
    log_rejection: bool = True,
) -> ToolException:
    if log_rejection:
        logger.warning(
            "Tool path rejected",
            extra={"tool": tool, "tool_path": path, "reason": error.reason},
        )
    if error.reason == "outside":
        return ToolException(f"Access denied: {path} is outside the workspace")
    return ToolException(f"Access denied: {path} is blocked by workspace policy")


def _resolve(
    workspace: Workspace,
    path: str,
    *,
    tool: str,
    log_rejection: bool = True,
) -> Path:
    try:
        return workspace.resolve_write_target(path)
    except WorkspaceAccessError as error:
        raise _access_error(
            path, error, tool=tool, log_rejection=log_rejection
        ) from error
    except WorkspaceError as error:
        raise ToolException(f"Cannot write {path}: {error}") from error


def _write_error(path: str, error: OSError) -> ToolException:
    if isinstance(error, FileNotFoundError):
        return ToolException(f"Parent directory not found for {path}")
    if isinstance(error, NotADirectoryError):
        return ToolException(f"Parent of {path} is not a directory")
    if isinstance(error, IsADirectoryError):
        return ToolException(f"Cannot write {path}: it is a directory")
    if isinstance(error, PermissionError):
        return ToolException(f"Permission denied: {path}")
    return ToolException(
        f"Cannot write {path}: {error.strerror or type(error).__name__}"
    )


def write_file(workspace: Workspace, path: str, content: str) -> str:
    """Atomically replace ``path`` with ``content`` after boundary checks."""
    if "\0" in content:
        raise ToolException("Cannot write text containing a NUL byte")
    _resolve(workspace, path, tool="write_file")
    try:
        workspace.write_text_atomic(path, content)
    except WorkspaceAccessError as error:
        raise _access_error(path, error, tool="write_file") from error
    except WorkspaceError as error:
        raise ToolException(f"Cannot write {path}: {error}") from error
    except OSError as error:
        raise _write_error(path, error) from error
    return f"Wrote {path}."


def _header_path(line: str, marker: str) -> str | None:
    value = line[len(marker) :].rstrip("\r\n").split("\t", 1)[0]
    if value == "/dev/null":
        return None
    if not value or value.startswith("/") or value.startswith('"'):
        raise PatchError("patch paths must be relative and unquoted")
    if value.startswith(("a/", "b/")):
        value = value[2:]
    if not value:
        raise PatchError("patch path is empty")
    return value


def parse_patch(patch: str) -> tuple[FilePatch, ...]:
    """Parse a conservative, text-only subset of unified diff."""
    if "\0" in patch:
        raise PatchError("patch text contains a NUL byte")
    lines = patch.splitlines(keepends=True)
    position = 0
    files: list[FilePatch] = []
    while position < len(lines):
        if not lines[position].startswith("--- "):
            raise PatchError("expected a --- file header")
        old_path = _header_path(lines[position], "--- ")
        position += 1
        if position == len(lines) or not lines[position].startswith("+++ "):
            raise PatchError("expected a +++ file header")
        new_path = _header_path(lines[position], "+++ ")
        position += 1
        if old_path is None and new_path is None:
            raise PatchError("a patch must name an old or new file")
        if old_path is not None and new_path is not None and old_path != new_path:
            raise PatchError("renaming files is not supported")

        hunks: list[Hunk] = []
        while position < len(lines) and not lines[position].startswith("--- "):
            match = _HUNK.match(lines[position])
            if match is None:
                raise PatchError("expected a @@ hunk header")
            old_start, old_count, new_start, new_count = match.groups()
            position += 1
            body: list[str] = []
            while position < len(lines) and not lines[position].startswith(
                ("@@ ", "--- ")
            ):
                line = lines[position]
                if line.startswith("\\"):
                    raise PatchError(
                        "patches without a final newline are not supported"
                    )
                if not line.startswith((" ", "+", "-")):
                    raise PatchError("invalid hunk line")
                body.append(line)
                position += 1
            old_lines = sum(line[0] in " -" for line in body)
            new_lines = sum(line[0] in " +" for line in body)
            expected_old = int(old_count or "1")
            expected_new = int(new_count or "1")
            if old_lines != expected_old or new_lines != expected_new:
                raise PatchError("hunk line counts do not match its header")
            hunks.append(
                Hunk(
                    int(old_start),
                    expected_old,
                    int(new_start),
                    expected_new,
                    tuple(body),
                )
            )
        if not hunks:
            raise PatchError("a file patch needs at least one hunk")
        files.append(FilePatch(old_path, new_path, tuple(hunks)))
    return tuple(files)


def _apply_hunks(content: str, patch: FilePatch) -> str:
    source = content.splitlines(keepends=True)
    result: list[str] = []
    source_position = 0
    for hunk in patch.hunks:
        start = hunk.old_start - 1
        if hunk.old_count == 0:
            start += 1
        if start < source_position or start > len(source):
            raise PatchError("patch hunks overlap or point beyond the file")
        result.extend(source[source_position:start])
        source_position = start
        for line in hunk.lines:
            value = line[1:]
            if line[0] == " ":
                if source_position >= len(source) or source[source_position] != value:
                    raise PatchError("patch does not apply: context changed")
                result.append(source[source_position])
                source_position += 1
            elif line[0] == "-":
                if source_position >= len(source) or source[source_position] != value:
                    raise PatchError("patch does not apply: removed text changed")
                source_position += 1
            else:
                result.append(value)
    result.extend(source[source_position:])
    return "".join(result)


def _read_source(workspace: Workspace, path: str) -> str:
    try:
        with workspace.open_file(path) as file:
            content = file.read()
            if b"\0" in content:
                raise ToolException(f"Binary file, cannot patch: {path}")
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ToolException(f"Binary file, cannot patch: {path}") from error
    except WorkspaceAccessError as error:
        raise _access_error(path, error, tool="apply_patch") from error
    except WorkspaceError as error:
        raise ToolException(f"Cannot patch {path}: {error}") from error
    except OSError as error:
        raise _write_error(path, error) from error


def apply_patch(workspace: Workspace, patch: str) -> str:
    """Apply a unified text diff after validating every source file first."""
    try:
        files = parse_patch(patch)
    except PatchError as error:
        raise ToolException(f"Invalid patch: {error}") from error

    changes: list[tuple[str, str | None]] = []
    for file_patch in files:
        source_path = file_patch.old_path
        target_path = file_patch.new_path
        if source_path is None:
            assert target_path is not None
            target = _resolve(workspace, target_path, tool="apply_patch")
            if target.exists():
                raise ToolException(
                    f"Patch does not apply: {target_path} already exists"
                )
            source = ""
        else:
            source_target = _resolve(workspace, source_path, tool="apply_patch")
            if not source_target.exists():
                raise ToolException(
                    f"Patch does not apply: {source_path} does not exist"
                )
            source = _read_source(workspace, source_path)
        if target_path is not None:
            _resolve(workspace, target_path, tool="apply_patch")
        try:
            changed_path = target_path if target_path is not None else source_path
            assert changed_path is not None
            patched = _apply_hunks(source, file_patch)
            replacement = patched if target_path is not None else None
            if target_path is None or patched != source:
                changes.append((changed_path, replacement))
        except PatchError as error:
            path = source_path or target_path
            assert path is not None
            raise ToolException(f"Patch does not apply to {path}: {error}") from error

    if not changes:
        return "Patch made no changes."

    try:
        workspace.write_text_transaction(changes)
    except WorkspaceAccessError as error:
        path = next(path for path, _ in changes)
        raise _access_error(path, error, tool="apply_patch") from error
    except WorkspaceTransactionError as error:
        path = error.path.relative_to(workspace.root).as_posix()
        raise _write_error(path, error.error) from error
    except WorkspaceError as error:
        raise ToolException(f"Patch could not be applied: {error}") from error
    except OSError as error:
        path = next(path for path, _ in changes)
        raise _write_error(path, error) from error
    changed_paths = [path for path, _ in changes]
    return f"Applied patch to {', '.join(changed_paths)}."


def _preflight_path(
    workspace: Workspace, path: object, *, tool: str
) -> PreflightFailure | None:
    if not isinstance(path, str) or not path:
        return PreflightFailure(
            "Invalid write arguments: path must be a non-empty string."
        )
    try:
        workspace.resolve_write_target(path)
    except WorkspaceAccessError as error:
        message = (
            f"Access denied: {path} is outside the workspace"
            if error.reason == "outside"
            else f"Access denied: {path} is blocked by workspace policy"
        )
        return PreflightFailure(message, path, error.reason)
    except WorkspaceError as error:
        return PreflightFailure(f"Cannot write {path}: {error}", path)
    return None


def _write_preflight(workspace: Workspace) -> ApprovalPreflight:
    def check(args: dict[str, Any]) -> PreflightFailure | None:
        path_error = _preflight_path(workspace, args.get("path"), tool="write_file")
        if path_error:
            return path_error
        content = args.get("content")
        if not isinstance(content, str):
            return PreflightFailure(
                "Invalid write arguments: content must be a string."
            )
        if "\0" in content:
            return PreflightFailure("Cannot write text containing a NUL byte")
        if len(content) > MAX_WRITE_CHARS:
            return PreflightFailure(
                f"Invalid write arguments: content exceeds {MAX_WRITE_CHARS} characters."
            )
        return None

    return check


def _patch_preflight(workspace: Workspace) -> ApprovalPreflight:
    def check(args: dict[str, Any]) -> PreflightFailure | None:
        patch = args.get("patch")
        if not isinstance(patch, str) or not patch:
            return PreflightFailure(
                "Invalid patch arguments: patch must be a non-empty string."
            )
        if len(patch) > MAX_WRITE_CHARS:
            return PreflightFailure(
                f"Invalid patch arguments: patch exceeds {MAX_WRITE_CHARS} characters."
            )
        try:
            files = parse_patch(patch)
        except PatchError as error:
            return PreflightFailure(f"Invalid patch: {error}")
        for file_patch in files:
            if file_patch.old_path is not None:
                source_error = _preflight_path(
                    workspace, file_patch.old_path, tool="apply_patch"
                )
                if source_error:
                    return source_error
                source = _resolve(
                    workspace,
                    file_patch.old_path,
                    tool="apply_patch",
                    log_rejection=False,
                )
                if not source.exists():
                    return PreflightFailure(
                        f"Patch does not apply: {file_patch.old_path} does not exist",
                        file_patch.old_path,
                    )
            if file_patch.new_path is not None:
                target_error = _preflight_path(
                    workspace, file_patch.new_path, tool="apply_patch"
                )
                if target_error:
                    return target_error
                if file_patch.old_path is None:
                    target = _resolve(
                        workspace,
                        file_patch.new_path,
                        tool="apply_patch",
                        log_rejection=False,
                    )
                    if target.exists():
                        return PreflightFailure(
                            f"Patch does not apply: {file_patch.new_path} already exists",
                            file_patch.new_path,
                        )
        return None

    return check


def make_write_tools(workspace: Workspace) -> list[BaseTool]:
    """Create the workspace-mutating tools, both gated by human approval."""

    def write(path: str, content: str) -> str:
        return write_file(workspace, path, content)

    def patch(patch: str) -> str:
        return apply_patch(workspace, patch)

    return [
        StructuredTool.from_function(
            func=write,
            name="write_file",
            description="Replace a text file atomically. Human approval is required.",
            args_schema=WriteFileArgs,
            metadata={
                REQUIRES_APPROVAL: True,
                APPROVAL_PREFLIGHT: _write_preflight(workspace),
            },
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=patch,
            name="apply_patch",
            description="Apply a unified text diff. Human approval is required.",
            args_schema=ApplyPatchArgs,
            metadata={
                REQUIRES_APPROVAL: True,
                APPROVAL_PREFLIGHT: _patch_preflight(workspace),
            },
            handle_tool_error=True,
        ),
    ]
