import sys
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field

from app.agent.approval import REQUIRES_APPROVAL
from app.agent.tools.files import resolve_for_tool, scan_dir
from app.agent.tools.output import MAX_BYTES, format_size, take_head, with_notices
from app.agent.workspace import Workspace, WorkspaceError

DEFAULT_LIMIT = 500

DESCRIPTION = (
    "List directory contents. Returns entries sorted alphabetically, with '/' "
    "suffix for directories. Includes dotfiles. Output is truncated to "
    f"{DEFAULT_LIMIT} entries or {MAX_BYTES // 1024}KB (whichever is hit first)."
)


class LsArgs(BaseModel):
    path: str = Field(
        default=".",
        description="Directory to list, relative to the project root (default: root)",
    )
    limit: int = Field(
        default=DEFAULT_LIMIT,
        ge=1,
        description=f"Maximum number of entries to return (default: {DEFAULT_LIMIT})",
    )


def list_dir(workspace: Workspace, path: str = ".", limit: int = DEFAULT_LIMIT) -> str:
    directory = resolve_for_tool(workspace, path, tool="ls")
    if not directory.is_dir():
        raise ToolException(f"Not a directory: {path}")

    names: list[str] = []
    limit_reached = False
    for entry in scan_dir(directory, path):
        # Denied entries and symlinks leaving the workspace aren't listed.
        try:
            target = workspace.resolve(Path(entry.path))
        except WorkspaceError:
            continue
        if len(names) == limit:
            limit_reached = True
            break
        names.append(entry.name + ("/" if target.is_dir() else ""))

    if not names:
        return "(empty directory)"

    head = take_head(names, max_lines=sys.maxsize)
    notices: list[str] = []
    if limit_reached:
        notices.append(f"{limit} entries limit reached. Use limit={limit * 2} for more")
    if head.truncated_by:
        notices.append(f"{format_size(MAX_BYTES)} limit reached")
    return with_notices("\n".join(head.lines), notices)


def make_ls_tool(workspace: Workspace) -> BaseTool:
    def ls(path: str = ".", limit: int = DEFAULT_LIMIT) -> str:
        return list_dir(workspace, path, limit)

    return StructuredTool.from_function(
        func=ls,
        name="ls",
        description=DESCRIPTION,
        args_schema=LsArgs,
        handle_tool_error=True,
        metadata={REQUIRES_APPROVAL: True},
    )
