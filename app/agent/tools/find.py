import sys
from pathlib import PurePosixPath

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field

from app.agent.tools.files import (
    display_path,
    glob_matcher,
    iter_files,
    resolve_for_tool,
)
from app.agent.tools.output import MAX_BYTES, format_size, take_head, with_notices
from app.agent.workspace import Workspace

DEFAULT_LIMIT = 1000

DESCRIPTION = (
    "Search for files by glob pattern. Returns matching file paths relative to "
    "the project root. Respects .gitignore. Output is truncated to "
    f"{DEFAULT_LIMIT} results or {MAX_BYTES // 1024}KB (whichever is hit first)."
)


class FindArgs(BaseModel):
    pattern: str = Field(
        min_length=1,
        description=(
            "Glob pattern to match files, e.g. '*.py', '**/*.json', or "
            "'src/**/test_*.py'"
        ),
    )
    path: str = Field(
        default=".",
        description="Directory to search in, relative to the project root (default: root)",
    )
    limit: int = Field(
        default=DEFAULT_LIMIT,
        ge=1,
        description=f"Maximum number of results (default: {DEFAULT_LIMIT})",
    )


def find_files(
    workspace: Workspace,
    pattern: str,
    path: str = ".",
    limit: int = DEFAULT_LIMIT,
) -> str:
    start = resolve_for_tool(workspace, path, tool="find")
    if not start.is_dir():
        raise ToolException(f"Not a directory: {path}")

    matches = glob_matcher(pattern)
    results: list[str] = []
    limit_reached = False
    for file in iter_files(workspace, start):
        if not matches(PurePosixPath(file.relative_to(start).as_posix())):
            continue
        # One result past the limit tells "exactly N" apart from "more than N".
        if len(results) == limit:
            limit_reached = True
            break
        results.append(display_path(workspace, file))

    if not results:
        return "No files found matching pattern"

    head = take_head(results, max_lines=sys.maxsize)
    notices: list[str] = []
    if limit_reached:
        notices.append(f"{limit} results limit reached")
    if head.truncated_by:
        notices.append(f"{format_size(MAX_BYTES)} limit reached")
    return with_notices("\n".join(head.lines), notices)


def make_find_tool(workspace: Workspace) -> BaseTool:
    def find(pattern: str, path: str = ".", limit: int = DEFAULT_LIMIT) -> str:
        return find_files(workspace, pattern, path, limit)

    return StructuredTool.from_function(
        func=find,
        name="find",
        description=DESCRIPTION,
        args_schema=FindArgs,
        handle_tool_error=True,
    )
