import io
import os
from itertools import islice

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field

from app.agent.tools.tool_utils.files import open_for_tool
from app.agent.tools.tool_utils.output import (
    BINARY_SNIFF_BYTES,
    MAX_BYTES,
    MAX_LINES,
    format_size,
    iter_lines,
    take_head,
    with_notices,
)
from app.agent.workspace import Workspace

DESCRIPTION = (
    "Read the contents of a text file. Output is truncated to "
    f"{MAX_LINES} lines or {MAX_BYTES // 1024}KB (whichever is hit first). "
    "Use offset/limit for large files. When you need the full file, continue "
    "with offset until complete."
)


class ReadArgs(BaseModel):
    path: str = Field(description="Path to the file, relative to the project root")
    offset: int | None = Field(
        default=None,
        ge=1,
        description="Line number to start reading from (1-indexed)",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of lines to read",
    )


def read_file(
    workspace: Workspace,
    path: str,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    start = offset or 1
    with open_for_tool(workspace, path, tool="read") as raw:
        size = os.fstat(raw.fileno()).st_size
        if b"\0" in raw.read(BINARY_SNIFF_BYTES):
            raise ToolException(f"Binary file, not shown: {path}")
        raw.seek(0)
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        # A line one character past the byte cap is already too long to show.
        lines = iter_lines(text, max_chars=MAX_BYTES + 1)
        skipped = sum(1 for _ in islice(lines, start - 1))
        head = take_head(lines if limit is None else islice(lines, limit))
        # Reading stops here: a hit cap already consumed one more line, and
        # otherwise one peek says whether the file goes on. Counting every
        # line for a total would read the whole file.
        more = head.truncated_by is not None or next(lines, None) is not None

    if not head.lines and head.truncated_by is None:
        # Nothing at the offset, so the skip reached the end and counted it all.
        if offset is not None:
            raise ToolException(
                f"Offset {offset} is beyond end of file ({skipped} lines total)"
            )
        return "(empty file)"
    if head.first_line_too_long:
        return (
            f"[Line {start} is over the {format_size(MAX_BYTES)} limit and can't "
            "be shown. Use grep to search within it.]"
        )

    end = start + len(head.lines) - 1
    notices: list[str] = []
    if head.truncated_by == "lines":
        notices.append(
            f"Showing lines {start}-{end} of a {format_size(size)} file. "
            f"Use offset={end + 1} to continue."
        )
    elif head.truncated_by == "bytes":
        notices.append(
            f"Showing lines {start}-{end} ({format_size(MAX_BYTES)} limit) of a "
            f"{format_size(size)} file. Use offset={end + 1} to continue."
        )
    elif more:
        notices.append(f"More lines in file. Use offset={end + 1} to continue.")
    return with_notices("\n".join(head.lines), notices)


def make_read_tool(workspace: Workspace) -> BaseTool:
    def read(path: str, offset: int | None = None, limit: int | None = None) -> str:
        return read_file(workspace, path, offset, limit)

    return StructuredTool.from_function(
        func=read,
        name="read",
        description=DESCRIPTION,
        args_schema=ReadArgs,
        handle_tool_error=True,
    )
