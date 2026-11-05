import io
import re
import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field

from app.agent.tools.tool_utils.files import (
    display_path,
    glob_matcher,
    iter_files,
    resolve_for_tool,
)
from app.agent.tools.tool_utils.output import (
    BINARY_SNIFF_BYTES,
    MAX_BYTES,
    MAX_LINE_CHARS,
    format_size,
    iter_lines,
    take_head,
    truncate_line,
    with_notices,
)
from app.agent.workspace import Workspace, WorkspaceError

DEFAULT_LIMIT = 100
# Only this much of each line is searched, so one huge line can't exhaust memory.
SCAN_LINE_CHARS = 1024 * 1024

DESCRIPTION = (
    "Search file contents for a pattern. Returns matching lines with file paths "
    "and line numbers. Respects .gitignore and skips binary files. Output is "
    f"truncated to {DEFAULT_LIMIT} matches or {MAX_BYTES // 1024}KB (whichever is "
    f"hit first). Long lines are truncated to {MAX_LINE_CHARS} chars."
)


class GrepArgs(BaseModel):
    pattern: str = Field(
        min_length=1,
        description="Search pattern (Python regular expression, or a literal string)",
    )
    path: str = Field(
        default=".",
        description=(
            "Directory or file to search, relative to the project root "
            "(default: root)"
        ),
    )
    glob: str | None = Field(
        default=None,
        description="Filter files by glob pattern, e.g. '*.py' or '**/test_*.py'",
    )
    ignore_case: bool = Field(
        default=False,
        description="Case-insensitive search (default: false)",
    )
    literal: bool = Field(
        default=False,
        description="Treat pattern as a literal string instead of a regex (default: false)",
    )
    context: int = Field(
        default=0,
        ge=0,
        description="Number of lines to show before and after each match (default: 0)",
    )
    limit: int = Field(
        default=DEFAULT_LIMIT,
        ge=1,
        description=f"Maximum number of matches to return (default: {DEFAULT_LIMIT})",
    )


@dataclass
class _Search:
    regex: re.Pattern[str]
    context: int
    limit: int
    lines: list[str] = field(default_factory=list)
    size: int = 0
    matches: int = 0
    limit_reached: bool = False
    lines_truncated: bool = False

    def _emit(self, label: str, number: int, text: str, *, is_match: bool) -> None:
        # Lines past the byte cap would be cut from the output anyway.
        if self.size > MAX_BYTES:
            return
        shown, truncated = truncate_line(text)
        self.lines_truncated |= truncated
        separator = ":" if is_match else "-"
        line = f"{label}{separator}{number}{separator} {shown}"
        self.lines.append(line)
        self.size += len(line.encode()) + 1

    def search(self, label: str, lines: Iterable[str]) -> None:
        before: deque[tuple[int, str]] = deque(maxlen=self.context)
        after = 0
        for number, line in enumerate(lines, start=1):
            if self.regex.search(line):
                # Only one more match proves the limit cut results off.
                if self.matches == self.limit:
                    self.limit_reached = True
                    return
                for previous_number, previous in before:
                    self._emit(label, previous_number, previous, is_match=False)
                before.clear()
                self._emit(label, number, line, is_match=True)
                self.matches += 1
                after = self.context
            elif after:
                self._emit(label, number, line, is_match=False)
                after -= 1
            elif self.matches < self.limit:
                before.append((number, line))


def _search_file(workspace: Workspace, search: _Search, file: Path) -> None:
    # Files that vanish or can't be opened while searching are skipped.
    try:
        with workspace.open_file(file) as raw:
            if b"\0" in raw.read(BINARY_SNIFF_BYTES):
                return
            raw.seek(0)
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            search.search(
                display_path(workspace, file),
                iter_lines(text, max_chars=SCAN_LINE_CHARS),
            )
    except (WorkspaceError, OSError):
        return


def grep_files(
    workspace: Workspace,
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    ignore_case: bool = False,
    literal: bool = False,
    context: int = 0,
    limit: int = DEFAULT_LIMIT,
) -> str:
    target = resolve_for_tool(workspace, path, tool="grep")
    try:
        regex = re.compile(
            re.escape(pattern) if literal else pattern,
            re.IGNORECASE if ignore_case else 0,
        )
    except re.error as error:
        raise ToolException(f"Invalid regular expression: {error}") from error

    if target.is_dir():
        base, files = target, iter_files(workspace, target)
    else:
        base, files = target.parent, iter([target])
    matches_glob = glob_matcher(glob) if glob else None

    search = _Search(regex, context, limit)
    for file in files:
        relative = PurePosixPath(file.relative_to(base).as_posix())
        if matches_glob and not matches_glob(relative):
            continue
        _search_file(workspace, search, file)
        if search.limit_reached:
            break

    if search.matches == 0:
        return "No matches found"

    head = take_head(search.lines, max_lines=sys.maxsize)
    notices: list[str] = []
    if search.limit_reached:
        notices.append(
            f"{limit} matches limit reached. Use limit={limit * 2} for more, "
            "or refine pattern"
        )
    if head.truncated_by or search.size > MAX_BYTES:
        notices.append(f"{format_size(MAX_BYTES)} limit reached")
    if search.lines_truncated:
        notices.append(
            f"Some lines truncated to {MAX_LINE_CHARS} chars. Use read tool to see "
            "full lines"
        )
    return with_notices("\n".join(head.lines), notices)


def make_grep_tool(workspace: Workspace) -> BaseTool:
    def grep(
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        ignore_case: bool = False,
        literal: bool = False,
        context: int = 0,
        limit: int = DEFAULT_LIMIT,
    ) -> str:
        return grep_files(
            workspace, pattern, path, glob, ignore_case, literal, context, limit
        )

    return StructuredTool.from_function(
        func=grep,
        name="grep",
        description=DESCRIPTION,
        args_schema=GrepArgs,
        handle_tool_error=True,
    )
