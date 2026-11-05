import asyncio
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from app.agent.approval import approval_tool_names
from app.agent.tools import make_read_tools, read
from app.agent.tools.output import format_size, iter_lines
from app.agent.workspace import Workspace


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def tools(root: Path) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in make_read_tools(Workspace(root))}


def test_no_read_tool_needs_approval(tools: dict[str, BaseTool]) -> None:
    # Only tools with side effects pause the run.
    assert approval_tool_names(list(tools.values())) == frozenset()


def call(tool: BaseTool, **args: object) -> ToolMessage:
    """Invoke a tool the way ToolNode does, so errors come back as results."""
    result = tool.invoke(
        {"type": "tool_call", "name": tool.name, "args": args, "id": "call-1"}
    )
    assert isinstance(result, ToolMessage)
    return result


def write(root: Path, path: str, content: str | bytes) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        target.write_bytes(content)
    else:
        target.write_text(content, encoding="utf-8")


def numbered(count: int) -> str:
    return "".join(f"line {number}\n" for number in range(1, count + 1))


# read


def test_read_returns_the_file_text(tools: dict[str, BaseTool], root: Path) -> None:
    write(root, "src/app.py", "import os\nprint('hi')\n")

    result = call(tools["read"], path="src/app.py")

    assert result.status == "success"
    assert result.content == "import os\nprint('hi')"


def test_read_offset_and_limit_point_to_the_rest(
    tools: dict[str, BaseTool], root: Path
) -> None:
    write(root, "notes.txt", numbered(10))

    result = call(tools["read"], path="notes.txt", offset=3, limit=2)

    assert result.content == (
        "line 3\nline 4\n\n[More lines in file. Use offset=5 to continue.]"
    )


def test_read_stops_at_the_line_cap(tools: dict[str, BaseTool], root: Path) -> None:
    write(root, "long.txt", numbered(2500))

    result = call(tools["read"], path="long.txt")

    text, notice = str(result.content).split("\n\n")
    size = format_size((root / "long.txt").stat().st_size)
    assert text.splitlines()[-1] == "line 2000"
    assert notice == (
        f"[Showing lines 1-2000 of a {size} file. Use offset=2001 to continue.]"
    )


def test_read_stops_at_the_byte_cap(tools: dict[str, BaseTool], root: Path) -> None:
    write(root, "wide.txt", ("x" * 999 + "\n") * 100)

    result = call(tools["read"], path="wide.txt")

    # 51 lines of 999 chars plus 50 newlines fit in 50KB; 52 don't.
    text, notice = str(result.content).split("\n\n")
    assert len(text.splitlines()) == 51
    assert notice == (
        "[Showing lines 1-51 (50.0KB limit) of a 97.7KB file. "
        "Use offset=52 to continue.]"
    )


@pytest.mark.parametrize(("args", "shown"), [({}, 2000), ({"limit": 5}, 5)])
def test_read_stops_reading_once_the_output_is_full(
    tools: dict[str, BaseTool],
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: dict[str, int],
    shown: int,
) -> None:
    write(root, "huge.log", numbered(10_000))
    yielded = 0

    def counting_lines(stream: TextIO, *, max_chars: int) -> Iterator[str]:
        nonlocal yielded
        for line in iter_lines(stream, max_chars=max_chars):
            yielded += 1
            yield line

    monkeypatch.setattr(read, "iter_lines", counting_lines)
    call(tools["read"], path="huge.log", **args)

    # The lines shown plus one more, to know the file goes on.
    assert yielded == shown + 1


def test_read_reports_a_line_too_long_to_show(
    tools: dict[str, BaseTool], root: Path
) -> None:
    write(root, "min.js", "x" * 60_000 + "\nshort\n")

    result = call(tools["read"], path="min.js")

    assert str(result.content).startswith("[Line 1 is over the 50.0KB limit")
    assert call(tools["read"], path="min.js", offset=2).content == "short"


def test_read_rejects_an_offset_past_the_end(
    tools: dict[str, BaseTool], root: Path
) -> None:
    write(root, "short.txt", numbered(3))

    result = call(tools["read"], path="short.txt", offset=5)

    assert result.status == "error"
    assert result.content == "Offset 5 is beyond end of file (3 lines total)"


def test_read_marks_an_empty_file(tools: dict[str, BaseTool], root: Path) -> None:
    write(root, "empty.txt", "")

    assert call(tools["read"], path="empty.txt").content == "(empty file)"


def test_read_refuses_binary_files(tools: dict[str, BaseTool], root: Path) -> None:
    write(root, "image.png", b"\x89PNG\x00\x01")

    result = call(tools["read"], path="image.png")

    assert result.status == "error"
    assert result.content == "Binary file, not shown: image.png"


@pytest.mark.parametrize(
    ("path", "message", "reason"),
    [
        (".env", "Access denied: .env is blocked by workspace policy", "denied"),
        (
            "../outside.txt",
            "Access denied: ../outside.txt is outside the workspace",
            "outside",
        ),
    ],
)
def test_read_rejects_denied_and_outside_paths(
    tools: dict[str, BaseTool],
    root: Path,
    caplog: pytest.LogCaptureFixture,
    path: str,
    message: str,
    reason: str,
) -> None:
    write(root, ".env", "SECRET=1")
    write(root.parent, "outside.txt", "SECRET=2")

    with caplog.at_level(logging.WARNING, logger="app.agent.tools"):
        result = call(tools["read"], path=path)

    assert result.status == "error"
    assert result.content == message
    rejected = next(r for r in caplog.records if r.message == "Tool path rejected")
    assert rejected.__dict__["tool"] == "read"
    assert rejected.__dict__["reason"] == reason


def test_read_reports_a_missing_file(tools: dict[str, BaseTool]) -> None:
    result = call(tools["read"], path="nope.txt")

    assert result.status == "error"
    assert result.content == "Path not found: nope.txt"


@pytest.mark.anyio
async def test_read_refuses_directories_and_fifos_without_blocking(
    tools: dict[str, BaseTool], root: Path
) -> None:
    (root / "src").mkdir()
    os.mkfifo(root / "pipe")

    for path, message in [
        ("src", "Cannot read src: it is a directory, use ls"),
        ("pipe", "Cannot read pipe: not a regular file"),
    ]:
        result = await asyncio.wait_for(
            asyncio.to_thread(call, tools["read"], path=path), timeout=5
        )
        assert result.status == "error"
        assert result.content == message


# ls


def test_ls_sorts_entries_and_marks_directories(
    tools: dict[str, BaseTool], root: Path
) -> None:
    write(root, "b.txt", "")
    write(root, "A.txt", "")
    write(root, ".hidden", "")
    write(root, "src/app.py", "")

    assert call(tools["ls"]).content == ".hidden\nA.txt\nb.txt\nsrc/"


def test_ls_hides_denied_entries_and_escaping_symlinks(
    tools: dict[str, BaseTool], root: Path
) -> None:
    write(root, ".env", "SECRET=1")
    write(root, ".git/config", "")
    write(root, ".claude/settings.local.json", "{}")
    write(root, "app.py", "")
    (root.parent / "outside").mkdir()
    (root / "escape").symlink_to(root.parent / "outside")

    assert call(tools["ls"]).content == "app.py"


def test_ls_says_when_the_entry_limit_cut_results(
    tools: dict[str, BaseTool], root: Path
) -> None:
    for name in ("a", "b", "c"):
        write(root, f"full/{name}", "")
        if name != "c":
            write(root, f"exact/{name}", "")

    assert call(tools["ls"], path="full", limit=2).content == (
        "a\nb\n\n[2 entries limit reached. Use limit=4 for more]"
    )
    assert call(tools["ls"], path="exact", limit=2).content == "a\nb"


def test_ls_reports_empty_and_non_directories(
    tools: dict[str, BaseTool], root: Path
) -> None:
    (root / "empty").mkdir()
    write(root, "file.txt", "")

    assert call(tools["ls"], path="empty").content == "(empty directory)"
    not_a_directory = call(tools["ls"], path="file.txt")
    assert not_a_directory.status == "error"
    assert not_a_directory.content == "Not a directory: file.txt"


# find


def test_find_matches_names_at_any_depth(
    tools: dict[str, BaseTool], root: Path
) -> None:
    write(root, "setup.py", "")
    write(root, "src/pkg/app.py", "")
    write(root, "src/pkg/app.txt", "")

    assert call(tools["find"], pattern="*.py").content == "setup.py\nsrc/pkg/app.py"
    # Results stay relative to the root when searching a subdirectory.
    assert call(tools["find"], pattern="*.py", path="src").content == "src/pkg/app.py"


def test_find_patterns_with_a_slash_match_paths(
    tools: dict[str, BaseTool], root: Path
) -> None:
    write(root, "src/test_app.py", "")
    write(root, "lib/src/test_lib.py", "")

    assert call(tools["find"], pattern="src/test_*.py").content == (
        "lib/src/test_lib.py\nsrc/test_app.py"
    )
    assert call(tools["find"], pattern="/src/test_*.py").content == "src/test_app.py"


def test_find_skips_denied_and_gitignored_files(
    tools: dict[str, BaseTool], root: Path
) -> None:
    write(root, ".gitignore", "build/\n*.log\n!keep.log\n")
    write(root, "sub/.gitignore", "local.txt\n")
    write(root, ".env.txt", "")
    write(root, "build/out.txt", "")
    write(root, "debug.log", "")
    write(root, "keep.log", "")
    write(root, "sub/local.txt", "")
    write(root, "sub/kept.txt", "")

    assert call(tools["find"], pattern="*.txt").content == "sub/kept.txt"
    assert call(tools["find"], pattern="*.log").content == "keep.log"
    # Naming an ignored directory searches it anyway.
    assert call(tools["find"], pattern="*", path="build").content == "build/out.txt"


def test_find_says_when_the_result_limit_cut_results(
    tools: dict[str, BaseTool], root: Path
) -> None:
    for name in ("a", "b", "c"):
        write(root, f"{name}.py", "")

    assert call(tools["find"], pattern="*.py", limit=2).content == (
        "a.py\nb.py\n\n[2 results limit reached]"
    )
    assert call(tools["find"], pattern="*.py", limit=3).content == "a.py\nb.py\nc.py"
    assert call(tools["find"], pattern="*.rs").content == (
        "No files found matching pattern"
    )


# grep


def test_grep_shows_matches_with_paths_and_line_numbers(
    tools: dict[str, BaseTool], root: Path
) -> None:
    write(root, "src/app.py", "import os\nprint('hi')\n")
    write(root, "README.md", "Say hi\n")

    assert call(tools["grep"], pattern="hi").content == (
        "README.md:1: Say hi\nsrc/app.py:2: print('hi')"
    )


def test_grep_shows_context_lines(tools: dict[str, BaseTool], root: Path) -> None:
    write(root, "app.py", "one\ntwo\nthree\nfour\nfive\n")

    assert call(tools["grep"], pattern="three", context=1).content == (
        "app.py-2- two\napp.py:3: three\napp.py-4- four"
    )


def test_grep_options_filter_matches(tools: dict[str, BaseTool], root: Path) -> None:
    write(root, "app.py", "Hello\naxb\n")
    write(root, "notes.md", "hello\n")

    assert call(
        tools["grep"], pattern="hello", ignore_case=True, glob="*.py"
    ).content == ("app.py:1: Hello")
    assert (
        call(tools["grep"], pattern="a.b", literal=True).content == "No matches found"
    )
    assert call(tools["grep"], pattern="hello", path="notes.md").content == (
        "notes.md:1: hello"
    )


def test_grep_says_when_the_match_limit_cut_results(
    tools: dict[str, BaseTool], root: Path
) -> None:
    write(root, "app.py", "hit 1\nhit 2\nhit 3\n")

    assert call(tools["grep"], pattern="hit", limit=2).content == (
        "app.py:1: hit 1\napp.py:2: hit 2\n\n"
        "[2 matches limit reached. Use limit=4 for more, or refine pattern]"
    )
    assert "limit reached" not in str(
        call(tools["grep"], pattern="hit", limit=3).content
    )


def test_grep_truncates_long_lines(tools: dict[str, BaseTool], root: Path) -> None:
    write(root, "min.js", "hit" + "x" * 600 + "\n")

    content = str(call(tools["grep"], pattern="hit").content)

    assert content.startswith("min.js:1: hit" + "x" * 497 + "... [truncated]")
    assert content.endswith(
        "[Some lines truncated to 500 chars. Use read tool to see full lines]"
    )


def test_grep_never_searches_denied_ignored_binary_or_outside_files(
    tools: dict[str, BaseTool], root: Path
) -> None:
    write(root, ".env", "SECRET=1\n")
    write(root, "nested/.env.local", "SECRET=2\n")
    write(root, ".git/config", "SECRET=3\n")
    write(root, ".claude/settings.local.json", "SECRET=7\n")
    write(root, ".gitignore", "ignored.txt\n")
    write(root, "ignored.txt", "SECRET=4\n")
    write(root, "blob.bin", b"SECRET=5\x00")
    write(root.parent, "outside.txt", "SECRET=6\n")
    (root / "link.txt").symlink_to(root.parent / "outside.txt")

    assert call(tools["grep"], pattern="SECRET").content == "No matches found"


def test_grep_reports_an_invalid_pattern(tools: dict[str, BaseTool]) -> None:
    result = call(tools["grep"], pattern="(")

    assert result.status == "error"
    assert str(result.content).startswith("Invalid regular expression:")
