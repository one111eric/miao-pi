import errno
import os
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from app.agent.approval import approval_tool_names
from app.agent.tools.write import make_write_tools
from app.agent.workspace import Workspace


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def tools(root: Path) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in make_write_tools(Workspace(root))}


def call(tool: BaseTool, **args: object) -> ToolMessage:
    result = tool.invoke(
        {"type": "tool_call", "name": tool.name, "args": args, "id": "call-1"}
    )
    assert isinstance(result, ToolMessage)
    return result


def test_write_tools_require_approval(tools: dict[str, BaseTool]) -> None:
    assert approval_tool_names(list(tools.values())) == {"write_file", "apply_patch"}


def test_write_file_creates_and_replaces_a_file(
    tools: dict[str, BaseTool], root: Path
) -> None:
    (root / "notes.txt").write_text("old", encoding="utf-8")

    result = call(tools["write_file"], path="notes.txt", content="new\n")

    assert result.status == "success"
    assert result.content == "Wrote notes.txt."
    assert (root / "notes.txt").read_text(encoding="utf-8") == "new\n"


def test_write_file_refuses_a_nul_byte(tools: dict[str, BaseTool]) -> None:
    result = call(tools["write_file"], path="notes.txt", content="bad\0text")

    assert result.status == "error"
    assert result.content == "Cannot write text containing a NUL byte"


def test_write_tools_reject_a_leaf_symlink(
    tools: dict[str, BaseTool], root: Path
) -> None:
    (root / "target.txt").write_text("old\n", encoding="utf-8")
    (root / "alias.txt").symlink_to("target.txt")

    write = call(tools["write_file"], path="alias.txt", content="changed\n")
    patch = call(
        tools["apply_patch"],
        patch="""--- a/alias.txt
+++ b/alias.txt
@@ -1 +1 @@
-old
+changed
""",
    )

    assert write.status == "error"
    assert patch.status == "error"
    assert (root / "target.txt").read_text(encoding="utf-8") == "old\n"
    assert (root / "alias.txt").is_symlink()


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (".env", "Access denied: .env is blocked by workspace policy"),
        ("../outside.txt", "Access denied: ../outside.txt is outside the workspace"),
    ],
)
def test_write_file_refuses_denied_and_outside_paths(
    tools: dict[str, BaseTool], root: Path, path: str, message: str
) -> None:
    (root / ".env").write_text("SECRET=1", encoding="utf-8")

    result = call(tools["write_file"], path=path, content="changed")

    assert result.status == "error"
    assert result.content == message
    assert (root / ".env").read_text(encoding="utf-8") == "SECRET=1"


def test_apply_patch_updates_adds_and_deletes_files(
    tools: dict[str, BaseTool], root: Path
) -> None:
    (root / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")
    (root / "remove.txt").write_text("bye\n", encoding="utf-8")
    patch = """--- a/notes.txt
+++ b/notes.txt
@@ -1,2 +1,2 @@
 one
-two
+changed
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+created
--- a/remove.txt
+++ /dev/null
@@ -1 +0,0 @@
-bye
"""

    result = call(tools["apply_patch"], patch=patch)

    assert result.status == "success"
    assert result.content == "Applied patch to notes.txt, new.txt, remove.txt."
    assert (root / "notes.txt").read_text(encoding="utf-8") == "one\nchanged\n"
    assert (root / "new.txt").read_text(encoding="utf-8") == "created\n"
    assert not (root / "remove.txt").exists()


def test_apply_patch_rejects_a_stale_base_without_changing_the_file(
    tools: dict[str, BaseTool], root: Path
) -> None:
    (root / "notes.txt").write_text("current\n", encoding="utf-8")
    patch = """--- a/notes.txt
+++ b/notes.txt
@@ -1 +1 @@
-old
+new
"""

    result = call(tools["apply_patch"], patch=patch)

    assert result.status == "error"
    assert "Patch does not apply to notes.txt" in str(result.content)
    assert (root / "notes.txt").read_text(encoding="utf-8") == "current\n"


def test_apply_patch_refuses_a_binary_source(
    tools: dict[str, BaseTool], root: Path
) -> None:
    (root / "data.bin").write_bytes(b"\xff")
    patch = """--- a/data.bin
+++ b/data.bin
@@ -1 +1 @@
-old
+new
"""

    result = call(tools["apply_patch"], patch=patch)

    assert result.status == "error"
    assert result.content == "Binary file, cannot patch: data.bin"


def test_apply_patch_rejects_a_missing_delete_target(
    tools: dict[str, BaseTool],
) -> None:
    patch = """--- a/gone.txt
+++ /dev/null
@@ -1 +0,0 @@
-gone
"""

    result = call(tools["apply_patch"], patch=patch)

    assert result.status == "error"
    assert result.content == "Patch does not apply: gone.txt does not exist"


def test_apply_patch_skips_an_unchanged_file(
    tools: dict[str, BaseTool], root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (root / "notes.txt").write_text("same\n", encoding="utf-8")
    patch = """--- a/notes.txt
+++ b/notes.txt
@@ -1 +1 @@
-same
+same
"""

    def transaction_was_not_needed(*args: object, **kwargs: object) -> None:
        raise AssertionError("a no-op patch should not write")

    monkeypatch.setattr(Workspace, "write_text_transaction", transaction_was_not_needed)

    result = call(tools["apply_patch"], patch=patch)

    assert result.status == "success"
    assert result.content == "Patch made no changes."


def test_apply_patch_rolls_back_when_a_later_write_fails(
    tools: dict[str, BaseTool], root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (root / "one.txt").write_text("one\n", encoding="utf-8")
    (root / "two.txt").write_text("two\n", encoding="utf-8")
    patch = """--- a/one.txt
+++ b/one.txt
@@ -1 +1 @@
-one
+first
--- a/two.txt
+++ b/two.txt
@@ -1 +1 @@
-two
+second
"""
    original_replace = os.replace
    failures = 0

    def fail_second_write(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal failures
        if source.startswith(".miao-pi-") and destination == "two.txt" and not failures:
            failures += 1
            raise OSError(errno.EACCES, "Permission denied")
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "replace", fail_second_write)

    result = call(tools["apply_patch"], patch=patch)

    assert result.status == "error"
    assert result.content == "Permission denied: two.txt"
    assert (root / "one.txt").read_text(encoding="utf-8") == "one\n"
    assert (root / "two.txt").read_text(encoding="utf-8") == "two\n"
