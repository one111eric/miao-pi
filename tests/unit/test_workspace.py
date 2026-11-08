import stat
from pathlib import Path

import pytest

from app.agent.workspace import (
    Workspace,
    WorkspaceAccessError,
    WorkspaceError,
    WorkspaceText,
)


def test_resolves_paths_inside_the_workspace(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = Workspace(root)

    assert workspace.resolve("src/main.py") == root / "src/main.py"
    assert workspace.resolve(root / "README.md") == root / "README.md"


@pytest.mark.parametrize("path", ["../outside.txt", "/tmp/outside.txt"])
def test_rejects_paths_outside_the_workspace(tmp_path: Path, path: str) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = Workspace(root)

    with pytest.raises(WorkspaceAccessError) as error:
        workspace.resolve(path)
    assert error.value.reason == "outside"


def test_rejects_symlinks_that_escape_the_workspace(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    workspace = Workspace(root)

    with pytest.raises(WorkspaceAccessError) as error:
        workspace.resolve("escape/secret.txt")
    assert error.value.reason == "outside"


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".ENV",
        ".env.local",
        ".ENV.PRODUCTION",
        ".envrc",
        ".env-prod",
        ".env_local",
        ".git",
        ".GIT/config",
        "nested/.env",
        "nested/.git/config",
        ".claude/settings.local.json",
        ".Claude",
    ],
)
def test_rejects_denied_paths(tmp_path: Path, path: str) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = Workspace(root)

    with pytest.raises(WorkspaceAccessError) as error:
        workspace.resolve(path)
    assert error.value.reason == "denied"


def test_rejects_a_symlink_alias_to_a_denied_path(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("secret", encoding="utf-8")
    (tmp_path / "alias").symlink_to(".env")
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspaceAccessError) as error:
        workspace.resolve("alias")
    assert error.value.reason == "denied"


def test_rejects_a_denied_name_that_links_to_an_allowed_file(tmp_path: Path) -> None:
    (tmp_path / "settings.txt").write_text("secret", encoding="utf-8")
    (tmp_path / ".env").symlink_to("settings.txt")
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspaceAccessError):
        workspace.resolve(".env")


@pytest.mark.parametrize(
    ("pattern", "path"),
    [
        ("*.pem", "certs/server.PEM"),
        ("config/private", "config/private"),
        ("config/private", "CONFIG/Private/key"),
        ("caf\u00e9", "CAF\u00c9"),
        ("caf\u00e9", "cafe\u0301"),  # decomposed accent
    ],
)
def test_rejects_paths_matching_configured_patterns(
    tmp_path: Path,
    pattern: str,
    path: str,
) -> None:
    workspace = Workspace(tmp_path, denied_paths=(pattern,))

    with pytest.raises(WorkspaceAccessError):
        workspace.resolve(path)


def test_configured_patterns_with_a_slash_are_rooted(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, denied_paths=("config/private",))

    assert (
        workspace.resolve("sub/config/private/key")
        == tmp_path / "sub/config/private/key"
    )


@pytest.mark.parametrize(
    "pattern",
    ["", ".", "./", "/absolute", "../outside", "safe/../../outside"],
)
def test_rejects_invalid_denied_patterns(tmp_path: Path, pattern: str) -> None:
    with pytest.raises(ValueError, match="workspace pattern"):
        Workspace(tmp_path, denied_paths=(pattern,))


def test_rejects_an_invalid_workspace_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workspace_root"):
        Workspace(tmp_path / "missing")


def test_caps_read_content_and_marks_truncation(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("abcdefghij", encoding="utf-8")
    workspace = Workspace(tmp_path)

    assert workspace.read_text("README.md", max_chars=5) == WorkspaceText(
        "abcde\n\n[Content truncated]", truncated=True
    )
    assert workspace.read_text("README.md", max_chars=10) == WorkspaceText(
        "abcdefghij", truncated=False
    )


@pytest.mark.parametrize(
    ("swapped", "message"),
    [
        ("", "workspace root changed"),
        ("sub", "path changed while opening"),
        ("sub/file.txt", "path changed while opening"),
    ],
)
def test_open_file_refuses_a_symlink_swapped_in_after_the_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swapped: str,
    message: str,
) -> None:
    root = tmp_path / "workspace"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "file.txt").write_text("inside", encoding="utf-8")
    # The outside tree mirrors the workspace, so every swap finds a file.
    outside = tmp_path / "outside"
    (outside / "sub").mkdir(parents=True)
    (outside / "sub" / "file.txt").write_text("secret", encoding="utf-8")
    workspace = Workspace(root)
    check = workspace.resolve

    def check_then_swap(path: str | Path = ".") -> Path:
        resolved = check(path)
        target = root / swapped
        target.rename(tmp_path / "moved")
        target.symlink_to(outside / swapped)
        return resolved

    monkeypatch.setattr(workspace, "resolve", check_then_swap)

    with pytest.raises(WorkspaceError, match=message):
        with workspace.open_file("sub/file.txt") as file:
            file.read()


def test_open_file_names_a_directory(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()

    with pytest.raises(IsADirectoryError):
        with Workspace(tmp_path).open_file("src"):
            pass


def test_write_and_remove_primitives_replace_regular_files(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    workspace = Workspace(tmp_path)

    assert workspace.write_text_atomic("notes.txt", "new\n") == target
    assert target.read_text(encoding="utf-8") == "new\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert workspace.remove_file("notes.txt") == target
    assert not target.exists()


def test_write_primitives_reject_a_leaf_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("original", encoding="utf-8")
    (tmp_path / "alias.txt").symlink_to(target.name)
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspaceError, match="symbolic link"):
        workspace.write_text_atomic("alias.txt", "changed")
    with pytest.raises(WorkspaceError, match="symbolic link"):
        workspace.remove_file("alias.txt")

    assert target.read_text(encoding="utf-8") == "original"
    assert (tmp_path / "alias.txt").is_symlink()


def test_transaction_applies_replacements_and_deletions(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two", encoding="utf-8")
    workspace = Workspace(tmp_path)

    changed = workspace.write_text_transaction(
        [("one.txt", "first"), ("new.txt", "new"), ("two.txt", None)]
    )

    assert changed == [tmp_path / "one.txt", tmp_path / "new.txt", tmp_path / "two.txt"]
    assert (tmp_path / "one.txt").read_text(encoding="utf-8") == "first"
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "new"
    assert not (tmp_path / "two.txt").exists()


def test_write_refuses_a_parent_swapped_to_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "workspace"
    (root / "sub").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = Workspace(root)
    check = workspace.resolve

    def check_then_swap(path: str | Path = ".") -> Path:
        resolved = check(path)
        (root / "sub").rename(tmp_path / "moved")
        (root / "sub").symlink_to(outside, target_is_directory=True)
        return resolved

    monkeypatch.setattr(workspace, "resolve", check_then_swap)

    with pytest.raises(WorkspaceError, match="path changed while opening"):
        workspace.write_text_atomic("sub/notes.txt", "safe")
    assert not (outside / "notes.txt").exists()
