"""Path handling shared by the read-only tools.

Every path goes through `Workspace`, so the tools inherit its boundary and
deny list. Errors become `ToolException`s, which reach the model as error
results instead of failing the run.
"""

import io
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

from langchain_core.tools import ToolException
from pathspec import GitIgnoreSpec

from app.agent.workspace import Workspace, WorkspaceAccessError, WorkspaceError

logger = logging.getLogger(__name__)

_GITIGNORE_MAX_CHARS = 1024 * 1024


def _access_denied(
    path: str | Path,
    error: WorkspaceAccessError,
    tool: str,
) -> ToolException:
    logger.warning(
        "Tool path rejected",
        extra={"tool": tool, "tool_path": str(path), "reason": error.reason},
    )
    if error.reason == "outside":
        return ToolException(f"Access denied: {path} is outside the workspace")
    return ToolException(f"Access denied: {path} is blocked by workspace policy")


def _os_error(path: str | Path, error: OSError) -> ToolException:
    # Only the reason is shown: the OS message would include the server's
    # absolute path.
    if isinstance(error, FileNotFoundError):
        return ToolException(f"Path not found: {path}")
    if isinstance(error, NotADirectoryError):
        return ToolException(f"Not a directory: {path}")
    if isinstance(error, IsADirectoryError):
        return ToolException(f"Cannot read {path}: it is a directory, use ls")
    if isinstance(error, PermissionError):
        return ToolException(f"Permission denied: {path}")
    return ToolException(
        f"Cannot read {path}: {error.strerror or type(error).__name__}"
    )


def resolve_for_tool(workspace: Workspace, path: str, *, tool: str) -> Path:
    """Resolve a path the model gave, or raise a `ToolException` it can read."""
    try:
        resolved = workspace.resolve(path)
    except WorkspaceAccessError as error:
        raise _access_denied(path, error, tool) from error
    except WorkspaceError as error:
        raise ToolException(f"Cannot read {path}: {error}") from error
    if not resolved.exists():
        raise ToolException(f"Path not found: {path}")
    return resolved


@contextmanager
def open_for_tool(
    workspace: Workspace,
    path: str,
    *,
    tool: str,
) -> Iterator[io.BufferedReader]:
    """Open a file for a tool, turning every failure into a `ToolException`."""
    try:
        with workspace.open_file(path) as file:
            yield file
    except WorkspaceAccessError as error:
        raise _access_denied(path, error, tool) from error
    except WorkspaceError as error:
        raise ToolException(f"Cannot read {path}: {error}") from error
    except OSError as error:
        raise _os_error(path, error) from error


def scan_dir(directory: Path, path: str) -> list[os.DirEntry[str]]:
    """List a directory's entries, sorted case-insensitively like pi's ls."""
    try:
        with os.scandir(directory) as entries:
            return sorted(
                entries, key=lambda entry: (entry.name.casefold(), entry.name)
            )
    except OSError as error:
        raise _os_error(path, error) from error


def display_path(workspace: Workspace, path: Path) -> str:
    """Show a path relative to the workspace root, as the model passes it back."""
    return path.relative_to(workspace.root).as_posix()


def is_visible(workspace: Workspace, path: Path) -> bool:
    """Whether a path inside the workspace is allowed, after following symlinks."""
    try:
        workspace.resolve(path)
    except WorkspaceError:
        return False
    return True


class _GitIgnore:
    """Applies each `.gitignore` from the workspace root down; deeper files win."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._specs: dict[Path, GitIgnoreSpec | None] = {}

    def _spec(self, directory: Path) -> GitIgnoreSpec | None:
        if directory not in self._specs:
            try:
                content = self._workspace.read_text(
                    directory / ".gitignore",
                    max_chars=_GITIGNORE_MAX_CHARS,
                )
            except (WorkspaceError, OSError):
                self._specs[directory] = None
            else:
                self._specs[directory] = GitIgnoreSpec.from_lines(
                    content.text.splitlines()
                )
        return self._specs[directory]

    def ignored(self, path: Path, *, is_dir: bool) -> bool:
        relative = path.relative_to(self._workspace.root)
        ignored = False
        for ancestor in reversed(relative.parents):
            directory = self._workspace.root / ancestor
            spec = self._spec(directory)
            if spec is None:
                continue
            # A trailing "/" lets directory-only patterns such as "build/" match.
            candidate = path.relative_to(directory).as_posix() + ("/" if is_dir else "")
            result = spec.check_file(candidate)
            if result.include is not None:
                ignored = result.include
        return ignored


def iter_files(workspace: Workspace, start: Path) -> Iterator[Path]:
    """Yield the files below ``start`` that a tool may read, in sorted order.

    Denied paths, symlinks leaving the workspace, and `.gitignore`d paths are
    skipped. Directory symlinks aren't followed. When ``start`` itself is
    ignored, the model asked for it explicitly, so nothing below it is.
    """
    gitignore = _GitIgnore(workspace)
    respect_gitignore = start == workspace.root or not gitignore.ignored(
        start, is_dir=True
    )

    def keep(path: Path, *, is_dir: bool) -> bool:
        if not is_visible(workspace, path):
            return False
        return not (respect_gitignore and gitignore.ignored(path, is_dir=is_dir))

    for dirpath, dirnames, filenames in os.walk(start):
        current = Path(dirpath)
        dirnames[:] = sorted(
            name for name in dirnames if keep(current / name, is_dir=True)
        )
        for name in sorted(filenames):
            path = current / name
            if keep(path, is_dir=False):
                yield path


def glob_matcher(pattern: str) -> Callable[[PurePosixPath], bool]:
    """Match paths below a search directory the way pi's find does.

    A pattern without "/" matches the file name. One with "/" matches the
    path at any depth, unless it starts with "/", which anchors it to the
    search directory.
    """
    if "/" not in pattern:
        return lambda path: fnmatchcase(path.name, pattern)
    if pattern.startswith("/"):
        anchored = pattern.lstrip("/")
        return lambda path: path.full_match(anchored)
    anywhere = pattern if pattern.startswith("**/") else f"**/{pattern}"
    return lambda path: path.full_match(anywhere)
