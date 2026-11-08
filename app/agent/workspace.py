import errno
import io
import os
import secrets
import stat
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Literal

# `.env*` also covers variants such as `.env.local`, `.envrc`, and `.env-prod`.
# `.claude` holds local coding-agent settings, whose permission allowlists can
# carry tokens; it's often kept out of git by a global ignore file, which the
# tools don't read, so it has to be denied here.
REQUIRED_DENIED_PATTERNS = (".env*", ".git", ".claude")


def _fold(text: str) -> str:
    # Canonical caseless form, so case or Unicode normalization variants that a
    # case-insensitive filesystem opens as the same file can't dodge a pattern.
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", text).casefold())


class WorkspaceError(Exception):
    """Base error for invalid or inaccessible workspace paths."""


type AccessDenialReason = Literal["outside", "denied"]


class WorkspaceAccessError(WorkspaceError):
    """Raised when a path escapes the workspace or enters a denied location."""

    def __init__(self, message: str, *, reason: AccessDenialReason) -> None:
        super().__init__(message)
        self.reason = reason


class WorkspaceTransactionError(WorkspaceError):
    """An I/O failure while committing a transaction, tied to its file."""

    def __init__(self, path: Path, error: OSError) -> None:
        super().__init__(str(error))
        self.path = path
        self.error = error


@dataclass(frozen=True)
class WorkspaceText:
    """Text read from the workspace; ``truncated`` is true when it was capped."""

    text: str
    truncated: bool


@dataclass
class _StagedWrite:
    path: Path
    directory: int
    name: str
    content: str | None
    temporary: str | None = None
    backup: str | None = None
    had_original: bool = False
    written: bool = False


class Workspace:
    """Resolve and read paths within one configured project directory."""

    def __init__(
        self,
        root: Path,
        denied_paths: tuple[str, ...] = (),
    ) -> None:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.is_dir():
            raise ValueError(f"workspace_root is not a directory: {resolved_root}")

        self.root = resolved_root
        # The root is reopened by path, so remember which directory it is.
        root_stat = resolved_root.stat()
        self._root_id = (root_stat.st_dev, root_stat.st_ino)
        # Configured patterns add to the required ones and never replace them.
        self._denied_patterns = tuple(
            dict.fromkeys(
                (
                    *(_fold(pattern) for pattern in REQUIRED_DENIED_PATTERNS),
                    *(self._validate_denied_pattern(p) for p in denied_paths),
                )
            )
        )

    @staticmethod
    def _validate_denied_pattern(pattern: str) -> str:
        denied = Path(pattern)
        # "", ".", and "./" have no parts, so they would silently match nothing.
        if not denied.parts:
            raise ValueError(f"denied workspace pattern is empty: {pattern!r}")
        if denied.is_absolute():
            raise ValueError(f"denied workspace pattern must be relative: {pattern}")
        if ".." in denied.parts:
            raise ValueError(
                f"denied workspace pattern escapes the workspace: {pattern}"
            )
        return _fold(denied.as_posix())

    @property
    def denied_patterns(self) -> tuple[str, ...]:
        """Every pattern in effect, required ones first, in folded form."""
        return self._denied_patterns

    @property
    def has_configured_patterns(self) -> bool:
        """Whether ``denied_paths`` added anything beyond the required patterns."""
        return len(self._denied_patterns) > len(REQUIRED_DENIED_PATTERNS)

    def is_denied(self, path: Path) -> bool:
        """Whether an absolute path inside the root matches a deny pattern.

        This checks names only and follows no symlinks; ``resolve`` checks
        both before and after following them.
        """
        relative = path.relative_to(self.root)
        folded_parts = tuple(_fold(part) for part in relative.parts)

        for pattern in self._denied_patterns:
            pattern_parts = Path(pattern).parts
            if len(pattern_parts) == 1:
                if any(fnmatchcase(part, pattern) for part in folded_parts):
                    return True
                continue

            # Multi-part patterns are rooted at the workspace and deny the
            # matched path plus everything beneath it.
            for end in range(1, len(folded_parts) + 1):
                prefix = "/".join(folded_parts[:end])
                if fnmatchcase(prefix, pattern):
                    return True
        return False

    def resolve(self, path: str | Path = ".") -> Path:
        """Resolve a path and reject escapes and denied locations."""
        if "\x00" in os.fspath(path):
            # The OS would raise ValueError; keep every bad path a WorkspaceError.
            raise WorkspaceError("path contains a NUL byte")
        candidate = Path(path)
        lexical = Path(
            os.path.abspath(
                candidate if candidate.is_absolute() else self.root / candidate
            )
        )

        if not lexical.is_relative_to(self.root):
            raise WorkspaceAccessError(
                f"path is outside the workspace: {path}", reason="outside"
            )

        if self.is_denied(lexical):
            raise WorkspaceAccessError(
                f"path is denied by workspace policy: {path}", reason="denied"
            )

        resolved = lexical.resolve()
        if not resolved.is_relative_to(self.root):
            raise WorkspaceAccessError(
                f"path is outside the workspace: {path}", reason="outside"
            )

        if self.is_denied(resolved):
            raise WorkspaceAccessError(
                f"path is denied by workspace policy: {path}", reason="denied"
            )

        return resolved

    def resolve_write_target(self, path: str | Path) -> Path:
        """Resolve a write target without allowing its leaf to be a symlink."""
        resolved = self.resolve(path)
        candidate = Path(path)
        lexical = Path(
            os.path.abspath(
                candidate if candidate.is_absolute() else self.root / candidate
            )
        )
        if lexical.is_symlink():
            raise WorkspaceError("cannot write a symbolic link")
        return resolved

    @contextmanager
    def open_file(self, path: str | Path) -> Iterator[io.BufferedReader]:
        """Open a regular file inside the workspace for binary reading.

        Opening is non-blocking so a FIFO can't hang the caller, and anything
        other than a regular file is rejected once open.
        """
        resolved = self.resolve(path)
        fd = self._open_beneath_root(resolved.relative_to(self.root))
        try:
            mode = os.fstat(fd).st_mode
            # POSIX opens directories without complaint; name them explicitly
            # rather than relying on the file object to refuse one.
            if stat.S_ISDIR(mode):
                raise IsADirectoryError(
                    errno.EISDIR, os.strerror(errno.EISDIR), str(path)
                )
            if not stat.S_ISREG(mode):
                raise WorkspaceError("not a regular file")
            os.set_blocking(fd, True)
            file = os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise
        with file:
            yield file

    def _open_beneath_root(self, relative: Path) -> int:
        """Open a resolved path one component at a time, never following links.

        ``resolve`` returns a path without symlinks, so meeting one on the way
        means a component was swapped after the check. Opening each component
        relative to its parent with ``O_NOFOLLOW`` makes that fail instead of
        escaping; opening the full path would follow a swapped parent. The root
        itself is opened by path, so it must still be the directory found at
        startup (same device and inode).
        """
        *parents, name = relative.parts or (".",)
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            root_stat = os.fstat(directory)
            if (root_stat.st_dev, root_stat.st_ino) != self._root_id:
                raise WorkspaceError("workspace root changed")
            for part in parents:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                os.close(directory)
                directory = child
            return os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
        except OSError as error:
            # A symlink (ELOOP) or a file in place of a directory (ENOTDIR).
            if error.errno in (errno.ELOOP, errno.ENOTDIR):
                raise WorkspaceError("path changed while opening") from error
            raise
        finally:
            os.close(directory)

    def _open_parent_beneath_root(self, relative: Path) -> tuple[int, str]:
        """Open a writable target's parent without following swapped links.

        The caller owns the returned directory descriptor.  This mirrors the
        read path's component-by-component walk, so an approved write cannot
        be redirected outside the workspace between validation and rename.
        """
        if not relative.parts:
            raise IsADirectoryError(
                errno.EISDIR, os.strerror(errno.EISDIR), str(relative)
            )
        *parents, name = relative.parts
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            root_stat = os.fstat(directory)
            if (root_stat.st_dev, root_stat.st_ino) != self._root_id:
                raise WorkspaceError("workspace root changed")
            for part in parents:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                os.close(directory)
                directory = child
        except OSError as error:
            os.close(directory)
            if error.errno in (errno.ELOOP, errno.ENOTDIR):
                raise WorkspaceError("path changed while opening") from error
            raise
        return directory, name

    def write_text_atomic(self, path: str | Path, content: str) -> Path:
        """Atomically replace a workspace text file after enforcing its boundary.

        Parent directories must already exist.  The temporary file and rename
        use a checked directory descriptor so a concurrent symlink swap can't
        send a write outside the workspace.
        """
        resolved = self.resolve_write_target(path)
        if resolved == self.root:
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), str(path))
        directory, name = self._open_parent_beneath_root(
            resolved.relative_to(self.root)
        )
        temporary: str | None = None
        try:
            try:
                destination = os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                mode = 0o666
            else:
                if not stat.S_ISREG(destination.st_mode):
                    raise IsADirectoryError(
                        errno.EISDIR, os.strerror(errno.EISDIR), str(path)
                    )
                mode = stat.S_IMODE(destination.st_mode)

            for _ in range(10):
                temporary = f".miao-pi-{secrets.token_hex(12)}.tmp"
                try:
                    descriptor = os.open(
                        temporary,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        mode,
                        dir_fd=directory,
                    )
                    break
                except FileExistsError:
                    temporary = None
            else:
                raise WorkspaceError("could not allocate a temporary file")

            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file:
                    file.write(content)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
                temporary = None
                os.fsync(directory)
            finally:
                if temporary is not None:
                    try:
                        os.unlink(temporary, dir_fd=directory)
                    except FileNotFoundError:
                        pass
        finally:
            os.close(directory)
        return resolved

    def remove_file(self, path: str | Path) -> Path:
        """Remove a regular workspace file without following swapped links."""
        resolved = self.resolve_write_target(path)
        if resolved == self.root:
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), str(path))
        directory, name = self._open_parent_beneath_root(
            resolved.relative_to(self.root)
        )
        try:
            target = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISREG(target.st_mode):
                raise IsADirectoryError(
                    errno.EISDIR, os.strerror(errno.EISDIR), str(path)
                )
            os.unlink(name, dir_fd=directory)
            os.fsync(directory)
        finally:
            os.close(directory)
        return resolved

    def write_text_transaction(
        self, changes: Sequence[tuple[str | Path, str | None]]
    ) -> list[Path]:
        """Apply file replacements and deletions, restoring all on a failure.

        Each replacement is staged in its destination directory before any
        visible path changes. Renames still happen one at a time, but every
        already-modified destination is restored if a later rename fails. A
        process killed between moving an original to its backup and moving in
        its replacement can temporarily leave that destination absent; this
        is not a durable, crash-safe transaction protocol.
        """
        if not changes:
            return []

        staged: list[_StagedWrite] = []
        seen: set[Path] = set()
        active_path: Path | None = None
        try:
            for path, content in changes:
                resolved = self.resolve_write_target(path)
                active_path = resolved
                if resolved == self.root:
                    raise IsADirectoryError(
                        errno.EISDIR, os.strerror(errno.EISDIR), str(path)
                    )
                if resolved in seen:
                    raise WorkspaceError("a transaction cannot change one file twice")
                seen.add(resolved)
                directory, name = self._open_parent_beneath_root(
                    resolved.relative_to(self.root)
                )
                entry = _StagedWrite(resolved, directory, name, content)
                staged.append(entry)
                try:
                    destination = os.stat(name, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    if content is None:
                        raise
                    mode = 0o666
                else:
                    if not stat.S_ISREG(destination.st_mode):
                        raise IsADirectoryError(
                            errno.EISDIR, os.strerror(errno.EISDIR), str(path)
                        )
                    entry.had_original = True
                    mode = stat.S_IMODE(destination.st_mode)

                if content is not None:
                    temporary = self._temporary_name(directory)
                    entry.temporary = temporary
                    descriptor = os.open(
                        temporary,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        mode,
                        dir_fd=directory,
                    )
                    with os.fdopen(
                        descriptor, "w", encoding="utf-8", newline=""
                    ) as file:
                        file.write(content)
                        file.flush()
                        os.fsync(file.fileno())

            for entry in staged:
                directory = entry.directory
                name = entry.name
                active_path = entry.path
                if entry.had_original:
                    backup = self._temporary_name(directory)
                    entry.backup = backup
                    os.replace(name, backup, src_dir_fd=directory, dst_dir_fd=directory)
                staged_temporary: str | None = entry.temporary
                if staged_temporary is not None:
                    os.replace(
                        staged_temporary,
                        name,
                        src_dir_fd=directory,
                        dst_dir_fd=directory,
                    )
                    entry.temporary = None
                    entry.written = True
                os.fsync(directory)
        except BaseException as original_error:
            rollback_error: OSError | None = None
            for entry in reversed(staged):
                directory = entry.directory
                name = entry.name
                try:
                    if entry.written:
                        try:
                            os.unlink(name, dir_fd=directory)
                        except FileNotFoundError:
                            pass
                    saved_backup: str | None = entry.backup
                    if saved_backup is not None:
                        os.replace(
                            saved_backup,
                            name,
                            src_dir_fd=directory,
                            dst_dir_fd=directory,
                        )
                        entry.backup = None
                    os.fsync(directory)
                except OSError as error:
                    rollback_error = error
            if rollback_error is not None:
                raise WorkspaceError("transaction failed and rollback was incomplete")
            if isinstance(original_error, OSError) and active_path is not None:
                raise WorkspaceTransactionError(
                    active_path, original_error
                ) from original_error
            raise
        finally:
            for entry in staged:
                directory = entry.directory
                for leftover in (entry.temporary, entry.backup):
                    if leftover is not None:
                        try:
                            os.unlink(leftover, dir_fd=directory)
                        except FileNotFoundError:
                            pass
                os.close(directory)
        return [entry.path for entry in staged]

    @staticmethod
    def _temporary_name(directory: int) -> str:
        for _ in range(10):
            name = f".miao-pi-{secrets.token_hex(12)}.tmp"
            try:
                os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                return name
        raise WorkspaceError("could not allocate a temporary file")

    def read_text(self, path: str | Path, *, max_chars: int) -> WorkspaceText:
        """Read at most ``max_chars`` and mark truncated content."""
        if max_chars < 1:
            raise ValueError("max_chars must be positive")

        with (
            self.open_file(path) as raw,
            io.TextIOWrapper(raw, encoding="utf-8", errors="replace") as file,
        ):
            content = file.read(max_chars + 1)

        if len(content) <= max_chars:
            return WorkspaceText(content, truncated=False)
        return WorkspaceText(
            f"{content[:max_chars]}\n\n[Content truncated]",
            truncated=True,
        )
