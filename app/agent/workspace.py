import os
import unicodedata
from fnmatch import fnmatchcase
from pathlib import Path

# `.env*` also covers variants such as `.env.local`, `.envrc`, and `.env-prod`.
REQUIRED_DENIED_PATTERNS = (".env*", ".git")


def _fold(text: str) -> str:
    # Canonical caseless form, so case or Unicode normalization variants that a
    # case-insensitive filesystem opens as the same file can't dodge a pattern.
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", text).casefold())


class WorkspaceError(Exception):
    """Base error for invalid or inaccessible workspace paths."""


class WorkspaceAccessError(WorkspaceError):
    """Raised when a path escapes the workspace or enters a denied location."""


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

    def _is_denied(self, path: Path) -> bool:
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
        candidate = Path(path)
        lexical = Path(
            os.path.abspath(
                candidate if candidate.is_absolute() else self.root / candidate
            )
        )

        if not lexical.is_relative_to(self.root):
            raise WorkspaceAccessError(f"path is outside the workspace: {path}")

        if self._is_denied(lexical):
            raise WorkspaceAccessError(f"path is denied by workspace policy: {path}")

        resolved = lexical.resolve()
        if not resolved.is_relative_to(self.root):
            raise WorkspaceAccessError(f"path is outside the workspace: {path}")

        if self._is_denied(resolved):
            raise WorkspaceAccessError(f"path is denied by workspace policy: {path}")

        return resolved

    def read_text(self, path: str | Path, *, max_chars: int) -> str:
        """Read at most ``max_chars`` and mark truncated content."""
        if max_chars < 1:
            raise ValueError("max_chars must be positive")

        resolved = self.resolve(path)
        with resolved.open(encoding="utf-8", errors="replace") as file:
            content = file.read(max_chars + 1)

        if len(content) <= max_chars:
            return content
        return f"{content[:max_chars]}\n\n[Content truncated]"
