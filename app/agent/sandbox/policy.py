from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.agent.workspace import Workspace


class SandboxConfigError(ValueError):
    """Raised at startup for a sandbox setting that can't be used."""


@dataclass(frozen=True)
class SandboxPolicy:
    """What a confined command may reach, independent of the backend.

    Commands read and write the workspace, minus its deny list, and a
    per-command scratch directory. They read the system directories each
    backend adds, plus ``read_paths``. Everything else is out of reach.
    """

    # The same instance the read tools use, so both enforce one deny list.
    workspace: Workspace
    # Extra read-only paths, for toolchains under the home directory.
    read_paths: tuple[Path, ...] = ()


def resolve_read_paths(raw_paths: Sequence[str]) -> tuple[Path, ...]:
    """Expand and check the configured read paths.

    The home directory itself, or anything containing it, is refused: it
    would hand every command ``~/.ssh`` and other credentials.
    """
    home = Path.home().resolve()
    resolved: list[Path] = []
    for raw in raw_paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise SandboxConfigError(f"sandbox read path doesn't exist: {raw}")
        if home.is_relative_to(path):
            raise SandboxConfigError(
                f"sandbox read path would expose the whole home directory: {raw}"
            )
        resolved.append(path)
    return tuple(resolved)
