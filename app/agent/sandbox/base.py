import logging
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol

from app.agent.sandbox.policy import SandboxPolicy
from app.agent.sandbox.runner import CommandResult, build_env, run_process
from common.config import DEFAULT_SANDBOX_PATH

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandSettings:
    """How confined commands run, as opposed to what they can reach."""

    path: str = DEFAULT_SANDBOX_PATH
    # Extra variables set for every command; nothing is inherited.
    env: Mapping[str, str] = field(default_factory=dict)
    max_output_bytes: int = 50 * 1024
    # Seconds, when a caller doesn't pass its own.
    timeout: float = 120


class Sandbox(Protocol):
    """Runs commands so they can't reach beyond the workspace."""

    @property
    def name(self) -> str: ...

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout: float | None = None,
    ) -> CommandResult: ...


class ConfinedSandbox(ABC):
    """A sandbox that confines each command with a backend's argument prefix.

    A backend only says how to confine a command; spawning, the clean
    environment, the scratch directory, timeouts, and output are shared.
    """

    name: ClassVar[str]

    def __init__(
        self,
        policy: SandboxPolicy,
        settings: CommandSettings | None = None,
    ) -> None:
        self.policy = policy
        self.settings = settings or CommandSettings()

    @abstractmethod
    def confine(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        scratch: Path,
    ) -> list[str]:
        """The full command line that runs ``argv`` inside the sandbox."""

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run ``argv`` (never through a shell) and return how it ended.

        ``cwd`` must be a directory inside the workspace that the deny list
        doesn't cover. Each command gets a fresh scratch directory as its
        ``HOME`` and ``TMPDIR``, removed afterwards.
        """
        # A leading "-" would be read as an option by the sandbox launcher.
        if not argv or not argv[0] or argv[0].startswith("-"):
            raise ValueError("argv must start with a program name")
        workdir = self.policy.workspace.resolve(cwd)
        if not workdir.is_dir():
            raise ValueError(f"not a directory: {cwd}")

        with tempfile.TemporaryDirectory(
            prefix="miao-pi-cmd-", ignore_cleanup_errors=True
        ) as raw_scratch:
            scratch = Path(raw_scratch).resolve()
            result = await run_process(
                self.confine(argv, cwd=workdir, scratch=scratch),
                cwd=workdir,
                env=build_env(
                    path=self.settings.path, scratch=scratch, extra=self.settings.env
                ),
                timeout=self.settings.timeout if timeout is None else timeout,
                max_output_bytes=self.settings.max_output_bytes,
            )
        logger.info(
            "Command completed",
            extra={
                # Only the program's name: arguments can carry secrets.
                "command": Path(argv[0]).name,
                "sandbox": self.name,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "truncated": result.truncated,
                "duration_ms": result.duration_ms,
            },
        )
        return result
