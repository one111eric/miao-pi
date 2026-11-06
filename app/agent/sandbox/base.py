import logging
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import ClassVar, Protocol

from app.agent.sandbox.policy import SandboxPolicy
from app.agent.sandbox.runner import (
    CommandResult,
    ContainmentError,
    build_env,
    run_process,
)
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


@dataclass(frozen=True)
class RunDir:
    """One command's private directory, removed when the command ends.

    Inside the sandbox only ``scratch`` (the command's ``HOME`` and
    ``TMPDIR``) is writable, not the directory around it, so a backend can
    keep files next to ``scratch`` that the command can't change.
    """

    path: Path

    @property
    def scratch(self) -> Path:
        return self.path / "home"


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
    environment, the scratch directory, timeouts, and output are shared. A
    backend whose confinement doesn't end every process the command started
    also implements ``sweep``.

    Once a sweep fails, the sandbox refuses every command until the service
    restarts, and stops the ones already running: what the failed one left
    running can still change the workspace, so no command's result can be
    trusted alongside it.
    """

    name: ClassVar[str]

    def __init__(
        self,
        policy: SandboxPolicy,
        settings: CommandSettings | None = None,
    ) -> None:
        self.policy = policy
        self.settings = settings or CommandSettings()
        # Why commands are refused, once a sweep failed; None until then.
        self.disabled_reason: str | None = None

    @abstractmethod
    def confine(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        run: RunDir,
    ) -> list[str]:
        """The full command line that runs ``argv`` inside the sandbox."""

    def prepare(self, run: RunDir) -> None:
        """Set up anything this backend keeps in a command's run directory."""

    def sweep(self, run: RunDir, pgid: int) -> None:
        """End the command's processes that the group kill missed.

        Called once, after its process group ``pgid`` is killed, in a
        thread. Raises ``ContainmentError`` when any may still be running.
        Nothing to do for a backend that ends the whole process tree itself.
        """

    async def self_check(self) -> str | None:
        """Backend-specific startup checks: what got through, or None.

        Runs on the throwaway sandbox in ``startup.self_test``.
        """
        return None

    @contextmanager
    def new_run(self) -> Iterator[RunDir]:
        """A fresh run directory, prepared for this backend."""
        with tempfile.TemporaryDirectory(
            prefix="miao-pi-cmd-", ignore_cleanup_errors=True
        ) as raw:
            run = RunDir(Path(raw).resolve())
            run.scratch.mkdir()
            self.prepare(run)
            yield run

    def command_env(self, run: RunDir) -> dict[str, str]:
        return build_env(
            path=self.settings.path, scratch=run.scratch, extra=self.settings.env
        )

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

        Raises ``ContainmentError`` when the command's processes may still
        be running, when another command's did while it ran, and for every
        command after that.
        """
        if (refusal := self._refusal()) is not None:
            raise ContainmentError(refusal)
        # A leading "-" would be read as an option by the sandbox launcher.
        if not argv or not argv[0] or argv[0].startswith("-"):
            raise ValueError("argv must start with a program name")
        workdir = self.policy.workspace.resolve(cwd)
        if not workdir.is_dir():
            raise ValueError(f"not a directory: {cwd}")

        with self.new_run() as run:
            result = await run_process(
                self.confine(argv, cwd=workdir, run=run),
                cwd=workdir,
                env=self.command_env(run),
                timeout=self.settings.timeout if timeout is None else timeout,
                max_output_bytes=self.settings.max_output_bytes,
                sweep=partial(self._sweep, run),
                stop=self._refusal,
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
        # It may have exited between the runner's checks, after another
        # command's sweep failed; its result can't be trusted either.
        if (refusal := self._refusal()) is not None:
            raise ContainmentError(refusal)
        return result

    def _refusal(self) -> str | None:
        """Why commands are refused, as an error message, or None."""
        if self.disabled_reason is None:
            return None
        return f"commands are off until the service restarts: {self.disabled_reason}"

    def _sweep(self, run: RunDir, pgid: int) -> None:
        """The backend's sweep, turning commands off when it fails."""
        try:
            self.sweep(run, pgid)
        except ContainmentError as error:
            self._disable(str(error))
            raise
        except Exception as error:
            # Then nothing says the command's processes are gone.
            reason = f"the sweep failed: {type(error).__name__}: {error}"
            self._disable(reason)
            raise ContainmentError(reason) from error

    def _disable(self, reason: str) -> None:
        if self.disabled_reason is not None:
            return
        self.disabled_reason = reason
        logger.error(
            "Sandbox commands turned off",
            extra={"sandbox": self.name, "reason": reason},
        )
