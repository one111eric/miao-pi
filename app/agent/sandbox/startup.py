"""Choosing a sandbox backend at startup, and proving it works first."""

import asyncio
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.agent.sandbox.base import CommandSettings, ConfinedSandbox, Sandbox
from app.agent.sandbox.bubblewrap import BubblewrapSandbox
from app.agent.sandbox.policy import SandboxPolicy, resolve_read_paths
from app.agent.sandbox.runner import CommandResult
from app.agent.sandbox.seatbelt import SeatbeltSandbox
from app.agent.workspace import Workspace
from common.config import SandboxBackend, SvcSettings

logger = logging.getLogger(__name__)

SELF_TEST_TIMEOUT = 10.0
_CANARY = "miao-pi sandbox canary"

type BackendClass = type[SeatbeltSandbox] | type[BubblewrapSandbox]

_BACKENDS: dict[SandboxBackend, BackendClass] = {
    SandboxBackend.SEATBELT: SeatbeltSandbox,
    SandboxBackend.BUBBLEWRAP: BubblewrapSandbox,
}


@dataclass(frozen=True)
class SandboxSetup:
    # None when commands can't run sandboxed; then no command tools exist.
    sandbox: Sandbox | None
    # The backend's name, "off", or "unavailable: <reason>".
    status: str


def _choose(setting: SandboxBackend) -> BackendClass | str:
    """A backend class that's available here, or why there isn't one."""
    if setting is SandboxBackend.AUTO:
        for backend in _BACKENDS.values():
            if backend.available():
                return backend
        return "no sandbox backend for this platform"
    backend = _BACKENDS[setting]
    return backend if backend.available() else f"{backend.name} isn't available here"


async def _network_reachable(sandbox: ConfinedSandbox, cwd: Path) -> bool | None:
    """Whether a command reached a listener on localhost; None if untestable."""
    connections = 0

    def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal connections
        connections += 1
        writer.close()

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        if shutil.which("bash", path=sandbox.settings.path):
            client = ["bash", "-c", f"exec 3<>/dev/tcp/127.0.0.1/{port}"]
        elif shutil.which("curl", path=sandbox.settings.path):
            client = ["curl", "-sS", "-m", "3", f"http://127.0.0.1:{port}/"]
        else:
            return None
        await sandbox.run(client, cwd=cwd, timeout=SELF_TEST_TIMEOUT)
        # Let a connection that did get through reach the accept callback.
        await asyncio.sleep(0.1)
    finally:
        server.close()
        await server.wait_closed()
    return connections > 0


async def self_test(sandbox: ConfinedSandbox) -> str | None:
    """Prove the sandbox confines commands, or say what got through.

    Only a plain command runs in the real workspace. The checks that write
    run in a throwaway workspace built the same way (same backend, deny
    patterns, read paths, and settings), so the real one is never touched.
    """

    async def run(argv: list[str], cwd: Path, on: ConfinedSandbox) -> CommandResult:
        return await on.run(argv, cwd=cwd, timeout=SELF_TEST_TIMEOUT)

    real_root = sandbox.policy.workspace.root
    if (await run(["true"], real_root, sandbox)).exit_code != 0:
        return "a plain command failed in the workspace"

    with tempfile.TemporaryDirectory() as raw_base:
        base = Path(raw_base).resolve()
        if base.is_relative_to(real_root):
            return "the workspace contains the temp directory"
        root, outside = base / "workspace", base / "outside"
        root.mkdir()
        outside.mkdir()
        (root / ".env").write_text(_CANARY, encoding="utf-8")
        (outside / "canary.txt").write_text(_CANARY, encoding="utf-8")
        trial = type(sandbox)(
            SandboxPolicy(
                Workspace(root, denied_paths=sandbox.policy.workspace.denied_patterns),
                sandbox.policy.read_paths,
            ),
            sandbox.settings,
        )

        await run(["/bin/sh", "-c", "echo ok > inside.txt"], root, trial)
        if not (root / "inside.txt").exists():
            return "a command couldn't write inside the workspace"
        denied_read = await run(["cat", ".env"], root, trial)
        if denied_read.exit_code == 0 or _CANARY in denied_read.output:
            return "a denied file was readable"
        await run(["/bin/sh", "-c", "echo changed >> .env"], root, trial)
        if (root / ".env").read_text(encoding="utf-8") != _CANARY:
            return "a denied file was writable"
        outside_read = await run(["cat", str(outside / "canary.txt")], root, trial)
        if outside_read.exit_code == 0 or _CANARY in outside_read.output:
            return "a file outside the workspace was readable"
        await run(["/bin/sh", "-c", f"echo x > '{outside}/new.txt'"], root, trial)
        if (outside / "new.txt").exists():
            return "a file outside the workspace was writable"

        reachable = await _network_reachable(trial, root)
        if reachable:
            return "a local network connection got through"
        if reachable is None:
            logger.warning("Sandbox network check skipped: no bash or curl")
        return await trial.self_check()


async def make_sandbox(settings: SvcSettings, workspace: Workspace) -> SandboxSetup:
    """Build the configured sandbox and self-test it.

    A read path that doesn't exist, or that would expose the home directory,
    raises ``SandboxConfigError`` and fails startup, like a bad workspace
    root. Anything else that goes wrong (no backend, a backend that can't
    express the deny list, one that fails to start or fails a check) only
    disables command tools.
    """
    if settings.sandbox_backend is SandboxBackend.OFF:
        return SandboxSetup(None, "off")
    backend = _choose(settings.sandbox_backend)
    if isinstance(backend, str):
        return SandboxSetup(None, f"unavailable: {backend}")
    read_paths = resolve_read_paths(settings.sandbox_read_paths)

    try:
        sandbox = backend(
            SandboxPolicy(workspace, read_paths),
            CommandSettings(
                path=settings.sandbox_path,
                env=dict(settings.sandbox_env),
                max_output_bytes=settings.sandbox_max_output_bytes,
                timeout=settings.sandbox_command_timeout,
            ),
        )
        problem = await self_test(sandbox)
    except Exception as error:
        # Unproven means unavailable, never a crashed service or a sandbox
        # trusted without its checks.
        problem = f"{type(error).__name__}: {error}"
    if problem is not None:
        logger.error(
            "Sandbox self-test failed",
            extra={"sandbox": backend.name, "reason": problem},
        )
        return SandboxSetup(None, f"unavailable: {problem}")
    return SandboxSetup(sandbox, sandbox.name)
