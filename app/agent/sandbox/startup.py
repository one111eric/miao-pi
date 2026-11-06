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


async def self_test(sandbox: ConfinedSandbox) -> str | None:
    """Prove the sandbox confines commands, or say what got through."""
    root = sandbox.policy.workspace.root
    plain = await sandbox.run(["true"], cwd=root, timeout=SELF_TEST_TIMEOUT)
    if plain.exit_code != 0:
        return "a plain command failed"

    with tempfile.TemporaryDirectory() as raw_outside:
        outside = Path(raw_outside).resolve()
        if outside.is_relative_to(root):
            return "the workspace contains the temp directory"
        canary = outside / "canary.txt"
        canary.write_text(_CANARY, encoding="utf-8")
        read = await sandbox.run(
            ["cat", str(canary)], cwd=root, timeout=SELF_TEST_TIMEOUT
        )
        if read.exit_code == 0 or _CANARY in read.output:
            return "a file outside the workspace was readable"

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
            logger.warning("Sandbox network check skipped: no bash or curl")
            return None
        await sandbox.run(client, cwd=root, timeout=SELF_TEST_TIMEOUT)
        # Let a connection that did get through reach the accept callback.
        await asyncio.sleep(0.1)
    finally:
        server.close()
        await server.wait_closed()
    if connections:
        return "a local network connection got through"
    return None


async def make_sandbox(settings: SvcSettings, workspace: Workspace) -> SandboxSetup:
    """Build the configured sandbox and self-test it.

    A misconfigured read path raises ``SandboxConfigError``, failing startup
    like a bad workspace root. A missing or failing backend only disables
    command tools.
    """
    if settings.sandbox_backend is SandboxBackend.OFF:
        return SandboxSetup(None, "off")
    backend = _choose(settings.sandbox_backend)
    if isinstance(backend, str):
        return SandboxSetup(None, f"unavailable: {backend}")

    sandbox = backend(
        SandboxPolicy(workspace, resolve_read_paths(settings.sandbox_read_paths)),
        CommandSettings(
            path=settings.sandbox_path,
            env=dict(settings.sandbox_env),
            max_output_bytes=settings.sandbox_max_output_bytes,
            timeout=settings.sandbox_command_timeout,
        ),
    )
    problem = await self_test(sandbox)
    if problem is not None:
        logger.error(
            "Sandbox self-test failed",
            extra={"sandbox": sandbox.name, "reason": problem},
        )
        return SandboxSetup(None, f"unavailable: {problem}")
    return SandboxSetup(sandbox, sandbox.name)
