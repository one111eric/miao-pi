"""The sandbox's guarantees, checked against the platform's real backend.

Opt-in, like the live LLM tests: RUN_SANDBOX_TESTS=1 uv run pytest -m sandbox
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

from app.agent.sandbox import ConfinedSandbox, SandboxPolicy
from app.agent.sandbox.bubblewrap import BubblewrapSandbox
from app.agent.sandbox.seatbelt import SeatbeltSandbox
from app.agent.sandbox.startup import self_test
from app.agent.workspace import Workspace

pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(
        os.environ.get("RUN_SANDBOX_TESTS") != "1",
        reason="Set RUN_SANDBOX_TESTS=1 to run commands in the real sandbox.",
    ),
]

REPO = Path(__file__).resolve().parents[2]
SECRET = "sandbox-test-secret"


def platform_backend() -> type[SeatbeltSandbox] | type[BubblewrapSandbox]:
    for backend in (SeatbeltSandbox, BubblewrapSandbox):
        if backend.available():
            return backend
    pytest.skip("No sandbox backend on this platform.")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    for name in (
        ".env",
        "sub/.ENV.local",
        ".git/config",
        ".claude/settings.local.json",
        "private/key",
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(SECRET, encoding="utf-8")
    (root / "notes.txt").write_text("hello", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file.txt").write_text(SECRET, encoding="utf-8")
    (root / "link.txt").symlink_to(outside / "file.txt")
    return root.resolve()


@pytest.fixture
def sandbox(root: Path) -> ConfinedSandbox:
    workspace = Workspace(root, denied_paths=("private",))
    return platform_backend()(SandboxPolicy(workspace))


@pytest.mark.anyio
async def test_the_self_test_passes(sandbox: ConfinedSandbox) -> None:
    assert await self_test(sandbox) is None


@pytest.mark.anyio
async def test_commands_read_and_write_inside_the_workspace(
    sandbox: ConfinedSandbox, root: Path
) -> None:
    read = await sandbox.run(["cat", "notes.txt"], cwd=root)
    write = await sandbox.run(["/bin/sh", "-c", "echo made > new.txt"], cwd=root)

    assert (read.exit_code, read.output) == (0, "hello")
    assert write.exit_code == 0
    assert (root / "new.txt").read_text(encoding="utf-8") == "made\n"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "sub/.ENV.local",
        ".git/config",
        ".claude/settings.local.json",
        "private/key",
    ],
)
async def test_denied_paths_can_be_neither_read_nor_written(
    sandbox: ConfinedSandbox, root: Path, path: str
) -> None:
    read = await sandbox.run(["cat", path], cwd=root)
    write = await sandbox.run(["/bin/sh", "-c", f"echo changed > '{path}'"], cwd=root)

    assert read.exit_code != 0
    assert SECRET not in read.output
    assert write.exit_code != 0
    assert (root / path).read_text(encoding="utf-8") == SECRET


@pytest.mark.anyio
async def test_nothing_outside_the_workspace_is_reachable(
    sandbox: ConfinedSandbox, root: Path
) -> None:
    outside = root.parent / "outside"

    direct = await sandbox.run(["cat", str(outside / "file.txt")], cwd=root)
    through_link = await sandbox.run(["cat", "link.txt"], cwd=root)
    home = await sandbox.run(["ls", str(Path.home())], cwd=root)
    write = await sandbox.run(
        ["/bin/sh", "-c", f"echo x > '{outside}/new.txt'"], cwd=root
    )

    for result in (direct, through_link, home, write):
        assert result.exit_code != 0
        assert SECRET not in result.output
    assert not (outside / "new.txt").exists()


@pytest.mark.anyio
async def test_there_is_no_network(sandbox: ConfinedSandbox, root: Path) -> None:
    connections = 0

    def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal connections
        connections += 1
        writer.close()

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        local = await sandbox.run(
            ["bash", "-c", f"exec 3<>/dev/tcp/127.0.0.1/{port}"], cwd=root
        )
        remote = await sandbox.run(
            ["bash", "-c", "exec 3<>/dev/tcp/1.1.1.1/443"], cwd=root, timeout=10
        )
        await asyncio.sleep(0.1)
    finally:
        server.close()
        await server.wait_closed()

    assert local.exit_code != 0
    assert remote.exit_code != 0
    assert connections == 0


@pytest.mark.anyio
async def test_the_environment_carries_nothing_from_the_service(
    sandbox: ConfinedSandbox, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")

    result = await sandbox.run(["/usr/bin/env"], cwd=root)

    assert result.exit_code == 0
    assert "OPENAI_API_KEY" not in result.output


@pytest.mark.anyio
async def test_a_timeout_kills_everything_the_command_started(
    sandbox: ConfinedSandbox, root: Path
) -> None:
    result = await sandbox.run(
        ["/bin/sh", "-c", 'sleep 60 & echo "$$ $!" > pids; wait'],
        cwd=root,
        timeout=1,
    )

    assert result.timed_out is True
    await asyncio.sleep(0.3)
    for pid in (root / "pids").read_text().split():
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid), 0)


@pytest.mark.anyio
async def test_this_repos_python_runs_with_its_interpreter_as_a_read_path() -> None:
    # The .venv's interpreter lives outside the repo (uv installs it under
    # the home directory), which is what sandbox_read_paths is for.
    interpreter = Path(sys.base_prefix).resolve()
    sandbox = platform_backend()(
        SandboxPolicy(Workspace(REPO), read_paths=(interpreter,))
    )

    result = await sandbox.run(
        [".venv/bin/python", "-c", "import pytest; print(pytest.__version__)"],
        cwd=REPO,
    )

    assert result.exit_code == 0, result.output
