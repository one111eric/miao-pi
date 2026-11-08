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
from app.agent.sandbox.runner import PIPE_CLOSE_SECONDS
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


@pytest.mark.anyio
async def test_a_command_cant_leave_its_process_group_under_seatbelt(
    sandbox: ConfinedSandbox, root: Path
) -> None:
    if not isinstance(sandbox, SeatbeltSandbox):
        pytest.skip("On Linux the PID namespace ends such processes instead.")
    # Leaving the group would escape the timeout's kill. (posix_spawn's
    # options still can; the sweep ends those, as the next test shows.) The
    # calls come from a forked child: the command itself already leads its
    # own session, so both would fail for it anyway.
    probe = (
        "use POSIX; $| = 1;"
        "if (fork() == 0) {"
        " print POSIX::setsid() == -1 ? qq(setsid blocked\\n) : qq(setsid ESCAPED\\n);"
        " exit 0 }"
        "wait;"
        "if (fork() == 0) {"
        " print POSIX::setpgid(0, 0) ? qq(setpgid ESCAPED\\n) : qq(setpgid blocked\\n);"
        " exit 0 }"
        "wait;"
    )

    result = await sandbox.run(["/usr/bin/perl", "-e", probe], cwd=root)

    assert result.output == "setsid blocked\nsetpgid blocked\n"


async def ended(pid: int) -> bool:
    """Whether a process is gone, within a second."""
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.mark.anyio
@pytest.mark.parametrize("option", ["setsid=True", "setpgroup=0"])
async def test_a_process_spawned_out_of_the_group_is_swept_under_seatbelt(
    root: Path, option: str, caplog: pytest.LogCaptureFixture
) -> None:
    if not SeatbeltSandbox.available():
        pytest.skip("On Linux the PID namespace ends such processes instead.")
    interpreter = Path(sys.base_prefix).resolve()
    sandbox = SeatbeltSandbox(SandboxPolicy(Workspace(root), read_paths=(interpreter,)))
    # The spawned process lands outside the command's process group, so the
    # group kill misses it, and it holds the output pipe open.
    spawn = (
        "import os\n"
        f"pid = os.posix_spawn('/bin/sleep', ['sleep', '60'], {{}}, {option})\n"
        "open('pid', 'w').write(str(pid))\n"
        "print('spawned')\n"
    )

    result = await sandbox.run(
        [str(interpreter / "bin" / "python3"), "-c", spawn], cwd=root
    )

    assert result.output == "spawned\n"
    assert result.duration_ms < PIPE_CLOSE_SECONDS * 1000
    assert await ended(int((root / "pid").read_text()))
    assert "Ended processes that left the command's process group" in caplog.text


@pytest.mark.anyio
async def test_the_sweep_leaves_other_commands_alone(
    sandbox: ConfinedSandbox, root: Path
) -> None:
    if not isinstance(sandbox, SeatbeltSandbox):
        pytest.skip("Only Seatbelt sweeps.")
    other = asyncio.create_task(sandbox.run(["/bin/sleep", "1"], cwd=root))
    await asyncio.sleep(0.3)

    # This command's sweep runs while the other one is still going.
    await sandbox.run(["true"], cwd=root)

    assert (await other).exit_code == 0


@pytest.mark.anyio
async def test_a_denied_name_reaches_only_what_it_points_to(
    sandbox: ConfinedSandbox, root: Path
) -> None:
    # The deny list protects files by where they are: a link named like a
    # denied file reaches its (allowed) target, which is readable under its
    # own name anyway, and a link to a denied file gets nothing.
    (root / ".env.link").symlink_to("notes.txt")
    (root / "alias").symlink_to(".env")

    through_denied_name = await sandbox.run(["cat", ".env.link"], cwd=root)
    to_denied_file = await sandbox.run(["cat", "alias"], cwd=root)

    assert (through_denied_name.exit_code, through_denied_name.output) == (0, "hello")
    assert SECRET not in to_denied_file.output
