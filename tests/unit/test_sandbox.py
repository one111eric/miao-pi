import asyncio
import logging
import os
import re
import shutil
import signal
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.agent.sandbox import (
    CommandResult,
    CommandSettings,
    ConfinedSandbox,
    ContainmentError,
    RunDir,
    Sandbox,
    SandboxConfigError,
    SandboxPolicy,
    resolve_read_paths,
    seatbelt_sweep,
    startup,
)
from app.agent.sandbox.bubblewrap import BubblewrapSandbox
from app.agent.sandbox.runner import BASE_ENV, OFFLINE_ENV, PIPE_CLOSE_SECONDS
from app.agent.sandbox.seatbelt import (
    SANDBOX_EXEC,
    SeatbeltSandbox,
    build_profile,
    deny_regexes,
)
from app.agent.workspace import Workspace, WorkspaceAccessError
from common.config import SandboxBackend, SvcSettings
from common.logging_helper import LOGGING_FORMAT, RequestFormatter
from tests.fakes import FakeSandbox, UnconfinedSandbox


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root.resolve()


@pytest.fixture
def unconfined(root: Path) -> UnconfinedSandbox:
    return UnconfinedSandbox(SandboxPolicy(Workspace(root)))


def contains(sequence: Sequence[str], part: Sequence[str]) -> bool:
    """Whether ``part`` appears in ``sequence`` as consecutive items."""
    size = len(part)
    return any(
        list(sequence[i : i + size]) == list(part)
        for i in range(len(sequence) - size + 1)
    )


async def gone(pid: int) -> bool:
    """Whether a process has exited and been reaped, within a second."""
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        await asyncio.sleep(0.05)
    return False


# Policy


def test_read_paths_are_expanded_and_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / ".tools").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    assert resolve_read_paths(["~/.tools"]) == ((home / ".tools").resolve(),)
    with pytest.raises(SandboxConfigError, match="doesn't exist"):
        resolve_read_paths(["~/missing"])
    for exposes_home in ("~", str(tmp_path), "/"):
        with pytest.raises(SandboxConfigError, match="whole home directory"):
            resolve_read_paths([exposes_home])


# Seatbelt


@pytest.mark.parametrize(
    ("pattern", "path", "denied"),
    [
        (".env*", ".env", True),
        (".env*", "sub/.ENV.local", True),
        (".env*", "env.txt", False),
        (".env*", "a.env", False),
        (".git", "x/.Git/config", True),
        (".git", ".github/workflows/ci.yml", False),
        ("*.pem", "certs/server.PEM", True),
        ("*.pem", "pem", False),
        ("config/private", "config/private/key", True),
        ("config/private", "CONFIG/Private", True),
        ("config/private", "sub/config/private", False),
        ("secret[0-9]", "secret7", True),
        ("secret[0-9]", "secretx", False),
        ("[!a]x.txt", "bx.txt", True),
        ("[!a]x.txt", "ax.txt", False),
    ],
)
def test_deny_regexes_match_what_the_workspace_denies(
    root: Path, pattern: str, path: str, denied: bool
) -> None:
    workspace = Workspace(root, denied_paths=(pattern,))
    full = f"{root}/{path}"

    regex_denies = any(
        re.fullmatch(regex, full) for regex in deny_regexes(SandboxPolicy(workspace))
    )

    assert regex_denies is denied
    # The same answer as the read tools give.
    assert workspace.is_denied(Path(full)) is denied


def test_only_a_safe_regex_subset_is_emitted(root: Path) -> None:
    workspace = Workspace(root, denied_paths=("*.pem", "config/priv?te", "k[a-z0-9]y"))

    for regex in deny_regexes(SandboxPolicy(workspace)):
        assert re.search(r"(?<!\\)[{}+|]", regex) is None, regex


def test_unsupported_class_characters_are_refused(root: Path) -> None:
    with pytest.raises(SandboxConfigError, match="unsupported deny pattern"):
        SeatbeltSandbox(SandboxPolicy(Workspace(root, denied_paths=("key[$]",))))


def test_the_profile_denies_by_default_and_puts_denies_last(root: Path) -> None:
    workspace = Workspace(root)
    profile = build_profile(SandboxPolicy(workspace, read_paths=(root,)))
    lines = profile.splitlines()

    assert "(deny default)" in lines
    assert "network" not in profile
    assert "mach-lookup" not in profile
    assert "(allow signal (target same-sandbox))" in lines
    # Leaving the process group would escape the timeout's kill.
    assert "(deny syscall-unix (syscall-number SYS_setsid SYS_setpgid))" in lines
    # How the sweep recognizes this run's processes.
    assert '(allow file-read-data (literal (param "MARKER")))' in lines
    assert '(subpath (param "READ_0"))' in profile
    assert profile.count('(regex #"') == len(workspace.denied_patterns)
    # In Seatbelt the last matching rule wins.
    last_allow = max(i for i, line in enumerate(lines) if line.startswith("(allow"))
    first_deny = min(i for i, line in enumerate(lines) if line.startswith("(deny file"))
    assert first_deny > last_allow


def test_seatbelt_passes_paths_as_parameters(root: Path) -> None:
    tools = root.parent / "tools"
    sandbox = SeatbeltSandbox(SandboxPolicy(Workspace(root), read_paths=(tools,)))

    run = RunDir(root.parent / "run")
    command = sandbox.confine(["make", "test"], cwd=root, run=run)

    assert command[:3] == [str(SANDBOX_EXEC), "-p", sandbox.profile]
    assert contains(command, ["-D", f"WORKSPACE={root}"])
    assert contains(command, ["-D", f"SCRATCH={run.path / 'home'}"])
    assert contains(command, ["-D", f"MARKER={run.path / 'marker'}"])
    assert contains(command, ["-D", f"READ_0={tools}"])
    assert command[-2:] == ["make", "test"]


def test_seatbelt_refuses_a_root_it_cant_quote(tmp_path: Path) -> None:
    quoted = tmp_path / 'say "hi"'
    quoted.mkdir()

    with pytest.raises(SandboxConfigError, match="quote"):
        SeatbeltSandbox(SandboxPolicy(Workspace(quoted)))


def test_seatbelt_keeps_its_marker_where_commands_cant_change_it(root: Path) -> None:
    sandbox = SeatbeltSandbox(SandboxPolicy(Workspace(root)))

    with sandbox.new_run() as run:
        assert (run.path / "marker").is_file()
        assert (run.path / "canary").is_file()
        # Only scratch is writable inside the sandbox; the files sit beside it.
        assert list(run.scratch.iterdir()) == []
        assert run.scratch.parent == run.path
    assert not run.path.exists()


class FakeProcesses:
    """Stands in for libSystem: each process's group and what it may read."""

    def __init__(self) -> None:
        self.table: dict[int, tuple[bool, set[bytes], int]] = {}
        self.unkillable: set[int] = set()
        self.respawn = False
        self.killed: list[int] = []

    def add(self, pid: int, *, sandboxed: bool, reads: set[bytes], pgid: int) -> None:
        self.table[pid] = (sandboxed, reads, pgid)

    def pids(self) -> list[int]:
        return list(self.table)

    def sandboxed(self, pid: int) -> bool:
        return self.table[pid][0]

    def may_read(self, pid: int, path: bytes) -> bool:
        return path in self.table[pid][1]

    def getpgid(self, pid: int) -> int:
        return self.table[pid][2]

    def kill(self, pid: int, sig: int) -> None:
        if pid not in self.table:
            raise ProcessLookupError(pid)
        if pid in self.unkillable:
            raise PermissionError(pid)
        self.killed.append(pid)
        entry = self.table.pop(pid)
        if self.respawn:
            # A fork loop: every kill finds a new child already started.
            self.table[pid + 1000] = entry


@pytest.fixture
def processes(monkeypatch: pytest.MonkeyPatch) -> FakeProcesses:
    fake = FakeProcesses()
    monkeypatch.setattr(seatbelt_sweep, "_libsystem", lambda: fake)
    monkeypatch.setattr(os, "kill", fake.kill)
    monkeypatch.setattr(os, "getpgid", fake.getpgid)
    monkeypatch.setattr(seatbelt_sweep, "_PASS_PAUSE_SECONDS", 0)
    return fake


def test_the_sweep_ends_only_this_runs_processes(
    tmp_path: Path, processes: FakeProcesses, caplog: pytest.LogCaptureFixture
) -> None:
    run_dir = tmp_path / "run"
    marker = os.fsencode(run_dir / "marker")
    canary = os.fsencode(run_dir / "canary")
    other_marker = os.fsencode(tmp_path / "other" / "marker")
    processes.add(10, sandboxed=True, reads={marker}, pgid=100)  # dying in group
    processes.add(11, sandboxed=True, reads={marker}, pgid=11)  # escaped
    processes.add(12, sandboxed=False, reads={marker, canary}, pgid=12)
    processes.add(13, sandboxed=True, reads={other_marker}, pgid=13)  # other run
    processes.add(14, sandboxed=True, reads={marker, canary}, pgid=14)  # other app

    seatbelt_sweep.end_run_processes(run_dir, pgid=100)

    assert sorted(processes.killed) == [10, 11]
    assert sorted(processes.table) == [12, 13, 14]
    messages = {record.getMessage(): record for record in caplog.records}
    escaped = messages["Ended processes that left the command's process group"]
    assert getattr(escaped, "count") == 1  # 10 was still in the group


def test_the_sweep_fails_when_a_process_cant_be_signalled(
    tmp_path: Path, processes: FakeProcesses
) -> None:
    marker = os.fsencode(tmp_path / "marker")
    processes.add(10, sandboxed=True, reads={marker}, pgid=10)
    processes.add(11, sandboxed=True, reads={marker}, pgid=11)  # setuid, say
    processes.unkillable = {11}

    with pytest.raises(ContainmentError, match="1 can't be signalled"):
        seatbelt_sweep.end_run_processes(tmp_path, pgid=10)

    assert processes.killed == [10]


def test_the_sweep_fails_when_a_fork_loop_outlasts_it(
    tmp_path: Path, processes: FakeProcesses, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(seatbelt_sweep, "MAX_PASSES", 3)
    marker = os.fsencode(tmp_path / "marker")
    processes.add(10, sandboxed=True, reads={marker}, pgid=10)
    processes.respawn = True

    with pytest.raises(ContainmentError, match="1 outlasted 3 passes"):
        seatbelt_sweep.end_run_processes(tmp_path, pgid=10)

    assert len(processes.killed) == 3


def test_a_sweep_that_ends_everything_on_its_last_pass_succeeds(
    tmp_path: Path, processes: FakeProcesses, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(seatbelt_sweep, "MAX_PASSES", 1)
    marker = os.fsencode(tmp_path / "marker")
    processes.add(10, sandboxed=True, reads={marker}, pgid=10)

    seatbelt_sweep.end_run_processes(tmp_path, pgid=10)

    assert processes.killed == [10]


# Bubblewrap


def test_bubblewrap_masks_denied_paths_and_unshares_everything(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name, path=None: "/usr/bin/bwrap")
    for name in (
        ".env",
        "sub/.ENV.local",
        ".git/config",
        ".git/objects/ab",
        "keep.txt",
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x", encoding="utf-8")
    (root / "alias").symlink_to(root / ".env")
    (root / ".env.link").symlink_to(root / "keep.txt")
    sandbox = BubblewrapSandbox(SandboxPolicy(Workspace(root)))
    run = RunDir(root.parent / "run")
    scratch = run.scratch

    command = sandbox.confine(["make", "test"], cwd=root, run=run)

    assert command[:4] == [
        "/usr/bin/bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
    ]
    assert contains(command, ["--bind", str(root), str(root)])
    assert contains(command, ["--bind", str(scratch), str(scratch)])
    assert contains(command, ["--ro-bind", "/dev/null", f"{root}/.env"])
    assert contains(command, ["--ro-bind", "/dev/null", f"{root}/sub/.ENV.local"])
    assert contains(command, ["--tmpfs", f"{root}/.git"])
    # Nothing inside a masked directory, and no symlinks, are mounted over.
    assert not any(arg.startswith(f"{root}/.git/") for arg in command)
    assert f"{root}/alias" not in command
    # A denied-name symlink reaches only its target, readable under its own
    # name anyway; mounting over the link would mount over the target.
    assert f"{root}/.env.link" not in command
    # Masks come after the workspace bind, so they sit on top of it.
    assert command.index("--tmpfs") > command.index(str(root))
    assert command[-5:] == ["--chdir", str(root), "--", "make", "test"]


# Running commands (shared by every backend)


@pytest.mark.anyio
async def test_commands_get_only_a_minimal_environment(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    sandbox = UnconfinedSandbox(
        SandboxPolicy(Workspace(root)), CommandSettings(env={"EXTRA": "1"})
    )

    result = await sandbox.run(["/usr/bin/env"], cwd=root)

    names = {line.split("=", 1)[0] for line in result.output.splitlines()}
    assert names == {
        *BASE_ENV,
        *OFFLINE_ENV,
        "PATH",
        "HOME",
        "TMPDIR",
        "EXTRA",
    }
    assert "sk-should-not-leak" not in result.output


@pytest.mark.anyio
async def test_home_is_a_scratch_directory_removed_afterwards(
    root: Path, unconfined: UnconfinedSandbox
) -> None:
    result = await unconfined.run(["/bin/sh", "-c", 'echo "$HOME"'], cwd=root)

    home = Path(result.output.strip())
    assert home.name == "home"
    assert home.parent.name.startswith("miao-pi-cmd-")
    assert not home.is_relative_to(root)
    assert not home.parent.exists()


@pytest.mark.anyio
async def test_a_timeout_kills_the_whole_process_tree(
    root: Path, unconfined: UnconfinedSandbox
) -> None:
    result = await unconfined.run(
        ["/bin/sh", "-c", 'sleep 60 & echo "$$ $!" > pids; wait'],
        cwd=root,
        timeout=0.5,
    )

    assert result.timed_out is True
    assert result.exit_code is None
    for pid in (root / "pids").read_text().split():
        assert await gone(int(pid))


@pytest.mark.anyio
async def test_processes_left_behind_are_killed_without_waiting(
    root: Path, unconfined: UnconfinedSandbox
) -> None:
    result = await unconfined.run(
        ["/bin/sh", "-c", 'sleep 60 & echo "$!" > pid'], cwd=root, timeout=10
    )

    assert result.exit_code == 0
    assert result.duration_ms < 5000
    assert await gone(int((root / "pid").read_text()))


@pytest.mark.anyio
async def test_a_process_that_leaves_its_group_cant_stall_the_call(
    root: Path, unconfined: UnconfinedSandbox
) -> None:
    # It calls setsid, so the group kill misses it, and it keeps the output
    # pipe open; the call must still return instead of waiting on the pipe.
    escape = (
        "import os, sys, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        "    open('pid', 'w').write(str(os.getpid()))\n"
        "    time.sleep(60)\n"
    )
    try:
        result = await asyncio.wait_for(
            unconfined.run([sys.executable, "-c", escape], cwd=root, timeout=10),
            timeout=8,
        )
        assert result.exit_code == 0
    finally:
        for _ in range(50):
            if (root / "pid").exists() and (root / "pid").read_text():
                os.kill(int((root / "pid").read_text()), signal.SIGKILL)
                break
            await asyncio.sleep(0.02)


class RecordsSweeps(UnconfinedSandbox):
    """Records each sweep: whether the run directory existed, and the group."""

    def __init__(self, policy: SandboxPolicy) -> None:
        super().__init__(policy)
        self.sweeps: list[tuple[bool, int]] = []

    def sweep(self, run: RunDir, pgid: int) -> None:
        self.sweeps.append((run.path.is_dir(), pgid))


class EndsEscapedProcess(RecordsSweeps):
    """Its sweep also ends the process whose id the command wrote to ``pid``."""

    def sweep(self, run: RunDir, pgid: int) -> None:
        super().sweep(run, pgid)
        pid_file = self.policy.workspace.root / "pid"
        for _ in range(100):
            if pid_file.exists() and pid_file.read_text():
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
                return
            time.sleep(0.01)


@pytest.mark.anyio
async def test_the_sweep_runs_before_waiting_for_output(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The escaped process holds the output pipe open. The sweep runs right
    # after the group kill, while the run directory still exists, so ending
    # it there means the call doesn't wait out the pipe grace period.
    sandbox = EndsEscapedProcess(SandboxPolicy(Workspace(root)))
    escape = (
        "import os, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        "    open('pid', 'w').write(str(os.getpid()))\n"
        "    time.sleep(60)\n"
    )

    result = await sandbox.run([sys.executable, "-c", escape], cwd=root, timeout=10)

    assert result.exit_code == 0
    assert result.duration_ms < PIPE_CLOSE_SECONDS * 1000
    assert "Command output stayed open" not in caplog.text
    assert len(sandbox.sweeps) == 1
    run_dir_existed, pgid = sandbox.sweeps[0]
    assert run_dir_existed and pgid > 0
    assert await gone(int((root / "pid").read_text()))


@pytest.mark.anyio
async def test_cancelling_the_caller_sweeps_too(root: Path) -> None:
    sandbox = RecordsSweeps(SandboxPolicy(Workspace(root)))
    run = asyncio.create_task(sandbox.run(["/bin/sleep", "60"], cwd=root))
    await asyncio.sleep(0.2)
    run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run
    assert [existed for existed, _ in sandbox.sweeps] == [True]


UNSIGNALLABLE = "the command left processes running: 1 can't be signalled"


class FailingSweep(UnconfinedSandbox):
    """A sweep that raises ``error``, optionally after a pause."""

    def __init__(
        self, policy: SandboxPolicy, error: Exception | None, pause: float = 0
    ) -> None:
        super().__init__(policy)
        self.error = error
        self.pause = pause
        self.started = threading.Event()
        self.threads: list[int] = []
        # Whether the run directory still existed when each sweep finished.
        self.finished: list[bool] = []

    def sweep(self, run: RunDir, pgid: int) -> None:
        self.threads.append(threading.get_ident())
        self.started.set()
        time.sleep(self.pause)
        self.finished.append(run.path.is_dir())
        if self.error is not None:
            raise self.error


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ContainmentError(UNSIGNALLABLE), id="processes-left"),
        pytest.param(OSError("proc_listallpids failed"), id="sweep-crashed"),
    ],
)
async def test_a_failed_sweep_fails_the_command_and_turns_commands_off(
    root: Path, error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    sandbox = FailingSweep(SandboxPolicy(Workspace(root)), error)

    with pytest.raises(ContainmentError):
        await sandbox.run(["true"], cwd=root)
    with pytest.raises(ContainmentError, match="until the service restarts"):
        await sandbox.run(["/bin/sh", "-c", "echo ran > ran.txt"], cwd=root)

    assert not (root / "ran.txt").exists()
    assert sandbox.disabled_reason is not None
    assert str(error) in sandbox.disabled_reason
    assert len(sandbox.finished) == 1
    assert "Sandbox commands turned off" in caplog.text


@pytest.mark.anyio
async def test_cancelling_during_the_sweep_waits_for_it_once(root: Path) -> None:
    sandbox = FailingSweep(SandboxPolicy(Workspace(root)), None, pause=0.3)
    run = asyncio.create_task(sandbox.run(["true"], cwd=root))
    await asyncio.to_thread(sandbox.started.wait, 5)
    # Twice: the second lands while the run is already waiting for the sweep.
    run.cancel()
    await asyncio.sleep(0.05)
    run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run
    # One sweep, in a thread (not blocking the event loop), finished before
    # the run returned and before its run directory was removed.
    assert len(sandbox.threads) == 1
    assert sandbox.threads[0] != threading.get_ident()
    assert sandbox.finished == [True]
    assert sandbox.disabled_reason is None


@pytest.mark.anyio
async def test_a_sweep_that_fails_after_a_cancel_still_turns_commands_off(
    root: Path,
) -> None:
    sandbox = FailingSweep(
        SandboxPolicy(Workspace(root)), ContainmentError(UNSIGNALLABLE), pause=0.2
    )
    run = asyncio.create_task(sandbox.run(["true"], cwd=root))
    await asyncio.to_thread(sandbox.started.wait, 5)
    run.cancel()

    # The cancellation wins over the sweep's error, but it's still recorded.
    with pytest.raises(asyncio.CancelledError):
        await run
    assert sandbox.disabled_reason == UNSIGNALLABLE


class FirstSweepFails(RecordsSweeps):
    """The first command's sweep fails; later ones succeed."""

    def sweep(self, run: RunDir, pgid: int) -> None:
        super().sweep(run, pgid)
        if len(self.sweeps) == 1:
            raise ContainmentError(UNSIGNALLABLE)


@pytest.mark.anyio
async def test_a_failed_sweep_stops_commands_already_running(root: Path) -> None:
    sandbox = FirstSweepFails(SandboxPolicy(Workspace(root)))
    running = asyncio.create_task(
        sandbox.run(
            ["/bin/sh", "-c", 'echo "$$" > pid; exec sleep 60'], cwd=root, timeout=30
        )
    )
    for _ in range(100):
        if (root / "pid").exists() and (root / "pid").read_text().strip():
            break
        await asyncio.sleep(0.02)

    with pytest.raises(ContainmentError):
        await sandbox.run(["true"], cwd=root)

    # The running command is stopped at once, not left to finish.
    with pytest.raises(ContainmentError, match="until the service restarts"):
        await asyncio.wait_for(running, timeout=5)
    assert await gone(int((root / "pid").read_text()))
    assert len(sandbox.sweeps) == 2


class FailsWhileSweeping(RecordsSweeps):
    """Another command's sweep fails while this one's runs."""

    def sweep(self, run: RunDir, pgid: int) -> None:
        super().sweep(run, pgid)
        self.disabled_reason = UNSIGNALLABLE


@pytest.mark.anyio
async def test_a_command_that_ends_after_a_failed_sweep_reports_it(
    root: Path,
) -> None:
    # It exited before the runner noticed, so only the final check can
    # catch it.
    sandbox = FailsWhileSweeping(SandboxPolicy(Workspace(root)))

    with pytest.raises(ContainmentError, match="until the service restarts"):
        await sandbox.run(["true"], cwd=root)


@pytest.mark.anyio
async def test_cancelling_the_caller_kills_the_command(
    root: Path, unconfined: UnconfinedSandbox
) -> None:
    run = asyncio.create_task(
        unconfined.run(
            ["/bin/sh", "-c", 'sleep 60 & echo "$$ $!" > pids; wait'], cwd=root
        )
    )
    for _ in range(100):
        if (root / "pids").exists() and (root / "pids").read_text().strip():
            break
        await asyncio.sleep(0.02)
    run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run
    for pid in (root / "pids").read_text().split():
        assert await gone(int(pid))


@pytest.mark.anyio
async def test_output_keeps_the_tail_and_interleaves_stderr(root: Path) -> None:
    small = UnconfinedSandbox(
        SandboxPolicy(Workspace(root)), CommandSettings(max_output_bytes=10)
    )
    roomy = UnconfinedSandbox(SandboxPolicy(Workspace(root)))

    cut = await small.run(["printf", "abcdefghijklmnopqrstuvwxyz"], cwd=root)
    both = await roomy.run(
        ["/bin/sh", "-c", "echo out; echo err >&2; exit 3"], cwd=root
    )

    assert (cut.output, cut.truncated, cut.output_bytes) == ("qrstuvwxyz", True, 26)
    assert (both.output, both.truncated, both.exit_code) == ("out\nerr\n", False, 3)


@pytest.mark.anyio
async def test_the_default_timeout_comes_from_settings(root: Path) -> None:
    sandbox = UnconfinedSandbox(
        SandboxPolicy(Workspace(root)), CommandSettings(timeout=0.3)
    )

    result = await sandbox.run(["sleep", "30"], cwd=root)

    assert result.timed_out is True


@pytest.mark.anyio
async def test_bad_commands_and_directories_are_refused(
    root: Path, unconfined: UnconfinedSandbox
) -> None:
    (root / ".git").mkdir()
    (root / "file.txt").write_text("x", encoding="utf-8")

    for argv in ([], [""], ["-rf"]):
        with pytest.raises(ValueError, match="program name"):
            await unconfined.run(argv, cwd=root)
    with pytest.raises(WorkspaceAccessError):
        await unconfined.run(["true"], cwd=root.parent)
    with pytest.raises(WorkspaceAccessError):
        await unconfined.run(["true"], cwd=root / ".git")
    with pytest.raises(ValueError, match="not a directory"):
        await unconfined.run(["true"], cwd=root / "file.txt")


@pytest.mark.anyio
async def test_the_fake_sandbox_records_calls(root: Path) -> None:
    result = CommandResult(
        exit_code=0,
        output="ok",
        truncated=False,
        output_bytes=2,
        timed_out=False,
        duration_ms=1.0,
    )
    sandbox: Sandbox = FakeSandbox(result)

    assert await sandbox.run(["make"], cwd=root, timeout=5) is result
    assert isinstance(sandbox, FakeSandbox)
    assert sandbox.calls == [(["make"], root, 5)]


# Logging

COMMAND_LOGS = {"Command completed", "Command failed", "Command cancelled"}


def command_logs(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() in COMMAND_LOGS]


@pytest.mark.anyio
async def test_a_completed_command_logs_its_full_output_size(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="app.agent.sandbox")
    small = UnconfinedSandbox(
        SandboxPolicy(Workspace(root)), CommandSettings(max_output_bytes=10)
    )

    await small.run(["printf", "abcdefghijklmnopqrstuvwxyz"], cwd=root)

    [record] = command_logs(caplog)
    assert record.getMessage() == "Command completed"
    assert getattr(record, "command") == "printf"
    assert getattr(record, "truncated") is True
    assert getattr(record, "output_bytes") == 26


@pytest.mark.anyio
async def test_a_cancelled_command_logs_that_it_was_cancelled(
    root: Path, unconfined: UnconfinedSandbox, caplog: pytest.LogCaptureFixture
) -> None:
    run = asyncio.create_task(unconfined.run(["/bin/sleep", "60"], cwd=root))
    await asyncio.sleep(0.2)
    run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run

    [record] = command_logs(caplog)
    assert record.getMessage() == "Command cancelled"
    assert getattr(record, "command") == "sleep"
    assert getattr(record, "duration_ms") >= 200


@pytest.mark.anyio
async def test_a_failed_command_logs_why_but_never_the_path(
    root: Path, unconfined: UnconfinedSandbox, caplog: pytest.LogCaptureFixture
) -> None:
    sandbox = FailingSweep(
        SandboxPolicy(Workspace(root)), ContainmentError(UNSIGNALLABLE)
    )

    with pytest.raises(ContainmentError):
        await sandbox.run(["true"], cwd=root)
    with pytest.raises(ContainmentError):
        await sandbox.run(["true"], cwd=root)
    with pytest.raises(WorkspaceAccessError):
        await unconfined.run(["true"], cwd=root / ".git")

    contained, refused, rejected = command_logs(caplog)
    for record in (contained, refused, rejected):
        assert record.getMessage() == "Command failed"
    assert getattr(contained, "reason") == UNSIGNALLABLE
    assert getattr(refused, "reason").startswith("commands are off")
    # Other errors' messages can quote the requested path; only the type.
    assert getattr(rejected, "error_type") == "WorkspaceAccessError"
    assert not hasattr(rejected, "reason")


class ContextLines(logging.Handler):
    """Each record as the service's text formatter writes it, context included."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(RequestFormatter(LOGGING_FORMAT))
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@pytest.mark.anyio
async def test_the_self_tests_commands_are_marked_as_such(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    unconfined: UnconfinedSandbox,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.agent.sandbox")
    handler = ContextLines()
    logger = logging.getLogger("app.agent.sandbox")
    logger.addHandler(handler)
    monkeypatch.setattr(startup, "_choose", lambda setting: UnconfinedSandbox)
    try:
        await startup.make_sandbox(
            SvcSettings(sandbox_backend=SandboxBackend.AUTO), Workspace(root)
        )
        during = [line for line in handler.lines if "msg=Command " in line]
        handler.lines.clear()
        await unconfined.run(["true"], cwd=root)
    finally:
        logger.removeHandler(handler)

    assert during
    assert all("sandbox_self_test=True" in line for line in during)
    assert "sandbox_self_test" not in handler.lines[0]


# Startup


class BlocksOnlyReads(UnconfinedSandbox):
    """A broken backend: it stops `cat` but lets every write through."""

    def confine(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        run: RunDir,
    ) -> list[str]:
        return ["false"] if argv[0] == "cat" else list(argv)


class FailsToStart(UnconfinedSandbox):
    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout: float | None = None,
    ) -> CommandResult:
        raise FileNotFoundError("sandbox launcher is gone")


@pytest.mark.anyio
async def test_the_self_test_catches_a_sandbox_that_confines_nothing(
    unconfined: UnconfinedSandbox,
) -> None:
    assert await startup.self_test(unconfined) == "a denied file was readable"


@pytest.mark.anyio
async def test_the_self_test_catches_writes_that_get_through(root: Path) -> None:
    sandbox = BlocksOnlyReads(SandboxPolicy(Workspace(root)))

    assert await startup.self_test(sandbox) == "a denied file was writable"


@pytest.mark.anyio
async def test_the_self_test_leaves_the_real_workspace_alone(
    root: Path, unconfined: UnconfinedSandbox
) -> None:
    await startup.self_test(unconfined)

    assert list(root.iterdir()) == []


@pytest.mark.anyio
async def test_startup_disables_commands_without_a_working_sandbox(
    root: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace = Workspace(root)
    monkeypatch.setattr(SeatbeltSandbox, "available", staticmethod(lambda: False))
    monkeypatch.setattr(BubblewrapSandbox, "available", staticmethod(lambda: False))

    off = await startup.make_sandbox(
        SvcSettings(sandbox_backend=SandboxBackend.OFF), workspace
    )
    none_here = await startup.make_sandbox(
        SvcSettings(sandbox_backend=SandboxBackend.AUTO), workspace
    )
    named = await startup.make_sandbox(
        SvcSettings(sandbox_backend=SandboxBackend.BUBBLEWRAP), workspace
    )
    monkeypatch.setattr(startup, "_choose", lambda setting: UnconfinedSandbox)
    failing = await startup.make_sandbox(
        SvcSettings(sandbox_backend=SandboxBackend.AUTO), workspace
    )

    assert (off.sandbox, off.status) == (None, "off")
    assert (none_here.sandbox, none_here.status) == (
        None,
        "unavailable: no sandbox backend for this platform",
    )
    assert named.status == "unavailable: bubblewrap isn't available here"
    assert (failing.sandbox, failing.status) == (
        None,
        "unavailable: a denied file was readable",
    )
    assert "Sandbox self-test failed" in caplog.text
    # A warning when there's no backend, but none when commands were turned off.
    unavailable = [r for r in caplog.records if r.getMessage() == "Sandbox unavailable"]
    assert [getattr(r, "reason") for r in unavailable] == [
        "no sandbox backend for this platform",
        "bubblewrap isn't available here",
    ]


@pytest.mark.anyio
async def test_a_working_sandbox_reports_its_read_paths(
    tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir()

    async def passes(sandbox: ConfinedSandbox) -> str | None:
        return None

    monkeypatch.setattr(startup, "_choose", lambda setting: UnconfinedSandbox)
    monkeypatch.setattr(startup, "self_test", passes)
    setup = await startup.make_sandbox(
        SvcSettings(
            sandbox_backend=SandboxBackend.AUTO, sandbox_read_paths=(str(toolchain),)
        ),
        Workspace(root),
    )

    assert setup.status == "unconfined"
    assert setup.read_paths == (toolchain.resolve(),)


@pytest.mark.anyio
async def test_backend_errors_disable_commands_instead_of_crashing(
    root: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = SvcSettings(sandbox_backend=SandboxBackend.AUTO)

    monkeypatch.setattr(startup, "_choose", lambda setting: FailsToStart)
    gone = await startup.make_sandbox(settings, Workspace(root))
    # A crash may be a bug, so its traceback is kept.
    [crashed] = [
        r for r in caplog.records if r.getMessage() == "Sandbox self-test failed"
    ]
    assert crashed.exc_info is not None
    monkeypatch.setattr(startup, "_choose", lambda setting: SeatbeltSandbox)
    inexpressible = await startup.make_sandbox(
        settings, Workspace(root, denied_paths=("key[$]",))
    )

    assert (gone.sandbox, gone.status) == (
        None,
        "unavailable: FileNotFoundError: sandbox launcher is gone",
    )
    assert inexpressible.sandbox is None
    assert inexpressible.status.startswith(
        "unavailable: SandboxConfigError: unsupported deny pattern"
    )


@pytest.mark.anyio
async def test_a_bad_read_path_fails_startup(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(startup, "_choose", lambda setting: UnconfinedSandbox)
    settings = SvcSettings(
        sandbox_backend=SandboxBackend.AUTO, sandbox_read_paths=("/missing/x",)
    )

    with pytest.raises(SandboxConfigError):
        await startup.make_sandbox(settings, Workspace(root))


def test_settings_accept_every_backend_name() -> None:
    assert {backend.value for backend in SandboxBackend} == {
        "auto",
        "seatbelt",
        "bubblewrap",
        "off",
    }
