import asyncio
import os
import re
import shutil
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.agent.sandbox import (
    CommandResult,
    CommandSettings,
    Sandbox,
    SandboxConfigError,
    SandboxPolicy,
    resolve_read_paths,
    startup,
)
from app.agent.sandbox.bubblewrap import BubblewrapSandbox
from app.agent.sandbox.runner import BASE_ENV, OFFLINE_ENV
from app.agent.sandbox.seatbelt import (
    SANDBOX_EXEC,
    SeatbeltSandbox,
    build_profile,
    deny_regexes,
)
from app.agent.workspace import Workspace, WorkspaceAccessError
from common.config import SandboxBackend, SvcSettings
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


def test_unsupported_class_characters_fail_at_startup(root: Path) -> None:
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
    assert '(subpath (param "READ_0"))' in profile
    assert profile.count('(regex #"') == len(workspace.denied_patterns)
    # In Seatbelt the last matching rule wins.
    last_allow = max(i for i, line in enumerate(lines) if line.startswith("(allow"))
    first_deny = min(i for i, line in enumerate(lines) if line.startswith("(deny file"))
    assert first_deny > last_allow


def test_seatbelt_passes_paths_as_parameters(root: Path) -> None:
    tools = root.parent / "tools"
    sandbox = SeatbeltSandbox(SandboxPolicy(Workspace(root), read_paths=(tools,)))

    command = sandbox.confine(["make", "test"], cwd=root, scratch=root.parent / "s")

    assert command[:3] == [str(SANDBOX_EXEC), "-p", sandbox.profile]
    assert contains(command, ["-D", f"WORKSPACE={root}"])
    assert contains(command, ["-D", f"SCRATCH={root.parent / 's'}"])
    assert contains(command, ["-D", f"READ_0={tools}"])
    assert command[-2:] == ["make", "test"]


def test_seatbelt_refuses_a_root_it_cant_quote(tmp_path: Path) -> None:
    quoted = tmp_path / 'say "hi"'
    quoted.mkdir()

    with pytest.raises(SandboxConfigError, match="quote"):
        SeatbeltSandbox(SandboxPolicy(Workspace(quoted)))


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
    sandbox = BubblewrapSandbox(SandboxPolicy(Workspace(root)))
    scratch = root.parent / "scratch"

    command = sandbox.confine(["make", "test"], cwd=root, scratch=scratch)

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
    assert home.name.startswith("miao-pi-cmd-")
    assert not home.is_relative_to(root)
    assert not home.exists()


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

    assert (cut.output, cut.truncated) == ("qrstuvwxyz", True)
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
        exit_code=0, output="ok", truncated=False, timed_out=False, duration_ms=1.0
    )
    sandbox: Sandbox = FakeSandbox(result)

    assert await sandbox.run(["make"], cwd=root, timeout=5) is result
    assert isinstance(sandbox, FakeSandbox)
    assert sandbox.calls == [(["make"], root, 5)]


# Startup


@pytest.mark.anyio
async def test_the_self_test_catches_a_sandbox_that_confines_nothing(
    unconfined: UnconfinedSandbox,
) -> None:
    assert await startup.self_test(unconfined) == (
        "a file outside the workspace was readable"
    )


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
        "unavailable: a file outside the workspace was readable",
    )
    assert "Sandbox self-test failed" in caplog.text


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
