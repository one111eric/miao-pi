"""Ending a Seatbelt command's processes after they leave its process group.

posix_spawn's setsid and setpgroup options move the new process to another
session or group inside the spawn call, which Seatbelt can't filter, so the
runner's group kill misses it. The process is still in the command's
sandbox, though, and that's how this finds it: macOS's ``sandbox_check``
answers, for any process, whether its sandbox allows an operation.

Each run directory holds a marker file that the run's profile lets commands
read, next to a canary file it doesn't. Only that run's processes are
sandboxed, may read the marker, and may not read the canary:

- unsandboxed processes may read both;
- another run's processes may read only their own run's marker;
- another app's sandbox doesn't allow this marker.

A command can't move or delete either file, since only its scratch
directory is writable, and it can't add a profile that hides the marker:
macOS refuses to sandbox a sandboxed process again (``sandbox_apply:
Operation not permitted``).

``sandbox_check`` is private but long-standing (Chromium uses it), so a
macOS update could break it. ``SeatbeltSandbox.self_check`` proves at
startup that it still finds and ends such a process. When the sweep can't
end every process (one running as another user, such as a setuid program,
can't be signalled, and a fork loop could outlast it), it raises
``ContainmentError``. One limit remains: a process id freed and reused
between the check and the kill, a window of microseconds.
"""

import ctypes
import logging
import os
import signal
import time
from functools import cache
from pathlib import Path

from app.agent.sandbox.runner import ContainmentError

logger = logging.getLogger(__name__)

LIBSYSTEM = "/usr/lib/libSystem.B.dylib"
MARKER = "marker"
CANARY = "canary"

# A process forking in a loop takes a few passes; see docs/sandbox.md.
MAX_PASSES = 50
_PASS_PAUSE_SECONDS = 0.005
_FILTER_NONE = 0
_FILTER_PATH = 1


class SweepUnavailableError(RuntimeError):
    """The macOS calls the sweep relies on aren't there."""


class _Libsystem:
    """The two libSystem calls the sweep needs."""

    def __init__(self) -> None:
        try:
            lib = ctypes.CDLL(LIBSYSTEM, use_errno=True)
            self._check = lib.sandbox_check
            self._list_pids = lib.proc_listallpids
            # Asks sandbox_check not to log a violation for each denial.
            self._no_report = ctypes.c_int.in_dll(lib, "SANDBOX_CHECK_NO_REPORT").value
        except (OSError, AttributeError, ValueError) as error:
            raise SweepUnavailableError(f"{LIBSYSTEM}: {error}") from error
        self._check.restype = ctypes.c_int
        # Only the fixed arguments. A path goes after them as a variadic
        # argument, which ctypes then passes the way Apple's arm64 calling
        # convention requires (on the stack, not in a register).
        self._check.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self._list_pids.restype = ctypes.c_int
        self._list_pids.argtypes = [ctypes.c_void_p, ctypes.c_int]

    def pids(self) -> list[int]:
        """Every process id on the system."""
        while True:
            count = int(self._list_pids(None, 0))
            if count <= 0:
                raise OSError(ctypes.get_errno(), "proc_listallpids failed")
            capacity = count + 64
            buffer = (ctypes.c_int * capacity)()
            listed = int(self._list_pids(buffer, ctypes.sizeof(buffer)))
            if listed < 0:
                raise OSError(ctypes.get_errno(), "proc_listallpids failed")
            # A full buffer may have cut the list short; ask again.
            if listed < capacity:
                return [pid for pid in buffer[:listed] if pid > 0]

    def sandboxed(self, pid: int) -> bool:
        return int(self._check(pid, None, _FILTER_NONE)) == 1

    def may_read(self, pid: int, path: bytes) -> bool:
        """Whether ``pid``'s sandbox lets it read ``path``.

        False, too, when ``path`` doesn't exist or ``pid`` has exited:
        sandbox_check looks the path up, and answers 1 either way.
        """
        flags = _FILTER_PATH | self._no_report
        return (
            int(self._check(pid, b"file-read-data", flags, ctypes.c_char_p(path))) == 0
        )


@cache
def _libsystem() -> _Libsystem:
    # Loaded on first use, so importing this module works on Linux.
    return _Libsystem()


def ensure_available() -> None:
    """Raise ``SweepUnavailableError`` unless the sweep can run here."""
    _libsystem()


def create_marker(run_dir: Path) -> None:
    (run_dir / MARKER).touch()
    (run_dir / CANARY).touch()


def run_processes(run_dir: Path) -> list[int]:
    """Ids of the processes in this run's sandbox."""
    api = _libsystem()
    own = os.getpid()
    marker = os.fsencode(run_dir / MARKER)
    canary = os.fsencode(run_dir / CANARY)
    return [
        pid
        for pid in api.pids()
        if pid != own
        and api.sandboxed(pid)
        and api.may_read(pid, marker)
        and not api.may_read(pid, canary)
    ]


def _group_of(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def end_run_processes(run_dir: Path, pgid: int) -> None:
    """Kill every process in this run's sandbox until none is left.

    Called once ``pgid`` (the command's group) has been killed. Processes
    still in that group are dying already; the ones outside it escaped,
    and are logged. Raises ``ContainmentError`` if any is still running.
    """
    escaped: set[int] = set()
    unkillable: set[int] = set()

    def left() -> list[int]:
        return [pid for pid in run_processes(run_dir) if pid not in unkillable]

    for _ in range(MAX_PASSES):
        remaining = left()
        if not remaining:
            break
        for pid in remaining:
            if _group_of(pid) != pgid:
                escaped.add(pid)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                unkillable.add(pid)
        time.sleep(_PASS_PAUSE_SECONDS)
    else:
        # The last pass's kills haven't been checked yet.
        remaining = left()
    if escaped:
        logger.warning(
            "Ended processes that left the command's process group",
            extra={"count": len(escaped)},
        )
    problems = []
    if unkillable:
        problems.append(f"{len(unkillable)} can't be signalled")
    if remaining:
        problems.append(f"{len(remaining)} outlasted {MAX_PASSES} passes")
    if problems:
        raise ContainmentError(
            f"the command left processes running: {', '.join(problems)}"
        )
