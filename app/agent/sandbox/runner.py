"""Running one confined command: environment, timeout, kill, and output.

Backends only decide how a command is confined (the argument prefix). This
module does the rest, so every backend behaves the same way.
"""

import asyncio
import logging
import os
import signal
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

logger = logging.getLogger(__name__)

# Seconds a timed-out command gets between SIGTERM and SIGKILL.
KILL_GRACE_SECONDS = 2.0
# Seconds to wait for the output pipe to close once everything is killed.
PIPE_CLOSE_SECONDS = 2.0
_POLL_SECONDS = 0.02
_READ_CHUNK = 64 * 1024

# Make package managers fail fast instead of hanging with no network.
OFFLINE_ENV = {
    "UV_OFFLINE": "1",
    "PIP_NO_INDEX": "1",
    "NPM_CONFIG_OFFLINE": "true",
}
BASE_ENV = {"LANG": "C.UTF-8", "TERM": "dumb", "NO_COLOR": "1"}


# Ends a command's processes that its group kill missed, given the group id.
# Raises when it can't be sure they're gone.
type Sweep = Callable[[int], None]
# Why a running command must be stopped now, or None to let it run.
type StopCheck = Callable[[], str | None]


class ContainmentError(RuntimeError):
    """A command's processes may still be running after it ended.

    Raised instead of a result, since anything left running can keep
    changing the workspace.
    """


@dataclass(frozen=True)
class CommandResult:
    # None when the command was killed.
    exit_code: int | None
    # stdout and stderr interleaved; only the tail when truncated.
    output: str
    truncated: bool
    # Everything the command wrote, including what was cut.
    output_bytes: int
    timed_out: bool
    duration_ms: float


def build_env(
    *,
    path: str,
    scratch: Path,
    extra: Mapping[str, str],
) -> dict[str, str]:
    """The whole environment a command sees; nothing is inherited.

    The service's own environment can hold API keys, so it's never passed on.
    """
    return {
        **BASE_ENV,
        **OFFLINE_ENV,
        **extra,
        "PATH": path,
        "HOME": str(scratch),
        "TMPDIR": f"{scratch}/",
    }


class _Tail:
    """Keeps the last ``limit`` bytes written to it, and counts the rest."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._data = bytearray()
        self.total = 0

    def add(self, chunk: bytes) -> None:
        self.total += len(chunk)
        self._data += chunk
        if len(self._data) > self._limit:
            del self._data[: len(self._data) - self._limit]

    def text(self) -> str:
        return self._data.decode("utf-8", errors="replace")


async def _drain(stream: asyncio.StreamReader, tail: _Tail) -> None:
    while chunk := await stream.read(_READ_CHUNK):
        tail.add(chunk)


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        # Nothing left to signal. macOS reports a group whose members have
        # all exited, and are waiting to be reaped, as EPERM.
        pass


async def _exited(
    process: asyncio.subprocess.Process, timeout: float, stop: StopCheck | None
) -> bool:
    """Whether the process exits within ``timeout`` seconds.

    This watches ``returncode`` instead of awaiting ``wait()``: on Python
    3.12 and later, ``wait()`` doesn't return until the output pipe closes,
    and a background process the command started can hold it open. Raises
    ``ContainmentError`` as soon as ``stop`` gives a reason.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while process.returncode is None:
        if stop is not None and (reason := stop()) is not None:
            raise ContainmentError(reason)
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(_POLL_SECONDS)
    return True


async def _end(pgid: int, sweep: Sweep | None) -> None:
    """Kill the command's group, then sweep, once and to the end.

    The sweep runs in a thread, since it can take a while against a fork
    loop. A cancellation doesn't cut it short: the thread would keep going
    anyway, and the run directory it relies on must outlast it. So this
    waits for it, and raises the cancellation once it's done.
    """
    _signal_group(pgid, signal.SIGKILL)
    if sweep is None:
        return
    sweeping = asyncio.create_task(asyncio.to_thread(sweep, pgid))
    cancelled: asyncio.CancelledError | None = None
    while not sweeping.done():
        try:
            await asyncio.shield(sweeping)
        except asyncio.CancelledError as error:
            cancelled = error
        except Exception:
            pass  # the sweep failed; raised below unless cancelled
    if cancelled is not None:
        if not sweeping.cancelled():
            sweeping.exception()  # retrieved, so asyncio doesn't report it
        raise cancelled
    sweeping.result()


async def run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
    max_output_bytes: int,
    sweep: Sweep | None = None,
    stop: StopCheck | None = None,
) -> CommandResult:
    """Run ``command`` in its own process group and always end it.

    The command is killed with everything it started when it times out, and
    when the caller is cancelled. Processes it leaves behind after exiting
    normally are killed too. A process that leaves the group (``setsid``)
    escapes that kill; ``sweep``, run right after it, is the backend's
    chance to end those. If one still holds the output open, the output is
    given up on after a short grace period instead of waiting for it.

    A sweep that fails raises its error here instead of a result. When the
    caller was cancelled, the cancellation is raised instead, so a sweep
    must also record its own failure (``ConfinedSandbox`` does). While the
    command runs, ``stop`` is checked every poll; once it gives a reason,
    the command is ended like a cancelled one and ``ContainmentError`` is
    raised.
    """
    started_at = perf_counter()
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        env=dict(env),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # The command leads its own process group, so one signal reaches
        # everything it starts.
        start_new_session=True,
    )
    if process.stdout is None:  # can't happen with stdout=PIPE; narrows the type
        raise RuntimeError("command output pipe missing")
    tail = _Tail(max_output_bytes)
    reader = asyncio.create_task(_drain(process.stdout, tail))
    timed_out = False
    try:
        if not await _exited(process, timeout, stop):
            timed_out = True
            _signal_group(process.pid, signal.SIGTERM)
            await _exited(process, KILL_GRACE_SECONDS, stop)
    except BaseException:
        # Cancelled, stopped, or failed mid-run: the command must not keep
        # running.
        # asyncio reaps the process once it's gone.
        reader.cancel()
        try:
            await _end(process.pid, sweep)
        except (Exception, asyncio.CancelledError):
            pass  # a failed sweep or a second cancellation; the first error wins
        raise
    try:
        # Also ends what the command left running, which holds the pipe open.
        await _end(process.pid, sweep)
    except BaseException:
        reader.cancel()
        raise
    # Cancelling here cancels the reader too; the command has ended already.
    try:
        async with asyncio.timeout(PIPE_CLOSE_SECONDS):
            await reader
    except TimeoutError:
        reader.cancel()
        logger.warning("Command output stayed open after the command ended")

    return CommandResult(
        exit_code=None if timed_out else process.returncode,
        output=tail.text(),
        truncated=tail.total > max_output_bytes,
        output_bytes=tail.total,
        timed_out=timed_out,
        duration_ms=round((perf_counter() - started_at) * 1000, 2),
    )
