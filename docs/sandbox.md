# Command sandbox

Detailed plan for step 5b in [next-steps.md](next-steps.md): where commands
the agent runs will execute, and what they can reach. Step 6 adds the
`run_command` tool on top of it; 5a's approval already gates that tool.

**Status: built (2026-11-05).** The Seatbelt backend passes every opt-in
test on macOS. The bubblewrap backend has unit tests only: its opt-in tests
haven't run on Linux yet. "What building it found", at the end, lists the
changes from this plan and the results of the checks.

**Goal:** a command the agent runs can't reach beyond the workspace, even if
the command itself is malicious. That covers a command the user approved
without seeing everything it does: `pytest` runs `conftest.py`, `npm test`
runs package scripts, and a repository the agent was pointed at can contain
both.

## What the sandbox must guarantee

Every backend enforces all of these, and an opt-in test proves each one:

1. **No network**, including `localhost`. That keeps a command from sending
   workspace contents anywhere, and from reaching other local services: the
   model server, the miao-pi API itself, a database.
2. **Writes only inside the workspace** and a per-command scratch directory.
3. **Reads only inside the workspace, the scratch directory, system
   directories, and configured toolchain paths.** Not the rest of the home
   directory (`~/.ssh`, `~/.aws`, browser profiles).
4. **The deny list still applies inside the workspace**: files and
   directories matched by `.env*`, `.git`, `.claude`, and configured
   patterns can't be read or written. The rules protect files where they
   are, not names: a symlink with a denied name, such as
   `.env.link -> notes.txt`, still reaches its target, which is readable
   under its own name anyway. A symlink to a denied file gets nothing, and
   one pointing outside the workspace leads nowhere.
5. **A clean environment.** Nothing from the service's own environment is
   passed through, API keys included.
6. **A known end, or a detected failure.** Every command finishes within
   its timeout or is killed, with its whole process tree, and the caller
   gets the outcome either way. On macOS, where a process can leave the
   command's process group in a way Seatbelt can't block, a sweep finds it
   by its sandbox and kills it (see "The sweep" under the Seatbelt backend).
   If the sweep can't end every one, that's a containment failure, and it's
   detected, not prevented: the leftover processes may keep running, still
   confined by the sandbox, and can still change the workspace. The command
   fails with `ContainmentError` instead of a result. Commands already
   running are stopped and fail the same way, and every later one is
   refused until the service restarts. So no command's result looks normal
   once containment has failed. For step 6's "every approved action ends
   with a known, saved result", that error is the saved result.

There's no unsandboxed fallback. If no backend works on the machine, command
tools aren't offered to the model at all.

## Decisions

Each has a recommendation; the first two are the big ones.

### 1. Native OS sandbox, not containers

| | Native (Seatbelt on macOS, bubblewrap on Linux) | Container per command (Docker/Podman) |
| --- | --- | --- |
| Needs installing | Nothing on macOS; the `bubblewrap` package on Linux | Docker Desktop, Colima, or Podman, plus a Linux VM on macOS |
| Runs the project's toolchain | Yes, the host's own (`.venv`, `node_modules`, compilers) | Only what's in the image; a `.venv` built on macOS won't run in a Linux container, and with no network nothing can be installed |
| Isolation strength | Kernel-enforced policy on the host | Separate namespaces, plus a VM boundary on macOS |
| Hard CPU and memory limits | Weak on macOS, better on Linux | Yes |

**Recommendation: native.** miao-pi runs on the user's own machine until
step 8, and a sandbox that can't run the project's tests is useless to a
coding agent. This is also what Claude Code does ("the built-in Seatbelt
framework" on macOS, "bubblewrap" on Linux). A container backend becomes the
right fit at step 8, when the service itself runs in a container with the
toolchain inside; the interface below allows adding it then.

Checked on this machine (macOS 26.6.2, 2026-11-05) with a deny-by-default
Seatbelt profile and no extra installs:

- Allowed: reading and writing inside the workspace.
- Refused: `.env`, `sub/.ENV.local` (the case-insensitive pattern works), a
  symlink to a file outside, reads and writes outside, and `ls ~/.ssh`.
- Network refused, including `localhost`: `curl http://127.0.0.1:9999` failed
  to connect to the running model server.
- Two findings that shape the plan: `sandbox-exec` passes the caller's whole
  environment through, so the runner must build one from scratch; and
  toolchains need explicit allowances (Apple's `/usr/bin/python3` wrapper
  writes a cache under `/var/folders/…/T`, ignoring `TMPDIR`, and a
  `~/.pyenv` shim is blocked along with the rest of the home directory).

`sandbox-exec` is marked deprecated in its man page (since 2017) but ships
and works in current macOS, and both Claude Code and Anthropic's
sandbox-runtime use it. If Apple removes it, the macOS fallback is the
container backend.

### 2. Our own small backends, not sandbox-runtime

Anthropic's [sandbox-runtime](https://github.com/anthropics/sandbox-runtime)
(Apache-2.0, `npm install -g @anthropic-ai/sandbox-runtime`, CLI `srt`) wraps
the same two mechanisms, and adds a proxy that allows chosen domains.

**Recommendation: write our own**, for three reasons:

- It needs Node, which would add a second runtime to a Python service.
- Its main complexity is the network proxy, and we want no network at all.
- Its defaults read everywhere ("By default, read access is allowed
  everywhere"); we want reads denied by default (decision 3). It can be
  configured that way, but then little of it is left.

The profiles and the argument lists we need are short and fully testable.
Revisit if maintaining them gets costly, or when per-command network access
is wanted.

### 3. Reads are denied by default

Claude Code allows reads across the whole machine and notes that this "still
allows reading credential files". Here, `read`, `ls`, `find`, and `grep`
already can't leave the workspace; a command that could `cat ~/.aws/credentials`
would undo that, and its output goes to the model provider.

**Recommendation: deny by default**, and allow the workspace, the scratch
directory, system directories (`/usr`, `/bin`, `/System`, `/Library`,
`/opt/homebrew`, `/etc`, and their Linux equivalents), and the configured
toolchain paths, read-only.

### 4. `.git` stays hidden

The deny list already hides `.git` from the read tools, and `.git/config`
can hold credentials (remote URLs with tokens). So `git status`, `git diff`,
and `git log` fail inside the sandbox.

**Recommendation: keep it hidden for 5b.** If the agent needs git
information, add dedicated read-only tools later (status, diff, log), run by
the service with fixed arguments, not by the agent. The alternative, `.git`
readable but not writable (Claude Code protects `.git/config` and
`.git/hooks` from writes), would expose those credentials.

### 5. No network at all

**Recommendation: none, not even opt-in, in 5b.** Installing dependencies
(`uv sync`, `npm install`) therefore fails inside the sandbox: the user runs
those outside, and the sandbox's environment sets the offline switches so
tools fail fast instead of hanging (`UV_OFFLINE=1`, `PIP_NO_INDEX=1`,
`NPM_CONFIG_OFFLINE=true`). Per-command network access with a domain
allowlist means a proxy, which is a later step of its own.

## Design

### Layout

```
app/agent/sandbox/
  __init__.py     # Sandbox protocol, CommandResult, make_sandbox()
  policy.py       # what's readable, writable, and denied (backend-neutral)
  runner.py       # spawning, environment, timeout and kill, output capture
  seatbelt.py     # macOS: policy -> Seatbelt profile -> sandbox-exec argv
  seatbelt_sweep.py  # macOS: ending processes that left the command's group
  bubblewrap.py   # Linux: policy -> bwrap argv
```

A backend only turns a policy into the argument list that confines a
command. Spawning, timeouts, and output are shared in `runner.py`, so both
backends behave the same way and most of the logic is tested once.

### Interface

```python
@dataclass(frozen=True)
class CommandResult:
    exit_code: int | None   # None when the command was killed
    output: str             # stdout and stderr interleaved, tail kept
    truncated: bool
    timed_out: bool
    duration_ms: float


class Sandbox(Protocol):
    name: str  # "seatbelt", "bubblewrap"

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,        # resolved through Workspace, inside the root
        timeout: float,
    ) -> CommandResult: ...
```

- `argv` is a list, executed directly, never through a shell. A model that
  wants a pipeline passes `["sh", "-c", "…"]`, which the approval card then
  shows exactly. The shell runs inside the sandbox, so this doesn't widen
  anything.
- stdout and stderr are interleaved into one stream, the way a terminal
  shows them, and capped by keeping the tail, where errors and summaries
  usually are. A notice says how much was cut.
- `make_sandbox(settings, workspace) -> Sandbox | None` picks and self-tests
  a backend at startup (see Startup).

### Policy (`policy.py`)

```python
@dataclass(frozen=True)
class SandboxPolicy:
    workspace: Path                    # read-write; Workspace.root
    scratch: Path                      # read-write; per command
    read_only: tuple[Path, ...]        # system dirs + sandbox_read_paths
    denied_patterns: tuple[str, ...]   # Workspace.denied_patterns
```

- `denied_patterns` comes from the `Workspace` instance, so the read tools
  and commands enforce one list.
- `sandbox_read_paths` entries are expanded (`~`), resolved, and must exist.
  The home directory itself, or anything that contains it, is refused at
  startup, since allowing it would undo decision 3.

### Runner (`runner.py`)

- **Spawn** with `asyncio.create_subprocess_exec(*backend_prefix, *argv)`:
  `stdin` from `/dev/null`, stdout and stderr to one pipe,
  `start_new_session=True` so the command leads its own process group, and
  `cwd` set.
- **Environment** built from scratch, never inherited:
  - `PATH` from `sandbox_path`.
  - `HOME` and `TMPDIR` set to the scratch directory.
  - `LANG=C.UTF-8`, `TERM=dumb`, `NO_COLOR=1`.
  - The offline switches from decision 5.
  - The variables in `sandbox_env`, with their values; empty by default.
- **Scratch directory:** a fresh one per command under a service-owned temp
  root, removed afterwards. Commands share nothing through it.
- **Timeout:** when the timeout expires, send `SIGTERM` to the process group,
  wait a short grace period (2s), then send `SIGKILL`. The result has
  `timed_out=True`, `exit_code=None`, and the output captured so far.
- **Cancellation:** if the caller is cancelled (step 6's backstop deadline,
  a client disconnect), the runner kills the process group and sweeps
  before re-raising. Unlike a tool in a worker thread, a subprocess can
  always be stopped, so nothing keeps running after its request ends.
- **Ending a command** (`_end`): the group kill, then the backend's sweep,
  once per command, in a thread so it never blocks the event loop. A
  cancellation, even a repeated one, doesn't cut the sweep short: the
  thread would keep going anyway, and the run directory it relies on must
  outlast it. The runner waits for it, then raises the cancellation.
- **A failed sweep:** `run_process` raises its error instead of returning a
  result. When the caller was cancelled, the cancellation wins, so
  `ConfinedSandbox` records the failure itself: any error from a sweep
  becomes `ContainmentError` and turns commands off until restart, logged
  once as `Sandbox commands turned off` with the reason.
- **Stopping other commands:** commands run concurrently (step 6's tool
  calls in one response, or several conversations), so turning commands
  off must also reach the ones already running. The runner's 20 ms poll
  checks `stop` (`ConfinedSandbox._refusal`). Once commands are off, it
  ends a running command the way it ends a cancelled one and raises
  `ContainmentError`. A command that exited between two polls is caught
  by one more check after it returns. A lock around `run()` would also
  close the race, but only one command could run at a time across the
  whole service.
- **Output cap:** read in chunks and keep only the last
  `sandbox_max_output_bytes` in memory (a bounded buffer), so a command that
  prints gigabytes can't exhaust memory. Decode as UTF-8 with replacement.
- **Logging:** `Command completed` with `command` (only `argv[0]`'s base
  name), `exit_code`, `timed_out`, `duration_ms`, and `output_bytes`. Never
  the full argv, which can carry secrets the user typed, and never the
  output, as with the read tools.

### Seatbelt backend (`seatbelt.py`)

Builds a profile and runs `sandbox-exec -p <profile> <argv…>`:

```
(version 1)
(deny default)
(allow process-exec process-fork sysctl-read)
(allow signal (target same-sandbox))
(allow ipc-posix-sem)
(allow file-read-metadata)
(deny syscall-unix (syscall-number SYS_setsid SYS_setpgid))
(allow file-read* <system dirs> <sandbox_read_paths>)
(allow file-read* file-write* (subpath WORKSPACE) (subpath SCRATCH))
(allow file-write* (literal "/dev/null"))
(allow file-read-data (literal MARKER))
(deny file-read* file-write* (regex <one per denied pattern>))
```

- `mach-lookup` (system services) isn't allowed: some services launch apps
  outside the sandbox, and none of the checked toolchains needed it.
- Signals reach only processes in the same sandbox, so a command can't kill
  the user's other programs.
- `ipc-posix-sem` is the one IPC rule needed, by Python's `multiprocessing`.
- `setsid` and `setpgid` are denied explicitly, because `(deny default)`
  doesn't cover system calls. Either one lets a process leave the command's
  process group and escape the timeout's kill. `posix_spawn`'s `setsid`
  and `setpgroup` options still do, so the sweep (below) ends those.
- `MARKER` is how the sweep recognizes this run's processes.

- No `network*` rule is allowed, so all network access is denied by
  `(deny default)`.
- **The deny rules must come last.** In Seatbelt the last matching rule
  wins, not the most specific one. Checked on 2026-11-05: with the `.env`
  deny placed after the workspace allow, `cat .env` is refused; placed
  before it, `cat .env` printed the secret. A unit test asserts that every
  deny follows every allow, and the opt-in tests prove the result.
- Paths go in as `-D` parameters (`(subpath (param "WORKSPACE"))`), so no
  path is ever spliced into the profile text. The regexes need the workspace
  path inside them, so it's regex-escaped there.
- **Converting deny patterns to regexes:** a glob becomes a regex anchored
  at the workspace. A pattern without `/` matches a name at any depth
  (`^WS/(.*/)?NAME(/.*)?$`); one with `/` is rooted (`^WS/PATH(/.*)?$`).
  Case-insensitivity is spelled out per letter (`[eE][nN][vV]`), since the
  profile language has no case flag. Only a small, tested subset of regex
  syntax is ever emitted: literal characters, character classes, `.*`, and
  `(…)?`.
- **Known gaps:** Unicode normalization variants of configured patterns
  aren't covered; hard links to a denied file aren't covered (the same gap
  as `Workspace`); there are no hard memory limits; and process count limits
  are skipped, because macOS counts processes per user, so a low limit
  would also hit the user's other programs. The wall-clock timeout bounds
  CPU time too.

#### The sweep (`seatbelt_sweep.py`)

`posix_spawn`'s `setsid` and `setpgroup` options move the new process out
of the command's group inside the spawn call, which Seatbelt can't filter,
so the group kill misses it. After every command, the sweep finds the
processes still in that command's sandbox and kills them, pass after pass,
until none is left. It runs before the run directory is removed.

- **How it recognizes a process:**
  - Each command gets a run directory holding `home/` (the scratch
    directory, its `HOME` and `TMPDIR`), with `marker` and `canary` beside
    it.
  - The profile lets commands read `marker` (the `MARKER` parameter), but
    not `canary`.
  - macOS's `sandbox_check(pid, "file-read-data", path)` answers that
    question for any process.
  - So a process belongs to this run when it's sandboxed, may read this
    run's marker, and may not read its canary. Unsandboxed processes may
    read both, and another run's processes may read only their own marker.
- **Why a command can't hide from it:**
  - Only `home/` is writable, so a command can't move or delete the marker,
    the canary, or the directory holding them. Each attempt got `Operation
    not permitted`.
  - It can't add a second profile that denies the marker. macOS refuses to
    sandbox an already-sandboxed process (`sandbox_apply: Operation not
    permitted`).
  - The first idea, checking "may write this run's scratch directory",
    could have been evaded. `sandbox_check` looks the path up, and a
    missing path answers "denied", so a command could move its own scratch
    directory away.
- **When it runs:**
  - Once per command, right after the group kill, in a thread, whether
    the command ended normally or the caller was cancelled (see "Ending a
    command" under the runner).
  - Before the runner waits for the output pipe, so an escaped process
    holding the pipe doesn't hold up the result.
- **Fork loops:** each pass kills every match. In the spike, a process
  forking in a loop took two passes, 232 processes in 0.4 s. After 50
  passes it checks once more, and if anything is left it gives up.
- **When it can't end everything:** a process it can't signal, or one
  still there after the last pass, makes it raise `ContainmentError`. The
  command fails and commands stay off until restart (see "A failed sweep"
  under the runner).
- **Logging:** `Ended processes that left the command's process group`,
  with a count, when it ended any. A failure is logged once, as `Sandbox
  commands turned off`.
- **Cost:** a pass over about 600 processes takes about 4 ms.
- **Limits:**
  - `sandbox_check` is private, though long-standing (Chromium uses it).
    The startup self-test plants a sandboxed `sleep` in a session of its
    own and requires the sweep to kill it. If a macOS update breaks the
    call, commands are turned off rather than run without the sweep.
  - A process id freed and reused between the check and the kill could be
    hit. The window is microseconds.
  - A process running as another user, such as a setuid program, can't be
    signalled. The command fails and commands are turned off.
  - Turning commands off doesn't stop what's left; it only keeps it from
    being mistaken for a finished command. For a command that must be
    assumed hostile, a VM or the Linux PID namespace is the stronger
    boundary.
  - A zombie doesn't match, so it can't stall the sweep.

### Bubblewrap backend (`bubblewrap.py`)

Runs `bwrap <args…> -- <argv…>` with:

- `--unshare-all` (network, PID, IPC, UTS, cgroup, and user namespaces),
  `--die-with-parent`, and `--new-session`.
- `--ro-bind` for the system dirs and `sandbox_read_paths`; `--bind` for the
  workspace and the scratch directory, at their real paths so tools see the
  same paths as on the host; `--dev /dev`, `--proc /proc`, and `--chdir`.
- **Masking denied paths:** bubblewrap has no pattern rules, so before each
  command, walk the workspace for existing denied paths. Mask each denied
  directory with `--tmpfs` and each denied file with `--ro-bind /dev/null`.
  The walk ignores `.gitignore`, since `.env` is usually ignored, and stops
  at denied directories. A denied file the command creates during its own
  run isn't masked, but it holds nothing the command didn't write itself.
- Anything outside the mounted paths, such as a symlink's target, doesn't
  exist inside the sandbox.
- **Requirements, per sandbox-runtime's README:** bubblewrap 0.4.0 or later,
  and on Ubuntu 24.04 and later, unprivileged user namespaces must be
  allowed (`kernel.apparmor_restrict_unprivileged_userns=0`, or an AppArmor
  profile for `bwrap`). The startup self-test detects both.

### Settings (`common/config.py`, `.env.example`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `sandbox_backend` | `auto` | `auto`, `seatbelt`, `bubblewrap`, or `off`. `off` disables command tools; there's no unsandboxed value. |
| `sandbox_command_timeout` | `120` | Seconds per command before it's killed. |
| `sandbox_max_output_bytes` | `51200` | Output kept per command (the tail). |
| `sandbox_read_paths` | `[]` | Extra read-only paths, for toolchains under the home directory (for example `~/.pyenv`, `~/.local/bin`, `~/.cargo`). |
| `sandbox_path` | system dirs | The `PATH` commands see. |
| `sandbox_env` | `{}` | Variables set for every command, with their values. Nothing is copied from the service's own environment. |

### Startup

- **Choosing a backend:** with `auto`, use Seatbelt on macOS if
  `/usr/bin/sandbox-exec` exists, or bubblewrap on Linux if `bwrap` is on
  `PATH`.
- **Self-test before trusting it:** a plain command (`true`) must run in the
  real workspace. Everything else runs in a throwaway workspace built the
  same way (same backend, deny patterns, read paths, and settings), so the
  real one is never written to. There, a command must be able to write a
  file. It must fail to read or change a canary `.env`, to read a canary
  outside the workspace, to write outside it, and to connect to a listener
  the service opens on `127.0.0.1`. On macOS the sweep must also kill a
  sandboxed process planted in a session of its own.
- **Failing closed:** a failed check, a backend that can't express the deny
  list, or a launcher that won't start all log an error and leave the
  sandbox unavailable; none of them stops the service. Only a mistyped
  `sandbox_read_paths` entry fails startup.
- **Logging:** `Startup complete` gains `sandbox` (the backend's name,
  `off`, or `unavailable: <reason>`).
- **Without a sandbox,** step 6 doesn't register `run_command`, so the model
  never sees it.

### How step 6 uses it

- **Approval card:** `run_command(argv, cwd?)` requires approval (5a). The
  card shows the exact argv, the working directory, and the timeout.
- **Results:** a `CommandResult` becomes the tool result: the exit code (or
  "killed after 120s"), the output tail, and a notice when it was cut. A
  `ContainmentError` becomes an error result saying commands are off until
  the service restarts.
- **Known end:** together with step 6's deadline between steps, every
  command ends with a saved result. It finishes, it's killed by its own
  timeout, or, if the backstop cancels the run, the runner kills it. If
  containment fails, the saved result is the `ContainmentError`, and
  processes may be left running in the sandbox.

## Tests

**Unit tests**, which run everywhere with no sandbox:

- **Policy:**
  - denied patterns come from the `Workspace`;
  - `sandbox_read_paths` are expanded and validated;
  - the home directory is refused.
- **Seatbelt profile:**
  - it denies by default and has no network rule;
  - the workspace and scratch directories are allowed;
  - there's one deny regex per pattern;
  - every deny rule comes after every allow rule, since the last match wins.
- **Regex conversion,** checked by running the generated regex with
  Python's `re` over a table of paths:
  - `.env*` matches `.env` and `sub/.ENV.local` but not `env.txt`;
  - `config/private` matches `config/private/x` but not
    `sub/config/private`;
  - a check that only the safe regex subset is emitted.
- **Bubblewrap arguments:**
  - existing denied files and directories are masked the right way;
  - the namespace flags and the binds are present.
- **Runner,** with a test-only backend that adds no confinement, used only
  for process handling:
  - the environment is exactly the minimal set, even with `OPENAI_API_KEY`
    set in the test's own environment;
  - a timeout kills the whole tree (`sh -c "sleep 60 & sleep 60 & wait"`
    leaves no survivors);
  - output keeps the tail and sets `truncated`;
  - exit codes come through;
  - cancelling the caller kills the command;
  - cancelling during the sweep, even twice, waits for that one sweep, run
    in a thread, before the run directory is removed;
  - a failed sweep fails the command and turns commands off, even when the
    caller was cancelled;
  - a failed sweep stops commands already running, and a command that
    ends just after it fails too.

**Opt-in integration tests** (`RUN_SANDBOX_TESTS=1`, a `sandbox` marker,
skipped when the platform's backend is unavailable), run against the real
backend:

- Reading and writing inside the workspace work.
- `.env`, `sub/.ENV.local`, `.git/config`, `.claude/settings.local.json`,
  and a configured pattern can be neither read nor written.
- A file outside the workspace, a symlink to it, and the home directory
  can't be read; nothing outside can be written.
- TCP to an outside address and to a local listener started by the test
  both fail.
- The command's environment has none of the service's variables.
- Timeouts kill children, and output is capped.
- A toolchain smoke test runs this repo's own `.venv/bin/python -m pytest
  --version`.

**A toolchain check by hand,** recorded in this file once done: in a
Python, a Node, and a compiled-language project, which common commands work
sandboxed (running tests, linters, builds) and which `sandbox_read_paths`
entries they needed.

## Order of work

1. Policy, runner, the interface, and a fake backend for step 6's graph
   tests, with unit tests.
2. Seatbelt backend, its opt-in tests, and the toolchain check on macOS (the
   platform the service runs on today).
3. Bubblewrap backend and its opt-in tests on Linux.
4. Settings, startup selection and self-test, logging, README, and
   AGENTS.md.

Steps 1 and 2 can land before 3 if Linux support isn't needed yet. Step 6
only needs the interface and one working backend.

**Done when:** on macOS, the opt-in tests pass against Seatbelt and this
repo's tests run inside the sandbox; on Linux, they pass against bubblewrap;
the service starts without command tools when no backend works; and the
decisions above are in AGENTS.md.

## What building it found (2026-11-05)

**Changes from the plan above**

- **No inherited variables, even on request:** `sandbox_env` sets
  variables with their values instead of naming ones to copy from the
  service's environment, which app code never reads directly.
- **Exit detection:** the runner watches the exit code instead of awaiting
  `wait()`. On Python 3.12 and later, `wait()` returns only once the output
  pipe closes, and a background process the command left behind holds it
  open: the first version waited out the whole timeout. Leftovers are now
  killed as soon as the command exits.
- **A process that leaves its group** escapes the group kill. The runner
  stops waiting for its output after 2 seconds instead of hanging (a unit
  test covers this). On Linux the PID namespace ends it with the sandbox.
  On macOS, a review found that both `setsid` and `setpgid` worked inside
  Seatbelt, and the survivor kept running, still confined, after the
  command's timeout. The profile now denies both system calls, and an
  opt-in test proves they fail. One way remained open: `posix_spawn` with
  its `setsid` or `setpgroup` option changes the session inside the spawn
  call, which Seatbelt can't filter, and a `sleep` started that way
  outlived the command. The sweep now closes it (see "The sweep"). The
  spike first confirmed what the design rests on: `sandbox_check` can be
  called from Python (ctypes passes its variadic path argument correctly on
  arm64), it identifies exactly one run's processes among about 600, a
  zombie doesn't match, and a command can neither move the marker nor stack
  a second profile. Breaking the sweep on purpose, by removing the marker
  rule or the sweep itself, fails the self-test, and the survivor then
  holds the output open for the full 2 seconds.
- **Denied-name symlinks reach their target, on both backends.** With
  `.env.link -> notes.txt`, `cat .env.link` prints `notes.txt`, while
  `Workspace.resolve` refuses the name. Nothing extra is exposed:
  `notes.txt` is readable under its own name in every channel. Masking the
  link isn't an option either, since mounting over a symlink mounts over its
  target. An opt-in test pins the behavior: the link reaches its target, and
  a link to a denied file gets nothing.
- **macOS reports an empty process group as `EPERM`,** not "no such
  process", so the runner treats both as already gone.
- **Smaller profile:** see the Seatbelt section.
- **Skipped:** the `take_tail` helper (the runner's own bounded buffer does
  the job) and `RLIMIT_CPU` (the wall-clock timeout bounds it).

**Opt-in tests:** all 17 pass against Seatbelt on macOS 26.6.2
(`RUN_SANDBOX_TESTS=1 uv run pytest -m sandbox`):
- the self-test;
- reads and writes inside the workspace;
- each denied path;
- outside paths and a symlink;
- no network;
- a clean environment;
- timeouts;
- `setsid` and `setpgid` refused;
- processes spawned out of the group with `posix_spawn` (`setsid` and
  `setpgroup`) swept, with other commands left alone;
- denied-name symlinks;
- this repo's Python, with its interpreter as a read path.

**Toolchain check, this repo, macOS.** Both read paths are under the home
directory, so they need listing:

```
~/.local/share/uv/python/cpython-3.13.7-macos-aarch64-none   # the .venv's interpreter
~/.local/bin                                                 # uv itself
```

| Command | Sandboxed |
| --- | --- |
| `.venv/bin/mypy`, `black --check .`, `isort --check-only .` | Works |
| `uv run pytest tests/unit/test_workspace.py` | Works (offline) |
| `make --version` | Works |
| `git status` | Fails: `.git` is hidden (decision 4) |
| The full test suite | 73 tests fail: `pydantic-settings` reads `.env`, which is denied |

The last row matters beyond this repo. A project whose code reads its own
`.env` (`python-dotenv`, `pydantic-settings`) can't run that code
sandboxed on macOS: reading the file raises `PermissionError`. On Linux,
bubblewrap masks the file with an empty one, so the same code reads
nothing instead. Both keep the secret out. If it becomes a problem, the
options are a test configuration that doesn't load `.env`, or keeping
secrets outside the workspace.

**Still to do:**
- **Linux:** run the opt-in tests on a machine where `bwrap` is installed
  and user namespaces are allowed.

## Later, not in 5b

- Per-command network access through a domain-allowlist proxy, with
  approval (what sandbox-runtime adds).
- A container backend for the step 8 deployment, with hard CPU, memory, and
  process limits.
- Read-only git tools (status, diff, log) run by the service, if the agent
  needs git information.
- Windows.
