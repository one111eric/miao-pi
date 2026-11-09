# AGENTS.md

Guidance for coding agents working in this repository. Human-facing setup and
usage live in `README.md`.

miao-pi is a coding agent served over HTTP: a FastAPI service wrapping a
LangGraph workflow that calls an OpenAI-compatible chat model.

## Commands

Python 3.13.7, managed with `uv`. Always run tools through `uv run`.

```bash
# Install dependencies
uv sync
# Start the development server
uv run uvicorn app.service.server:app --reload --port 9091
```

All four checks must pass before committing (CI runs the same):

```bash
uv run black --check .
uv run isort --check-only .
uv run mypy
uv run pytest
```

Fix formatting with `uv run isort . && uv run black .`.

## Layout

- `app/service/` — `create_app()`, lifespan hooks, and the uvicorn entry
  point (`server.py`). Startup builds the LLM once and compiles the agent
  graph into `app.state`; `create_app(llm=...)` overrides the model.
  `middlewares/` holds the request-id middleware, and `error_handling/` maps
  provider errors to HTTP statuses.
- `app/<feature>/` — one package per API feature (`agent`, `health`, `ui`),
  each with `router.py` and, where it has request or response bodies,
  `models.py` (Pydantic models).
- `app/ui/` — the developer test page: `static/` (`index.html`, `app.js`,
  `app.css`, no build step), `router.py` serving it under `/ui/`, and
  `gate.py` (`require_dev_ui`, which hides developer-only routes).
- `app/agent/` — the LangGraph workflow: `state.py` (graph state),
  `nodes.py` (async node functions), `graph.py` (wiring and compile),
  `service.py` (runs or resumes the graph for a request), `approval.py`
  (pausing for a human before tools with side effects), `workspace.py` (the
  boundary for project files), `tools/` (read and approval-gated write tools,
  one module each, plus `tool_utils/` with shared read helpers), and
  `sandbox/` (running commands confined to the workspace).
- `common/` — code shared across features: `config.py` (`SvcSettings`),
  `llm/` (`LLM`, a `ChatOpenAI` subclass), `logging_helper.py`.
- `tests/unit/` — no network, no app lifespan.
- `tests/integration/` — go through the ASGI app via the `client` fixture in
  `tests/conftest.py`.

## Architecture

Request flow: `POST /invoke` → `app/agent/router.py` → `invoke_agent()` →
compiled graph `preplanning → react ⇄ [approve →] tools → finalize`.
`POST /threads/{thread_id}/resume` → `resume_agent()` continues a run paused
in `approve`.

Streaming (step 7a, payloads in [docs/streaming.md](docs/streaming.md)):

- `POST /stream` and `POST /threads/{thread_id}/resume/stream` use FastAPI's
  `EventSourceResponse` and emit JSON SSE events. `prepare_invoke` and
  `prepare_resume` share validation and thread holding with the JSON routes;
  request-scoped dependencies preserve HTTP errors before headers are sent.
- `streaming.py` maps `graph.astream(stream_mode=["messages", "updates"])`
  to token, complete tool-call, and tool-result events. A checkpoint read
  after exhaustion produces the terminal `done` or `awaiting_approval` event.
  Mid-stream errors produce a safe `error` event without raw exception text.
- `open_agent_stream` owns a producer task and a bounded queue. Request
  teardown cancels and joins that task before releasing the thread, including
  on disconnect. Keep the timeout and logging context inside the producer;
  generators suspended at yield may be closed from another task.
- Stream tokens are provisional. The final response can be a step/time-limit
  message and must replace the displayed partial answer. There is no replay
  or automatic retry of interrupted side effects.

- `preplanning` reloads `AGENTS.md` (falling back to `README.md`) from the
  workspace on every turn, with a configured size cap.
- `react` calls the LLM, which `build_agent_graph(llm, tools=...)` injects
  through `make_react(llm, tools)`, binding the tools there. It builds a
  system message from `SYSTEM_PROMPT`, the tool names, and the current
  workspace context at call time; none of it is stored in message history.
- `tools` is LangGraph's `ToolNode`, reached through
  `make_route_after_react` and looping back to `react` until the model
  answers without a tool call. Each call goes through `log_tool_call`, then
  the approval gate. Without tools, `react` goes straight to `finalize`.

Approval (`app/agent/approval.py`):

- A tool opts in with `metadata={REQUIRES_APPROVAL: True}`; none of the read
  tools do. If any call in a model response needs approval, the route goes
  to `approve` instead of `tools`.
- `approve` calls LangGraph's `interrupt` once, listing every
  approval-requiring call in that response, and the run pauses in the
  checkpoint. `invoke_agent` sees `__interrupt__` and returns
  `awaiting_approval` with the actions. One decision covers them all.
- On resume (`Command(resume=decision)`), `approve` runs again from the top;
  `interrupt` then returns the decision instead of pausing, and `approve`
  records it in `state["approvals"]` by tool call id. Keep `approve` free of
  side effects before `interrupt`, since it always runs twice.
- The gate in front of every tool call refuses an approval-requiring call
  unless its recorded decision approved it, returning an error
  `ToolMessage` (`rejection_text`), so a routing bug fails closed. Rejected
  calls reach the model as that error result; other calls in the same
  response run normally.
- A tool may set an `APPROVAL_PREFLIGHT` callback for side-effect-free
  validation. A failed preflight becomes an error result without asking for
  approval. Its structured failure carries the rejected path and (for a
  boundary rejection) its reason for safe logging. Results are cached by
  tool arguments for the graph's lifetime, so routing, interrupt creation,
  and the gate agree without repeating expensive checks; the tool still
  revalidates its workspace boundary when it executes because the filesystem
  can change while approval is pending.
- The interrupt lives in its own node rather than inside each tool: on
  resume LangGraph reruns the whole interrupted node, and `ToolNode` would
  rerun every call in it.
- A new `/invoke` on a waiting thread gets `409` (`ThreadAwaitingApprovalError`),
  because a fresh run would silently drop the pending one. Resume returns
  `404` for an unknown thread and `409` when nothing waits. Resume holds
  the thread with `ThreadStore.use_existing()`, which raises
  `ThreadNotFoundError` without tracking the id: with `use()`, an unknown id
  would count toward `max_threads` and its cleanup could evict a real
  thread.
- `react` gives any tool call that arrives without an id one
  (`call_<hex>`), because approval, the gate, and tool results all match
  calls by id. If a run still pauses with no actions, `invoke_agent` raises
  rather than report it as completed.
- An approval round needs no extra step budget: the run pauses in `approve`,
  and each resume gets a fresh step budget.
- Decisions are logged as `Approval decision received` (approved, tool
  names), never with arguments.
- `finalize` is a placeholder for post-processing.

Tools (`app/agent/tools/`):

- `make_read_tools(workspace)` builds `read`, `ls`, `find`, and `grep`,
  shaped after pi's coding agent: same names, parameters, caps (2000 lines or
  50KB, 500 entries, 1000 results, 100 matches), and actionable truncation
  notices. Results are relative to the workspace root.
- Every path goes through `Workspace`. `tool_utils.files.resolve_for_tool`
  and `tool_utils.files.open_for_tool` turn workspace and OS errors into
  `ToolException`s; each tool sets `handle_tool_error=True`, so they reach
  the model as error results. Only bugs propagate. Never put an absolute
  server path in a tool result.
- `find` and `grep` walk the tree in Python (`tool_utils.files.iter_files`)
  instead of running `fd`/`rg`, so the deny list and `.gitignore` (via
  `pathspec`) apply exactly. Denied and escaping entries are hidden, not
  reported.
- `make_write_tools(workspace)` builds `write_file` and `apply_patch`.
  `make_command_tool(workspace, sandbox)` builds `run_command`, and
  `make_agent_tools(workspace, sandbox)` combines it with the read and write
  tools only when startup supplied a self-tested sandbox. All three require
  approval. `write_file` atomically replaces one UTF-8 text
  file, rejects NUL bytes, caps input at 1,000,000 characters, and requires
  its parent directory to exist. Both write primitives reject a leaf symlink
  rather than replacing its target. `apply_patch` has the same cap, accepts a
  conservative text-only unified diff (no renames or missing final newlines),
  rejects NUL-containing sources, checks every hunk's base, and stages all
  changes so an I/O failure restores earlier files. Neither tool accepts
  denied or outside paths.
- Multi-file patch commits use two renames per existing file (original to a
  backup, then staged replacement into place). A normal I/O failure rolls
  back, but process death in that small window can leave a destination absent;
  this is not crash-safe transaction durability.
- `run_command` receives a non-empty argv array and does not invoke a shell
  automatically; an explicit shell can interpret shell syntax. It
  runs from the workspace root, and reuses the sandbox's timeout and bounded
  output. Its result states the exit code or timeout and whether output was
  truncated. Invalid argv is rejected before approval; sandbox setup failure
  or a disabled sandbox means this tool is not registered at all.
- `Workspace.open_file` is the only way to read file contents. It walks
  from the root one component at a time, each opened relative to its parent
  (`dir_fd`) with `O_NOFOLLOW`, so a symlink swapped in anywhere along the
  path after `resolve()` fails with "path changed while opening" instead of
  escaping. The root is opened by path, so its device and inode must match
  the ones recorded at startup ("workspace root changed" otherwise). Never
  `open()` a resolved path directly: that follows a swapped parent. It opens
  non-blocking, raises `IsADirectoryError` for directories, and accepts only
  regular files, so a FIFO can't hang a read.
- `ls` and the `find`/`grep` walk still list directories by path. Every
  entry is re-checked with `resolve()` before it's shown, but a swap and
  swap-back between listing and checking could briefly show the names (never
  the contents) of an outside directory.
- `read` stops at the first cap: a cut notice gives the file's size, not its
  line count, because counting would read the whole file.

Command sandbox (`app/agent/sandbox/`, plan and findings in
`docs/sandbox.md`) powers step 6's approval-gated `run_command` tool:

- At startup, `make_sandbox` picks Seatbelt on macOS or bubblewrap on Linux
  and self-tests it. A plain command runs in the real workspace; the rest
  runs in a throwaway workspace built the same way, never the real one.
  There, a write must work, and these must fail: reading or changing a
  denied `.env`, reading or writing outside, and reaching a localhost
  listener. It sets `app.state.sandbox` to the sandbox, or to `None` when
  none works. Any exception while building or testing a backend also
  means `None`; only a bad `sandbox_read_paths` entry fails startup. There's no unsandboxed mode: without a sandbox, command tools
  aren't offered. `Startup complete` logs `sandbox` (the backend, `off`, or
  `unavailable: <reason>`) and `sandbox_read_paths`. No backend at all logs
  a `Sandbox unavailable` warning (not for `off`), and a self-test that
  raises logs its traceback.
- Every `ConfinedSandbox.run` call logs `Command started` with its effective
  timeout, then one terminal event: `Command completed`, `Command timed out`,
  `Command failed`, or `Command cancelled`. Timeouts, failures, and cancellation
  are warnings; normal completion includes exit code and output size.
  A unique `command_id` context field links all runner and cleanup logs, and
  `tool_call_id` links agent tool logs to approvals. Self-test commands carry
  `sandbox_self_test=True`. Never log argv beyond the program's name, or output.
- Approval refusals log `Tool call refused` with a safe reason code
  (`invalid_arguments`, `path_rejected`, `approval_rejected`, or
  `approval_missing`), without argument values or rejection notes. A failed
  Seatbelt sweep logs `Process cleanup incomplete` with counts; successful
  cleanup of escaped processes is reported only when the sweep succeeds.
- A backend only builds the command line that confines a command
  (`confine`). `ConfinedSandbox.run` and `runner.run_process` do the rest
  for every backend:
  - an environment built from scratch, never inherited;
  - a fresh run directory (`RunDir`), whose `home/` is `HOME` and `TMPDIR`;
  - the timeout: `SIGTERM`, then `SIGKILL`, to the whole process group;
  - killing what the command left running;
  - keeping the last `sandbox_max_output_bytes` of output.
- Never `await process.wait()` in the runner: on Python 3.12 and later it
  waits for the output pipe to close, which a leftover background process
  can delay forever. Watch `returncode` instead.
- Seatbelt rules:
  - Deny by default.
  - Deny rules must come after the allows, because the last matching rule
    wins (`test_the_profile_denies_by_default_and_puts_denies_last`).
  - No `mach-lookup`, and signals only within the sandbox.
  - `setsid` and `setpgid` are denied with a `syscall-unix` rule, since
    `(deny default)` doesn't cover system calls.
  - `posix_spawn`'s `setsid` and `setpgroup` options still leave the group,
    so after every command `seatbelt_sweep` kills what's left in that
    command's sandbox. It recognizes a process with the private
    `sandbox_check`: sandboxed, may read the run's `marker`, may not read
    its `canary`. Both files sit in the run directory next to `home/` (the
    scratch directory), where commands can't change them.
  - The runner's `_end` kills the group, then sweeps once, in a thread,
    before waiting for output. A cancellation, even a repeated one, waits
    for the sweep to finish (it must finish before the run directory is
    removed) and is raised afterwards. Never call a sweep on the event
    loop thread.
  - A sweep that can't end everything raises `ContainmentError`, and
    `ConfinedSandbox._sweep` turns any sweep error into one. The command
    fails instead of returning a result, and `disabled_reason` makes the
    sandbox refuse every later command until restart, since what's left can
    still change the workspace. On the cancel path the cancellation is
    raised, but the sandbox still turns commands off.
  - Commands run concurrently, so turning commands off also stops the ones
    already running: the runner's poll checks `stop`
    (`ConfinedSandbox._refusal`), and `run` checks once more after
    `run_process` returns. Don't serialize `run()` with a lock instead;
    one long command would block every other conversation.
  - That's detection, not containment: a failed sweep's processes may keep
    running (still sandboxed). Don't describe it as a guaranteed end.
  - The self-test plants a sandboxed process in a session of its own and
    requires the sweep to kill it, so a macOS update that breaks
    `sandbox_check` turns commands off.
  - Deny patterns become case-insensitive regexes (`glob_to_regex`), and
    paths are passed as `-D` parameters.
- bubblewrap unshares every namespace and, before each command, masks the
  workspace's existing denied paths using `Workspace.is_denied`. It has
  unit tests only; it hasn't run on Linux yet.

Run limits (`AgentLimits`, from `agent_max_steps`, `agent_run_timeout`, and
`agent_run_backstop_timeout`):

- `agent_max_steps` is the run's `recursion_limit`. `react` reads
  `remaining_steps` and, when another tool round can't fit, replaces the
  tool-calling response with `STEP_LIMIT_MESSAGE`, so a run never ends in
  `GraphRecursionError` with unanswered tool calls.
- `agent_run_timeout` is a monotonic, process-local deadline checked before
  model calls and after a model response that requests tools. When it
  expires, `react` ends the graph between steps with `TIME_LIMIT_MESSAGE`,
  preserving any completed tool result or final model answer.
  Resuming an approved action refreshes this deadline. The larger
  `agent_run_backstop_timeout` still wraps the whole run in `asyncio.timeout`;
  it can cancel a tool mid-step, so `_close_dangling_tool_calls` sends an
  ambiguous-result placeholder on the next turn. Do not persist the absolute
  monotonic deadline across process restarts; a durable checkpointer needs a
  different deadline representation.

State and memory:

- `AgentState.messages` uses the `add_messages` reducer. Nodes return only
  what they change, typed as `AgentUpdate` (`{"messages": [new]}`), never
  the whole state.
- Conversations are keyed by `thread_id` in the run config
  (`config={"configurable": {"thread_id": ...}}`), not in the state.
  `invoke_agent` generates one when the request has none, and always returns
  it.
- `ThreadStore` (`app/agent/threads.py`) owns the `InMemorySaver` and its
  retention. Wrap every run in `async with threads.use(thread_id):`, or
  `threads.use_existing(thread_id)` for a request that can only continue a
  thread (it never tracks an unknown id).
  - One run per thread at a time: a second `use()` on a busy thread raises
    `ThreadBusyError`, which `/invoke` returns as `409`.
  - Idle threads unused for `thread_ttl` seconds are deleted when a run
    starts or finishes. When a run finishes, idle threads beyond
    `max_threads` are deleted, least recently used first. Threads with a run
    in progress are never deleted, so the cap can be exceeded by the number
    of concurrent runs. There's no timer.
  - Starting a run and evicting take `ThreadStore._lock`. A finishing run
    frees its thread synchronously without the lock, so freeing can't be
    cancelled or delayed. Eviction therefore re-reads the state after every
    await, never working from a copy. Deletes can await I/O.
  - A failed delete is logged, and the thread stays tracked for a retry. It
    never fails the run that triggered the cleanup.

Workspace boundary:

- `Workspace.resolve()` is the only path boundary for agent code. It rejects
  paths and symlinks outside `workspace_root`, and denied paths both before
  and after following symlinks.
  - `.env*`, `.git`, and `.claude` are always denied
    (`REQUIRED_DENIED_PATTERNS`); `workspace_denied_paths` adds to them. The
    deny list is the only protection for secrets: `.gitignore` handling is a
    noise filter for `find` and `grep`, and global or `.git/info/exclude`
    ignore rules aren't read at all.
  - `react`'s system prompt names the required patterns, so the model
    doesn't waste a tool call on them or offer files it can't read. It never
    names configured patterns, only that others exist: operators deny those
    for being sensitive, and a hosted provider sees every prompt.
  - Patterns are globs matched case-insensitively and Unicode-normalized.
    One without `/` matches a name at any depth; one with `/` is rooted at
    the workspace and covers everything beneath it.
- The workspace is validated and built at startup, then injected into the
  graph. Future tools must receive this instance instead of constructing
  paths directly.
- Project context is capped by `workspace_context_max_chars` and reloaded
  every turn so instruction changes take effect without restarting. An
  inaccessible `AGENTS.md` is skipped in favor of `README.md`; if neither can
  be read, the base prompt is used.
- Each turn logs which context file was loaded, its size, and whether it was
  truncated, or that none was found. A skipped file is a warning with its
  error type and, for `WorkspaceAccessError`, its `reason` (`outside` or
  `denied`). Never log the context text itself.

Around every request:

- `RequestContextMiddleware` takes a valid `X-Request-ID` (or generates
  one), adds it to the logging context, and returns it in the response
  header. Unhandled errors are logged there and returned as a generic `500`
  that still carries the header. It also logs `Request completed` for every
  request, with method, path (never the query string), status, and duration.
  uvicorn's access log repeats this without the request id; pass
  `--no-access-log` where the duplicate is noise.
- OpenAI provider errors map to `504` (timeout) or `502` (anything else),
  with a generic message. Never put provider error text in a response.
- Model calls use `llm_timeout` and `llm_max_retries` from `SvcSettings`;
  arguments passed to `LLM(...)` override them. `react` logs each call's
  duration and token usage, and a failed call's duration and error type
  (inside the thread's logging context, which the HTTP error handlers are
  outside of).
- `Agent invocation started` carries `thread_supplied`. A supplied thread
  that finishes with `conversation_message_count=2` had no history: it was
  new or had expired.
- Each tool call logs `Tool call completed` with its name, status, and
  duration. A rejected path logs `Tool path rejected` with the tool, the
  requested path, and its `reason`. Never log tool output.

Test UI (step 4b, plan in `docs/test-ui.md`):

- `dev_ui_enabled` (default off) gates the page and
  `GET /threads/{thread_id}/messages`. Startup copies it to `app.state`, and
  `require_dev_ui` returns FastAPI's own `404` when it's off. Put any other
  developer-only route behind the same dependency.
- The history endpoint (`read_thread_history` in `service.py`) reads the
  last checkpoint with `graph.aget_state`, not through `ThreadStore.use()`,
  so it works during a run and doesn't refresh the thread's TTL. It sends
  `Cache-Control: no-store`, because tool results include file contents.
- The page renders every reply and tool result with `textContent`; never use
  `innerHTML` or anything else that parses text as HTML or code
  (`tests/unit/test_ui_static.py` checks). It's served with a strict CSP and
  no inline script or style, so keep JavaScript in `app.js` and CSS in
  `app.css`. All API calls go through `apiBase` in `app.js`.
- The page uses POST streams for new turns and approval continuations. It
  parses fragmented UTF-8 SSE with `sse.js`, renders text and tool activity
  as it arrives, and fetches history on page reload or error recovery.
- The page picks a new conversation's `thread_id` itself
  (`crypto.randomUUID()`), so a first turn that the server saved but answered
  with an error (`502`, `504`) still shows up. Error responses carry no
  `thread_id`.

The graph is still a skeleton and will change. Update this section when you
change its shape.

## Conventions

- mypy runs in strict mode over `app`, `common`, and `tests`. Annotate
  everything, including tests and fixtures. Avoid `Any` and
  `# type: ignore`; if one is unavoidable, add a comment explaining why
  (see the `cast` in `app/service/lifespan.py`).
- Use modern typing: `type Alias = ...`, `X | None`, `collections.abc`.
- Graph nodes and route handlers are `async`.
- Read configuration only through `SvcSettings` in `common/config.py`. Never
  read `.env` or `os.environ` directly in app code. New settings get a field
  there and, if user-facing, an entry in `.env.example`.
- Keep secrets as `SecretStr`. Never log them.
- Use `logging.getLogger(__name__)`. Pass per-request fields with
  `logging_context(...)` from `common/logging_helper.py`.
- Add new endpoints as a feature package with its own router, and register
  it in `create_app()`.

## Testing

- Default test runs must never call a real model. The `app` fixture injects
  `EchoChatModel` from `tests/fakes.py` through `create_app(llm=...)`.
  Pass a fake model the same way instead of monkeypatching `LLM`. The
  `echo_model` fixture is the same instance; its `calls` record what the
  model received.
- Fakes must implement `bind_tools`, because startup binds the read tools.
  `EchoChatModel` accepts tools but never calls them. For tool runs, use
  `ScriptedChatModel(responses=[...])`, which returns its responses in order
  and fails when they run out, with `tool_call(name, id, **args)` to build a
  tool-calling response.
- Unit-test tools by invoking them with a tool-call dict, the way `ToolNode`
  does, so errors come back as `ToolMessage(status="error")` (see `call` in
  `tests/unit/test_read_tools.py`).
- `tests/conftest.py` sets `SANDBOX_BACKEND=off` for every test, so app
  startup skips the sandbox self-test. Tests that need a sandbox build one.
  `UnconfinedSandbox` in `tests/fakes.py` confines nothing and exists only to
  test process handling; `FakeSandbox` runs nothing and records calls.
  The opt-in tests against the real backend need
  `RUN_SANDBOX_TESTS=1 uv run pytest -m sandbox`.
- `serve(model, root, extra_tools)` in `tests/apps.py` runs the app with its
  lifespan, a given model, and a workspace, when the `client` fixture's
  `EchoChatModel` won't do. `recording_tool(effects)` in `tests/fakes.py` is
  a tool that needs approval and records what it did.
- A graph swapped into `app.state.agent_graph` needs
  `checkpointer=app.state.thread_store.checkpointer`, like the real one:
  `invoke_agent` reads saved state to check for a pending approval. `GatedChatModel` in `tests/fakes.py` holds every call until
  `release` is set, to test behavior while a run is in progress.
- Tests that need the test UI set `app.state.dev_ui_enabled` explicitly, so a
  developer's `.env` can't change the result. SSE parsing and streamed UI
  state have dependency-free Node tests: `uv run node --test tests/js/streaming.test.cjs`.
  Check visual layout by hand (see `docs/test-ui.md`).
- Tests that make live LLM calls need a pytest marker (`openai_live`,
  `local_llm_live`) and an env-var `skipif` gate, like
  `tests/integration/test_openai_llm.py`.
- Unit tests that touch settings must clear the `LLM_*` and `OPENAI_API_KEY`
  environment variables and pass explicit `SvcSettings` (see
  `tests/unit/test_llm.py`), so they don't depend on a developer's `.env`.
- Async tests use `@pytest.mark.anyio`.

## Gotchas

- The LLM is built at startup, so the server refuses to start when the
  provider is misconfigured (no `OPENAI_API_KEY` for `openai`, or no
  `LLM_BASE_URL` for `local`).
- Conversation memory lives in the server process: it's lost on restart and
  isn't shared between workers, so run a single worker until step 8.
- Any caller can continue any `thread_id` they know. There's no ownership
  check until authentication lands (step 7b, before anything is reachable
  beyond the user's machine).
- The service is meant to run only on the user's machine, bound to
  `127.0.0.1`. Never start it with `--host 0.0.0.0`. Even so, it accepts any
  `Host` header, so a web page open in the user's browser can reach it
  through DNS rebinding. That's an accepted risk until authentication lands
  in step 7b, including once step 6 adds write and command tools (see
  `docs/next-steps.md`).
- The configured workspace root must exist and be a directory. Context files
  are optional; without either `AGENTS.md` or `README.md`, the base system
  prompt is used.
- `workspace_root` defaults to the process working directory. Set it
  explicitly in deployment before file-writing tools are enabled, or the
  service may be able to edit its own checkout. The `Startup complete` log
  shows the resolved root and the deny patterns in effect.
- `agent_run_timeout` is a deadline checked before each model invocation and
  after a response that requests tools, so the graph stops between steps
  rather than cancelling an approved side-effecting tool mid-step. A final
  answer returned after the deadline is retained. `agent_run_backstop_timeout` is a
  larger last resort for a genuinely stuck run and can still cancel a tool
  running in a worker thread. In that case its side effect may or may not
  have happened, and the next turn receives the ambiguous-result placeholder.
- `grep` uses Python's `re`, which has no timeout: a pathological pattern can
  keep a worker thread busy after the run deadline cancels the request.
  Tools run in threads, and threads can't be cancelled.
- Project files reach the model as tool results, and the model treats them
  as input. A malicious repository can try to steer the agent through them;
  approval (step 5a) is the safeguard before any tool has side effects.
- With `dev_ui_enabled`, anyone who can reach the server can read any
  thread's history, tool results included. Keep it off outside local
  development until 7b adds authentication and ownership checks.
- Inside the sandbox, a project's `.env` is unreadable on macOS
  (`PermissionError`) and empty on Linux. Code that loads `.env`, like this
  repo's `SvcSettings`, therefore fails sandboxed on macOS: 73 of this
  repo's tests do.
- Toolchains installed under the home directory (uv's Python, pyenv, cargo)
  need their paths in `sandbox_read_paths`; the home directory itself is
  refused.
- On macOS, a process started with `posix_spawn`'s `setsid` or `setpgroup`
  option escapes the group kill; only the sweep ends it. A process that
  runs as another user (a setuid program) can't be signalled: the command
  fails with `ContainmentError` and commands stay off until restart. On
  Linux the PID namespace ends everything.
- Inside the sandbox a denied-name symlink reaches its target
  (`.env.link -> notes.txt` reads `notes.txt`), on both backends, though
  `Workspace.resolve` refuses the name. The deny list protects files where
  they are; the target is readable under its own name anyway.

## Rules

- Never commit `.env` or real API keys. `.env.example` holds placeholders only.
- Don't add dependencies without saying so. Use `uv add` (or
  `uv add --dev`) so `uv.lock` stays in sync. CI installs with `--locked`.
