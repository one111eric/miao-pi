# AGENTS.md

Guidance for coding agents working in this repository. Human-facing setup and
usage live in `README.md`.

miao-pi is a coding agent served over HTTP: a FastAPI service wrapping a
LangGraph workflow that calls an OpenAI-compatible chat model.

## Commands

Python 3.13.7, managed with `uv`. Always run tools through `uv run`.

```bash
uv sync                                                    # install deps
uv run uvicorn app.service.server:app --reload --port 9091 # dev server
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
- `app/<feature>/` — one package per API feature (`agent`, `health`), each
  with `router.py` and `models.py` (Pydantic request/response models).
- `app/agent/` — the LangGraph workflow: `state.py` (graph state),
  `nodes.py` (async node functions), `graph.py` (wiring and compile),
  `service.py` (runs the graph for a request), `workspace.py` (the
  boundary for project files), and `tools/` (the read-only tools, one module
  each, plus shared `files.py` and `output.py`).
- `common/` — code shared across features: `config.py` (`SvcSettings`),
  `llm/` (`LLM`, a `ChatOpenAI` subclass), `logging_helper.py`.
- `tests/unit/` — no network, no app lifespan.
- `tests/integration/` — go through the ASGI app via the `client` fixture in
  `tests/conftest.py`.

## Architecture

Request flow: `POST /invoke` → `app/agent/router.py` → `invoke_agent()` →
compiled graph `preplanning → react ⇄ tools → finalize`.

- `preplanning` reloads `AGENTS.md` (falling back to `README.md`) from the
  workspace on every turn, with a configured size cap.
- `react` calls the LLM, which `build_agent_graph(llm, tools=...)` injects
  through `make_react(llm, tools)`, binding the tools there. It builds a
  system message from `SYSTEM_PROMPT`, the tool names, and the current
  workspace context at call time; none of it is stored in message history.
- `tools` is LangGraph's `ToolNode`, reached through `tools_condition` and
  looping back to `react` until the model answers without a tool call.
  `log_tool_call` wraps every call. Without tools, `react` goes straight to
  `finalize`.
- `finalize` is a placeholder for post-processing.

Tools (`app/agent/tools/`):

- `make_read_tools(workspace)` builds `read`, `ls`, `find`, and `grep`,
  shaped after pi's coding agent: same names, parameters, caps (2000 lines or
  50KB, 500 entries, 1000 results, 100 matches), and actionable truncation
  notices. Results are relative to the workspace root.
- Every path goes through `Workspace`. `files.resolve_for_tool` and
  `files.open_for_tool` turn workspace and OS errors into `ToolException`s;
  each tool sets `handle_tool_error=True`, so they reach the model as error
  results. Only bugs propagate. Never put an absolute server path in a tool
  result.
- `find` and `grep` walk the tree in Python (`files.iter_files`) instead of
  running `fd`/`rg`, so the deny list and `.gitignore` (via `pathspec`)
  apply exactly. Denied and escaping entries are hidden, not reported.
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

Run limits (`AgentLimits`, from `agent_max_steps` and `agent_run_timeout`):

- `agent_max_steps` is the run's `recursion_limit`. `react` reads
  `remaining_steps` and, when another tool round can't fit, replaces the
  tool-calling response with `STEP_LIMIT_MESSAGE`, so a run never ends in
  `GraphRecursionError` with unanswered tool calls.
- `agent_run_timeout` wraps the whole run in `asyncio.timeout`; expiry raises
  `AgentRunTimeoutError`, which `/invoke` returns as `504`. A cancelled run
  can leave a tool call without a result in the checkpoint, so `react`
  sends the model a placeholder `ToolMessage` for each one
  (`_close_dangling_tool_calls`) without changing the saved history.

State and memory:

- `AgentState.messages` uses the `add_messages` reducer. Nodes return only
  what they change, typed as `AgentUpdate` (`{"messages": [new]}`), never
  the whole state.
- Conversations are keyed by `thread_id` in the run config
  (`config={"configurable": {"thread_id": ...}}`), not in the state.
  `invoke_agent` generates one when the request has none, and always returns
  it.
- `ThreadStore` (`app/agent/threads.py`) owns the `InMemorySaver` and its
  retention. Wrap every run in `async with threads.use(thread_id):`.
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
  - `.env*` and `.git` are always denied (`REQUIRED_DENIED_PATTERNS`);
    `workspace_denied_paths` adds to them.
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
  check until authentication lands (step 5a).
- The configured workspace root must exist and be a directory. Context files
  are optional; without either `AGENTS.md` or `README.md`, the base system
  prompt is used.
- `workspace_root` defaults to the process working directory. Set it
  explicitly in deployment before file-writing tools are enabled, or the
  service may be able to edit its own checkout. The `Startup complete` log
  shows the resolved root and the deny patterns in effect.
- `grep` uses Python's `re`, which has no timeout: a pathological pattern can
  keep a worker thread busy after the run deadline cancels the request.
  Tools run in threads, and threads can't be cancelled.
- Project files reach the model as tool results, and the model treats them
  as input. A malicious repository can try to steer the agent through them;
  approval (step 5b) is the safeguard before any tool has side effects.

## Rules

- Never commit `.env` or real API keys. `.env.example` holds placeholders only.
- Don't add dependencies without saying so. Use `uv add` (or
  `uv add --dev`) so `uv.lock` stays in sync. CI installs with `--locked`.
