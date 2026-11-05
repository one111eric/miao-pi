# Next steps

Roadmap from the current skeleton to a working coding agent. Last reviewed
2026-11-03, against langgraph 1.2.17, langchain-core 1.7.0,
langchain-openai 1.7.2, and fastapi 0.149.0.

## Where things stand

- FastAPI service with `/health` and `POST /invoke`; strict mypy, black,
  isort, and pytest all pass in CI.
- The LLM (`common/llm/LLM`) is built once at startup and injected into the
  graph: `create_app(llm=...)` → `build_agent_graph(llm)` → `make_react(llm)`.
  A misconfigured provider stops the server at startup.
- Graph: `preplanning → react ⇄ tools → finalize`. The model can loop
  through four read-only tools (`read`, `ls`, `find`, `grep`) within a step
  limit and a run timeout (step 4).
- With `DEV_UI_ENABLED=true`, a test page at `/ui/` shows each turn's reply
  and tool calls, backed by `GET /threads/{thread_id}/messages` (step 4b).
- Tools marked as needing approval pause the run; `/invoke` returns
  `awaiting_approval` with the actions, and
  `POST /threads/{thread_id}/resume` approves or rejects them (step 5a). No
  shipped tool needs it yet.
- Conversations continue across requests that share a `thread_id`, kept in
  an in-memory checkpointer with a TTL and a thread cap (step 2).
- A validated workspace boundary rejects escapes and denied paths; project
  instructions are reloaded into the system prompt every turn (step 3).
- Requests carry an `X-Request-ID` through the logs, provider errors map to
  `502`/`504`, and model calls use the `llm_timeout`/`llm_max_retries`
  settings (step 1).
- Tests inject `EchoChatModel` (`tests/conftest.py`), so default runs never
  call a real model.

## Order

```
0.  Land current work
1.  Request IDs, error mapping, model timeout/retries
2.  Message reducer + in-memory checkpointer (with retention cap)
3.  Workspace boundary + project context
4.  ReAct loop + read-only tools
4b. Test UI (a developer page for trying the agent)
5a. Approval (interrupt + resume API)
5b. Command sandbox
6.  Write and command tools
7.  Streaming
7b. Authentication + thread ownership
8.  Persistent checkpointer and deployment
```

Do the steps in order. The guardrails come before the capabilities they
protect:

- Timeouts and error mapping (1) exist before the loop (4) can make many
  paid calls.
- The workspace boundary (3) exists before any tool touches the file system.
- Approval and the sandbox (5a, 5b) exist before the agent can write files
  or run commands (6).
- Authentication (7b) exists before the service, or a hosted UI, can be
  reached from anywhere but the user's own machine (8).

**Accepted risk until 7b:** authentication waits because, until step 8, the
service runs only on the user's own machine, bound to `127.0.0.1`. That
isn't fully private: the server accepts any `Host` header, so through DNS
rebinding a web page open in the user's browser can reach the API (checked
on 2026-11-04: a request with `Host: evil.example` got `200`). Today that
page could have the agent read workspace files. From 5a it could also
approve its own pending action, and from step 6 it could have the agent
write files and run commands. If that becomes unacceptable before 7b,
Starlette's `TrustedHostMiddleware`, allowing only `localhost`, `127.0.0.1`,
and `[::1]`, closes most of the gap in one line.

**Gate on step 6:** don't merge write or command tools until 5a and 5b have
both landed. An executable allowlist alone isn't a security boundary:
`pytest` runs `conftest.py`, `git` runs hooks, `npm` runs scripts, and
`python` runs anything. Combined with `write_file`, the agent can write a
file and then execute it.

When a step lands, update the Architecture and Gotchas sections in
`AGENTS.md`, and tick it off here.

---

## 0. Land current work

**Status: done (2026-10-30).**

`master` has had nothing merged since the first commit. `agent-setup` carries
the server, LLM client, graph skeleton, the small fixes, `AGENTS.md`,
`CLAUDE.md`, and this document.

- Commit the uncommitted work on `agent-setup`.
- Open a PR into `master` and merge it once CI passes. CI now runs on pushes
  to `master` too.
- Delete the stale local branches (`server-setup`, `llm-setup`) once they're
  merged.

**Done when:** `master` contains this branch, and CI is green on `master`.

---

## 1. Request IDs, error mapping, model timeout

**Status: done (2026-10-31).**

**Goal:** every request is traceable in the logs, model calls can't hang,
and provider failures return meaningful status codes.

**Changes**

- `common/config.py`: add `llm_timeout` (seconds, for example 60) and
  `llm_max_retries` (for example 2), and add both to `.env.example`.
- `common/llm/llm.py`: pass them to `ChatOpenAI` as `timeout` (the alias for
  `request_timeout`) and `max_retries`, unless the caller overrides them.
- A middleware (for example `app/service/middleware.py`):
  - Takes the request id from `X-Request-ID`, or generates one.
  - Wraps the request in `logging_context(request_id=...)`.
  - Echoes the header back on the response.
- Exception handlers registered in `create_app()`:
  - `openai.APITimeoutError` → `504`.
  - Other `openai.APIError` → `502`.
  - `APITimeoutError` is itself a subclass of `APIError`. FastAPI picks the
    most specific handler, but test both cases to be sure.
  - Return a short generic message. Never echo provider error text, which
    can contain request details.
- Log model-call duration and token usage (from the response's
  `usage_metadata`) at info level.

**Tests**

- The request id appears in captured log records and in the response
  header. A supplied `X-Request-ID` is preserved.
- A fake model that raises `APITimeoutError` → `504`; one that raises
  another `APIError` → `502`; the response body has no provider text.
- `LLM` picks up the timeout and retry settings (unit test, like the
  existing ones in `tests/unit/test_llm.py`).

**Done when:** the tests pass, and the new settings are in `.env.example`.

---

## 2. Message reducer and in-memory checkpointer

**Status: done (2026-11-01).** Retention lives in `ThreadStore`
(`app/agent/threads.py`); runs go through `async with threads.use(...)`.
Beyond the plan below: a thread runs one request at a time (a concurrent
request gets `409`), and threads with a run in progress are never evicted.

**Goal:** a conversation continues across requests that share a `thread_id`,
without memory growing forever.

**Changes**

- `app/agent/state.py`
  - `messages: Annotated[list[AnyMessage], add_messages]`
    (`from langgraph.graph.message import add_messages`).
  - Remove `thread_id` from the state. It belongs in the run config, not the
    state.
  - Add a partial update type for node return values, for example an
    `AgentUpdate` `TypedDict` with `total=False`.
- `app/agent/nodes.py`
  - Nodes return only what they change: `react` returns
    `{"messages": [response]}`.
  - **Update the `AgentNode` protocol** to return `Awaitable[AgentUpdate]`,
    and change the return types of `preplanning` and `finalize` to match.
    Today it requires `Awaitable[AgentState]`, which strict mypy will reject
    for partial returns.
  - **Move the system prompt out of the state.** With `add_messages`, a
    `SystemMessage` returned from `preplanning` is *appended* after the
    user's message, and the checkpointer would save it into every thread.
    Instead, `react` adds it at call time:
    `await llm.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])`.
  - `preplanning` becomes a no-op placeholder until step 3.
- `app/agent/graph.py`
  - `build_agent_graph(llm, checkpointer: BaseCheckpointSaver | None = None)`,
    then `compile(checkpointer=checkpointer)`.
- `app/service/__init__.py`
  - Create an `InMemorySaver` (`from langgraph.checkpoint.memory import
    InMemorySaver`) at startup and pass it to the graph.
- **Retention cap**
  - Track each thread's last-use time, and evict old threads with
    `checkpointer.adelete_thread(thread_id)`.
  - Evict both by age (a `thread_ttl` setting) and by count (`max_threads`,
    dropping the least recently used).
  - Keep it in a small class next to the checkpointer, so step 8 can replace
    both together.
- `app/agent/service.py`
  - `thread_id = request.thread_id or str(uuid4())`.
  - `graph.ainvoke({"messages": [HumanMessage(...)]},
    config={"configurable": {"thread_id": thread_id}})`.
- `app/agent/models.py`
  - `AgentResponse.thread_id` becomes a required `str`, so clients always get
    an id they can continue with. This is an API change; note it in the PR.

**Tests**

- Make the fake model record the messages it receives.
- Two `/invoke` calls with the same `thread_id`: the second model call sees
  the first exchange (human, AI) plus the new human message.
- Different `thread_id`s don't share history.
- A request without `thread_id` gets a generated one back.
- The system prompt is the first message the model receives, and it isn't
  stored in the checkpointed state.
- Eviction: threads past the TTL, or beyond `max_threads`, are deleted, and
  a later request on an evicted id starts a fresh conversation.

**Done when:** the tests pass, and the `thread_id` gotcha is removed from
`AGENTS.md`.

**Notes**

- Everything is still lost on restart and isn't shared between workers;
  step 8 fixes that.
- Thread ownership (stopping one caller from using another's `thread_id`)
  only means something once there are authenticated callers, so it lands in
  7b. Until then, generated ids are unguessable `uuid4`s, but a client can
  still supply any id it likes.

---

## 3. Workspace boundary and project context

**Status: done (2026-11-02).** `Workspace` is created at startup and injected
into the graph; context is capped and never stored in message history.

**Goal:** one place decides which files the agent may touch, and the agent
knows the project it's working in before its first model call.

**Decisions**

- One fixed `workspace_root` setting, or a path per request?
  Recommendation: a fixed setting for now. A per-request workspace can come
  later, behind authentication.

**Changes**

- New `app/agent/workspace.py` with a `Workspace` class built from
  `workspace_root`:
  - `resolve(path) -> Path` resolves with `Path.resolve()` and raises a
    dedicated error for anything outside the root (`is_relative_to`). This
    covers `..`, absolute paths, and symlinks that escape.
  - Every later tool goes through it. No tool builds paths on its own.
  - Required case-insensitive name patterns `.env*`, `.git`, and `.claude`
    reject matching names at every depth, before and after following
    symlinks. Configured deny patterns are additive. (`.claude` was added
    after the first test against a real model: local coding-agent settings
    are often hidden from git only by a global ignore file.)
- `common/config.py`: add `workspace_root` and the deny list, and add both to
  `.env.example`. Validate at startup that the root exists and is a
  directory.
- `AgentState` gains `context: NotRequired[str]`.
- `preplanning` loads the workspace's `AGENTS.md` (falling back to
  `README.md`) through `Workspace`, with a size cap, and returns
  `{"context": ...}`. Later it can add a directory overview or the git
  branch. Inaccessible context files are skipped rather than failing a
  request.
- `react` builds the system message from `SYSTEM_PROMPT` plus `context`. It
  stays out of `messages`, for the reasons in step 2.
- Reload context every turn. It's cheap and the files may change; cache
  later if needed.

**Tests**

- `Workspace.resolve` in a `tmp_path`: normal paths pass; `../`, absolute
  paths outside the root, symlinks pointing out, and denied paths are all
  rejected.
- Context: an `AGENTS.md` in the workspace reaches the model's system
  message; a missing file is handled; the size cap is applied; nothing from
  the context lands in the message history.

**Done when:** `Workspace` passes the escape tests, and the model receives
workspace context.

---

## 4. ReAct loop and read-only tools

**Status: done (2026-11-03).** The tools follow pi's coding agent
(<https://github.com/earendil-works/pi/tree/main/packages/coding-agent>,
`src/core/tools/`); the differences are listed below.

**Goal:** the model can inspect the workspace with tools, repeatedly, until
it gives a final answer.

**Tools** (all read-only, all through `Workspace`), with pi's names,
parameters, and caps:

1. `read(path, offset?, limit?)`: 2000 lines or 50KB, whichever comes
   first; a cut ends with "Use offset=N to continue."
2. `ls(path?, limit?)`: sorted, `/` after directories, dotfiles included,
   500 entries.
3. `find(pattern, path?, limit?)`: glob; a pattern without `/` matches the
   file name, one with `/` matches the path. 1000 results.
4. `grep(pattern, path?, glob?, ignore_case?, literal?, context?, limit?)`:
   `path:line: text`, context lines as `path-line- text`, 100 matches, 500
   characters per line.

Shared rules:

- Truncate large outputs and say how to get the rest, so the model knows.
  Fetch one result past a limit to tell "exactly N" from "more than N".
- Return errors (missing file, path outside the workspace) as tool results
  the model can read and react to. Don't raise through the graph.
- Log each tool call with its name, duration, and outcome, and log a
  rejected path with `WorkspaceAccessError.reason`. Never log file contents.

Where miao-pi differs from pi, and why:

- pi trusts the local machine. Here every path goes through `Workspace`:
  `read` rejects denied and escaping paths, and `ls`, `find`, and `grep`
  leave them out.
- pi runs `fd` and `rg`. Here `find` and `grep` walk the tree in Python, so
  the deny list (case-insensitive, Unicode-normalized, symlink-aware) applies
  exactly, and nothing needs installing. `.gitignore` support comes from
  `pathspec`. `grep` uses Python's `re` syntax.
- pi shows paths relative to the search directory. Here they're relative to
  the workspace root, so a result can be passed straight back to `read`.
- pi reads images and suggests `bash` for over-long lines. Here binary files
  are refused and there's no `bash` until step 6.
- pi reports a cut read as "lines X-Y of N". Here the notice gives the file
  size instead, so `read` stops at the cap rather than reading a huge file to
  the end just to count its lines.
- pi's local conveniences (`~` expansion, `@` prefixes, macOS screenshot
  file names) are left out; they don't apply to a server.

**Changes**

- `app/agent/tools/`: `make_read_tools(workspace)`; one module per tool,
  plus `files.py` (path checks, error mapping, the walker) and `output.py`
  (caps and notices).
- `Workspace.open_file`: checks that the root is still the directory found
  at startup (device and inode), opens each path component relative to its
  parent with `O_NOFOLLOW`, then the file with `O_NONBLOCK`, and accepts only
  regular files. That closes the symlink-swap and FIFO gaps noted in step 3:
  `O_NOFOLLOW` on the full path alone protects only its last component, and
  reopening the root by path would follow a swapped root. A path with a NUL
  byte is a `WorkspaceError`.
- `make_react(llm, tools)` binds the tools; `ToolNode` with
  `awrap_tool_call=log_tool_call`; `react → tools_condition → {tools,
  finalize}`, then `tools → react`.
- `agent_max_steps` (default 25) is the run's `recursion_limit`. Instead of
  catching `GraphRecursionError`, `react` reads LangGraph's `RemainingSteps`
  and ends with `STEP_LIMIT_MESSAGE` when another tool round can't fit. A
  `GraphRecursionError` would have left unanswered tool calls in the thread,
  and providers reject that history on the next turn.
- `agent_run_timeout` (default 300 seconds) bounds the whole run and maps to
  `504`. A timed-out run can still leave unanswered tool calls in the
  checkpoint, so `react` adds placeholder results when it sends history to
  the model.

**Tests:** `ScriptedChatModel` and `tool_call()` in `tests/fakes.py`
(`EchoChatModel` gained a no-op `bind_tools`); unit tests per tool for caps,
notices, denied, escaping, ignored, and binary files, and FIFOs; scripted
runs for read → answer, error results, the step limit at several values,
and a timeout followed by a recovered turn.

**Notes:**

- For the local provider, check that the chosen model supports tool calling
  over the OpenAI-compatible API (many Ollama models do, but not all).
  Document which models are known to work.
- `re` has no timeout, so a pathological `grep` pattern can keep a worker
  thread busy after the deadline. If that matters, switch to `google-re2`
  or run `grep` in the step 5b sandbox.

---

## 4b. Test UI

**Status: built (2026-11-04)**; the manual check against a real model is
still to do.

A developer page for trying the agent in a browser, with the tool calls
and results it made. The plan is in [test-ui.md](test-ui.md).

---

## 5a. Approval: interrupt and resume

**Status: done (2026-11-04).** Two changes from the plan below:

- The `interrupt` lives in its own `approve` node, between `react` and
  `tools`, not inside each tool. On resume LangGraph reruns the whole
  interrupted node, and `ToolNode` would rerun every call in it. A gate in
  front of each tool call also refuses an approval-requiring call without an
  approved decision, so a routing mistake fails closed.
- A paused run returns `actions` (a list): one model response can ask for
  several tools at once, and one decision covers them all. Diffs come with
  the write tools in step 6.

`/invoke` on a waiting thread returns `409`, because a new run would
silently drop the pending one.

**Goal:** the agent pauses before any action with side effects, and a human
approves or rejects it.

**Changes**

- Tools that change things (introduced in step 6) call LangGraph
  `interrupt(...)` (`from langgraph.types import interrupt, Command`) with a
  description of the pending action: the tool, its arguments, and for writes
  the diff. The checkpointer from step 2 holds the paused run.
- API changes:
  - `/invoke` returns either a final response or a pending action:
    `{"status": "awaiting_approval", "thread_id": ..., "action": {...}}`.
  - A new `POST /threads/{thread_id}/resume` takes
    `{"approved": bool, "note"?: str}` and continues the run with
    `Command(resume=...)`.
  - A rejection reaches the model as a tool result, so it can change its
    plan.
- Design the response shapes so step 7's streaming endpoint can emit the
  same pending-action payload as an event, and reuse the same resume
  endpoint.
- Build and test this with a test-only tool that has a side effect. The real
  write and command tools come in step 6.
- The 4b test page shows a pending action (tool, arguments, diff) with
  Approve and Reject buttons that call the resume endpoint.

**Tests:** scripted run → pending action returned → approve → the tool runs
and the run finishes; reject → the tool doesn't run, and the model sees the
rejection; resuming a thread with nothing pending → `409`.

**Note:** until 7b, anything that can reach the API can approve an action as
easily as request one (see the accepted risk under Order).

**Done when:** a side-effecting test tool can't run without an approval
through the API.

---

## 5b. Command sandbox

**Goal:** commands the agent runs can't reach beyond the workspace, even if
the command itself is malicious.

**Decisions**

- Container per command (Docker or Podman) or a long-lived sandbox per
  thread? Recommendation: a container per command with the workspace mounted
  read-write. It's simplest, and nothing carries over between commands.
- Is network access needed (installing dependencies)? Recommendation: no
  network by default, with a per-command opt-in that also requires approval.

**Changes**

- New `app/agent/sandbox.py` with one narrow interface, for example
  `async run(argv: list[str], timeout: float) -> CommandResult`, so the
  backend can change later.
- Container settings:
  - No network.
  - Only the workspace mounted, and not `.env` or anything else from the
    deny list.
  - A non-root user, a read-only root file system, and CPU, memory, and
    process limits.
  - A wall-clock timeout that kills the container.
- Settings for the image and the limits in `SvcSettings` and `.env.example`.
- Check at startup that the container runtime is available, or refuse to
  enable command tools.

**Tests**

- Interface-level tests against a fake backend.
- Real-container integration tests, opt-in behind an env flag like the live
  LLM tests: a timeout kills the command; there's no network; files outside
  the workspace aren't visible; output is captured and capped.

**Done when:** the sandbox passes its opt-in integration tests, and the
decisions above are written down in `AGENTS.md`.

---

## 6. Write and command tools

**Goal:** the agent can change code and run commands, safely.

**Gate:** 5a and 5b have both landed. Authentication (7b) comes later; see
the accepted risk under Order.

**Tools**

1. `write_file(path, content)` and/or `apply_patch(diff)`, through
   `Workspace`. Every write requires approval (5a), showing the diff.
2. `run_command(argv)`, executed only through the sandbox (5b), never on the
   host. Every command requires approval (5a), showing the exact argv. A
   short allowlist can stay as a convenience filter, but it's not the
   security boundary.

**Rules**

- Writes into the deny list (`.env`, `.git/`) are rejected before the
  approval step.
- Command output is truncated and capped like the read tools.
- Combine with the read tools into `TOOLS` in `app/agent/tools/`.

**Tests**

- Unit tests per tool in a `tmp_path` workspace.
- A scripted graph run: read a file → write a change (approved) → run tests
  in the sandbox (approved) → final answer. Run it against the fake sandbox
  backend by default.
- Rejecting the write leaves the file unchanged.

**Done when:** the scripted run passes, and neither tool can act without
approval.

---

## 7. Streaming

**Goal:** clients see tokens, tool activity, and approval requests as they
happen.

**Changes**

- `POST /stream` in `app/agent/router.py`, with the same request model as
  `/invoke` (and 7b's token check, once it lands), returning
  `fastapi.sse.EventSourceResponse` (available in the installed FastAPI).
- Drive it with `graph.astream(..., stream_mode=["messages", "updates"])`.
  It's simpler to map than `astream_events`; switch only if finer events are
  needed.
- Event vocabulary, documented in the OpenAPI description:
  - `token`
  - `tool_call`
  - `tool_result`
  - `awaiting_approval`, with the same payload as 5a
  - `done`, with the `thread_id`
  - `error`
- After an approval, the client uses the 5a resume endpoint; add a streaming
  variant of resume if needed.
- Errors after the stream has started must go out as an `error` event, not
  an HTTP status.
- Switch the 4b test page to the stream: show tokens and tool activity as
  they arrive, instead of fetching the history after each reply. Keep the
  history endpoint for reloading a conversation.

**Tests:** read the response with httpx streaming; with the scripted fake,
assert the event order for a run with one tool call, a run that pauses for
approval, and that `done` carries the `thread_id`.

**Done when:** scripted runs stream the expected events in order, including
the approval pause.

---

## 7b. Authentication and thread ownership

**Goal:** only known callers can use the agent, and each caller can only
reach their own threads.

**When:** before the service, or a hosted UI, can be reached from anywhere
but the user's own machine: step 8's deployment, a shared server, or the
hosting described in [test-ui.md](test-ui.md). It used to come right after
the test UI; it moved here on 2026-11-04 because a service that runs only on
the user's computer gains little from it. The risk that leaves open is described under Order.

**Changes**

- A bearer-token check as a FastAPI dependency on the agent routes (not
  `/health`).
  - Start with one token, or a small set, in a `SecretStr` setting.
  - Compare with `secrets.compare_digest`.
  - Refuse to start if no token is configured, unless an explicit
    `auth_disabled` setting is set for local development.
- Each token maps to a caller id. Record which caller created each thread
  (next to the retention tracking from step 2). A request for a thread owned
  by a different caller gets `404`, not `403`, so it doesn't reveal that the
  thread exists.
- Add the caller id to the logging context.
- Streaming (7) and the 5a resume endpoint get the same token check as
  `/invoke`.
- Add a `Host` allowlist (`TrustedHostMiddleware`) at the same time: it
  blocks DNS rebinding, and a deployment needs it anyway.
- The 4b history endpoint gets the same token check and ownership rule as
  `/invoke`. The test page gains a token field, keeps the token in
  `sessionStorage`, and sends it with every request.

**Tests:** missing or wrong token → `401`; `/health` stays open; caller B
can't continue or read caller A's thread; with auth disabled, requests work
without a token.

**Done when:** the agent routes require a token, and cross-caller access is
blocked.

---

## 8. Persistent checkpointer and deployment

**Goal:** threads survive restarts, and the service can run beyond a
developer laptop.

**Gate:** 7b has landed. Nothing is reachable beyond the user's machine
without authentication.

- Replace `InMemorySaver` with a persistent checkpointer: SQLite for a single
  instance, Postgres for several workers. Move thread ownership and retention
  into the same store, and replace step 2's in-memory eviction with a
  scheduled cleanup.
- Stop saving project context with every checkpoint. `context` lives in the
  graph state, so each turn stores another copy of up to
  `workspace_context_max_chars`. Load it inside `react` instead. Don't use
  `UntrackedValue`: a run resumed after a 5a approval continues at `react`
  without rerunning `preplanning`, so it would have no context.
- A Dockerfile for the service itself, separate from the command sandbox
  image.
- A run configuration with settings from the environment, and health checks
  wired to `/health`.

---

## Later

Not planned yet; revisit after step 6.

- Cost limits: `max_tokens`, and a per-thread or per-caller token budget.
- An eval set: a handful of real coding tasks against fixture repos, run on
  demand like the live LLM tests.
- Tracing (for example LangSmith) behind an opt-in setting.
- Per-request workspaces, building on 7b and 5b.
