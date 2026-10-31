# Next steps

Roadmap from the current skeleton to a working coding agent. Last reviewed
2026-10-31, against langgraph 1.2.17, langchain-core 1.7.0,
langchain-openai 1.7.2, and fastapi 0.149.0.

## Where things stand

- FastAPI service with `/health` and `POST /invoke`; strict mypy, black,
  isort, and pytest all pass in CI.
- The LLM (`common/llm/LLM`) is built once at startup and injected into the
  graph: `create_app(llm=...)` → `build_agent_graph(llm)` → `make_react(llm)`.
  A misconfigured provider stops the server at startup.
- Graph: `preplanning → react → finalize`, run once per request. No tools,
  no loop, no memory.
- `thread_id` is accepted and echoed back but not persisted.
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
5a. Authentication + thread ownership
5b. Approval (interrupt + resume API)
5c. Command sandbox
6.  Write and command tools
7.  Streaming
8.  Persistent checkpointer and deployment
```

Do the steps in order. The guardrails come before the capabilities they
protect:

- Timeouts and error mapping (1) exist before the loop (4) can make many
  paid calls.
- The workspace boundary (3) exists before any tool touches the file system.
- Authentication, approval, and the sandbox (5a–5c) all exist before the
  agent can write files or run commands (6).

**Gate on step 6:** don't merge write or command tools until 5a, 5b, and 5c
have all landed. An executable allowlist alone isn't a security boundary:
`pytest` runs `conftest.py`, `git` runs hooks, `npm` runs scripts, and
`python` runs anything. Combined with `write_file`, the agent can write a
file and then execute it.

When a step lands, update the Architecture and Gotchas sections in
`AGENTS.md`, and tick it off here.

---

## 0. Land current work

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
  5a. Until then, generated ids are unguessable `uuid4`s, but a client can
  still supply any id it likes.

---

## 3. Workspace boundary and project context

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
  - A deny list inside the workspace for paths the agent must never read or
    write, at least `.env` and `.git/`. Make it a setting.
- `common/config.py`: add `workspace_root` and the deny list, and add both to
  `.env.example`. Validate at startup that the root exists and is a
  directory.
- `AgentState` gains `context: NotRequired[str]`.
- `preplanning` loads the workspace's `AGENTS.md` (falling back to
  `README.md`) through `Workspace`, with a size cap, and returns
  `{"context": ...}`. Later it can add a directory overview or the git
  branch.
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

**Goal:** the model can inspect the workspace with tools, repeatedly, until
it gives a final answer.

**Tools** (all read-only, all through `Workspace`)

1. `list_dir(path)`
2. `read_file(path, start_line?, end_line?)`, with a size cap
3. `search(pattern, path?)`, a grep-like search with a result cap.
   Implement it in Python, or run a fixed `rg`/`grep` command with the
   pattern passed as an argument. Never let the model build a shell command.

Shared rules:

- Truncate large outputs and say they were truncated, so the model knows.
- Return errors (missing file, path outside the workspace) as tool results
  the model can read and react to. Don't raise through the graph.

**Changes**

- New `app/agent/tools/` package exporting a `READ_TOOLS: list[BaseTool]`.
  The tools take the `Workspace` from a factory, not from a global.
- `app/agent/nodes.py`
  - `make_react(llm, tools)` binds the tools inside:
    `model = llm.bind_tools(tools)`. Keep the parameter typed as
    `BaseChatModel`: `bind_tools` returns a `Runnable`, not a chat model.
- `app/agent/graph.py`
  - Add a `tools` node: `ToolNode(tools)` (`from langgraph.prebuilt import
    ToolNode, tools_condition`).
  - `react → tools_condition → {tools, finalize}`, then `tools → react`.
- `app/agent/service.py`
  - Pass `recursion_limit` in the run config (a new setting, for example
    `agent_max_steps`, default about 25).
  - Catch `GraphRecursionError` (`from langgraph.errors import
    GraphRecursionError`) and return a clear error instead of a bare 500.
  - Add an overall deadline for the whole run (for example
    `agent_run_timeout` with `asyncio.timeout`). Each model call can take
    `llm_timeout × (llm_max_retries + 1)`, about 4 minutes with the defaults,
    and the loop multiplies that by the number of steps.

**Tests**

- `EchoChatModel` can't do this: `BaseChatModel.bind_tools` raises
  `NotImplementedError` by default, and langchain's `GenericFakeChatModel`
  doesn't override it either. Replace it with a scripted fake:
  - It takes a list of `AIMessage`s and returns them in order.
  - `bind_tools` returns `self`.
  - It records the messages it receives.
  - Move it to `tests/fakes.py`.
- Unit tests per tool in a `tmp_path` workspace, including size caps,
  truncation, and an escape attempt returning an error result.
- Scripted run "`read_file` call → final answer": the tool runs, its
  `ToolMessage` reaches the model, and the final text is returned.
- A model that keeps calling tools stops at the step limit with a clean
  error.

**Done when:** a scripted multi-step run passes, and the step limit is
enforced.

**Notes:** for the local provider, check that the chosen model supports tool
calling over the OpenAI-compatible API (many Ollama models do, but not all).
Document which models are known to work.

---

## 5a. Authentication and thread ownership

**Goal:** only known callers can use the agent, and each caller can only
reach their own threads.

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

**Tests:** missing or wrong token → `401`; `/health` stays open; caller B
can't continue caller A's thread; with auth disabled, requests work without
a token.

**Done when:** the agent routes require a token, and cross-caller access is
blocked.

---

## 5b. Approval: interrupt and resume

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

**Tests:** scripted run → pending action returned → approve → the tool runs
and the run finishes; reject → the tool doesn't run, and the model sees the
rejection; resuming a thread with nothing pending → `409`; ownership from 5a
applies to resume.

**Done when:** a side-effecting test tool can't run without an approval
through the API.

---

## 5c. Command sandbox

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

**Gate:** 5a, 5b, and 5c have all landed.

**Tools**

1. `write_file(path, content)` and/or `apply_patch(diff)`, through
   `Workspace`. Every write requires approval (5b), showing the diff.
2. `run_command(argv)`, executed only through the sandbox (5c), never on the
   host. Every command requires approval (5b), showing the exact argv. A
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

- `POST /stream` in `app/agent/router.py`, with the same request model and
  auth as `/invoke`, returning `fastapi.sse.EventSourceResponse` (available
  in the installed FastAPI).
- Drive it with `graph.astream(..., stream_mode=["messages", "updates"])`.
  It's simpler to map than `astream_events`; switch only if finer events are
  needed.
- Event vocabulary, documented in the OpenAPI description:
  - `token`
  - `tool_call`
  - `tool_result`
  - `awaiting_approval`, with the same payload as 5b
  - `done`, with the `thread_id`
  - `error`
- After an approval, the client uses the 5b resume endpoint; add a streaming
  variant of resume if needed.
- Errors after the stream has started must go out as an `error` event, not
  an HTTP status.

**Tests:** read the response with httpx streaming; with the scripted fake,
assert the event order for a run with one tool call, a run that pauses for
approval, and that `done` carries the `thread_id`.

**Done when:** scripted runs stream the expected events in order, including
the approval pause.

---

## 8. Persistent checkpointer and deployment

**Goal:** threads survive restarts, and the service can run beyond a
developer laptop.

- Replace `InMemorySaver` with a persistent checkpointer: SQLite for a single
  instance, Postgres for several workers. Move thread ownership and retention
  into the same store, and replace step 2's in-memory eviction with a
  scheduled cleanup.
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
- Per-request workspaces, building on 5a and 5c.
