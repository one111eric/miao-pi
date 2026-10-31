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
  `service.py` (runs the graph for a request).
- `common/` — code shared across features: `config.py` (`SvcSettings`),
  `llm/` (`LLM`, a `ChatOpenAI` subclass), `logging_helper.py`.
- `tests/unit/` — no network, no app lifespan.
- `tests/integration/` — go through the ASGI app via the `client` fixture in
  `tests/conftest.py`.

## Architecture

Request flow: `POST /invoke` → `app/agent/router.py` → `invoke_agent()` →
compiled graph `preplanning → react → finalize`.

- `preplanning` prepends the system prompt.
- `react` calls the LLM, which `build_agent_graph(llm)` injects through
  `make_react(llm)`. There are no tools or loop yet.
- `finalize` is a placeholder for post-processing.

Around every request:

- `RequestContextMiddleware` takes a valid `X-Request-ID` (or generates
  one), adds it to the logging context, and returns it in the response
  header. Unhandled errors are logged there and returned as a generic `500`
  that still carries the header.
- OpenAI provider errors map to `504` (timeout) or `502` (anything else),
  with a generic message. Never put provider error text in a response.
- Model calls use `llm_timeout` and `llm_max_retries` from `SvcSettings`;
  arguments passed to `LLM(...)` override them. `react` logs each call's
  duration and token usage.

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
  `EchoChatModel` from `tests/conftest.py` through `create_app(llm=...)`.
  Pass a fake model the same way instead of monkeypatching `LLM`.
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
- `thread_id` is echoed back but not persisted. There is no checkpointer yet.

## Rules

- Never commit `.env` or real API keys. `.env.example` holds placeholders only.
- Don't add dependencies without saying so. Use `uv add` (or
  `uv add --dev`) so `uv.lock` stays in sync. CI installs with `--locked`.
