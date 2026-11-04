# miao-pi

A simple coding agent.

## Setup

Install Python 3.13.7 and synchronize the project environment:

```bash
uv python install 3.13.7
uv sync
```

## Run

Start the FastAPI development server:

```bash
uv run uvicorn app.service.server:app --reload --port 9091
```

The health endpoint is available at <http://127.0.0.1:9091/health>, and the
OpenAPI documentation is at <http://127.0.0.1:9091/api/v1/docs>.

Invoke the agent:

```bash
curl -X POST http://127.0.0.1:9091/invoke \
  -H 'Content-Type: application/json' \
  -d '{"message":"Explain recursion in one sentence."}'
```

The response includes a `thread_id`. Send it back to continue the same
conversation:

```bash
curl -X POST http://127.0.0.1:9091/invoke \
  -H 'Content-Type: application/json' \
  -d '{"message":"Now give an example.","thread_id":"<thread_id from the first response>"}'
```

A thread handles one request at a time: sending another message to a thread
whose previous request is still running returns `409 Conflict`.

Conversations are kept in memory only: they're lost on restart, and threads
unused for `THREAD_TTL` seconds (or beyond `MAX_THREADS`) are deleted.

## Workspace context

The agent is restricted to one workspace and reads `AGENTS.md`, falling back
to `README.md`, before each model call:

```env
WORKSPACE_ROOT=/path/to/project
WORKSPACE_DENIED_PATHS=["*.pem", "config/private"]
WORKSPACE_CONTEXT_MAX_CHARS=20000
```

Paths outside the workspace and symlink escapes are rejected. The patterns
`.env*`, `.git`, and `.claude` (local coding-agent settings) are always denied
at any depth; `WORKSPACE_DENIED_PATHS` adds to them. Secrets must be covered
by these patterns: `.gitignore` only keeps noise out of `find` and `grep`, and
the agent doesn't read your global git ignore file. Patterns are case-insensitive globs. One without a `/`, such as
`*.pem`, matches a name at any depth. One with a `/`, such as
`config/private`, is relative to the workspace root and also denies
everything beneath it. `.env*` intentionally includes `.env.example`, so the
agent can't read environment templates either.

`WORKSPACE_ROOT` defaults to the server's current directory. Set it explicitly
in deployment so future file tools cannot modify the service itself by
accident.

## Tools

The agent inspects the workspace with four read-only tools, modeled on
[pi's coding agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent):

| Tool | Does | Caps |
| --- | --- | --- |
| `read` | Reads a text file, with `offset`/`limit` for large ones | 2000 lines or 50KB |
| `ls` | Lists a directory, including dotfiles | 500 entries |
| `find` | Finds files by glob pattern | 1000 results |
| `grep` | Searches file contents (Python regex or literal) | 100 matches, 500 chars per line |

Every output also stops at 50KB, and a cut output ends with a notice telling
the model how to get the rest. Paths are relative to the workspace root.
Denied paths are rejected by `read` and never appear in `ls`, `find`, or
`grep`. `find` and `grep` skip files matched by `.gitignore`, and binary
files are never shown.

The model can call tools repeatedly within one request, bounded by:

```env
AGENT_MAX_STEPS=25
AGENT_RUN_TIMEOUT=300
```

`AGENT_MAX_STEPS` counts graph steps; each tool round takes two. When it runs
out, the agent stops and says so, and asking it to continue resumes with a
fresh budget. A run that exceeds `AGENT_RUN_TIMEOUT` seconds returns
`504 Gateway Timeout`.

## Test UI

A developer page for trying the agent in a browser. It shows each reply, the
tool calls behind it with their results, and each turn's status, request id,
and duration. It's off by default; turn it on in `.env`:

```env
DEV_UI_ENABLED=true
```

Then start the server as above and open <http://127.0.0.1:9091/ui/>. The
page also enables `GET /threads/{thread_id}/messages`, which returns a
conversation's saved messages, tool calls included.

There's no authentication yet, and the page shows file contents the tools
read, so keep the server bound to `127.0.0.1` (the default) while it's on.

## LLM client

`LLM` is a `langchain_openai.ChatOpenAI` subclass, so it can be passed directly
to LangChain chains and LangGraph nodes. It supports the OpenAI API and local
servers implementing the OpenAI-compatible chat-completions interface:

```python
from common.llm import LLM

# OpenAI (uses OPENAI_API_KEY when api_key is omitted)
openai_llm = LLM("gpt-4o-mini", temperature=0)

# OpenAI-compatible local server
local_llm = LLM("qwen2.5", base_url="http://localhost:11434/v1")

print(local_llm.invoke("Explain recursion in one sentence.").text)
```

Set the provider at application startup with environment variables. `LLM()`
then reads these values from `SvcSettings`; its explicit arguments still take
precedence when needed.

For local development, put them in `.env`; it is ignored by Git. Start from
`.env.example` and add the real `OPENAI_API_KEY` only to `.env`.

```bash
# Hosted OpenAI (the OpenAI SDK reads OPENAI_API_KEY)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# OpenAI-compatible local server
LLM_PROVIDER=local
LLM_MODEL=qwen2.5
LLM_BASE_URL=http://localhost:11434/v1
```

After setting `OPENAI_API_KEY` in `.env`, run the opt-in live smoke test:

```bash
RUN_OPENAI_LIVE_TESTS=1 uv run pytest tests/integration/test_openai_llm.py
```

To test a local server, set `LLM_PROVIDER=local` and `LLM_BASE_URL` in `.env`,
then run:

```bash
RUN_LOCAL_LLM_LIVE_TESTS=1 uv run pytest tests/integration/test_openai_llm.py
```

## Quality checks

```bash
uv run black --check .
uv run isort --check-only .
uv run mypy
uv run pytest
```

To apply import and code formatting:

```bash
uv run isort .
uv run black .
```
