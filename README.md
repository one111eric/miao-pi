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
