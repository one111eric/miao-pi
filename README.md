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