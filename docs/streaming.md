# Streaming API (step 7a)

`POST /stream` accepts `AgentRequest`, just like `/invoke`.
`POST /threads/{thread_id}/resume/stream` accepts `ResumeRequest`, just like
the JSON resume endpoint. Both return `text/event-stream` with JSON data and
keep-alive comments during idle periods. Responses include `X-Request-ID`,
`X-Thread-ID`, `Cache-Control: no-store`, and `X-Accel-Buffering: no`.

Every data payload includes `thread_id`:

| Event | Additional fields and meaning |
| --- | --- |
| `token` | `message_id`, `text`: a fragment of provisional assistant text. Models without token streaming may emit the whole text at once. |
| `tool_call` | `message_id`, `text`, `id`, `name`, `args`: a complete model tool call. `text` is the canonical preamble for that assistant message. This event does not grant approval or indicate execution. |
| `tool_result` | `tool_call_id`, `name`, `text`, `status`: the tool's result, including validation or approval refusals. |
| `awaiting_approval` | Full `/invoke` response: `status="awaiting_approval"`, `response=null`, `actions`. This ends the stream. |
| `done` | Full `/invoke` response: `status="completed"`, `response`, `actions=[]`. Emitted after checkpoint writes finish. This ends the stream. |
| `error` | `code`, `detail`: safe error text. This ends the stream without a subsequent `done`. |

Match tool results by `tool_call_id`, since parallel calls can complete in
any order. Assemble tokens by `message_id`, but use the final `done.response`
as the authoritative answer: a step/time limit can replace a model response
after tokens have already arrived. Tool arguments are emitted as complete
calls rather than partial JSON fragments. Project context and system messages
are never emitted as standalone events.

Invalid request bodies, unknown resume threads, busy threads, and invalid
approval state are rejected with the same HTTP status as the JSON endpoints
before streaming starts. Once the stream starts, errors use the `error` event
with one of `provider_error`, `provider_timeout`, `run_timeout`, or
`internal_error`. Raw exception text is never included in these events.

The normal run deadline still stops between graph steps. The larger backstop
cancels stuck runs. The thread remains held for the response's lifetime. On
disconnect, response teardown cancels and joins the graph producer before
releasing that thread, allowing sandbox process cleanup to finish. The same
uncertain-result history handling as `/invoke` applies when cancellation
interrupts a tool. Check saved history before repeating a side effect.

There is no event replay or automatic reconnect. Use `fetch` to consume the
POST response (browser `EventSource` uses GET). Reload a conversation through
the existing developer history endpoint when needed. The test page consumes
the events directly and uses history for reload and error recovery.

Verification:

```bash
uv run pytest tests/integration/test_streaming.py
uv run node --test tests/js/streaming.test.cjs
```

The JavaScript tests use Node's built-in runner without npm dependencies.
