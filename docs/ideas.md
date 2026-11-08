# Ideas

Candidate additions to the roadmap, not yet planned or scheduled. Each names
the step it would fit. Move one into `docs/next-steps.md` once it's agreed.

## A ledger for side-effecting tool calls (step 6)

Record every approval-requiring call before it runs, so a call is never run
twice and its outcome is never lost. Step 6 needs "every approved action ends
with a known, saved result"; this is how to get it.

- **Key:** a hash of the thread, the turn, the tool name, and the call's
  arguments in canonical form. The turn is the id of the last user message
  before the call, so the key is the same when the graph replays the call
  and changes with every new user message.
- **States:** `SENDING` before the call runs, then `COMMITTED` (with a
  snapshot of the result) or `UNCERTAIN`.
- **On a repeat:**
  - `COMMITTED` with the same tool call id (a crash after the call ran but
    before the checkpoint was saved): return the saved result.
  - `COMMITTED` with a different id: answer "already done" instead of
    running again.
  - `SENDING` whose lease expired, or `UNCERTAIN`: never run again; tell the
    model to check the current state with a read tool first.
- **Fails closed:** if the ledger can't be read or written, refuse the call
  rather than run it unprotected.
- **Claims and expiry:** a claim has a lease (minutes), a record a lifetime
  (a day). A run that pauses for approval releases its claim.
- **Limits to document:** no protection across turns (the same change asked
  for again in a new message runs again) or across threads.

This also fixes the run-timeout gotcha: a call cut off by the deadline
becomes `UNCERTAIN`, so the next turn can say so precisely instead of the
placeholder's "may or may not have run". A process-local store is enough
until step 8 makes it persistent.

Related rule: never retry a side-effecting tool automatically. A dropped
connection after the change was applied would apply it twice.

## Context compaction (new step, before or with step 6)

A coding agent fills its context quickly with file reads, and nothing on the
roadmap handles that yet.

- Compact only the request sent to the model; the checkpoint keeps the full
  history. Never rewrite saved messages.
- Run on every model call, in `react`, triggered by a token estimate. Calibrate
  the estimate against the usage the provider reports, which `react` already
  logs.
- Shrink in stages: old tool results first become short stubs (tool, path,
  size), then summaries. The current turn is never touched.
- Keep a tool call and its result together, or drop both. Check before
  sending that no result is left without its call.
- If the provider still rejects the request as too long, rebuild a smaller
  one and retry once.
- Tell history apart from messages `react` adds at call time by message id,
  not by position.

## Streaming (step 7)

- Use `astream(stream_mode=["messages", "updates"])`, not `astream_events`.
- Send a small, fixed set of events: start, text delta, tool input, tool
  output, approval needed, error, finish. Always end the stream with a
  finish event, even after an error.
- Tell the final answer apart from tool progress by node name.
- Let a run keep going when the client disconnects, with an endpoint to
  reconnect and pick up its events. Once commands have side effects, closing
  a tab mustn't kill an approved command halfway.

## Persistent checkpointer (step 8)

- Refresh a thread's expiry on every checkpoint write, so an active thread
  never expires mid-conversation.
- Test the persistent saver against `InMemorySaver` for identical behavior
  (same writes, same reads, same listing), using a fake backend so the tests
  need no server.

## Smaller ideas

- **Fixed message for unexpected tool errors.** Our `ToolException`s are
  written for the model, but an unexpected exception's text can leak paths
  or internals. Send the model a fixed message for those, and log the type.
- **One summary log line per run:** model calls and their total time, tool
  calls and their total time, total duration, and outcome (completed,
  awaiting approval, step limit, timeout, error).
- **Tool timings in the test page:** store each call's `duration_ms` in the
  `ToolMessage`'s `additional_kwargs`, and show it next to the result.
- **Prompt caching:** send a per-thread `prompt_cache_key` on OpenAI
  requests. Keep the start of the system prompt byte-for-byte stable and put
  text that changes every turn (the workspace context) last.
- **Per-thread approval grants:** besides approve and reject, "always allow
  this tool in this thread", kept in checkpointed state. Never for command
  tools without a per-command check.
- **Large results:** instead of only cutting a long tool result, keep the
  full result outside the message history and give the model a preview, its
  size, and (for JSON) a summary of its structure, with a way to read more.
- **Tool mocks for evals:** for a future eval harness, canned tool results
  matched by tool name and arguments, so a scenario runs without a real
  workspace or sandbox.
