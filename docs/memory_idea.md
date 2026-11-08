# Memory ideas

Candidate memory features, not yet planned or scheduled. They all need
step 8 first: memory is useless while conversations are lost on restart.
Move one into `docs/next-steps.md` once it's agreed.

## Recent session summaries, with transcripts on demand

The most value for the least machinery: most "you forgot what we did"
failures are fixed by knowing what recent sessions were about.

- When a thread goes idle, summarize it: a few sentences (goal, what was
  done, decisions, what's left) and a few short search terms.
- A new thread's system prompt lists the last few summaries for the same
  workspace, newest first, each with its thread id. It leaves out the
  current thread.
- A read-only tool, such as `read_past_session(id)`, returns that thread's
  messages, capped and truncated like `read`. Fetching on demand beats
  attaching transcripts to every summary: most turns don't need them.
- The natural trigger is `ThreadStore` eviction. Today an expired thread is
  deleted without a trace; summarize it in a background task first, and
  delete it once the summary is saved.
- Start with a list ordered by recency. Search (embeddings, keywords) can
  wait until there are too many sessions to list, and it would add
  dependencies.
- Cache the rendered block for about a minute, and recompute it after a
  session ends, so work that just finished shows up.
- Let a request pass a reference date that excludes later sessions, so evals
  and replays see only what existed at the time.

## A durable memory file per workspace

For preferences and decisions that should outlive any one session: "use
pytest fixtures, not setUp", "never touch the generated client", "we chose
SQLite for step 8".

- **Entries:** an id, a kind (`remember` when asked to, `correction` when
  the agent was told it was wrong, `stated` for a preference said in
  passing), the text, when it was said, and the source thread ids.
- **Size:** a byte budget on the rendered text, with the oldest entries
  evicted first. Cap each entry's length well below the budget, so a new
  entry is never the one evicted. Render one line per entry, with its date.
- **Schema version:** stored as a `Literal`, so a file written by a newer
  schema fails to load instead of being silently rewritten by an older one.
- **Writes:** conditional on the version that was read (compare a hash or
  modification stamp before replacing), so two runs can't lose each other's
  changes. A loser reloads and retries.
- **Reads:** only a missing file counts as empty. An unreadable or
  unparseable file raises, so a writer never builds on an empty file after
  a transient failure and wipes the real one. The error names the file,
  never the parse error, which can quote its contents.
- **Where it lives:** outside the workspace, in the service's data
  directory, keyed by the workspace root. The agent's file tools can't
  reach it.
- **How it's written:** only through a `remember` tool that needs approval,
  plus API endpoints to list, edit, and delete entries (and a panel in the
  test page). Never extracted automatically. Without approval, a malicious
  repository could get the agent to "remember" instructions that load into
  every later session.
- **Relation to `AGENTS.md`:** `AGENTS.md` is the project's shared,
  committed instructions; this file is the operator's own, uncommitted
  notes. Load both, `AGENTS.md` first.

## Fail open

Memory is an optional block in the prompt, never a reason to fail a turn.

- Each load gets a short timeout. A failed or slow load logs a warning
  with the error type and the turn goes on without it.
- Log entry ids, counts, and sizes, never entry text or summaries.
- Use a deadline (`asyncio.timeout`) rather than `wait_for`, so a
  `TimeoutError` raised inside the load is reported as an error, not as the
  deadline expiring.

## Background work that can't loop

Summarizing runs after the turn, in the background.

- Bounded retries, then one fallback (a plain truncated transcript instead
  of a summary), then stop. Never reset the attempt count in the fallback
  path: that retries a stuck thread forever.
- Each extractor gets its own timeout. One that fails or times out produces
  nothing; it never blocks the others.
- A malformed input that can never succeed is logged and dropped, not
  retried.

## Lessons

- **Stale memories beat fresh data.** A remembered number, file content,
  line number, or test result can override what a tool just returned. Store
  decisions and preferences, not code facts. Say in the prompt that tool
  results are newer than memory and win when they disagree. If facts with
  numbers are ever stored, strip the numbers before showing them.
- **Scope everything.** Memory from one workspace must never appear in
  another. Key every summary and entry by workspace root, and later by user
  too (step 7b). A memory filed under the wrong scope leaks names and facts
  into unrelated work.
- **Automatic fact extraction is expensive and unreliable.** It's the most
  complex part of a memory system and still keeps wrong or duplicate facts.
  Recent summaries plus explicit entries do more for less.
- **Evaluate memory both on and off.** Run the same eval set with memory
  enabled and disabled. Judge summaries with an LLM against fixed criteria
  (accurate, covers the decisions, nothing invented), using a small golden
  set of threads.

## Beyond coding: a fuller taxonomy

If the agent grows past coding (a general assistant, or a product with many
users), a richer model of memory becomes worth it. Not needed for a coding
agent, where one thread is one task and the repository holds the facts.

- **Episodic:** what happened. A summary of each interaction, with search
  terms and embeddings of both, so it can be found by meaning or by
  keyword. The summaries above are the simple form of this.
- **Semantic:** facts about the user or their world, stored as
  type, key, and value ("business: industry: landscaping").
  - Each new fact is embedded and compared with the most similar stored
    facts. An LLM labels each pair `NEW`, `DUPLICATE`, or `UPDATE`;
    pairs are judged independently, in one batched call.
  - `UPDATE` writes a new version of the fact (a merged value) and marks
    the old version not current, so history is kept and can be audited or
    rolled back.
  - Before showing facts, drop near-duplicates and strip stale numbers.
- **Instructional:** rules the user stated about how the agent should
  behave ("always show amounts in euros"). Extracted, compared, and updated
  like semantic facts. The durable memory file above is the explicit,
  hand-curated form of this.
- **Topic segmentation:** long, wandering conversations are split into
  interactions by an LLM before extraction, one per topic, so each summary
  covers one subject. The rule that works: default to not splitting; keep a
  workflow (view, then act on it) and follow-up questions together; split
  only on a clear change of subject, and after a "what can you do" overview.
- **Retrieval:** a hybrid search per turn, running the embedding of the
  question, the embedding of its terms, and a keyword match on terms in
  parallel, then merging. Optionally an LLM first plans the query
  (rewritten question, terms, which memory types to search, how many of
  each).
- **Consolidation offline:** a periodic pass that updates stale memories,
  merges redundant ones, and finds patterns across sessions, instead of
  doing it all at write time.
- **Visibility:** show users what's remembered and when it was used, and
  let them correct or delete it. Silent memory that turns out wrong costs
  more trust than no memory.
