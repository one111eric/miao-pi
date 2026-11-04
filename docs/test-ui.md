# Test UI

Plan for a developer page to try the agent in a browser. It's step 4b in
[next-steps.md](next-steps.md): after the ReAct loop and read-only tools,
before authentication.

**Status: built (2026-11-04).** The endpoint tests pass, and the page was
checked by hand against a scripted model; the manual check below still needs
a run against a real model. The page lives at `/ui/`, and `/ui` redirects
there, so the page's relative links to `app.js` and `app.css` stay valid
if it's ever served from a host's root instead.

**Goal:** try the agent from a browser: send messages, see which tools it
called and what they returned, and continue or restart a conversation,
without `curl`.

**Decisions**

- Serve it from the FastAPI app as static files: plain HTML, CSS, and
  JavaScript, with no build step, framework, or CDN. It shares the API's
  origin, so there's no CORS and nothing extra to run. Streamlit or Gradio
  would add dependencies and a second process; a React app would add a Node
  toolchain.
- No framework such as Astro, even with hosting in mind: any static host,
  Cloudflare Pages included, serves these files as they are. Astro suits
  content sites with a few interactive parts, and this page is interactive
  throughout. See "Hosting later" for what hosting actually needs.
- Off by default, behind a `dev_ui_enabled` setting. There's no
  authentication until 7b, and the page shows file contents the tools read.
- `/invoke` stays as it is. The page gets tool activity from a separate,
  read-only history endpoint, which step 7's stream can replace later.

**Changes**

- `common/config.py`: `dev_ui_enabled: bool = False`, and a commented-out
  `DEV_UI_ENABLED=true` in `.env.example`.
- Startup copies the setting into `app.state`. One shared dependency returns
  `404` when it's off, so a service with the page disabled looks like it has
  no such routes.
- `GET /threads/{thread_id}/messages` in `app/agent/router.py`, behind the
  gate:
  - Reads the thread with `graph.aget_state(...)`. It doesn't go through
    `ThreadStore.use()`, so it works while a run is in progress and never
    returns `409`.
  - An unknown or evicted thread returns `404`. Validate `thread_id` with the
    same pattern as `AgentRequest`.
  - Send `Cache-Control: no-store`: tool results include file contents, which
    no browser or proxy cache should keep.
  - Response: the messages in order, each `{"type": "human" | "ai" | "tool",
    "text": ...}`. AI messages add `tool_calls` (`id`, `name`, `args`); tool
    messages add `tool_call_id`, `name`, and `status`. Leave out the system
    prompt and `context`; the logs already record which file was loaded.
- New `app/ui/` feature package:
  - `router.py` serves `GET /ui/` (`index.html`), `GET /ui/app.js`, and
    `GET /ui/app.css` from `app/ui/static/`, and redirects `GET /ui` to
    `/ui/`, all behind the gate, with `include_in_schema=False`.
  - The page gets a strict `Content-Security-Policy`: `default-src 'none';
    script-src 'self'; style-src 'self'; connect-src 'self'; base-uri
    'none'; form-action 'none'; frame-ancestors 'none'`. Scripts and styles
    live in their own files, so it needs no `'unsafe-inline'`.
- The page (`app.js`):
  - A transcript, a message box (Send, or Ctrl/Cmd+Enter), and a New
    conversation button.
  - After each reply, fetch the thread's messages and show each tool call as
    a collapsible block: the tool name and arguments, the result's status,
    and its text.
  - Under each turn: the HTTP status, the `X-Request-ID` (to find the turn in
    the logs), and how long it took.
  - Keep the current `thread_id` in `localStorage`, and load the transcript
    from the history endpoint when the page opens.
  - Pick a new conversation's `thread_id` in the page
    (`crypto.randomUUID()`). Error responses carry no `thread_id`, so
    otherwise a first turn that was saved but failed (`502`, `504`) couldn't
    be shown.
  - Disable Send while a request is running, so the page never causes a
    `409`.
  - Show error responses plainly: `409`, `422`, `502`, `504` (including
    "Agent run timed out"), and `500` with its request id.
  - Render everything with `textContent`, never `innerHTML`. Model replies
    and tool results are untrusted: they can contain whatever HTML the
    workspace's files hold.
  - Send every API call through one `apiBase` value, defaulting to the
    page's own origin, so the same files can later point at a hosted API.
- README: how to enable the page and open <http://127.0.0.1:9091/ui/>. The
  README's run command binds to `127.0.0.1`; keep it that way while the page
  is enabled.

**Tests**

- With the setting off, `/ui`, its two assets, and the history endpoint all
  return `404`.
- With it on, `/ui` returns HTML with the CSP header, and `app.js` and
  `app.css` are served with the right content types.
- History endpoint:
  - A scripted `read` → answer run returns, in order, the human message, the
    AI tool call, the tool result with its `status`, and the final answer.
  - An unknown thread returns `404`; an invalid `thread_id` returns `422`.
  - A read while a run is in progress (the `GatedChatModel` pattern from
    `tests/integration/test_conversation.py`) returns the last saved state,
    not `409`.
- A guard test that nothing in `app/ui/static/` uses `innerHTML`,
  `outerHTML`, `insertAdjacentHTML`, or `document.write`.
- The page's behavior is checked by hand (below). A browser test runner
  isn't worth adding for a test page.

**Manual check**, with a real model and `DEV_UI_ENABLED=true`:

- Ask "What does this project do?": the transcript shows `ls` and `read`
  calls with their results.
- Ask it to read `.env`: an error result shows, with no contents.
- Reload mid-conversation: the transcript comes back. New conversation
  starts an empty one.
- With `AGENT_MAX_STEPS=5`, a task that needs several tool rounds ends with
  the step-limit message. With `AGENT_RUN_TIMEOUT=1`, a turn shows the `504`.

**Done when:** the endpoint tests pass, and the manual check works against a
real model.

**Hosting later** (for example on Cloudflare Pages): the static files deploy
as they are. The API they call is the hard part, and needs:

- Authentication (7b). A hosted page means a publicly reachable API, and
  without a token anyone could read the workspace through the agent.
- An HTTPS deployment of the API (step 8). It can't run on Cloudflare
  Workers: it needs a real file system and, from 5b, containers. Cloudflare
  Tunnel is one way to expose it.
- A setting listing the page origins allowed by CORS on the agent routes.
  Never `*`.
- `apiBase` set to the API's URL, and the CSP's `connect-src` set to that
  origin instead of `'self'`.

Pointing a hosted page at a local `http://127.0.0.1` server isn't a
shortcut: browsers restrict public pages from calling private addresses.

If the page grows into a product UI, with several views, login, or markdown
and code rendering, choose a framework then, as its own package with its own
CI.

**Notes:** steps 5a, 7, and 7b in [next-steps.md](next-steps.md) each
extend the page; their sections say how.
