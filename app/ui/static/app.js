"use strict";

// Every request goes through apiBase, so these files can later point at a
// hosted API. Empty means the page's own origin.
const apiBase = "";
const THREAD_KEY = "miao-pi.thread-id";

const transcript = document.getElementById("transcript");
const threadLabel = document.getElementById("thread-id");
const form = document.getElementById("composer");
const input = document.getElementById("message");
const sendButton = document.getElementById("send");
const newThreadButton = document.getElementById("new-thread");
const statusLine = document.getElementById("status");

let threadId = loadThreadId();
let history = [];
// HTTP status, request id, and duration of each turn sent from this page,
// keyed by the turn's position among the thread's human messages.
const turnInfo = new Map();
// An error to show after the transcript, such as a turn that wasn't saved.
let notice = null;
let busy = false;

// Everything is shown with textContent, never parsed as HTML: replies and
// tool results can contain whatever the workspace's files hold.
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function loadThreadId() {
  try {
    return localStorage.getItem(THREAD_KEY);
  } catch {
    return null;
  }
}

function saveThreadId(id) {
  threadId = id;
  threadLabel.textContent = id ?? "new";
  try {
    if (id) localStorage.setItem(THREAD_KEY, id);
    else localStorage.removeItem(THREAD_KEY);
  } catch {
    // Storage can be unavailable; the page still works for this load.
  }
}

// The page picks the id of a new conversation itself, so it can still show a
// first turn that the server saved but answered with an error (no thread_id
// in the response). randomUUID needs a secure context such as localhost;
// without one, the server picks the id.
function newThreadId() {
  return globalThis.crypto?.randomUUID?.() ?? null;
}

function humanCount() {
  return history.filter((message) => message.type === "human").length;
}

function formatArgs(args) {
  const text = Object.entries(args)
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(", ");
  return text.length > 120 ? `${text.slice(0, 117)}...` : text;
}

function formatInfo(info) {
  const parts = [info.status ? `HTTP ${info.status}` : "no response"];
  if (info.requestId) parts.push(`request ${info.requestId}`);
  parts.push(`${(info.ms / 1000).toFixed(1)}s`);
  return parts.join(" · ");
}

function renderToolCall(call, result) {
  const block = element("details", "tool-call");
  const summary = element("summary");
  const state = result ? (result.status ?? "success") : "no result";
  summary.append(
    element("span", "tool-name", call.name),
    element("span", "tool-args", `(${formatArgs(call.args)})`),
    element("span", `tool-status ${state.replace(" ", "-")}`, state),
  );
  const output = result
    ? result.text
    : "The run ended before this tool call finished.";
  block.append(summary, element("pre", "tool-result", output));
  return block;
}

function renderNotice(current) {
  const node = element("div", "notice", current.text);
  if (current.info) node.append(element("div", "turn-info", formatInfo(current.info)));
  return node;
}

function renderTranscript() {
  const results = new Map(
    history
      .filter((message) => message.type === "tool")
      .map((message) => [message.tool_call_id, message]),
  );
  const shown = new Set();
  transcript.replaceChildren();

  let turn = -1;
  let turnNode = null;
  const closeTurn = () => {
    if (turnNode && turnInfo.has(turn)) {
      turnNode.append(element("div", "turn-info", formatInfo(turnInfo.get(turn))));
    }
  };
  const currentTurn = () => {
    if (!turnNode) {
      turnNode = element("section", "turn");
      transcript.append(turnNode);
    }
    return turnNode;
  };

  for (const message of history) {
    if (message.type === "human") {
      closeTurn();
      turn += 1;
      turnNode = null;
      currentTurn().append(element("div", "message human", message.text));
    } else if (message.type === "ai") {
      if (message.text) currentTurn().append(element("div", "message ai", message.text));
      for (const call of message.tool_calls) {
        const result = results.get(call.id);
        if (result) shown.add(result);
        currentTurn().append(renderToolCall(call, result));
      }
    } else if (!shown.has(message)) {
      // A result whose call isn't in the history; show it on its own.
      currentTurn().append(renderToolCall({ name: message.name ?? "tool", args: {} }, message));
    }
  }
  closeTurn();

  if (notice) transcript.append(renderNotice(notice));
  transcript.lastElementChild?.scrollIntoView({ block: "end" });
}

async function errorText(response) {
  let detail = null;
  try {
    detail = (await response.json()).detail;
  } catch {
    // Not JSON; fall back to the status below.
  }
  if (Array.isArray(detail)) detail = detail.map((item) => item.msg).join("; ");
  const known = {
    409: "This conversation already has a request running.",
    500: "The server hit an unexpected error. Look up the request id in the logs.",
    502: "The LLM provider request failed.",
  };
  return known[response.status] ?? detail ?? `Request failed with HTTP ${response.status}.`;
}

async function refreshHistory({ isNew = false } = {}) {
  if (!threadId) {
    history = [];
    return;
  }
  try {
    const response = await fetch(
      `${apiBase}/threads/${encodeURIComponent(threadId)}/messages`,
    );
    if (response.status === 404 && isNew) {
      // A first message that never reached the thread; keep the id for a retry.
      history = [];
      return;
    }
    if (response.status === 404) {
      history = [];
      turnInfo.clear();
      saveThreadId(null);
      notice = {
        text: "That conversation no longer exists (it expired, was evicted, or the server restarted). The next message starts a new one.",
      };
      return;
    }
    if (!response.ok) {
      notice = { text: `Couldn't load the conversation: ${await errorText(response)}` };
      return;
    }
    history = (await response.json()).messages;
  } catch {
    notice = { text: "Couldn't reach the server to load the conversation." };
  }
}

function setBusy(value) {
  busy = value;
  sendButton.disabled = value;
  newThreadButton.disabled = value;
  if (!value) statusLine.textContent = "";
}

async function send(text) {
  setBusy(true);
  notice = null;
  const before = humanCount();
  const isNew = !threadId;
  if (isNew) saveThreadId(newThreadId());
  const started = performance.now();
  const timer = setInterval(() => {
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    statusLine.textContent = `Waiting for the agent... ${seconds}s`;
  }, 100);

  const pending = element("section", "turn pending");
  pending.append(element("div", "message human", text));
  transcript.append(pending);
  pending.scrollIntoView({ block: "end" });

  const body = threadId ? { message: text, thread_id: threadId } : { message: text };
  let info;
  try {
    const response = await fetch(`${apiBase}/invoke`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    info = {
      status: response.status,
      requestId: response.headers.get("X-Request-ID"),
      ms: performance.now() - started,
    };
    if (response.ok) saveThreadId((await response.json()).thread_id);
    else info.error = await errorText(response);
  } catch {
    info = { status: null, requestId: null, ms: performance.now() - started };
    info.error = "Couldn't reach the server.";
  } finally {
    clearInterval(timer);
  }

  await refreshHistory({ isNew });
  const saved = humanCount() > before;
  if (saved) turnInfo.set(humanCount() - 1, info);
  if (info.error) {
    // A saved turn already shows its status line; don't repeat it.
    notice = { text: info.error, info: saved ? null : info };
    // The message never reached the thread, so give it back for a retry.
    if (!saved && !input.value) input.value = text;
  }
  renderTranscript();
  setBusy(false);
  input.focus();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (busy || !text) return;
  input.value = "";
  send(text);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});

newThreadButton.addEventListener("click", () => {
  saveThreadId(null);
  history = [];
  turnInfo.clear();
  notice = null;
  renderTranscript();
  input.focus();
});

saveThreadId(threadId);
refreshHistory().then(renderTranscript);
