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
// Tool calls the paused run is waiting on, from the history endpoint.
let pendingActions = [];
// Status lines for requests sent from this page (a message, then any
// approvals), keyed by the turn's position among the thread's human messages.
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

function addTurnInfo(turn, info) {
  turnInfo.set(turn, [...(turnInfo.get(turn) ?? []), info]);
}

function formatArgs(args) {
  const text = Object.entries(args)
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(", ");
  return text.length > 120 ? `${text.slice(0, 117)}...` : text;
}

function formatInfo(info) {
  const parts = [info.status ? `HTTP ${info.status}` : "no response"];
  if (info.label) parts.unshift(info.label);
  if (info.requestId) parts.push(`request ${info.requestId}`);
  parts.push(`${(info.ms / 1000).toFixed(1)}s`);
  return parts.join(" · ");
}

function renderToolCall(call, result, waiting) {
  const block = element("details", "tool-call");
  const summary = element("summary");
  let state = "no result";
  let output = "The run ended before this tool call finished.";
  if (result) {
    state = result.status ?? "success";
    output = result.text;
  } else if (waiting) {
    state = "awaiting approval";
    output = "Waiting for your decision below.";
  } else if (busy) {
    state = "running";
    output = "Waiting for the tool result.";
  }
  summary.append(
    element("span", "tool-name", call.name),
    element("span", "tool-args", `(${formatArgs(call.args)})`),
    element("span", `tool-status ${state.replaceAll(" ", "-")}`, state),
  );
  block.append(summary, element("pre", "tool-result", output));
  return block;
}

function renderApproval() {
  const card = element("section", "approval");
  card.append(element("div", "approval-title", "The agent wants to run:"));
  for (const action of pendingActions) {
    const item = element("div", "approval-action");
    item.append(
      element("span", "tool-name", action.tool),
      element("pre", "approval-args", JSON.stringify(action.args, null, 2)),
    );
    card.append(item);
  }
  const note = element("textarea", "approval-note");
  note.rows = 2;
  note.placeholder = "Optional note for the agent, sent with a rejection";
  const reject = element("button", "reject", "Reject");
  const approve = element("button", "approve", "Approve");
  for (const button of [reject, approve]) {
    button.type = "button";
    button.disabled = busy;
  }
  reject.addEventListener("click", () => decide(false, note.value.trim()));
  approve.addEventListener("click", () => decide(true, note.value.trim()));
  const buttons = element("div", "actions");
  buttons.append(reject, approve);
  card.append(note, buttons);
  return card;
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
  const waiting = new Set(pendingActions.map((action) => action.tool_call_id));
  const shown = new Set();
  transcript.replaceChildren();

  let turn = -1;
  let turnNode = null;
  const closeTurn = () => {
    for (const info of (turnNode && turnInfo.get(turn)) ?? []) {
      turnNode.append(element("div", "turn-info", formatInfo(info)));
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
        currentTurn().append(renderToolCall(call, result, waiting.has(call.id)));
      }
    } else if (!shown.has(message)) {
      // A result whose call isn't in the history; show it on its own.
      currentTurn().append(
        renderToolCall({ name: message.name ?? "tool", args: {} }, message, false),
      );
    }
  }
  closeTurn();

  if (pendingActions.length) transcript.append(renderApproval());
  if (notice) transcript.append(renderNotice(notice));
  transcript.lastElementChild?.scrollIntoView({ block: "end" });
}

async function errorText(response) {
  const known = {
    500: "The server hit an unexpected error. Look up the request id in the logs.",
    502: "The LLM provider request failed.",
  };
  if (known[response.status]) return known[response.status];
  const fallback = `Request failed with HTTP ${response.status}.`;
  const contentType = (response.headers.get("Content-Type") ?? "")
    .split(";", 1)[0].trim().toLowerCase();
  if (contentType !== "application/json" && !contentType.endsWith("+json")) return fallback;
  try {
    const detail = (await response.json())?.detail;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => item?.msg)
        .filter((message) => typeof message === "string" && message.trim());
      if (messages.length) return messages.join("; ");
    }
  } catch {
    // A malformed JSON error response still has a usable HTTP status.
  }
  return fallback;
}

async function refreshHistory({ isNew = false } = {}) {
  if (!threadId) {
    history = [];
    pendingActions = [];
    return;
  }
  try {
    const response = await fetch(
      `${apiBase}/threads/${encodeURIComponent(threadId)}/messages`,
    );
    if (response.status === 404 && isNew) {
      // A first message that never reached the thread; keep the id for a retry.
      history = [];
      pendingActions = [];
      return;
    }
    if (response.status === 404) {
      history = [];
      pendingActions = [];
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
    const body = await response.json();
    history = body.messages;
    pendingActions = body.pending_actions ?? [];
  } catch {
    notice = { text: "Couldn't reach the server to load the conversation." };
  }
}

function updateControls() {
  // A paused run takes a decision, not a new message.
  sendButton.disabled = busy || pendingActions.length > 0;
  newThreadButton.disabled = busy;
  // The card is drawn while the request that paused the run is still busy.
  for (const button of transcript.querySelectorAll(".approval button")) {
    button.disabled = busy;
  }
  if (!busy) {
    statusLine.textContent = pendingActions.length ? "Waiting for your approval" : "";
  }
}

function setBusy(value) {
  busy = value;
  updateControls();
}

async function post(path, body, waitingText) {
  const started = performance.now();
  const timer = setInterval(() => {
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    statusLine.textContent = `${waitingText} ${seconds}s`;
  }, 100);
  const info = { status: null, requestId: null, ms: 0, accepted: false };
  const messages = new Map();
  function assistant(id) {
    if (!messages.has(id)) {
      const message = { type: "ai", text: "", tool_calls: [] };
      messages.set(id, message);
      history.push(message);
    }
    return messages.get(id);
  }

  try {
    const response = await fetch(`${apiBase}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    info.status = response.status;
    info.requestId = response.headers.get("X-Request-ID");
    info.accepted = response.ok;
    if (!response.ok) {
      info.error = await errorText(response);
      return info;
    }
    const returnedThread = response.headers.get("X-Thread-ID");
    if (returnedThread) saveThreadId(returnedThread);
    await readSSE(response, (event, data) => {
      if (data.thread_id) saveThreadId(data.thread_id);
      if (event === "token") {
        assistant(data.message_id).text += data.text;
      } else if (event === "tool_call") {
        const message = assistant(data.message_id);
        message.text = data.text;
        message.tool_calls.push({ id: data.id, name: data.name, args: data.args });
      } else if (event === "tool_result") {
        history.push({ type: "tool", ...data });
      } else if (event === "awaiting_approval") {
        pendingActions = data.actions;
        info.data = data;
      } else if (event === "done") {
        pendingActions = [];
        // The final saved answer replaces provisional text, including a
        // response replaced by the graph's time or step limit.
        const provisional = [...messages.values()].filter((message) => !message.tool_calls.length);
        const finalMessage = provisional[0] ?? assistant("final");
        history = history.filter((message) => !provisional.slice(1).includes(message));
        finalMessage.text = data.response ?? "";
        info.data = data;
      } else if (event === "error") {
        info.error = data.detail;
      }
      renderTranscript();
    });
    if (!info.data && !info.error) {
      info.error = "The stream ended before the run finished. Reload the conversation before retrying.";
    }
    return info;
  } catch {
    info.error = "The connection was interrupted. Reload the conversation before retrying.";
    return info;
  } finally {
    info.ms = performance.now() - started;
    clearInterval(timer);
  }
}

async function send(text) {
  setBusy(true);
  notice = null;
  const before = humanCount();
  const isNew = !threadId;
  if (isNew) saveThreadId(newThreadId());

  const pending = { type: "human", text, tool_calls: [] };
  history.push(pending);
  renderTranscript();

  const body = threadId ? { message: text, thread_id: threadId } : { message: text };
  const info = await post("/stream", body, "Waiting for the agent...");
  if (info.data) saveThreadId(info.data.thread_id);

  if (!info.accepted) history = history.filter((message) => message !== pending);
  if (info.error && info.accepted) await refreshHistory({ isNew });
  const saved = humanCount() > before;
  if (saved) addTurnInfo(humanCount() - 1, info);
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

async function decide(approved, note) {
  if (busy || !threadId) return;
  setBusy(true);
  notice = null;
  renderTranscript();

  const info = await post(
    `/threads/${encodeURIComponent(threadId)}/resume/stream`,
    { approved, note: note || null },
    approved ? "Approved; continuing..." : "Rejected; continuing...",
  );
  info.label = approved ? "Approved" : "Rejected";

  if (info.error) await refreshHistory();
  addTurnInfo(humanCount() - 1, info);
  if (info.error) notice = { text: info.error, info: null };
  renderTranscript();
  setBusy(false);
  input.focus();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (busy || pendingActions.length || !text) return;
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
  pendingActions = [];
  turnInfo.clear();
  notice = null;
  renderTranscript();
  updateControls();
  input.focus();
});

saveThreadId(threadId);
refreshHistory().then(() => {
  renderTranscript();
  updateControls();
});
