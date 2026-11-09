const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const staticDir = path.resolve(__dirname, "../../app/ui/static");
const parser = fs.readFileSync(path.join(staticDir, "sse.js"), "utf8");

function response(text, splitBytes = false) {
  const bytes = new TextEncoder().encode(text);
  return new Response(new ReadableStream({
    start(controller) {
      if (splitBytes) {
        for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
      } else controller.enqueue(bytes);
      controller.close();
    },
  }), { headers: { "X-Thread-ID": "test-thread" } });
}

test("SSE parser handles fragmented UTF-8, CRLF, comments and multiline data", async () => {
  const context = vm.createContext({ TextDecoder });
  vm.runInContext(parser, context);
  const events = [];
  await context.readSSE(response(
    ': ping\r\nevent: token\r\ndata: {"text":\r\ndata: "你好"}\r\n\r\n' +
    'event: done\ndata: {"thread_id":"one"}\n\n', true,
  ), (name, data) => events.push([name, JSON.parse(JSON.stringify(data))]));
  assert.deepEqual(events, [["token", { text: "你好" }], ["done", { thread_id: "one" }]]);
});

function node() {
  return {
    value: "", textContent: "", lastElementChild: null,
    append() {}, replaceChildren() {}, addEventListener() {},
    scrollIntoView() {}, focus() {}, querySelectorAll() { return []; },
  };
}

function appContext(events) {
  const context = vm.createContext({
    TextDecoder, performance, setInterval, clearInterval,
    document: { getElementById: node, createElement: node },
    localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
    fetch: async () => response(events.map(([name, data]) =>
      `event: ${name}\ndata: ${JSON.stringify({ thread_id: "test-thread", ...data })}\n\n`,
    ).join(""), true),
  });
  vm.runInContext(parser, context);
  vm.runInContext(fs.readFileSync(path.join(staticDir, "app.js"), "utf8"), context);
  return context;
}

test("UI combines streamed tool activity and the final answer", async () => {
  const context = appContext([
    ["token", { message_id: "first", text: "Checking." }],
    ["tool_call", { message_id: "first", text: "Checking.", id: "call", name: "ls", args: {} }],
    ["tool_result", { tool_call_id: "call", name: "ls", text: "notes.txt", status: "success" }],
    ["token", { message_id: "second", text: "Done" }],
    ["done", { status: "completed", response: "Done.", actions: [] }],
  ]);
  const result = await vm.runInContext('post("/stream", {}, "Running")', context);
  assert.equal(result.error, undefined);
  const history = JSON.parse(vm.runInContext("JSON.stringify(history)", context));
  assert.equal(history.length, 3);
  assert.equal(history[0].tool_calls[0].id, "call");
  assert.equal(history[1].text, "notes.txt");
  assert.equal(history[2].text, "Done.");
});

test("UI replaces provisional tokens with the authoritative limit response", async () => {
  const context = appContext([
    ["token", { message_id: "first", text: "Unfinished text" }],
    ["token", { message_id: "limit", text: "Time limit reached." }],
    ["done", { status: "completed", response: "Time limit reached.", actions: [] }],
  ]);
  await vm.runInContext('post("/stream", {}, "Running")', context);
  assert.equal(vm.runInContext("history.length", context), 1);
  assert.equal(vm.runInContext("history[0].text", context), "Time limit reached.");
});

test("UI reports an incomplete stream rather than treating it as completed", async () => {
  const context = appContext([["token", { message_id: "first", text: "Partial" }]]);
  const result = await vm.runInContext('post("/stream", {}, "Running")', context);
  assert.match(result.error, /ended before the run finished/);
});

test("HTTP errors preserve useful JSON detail and fall back safely", async () => {
  const context = appContext([]);
  for (const [body, expected] of [
    [{ detail: "Thread already has a run in progress" }, "Thread already has a run in progress"],
    [{ detail: [{ msg: "Field required" }, null, { msg: {} }] }, "Field required"],
    [{ detail: {} }, "Request failed with HTTP 409."],
    [{ detail: "" }, "Request failed with HTTP 409."],
    [null, "Request failed with HTTP 409."],
  ]) {
    const result = await context.errorText(new Response(JSON.stringify(body), {
      status: 409, headers: { "Content-Type": "application/json" },
    }));
    assert.equal(result, expected);
  }
  assert.equal(await context.errorText(new Response("broken JSON", {
    status: 503, headers: { "Content-Type": "application/json" },
  })), "Request failed with HTTP 503.");

  const html = new Response("<h1>Unavailable</h1>", {
    status: 503, headers: { "Content-Type": "text/html" },
  });
  html.json = () => { throw new Error("Must not attempt JSON parsing for HTML"); };
  assert.equal(await context.errorText(html), "Request failed with HTTP 503.");
  assert.equal(await context.errorText(new Response('{"detail":"private details"}', {
    status: 500, headers: { "Content-Type": "application/json" },
  })), "The server hit an unexpected error. Look up the request id in the logs.");
});
