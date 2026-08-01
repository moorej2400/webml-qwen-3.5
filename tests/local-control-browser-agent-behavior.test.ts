import assert from "node:assert/strict";
import vm from "node:vm";
import test from "node:test";

import { createBrowserAgentSource } from "../dev/control/browser-agent-source.js";
import { ControlPlane } from "../dev/control/control-plane.js";
import { SessionTicketAuthority } from "../dev/control/session-ticket-authority.js";
import {
  consumePhoneTicketProtocols,
  connectTicketBoundPhone,
} from "../dev/control/server.js";

class FakeStorage {
  readonly values = new Map<string, string>();
  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
  removeItem(key: string): void {
    this.values.delete(key);
  }
}

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static instances: FakeWebSocket[] = [];
  readyState = 0;
  readonly sent: unknown[] = [];
  readonly listeners = new Map<string, ((event: { data?: string }) => void)[]>();
  closeCode?: number;
  closeReason?: string;

  constructor(readonly url: string, readonly protocols: string[]) {
    FakeWebSocket.instances.push(this);
  }

  addEventListener(type: string, listener: (event: { data?: string }) => void): void {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  send(value: string): void {
    this.sent.push(JSON.parse(value));
  }

  close(code?: number, reason?: string): void {
    this.readyState = 3;
    this.closeCode = code;
    this.closeReason = reason;
  }

  emit(type: string, value?: unknown): void {
    if (type === "open") this.readyState = FakeWebSocket.OPEN;
    const event = value === undefined ? {} : { data: JSON.stringify(value) };
    for (const listener of this.listeners.get(type) ?? []) listener(event);
  }
}

const flush = async (): Promise<void> => {
  for (let index = 0; index < 8; index += 1) await Promise.resolve();
  await new Promise<void>((resolve) => setImmediate(resolve));
};

const createAgentHarness = () => {
  FakeWebSocket.instances = [];
  const localStorage = new FakeStorage();
  const sessionStorage = new FakeStorage();
  const lifecycle = new Map<string, (() => void)[]>();
  const fetches: string[] = [];
  let reloads = 0;
  let uuid = 0;
  const context = {
    BroadcastChannel: class {
      addEventListener(): void {}
      postMessage(): void {}
    },
    CustomEvent: class {
      constructor(readonly type: string) {}
    },
    Object,
    Promise,
    JSON,
    Map,
    Set,
    Number,
    Array,
    String,
    crypto: { randomUUID: () => `00000000-0000-4000-8000-${String(++uuid).padStart(12, "0")}` },
    localStorage,
    sessionStorage,
    WebSocket: FakeWebSocket,
    location: {
      protocol: "https:",
      host: "development.invalid",
      reload() {
        reloads += 1;
      },
    },
    document: {
      getElementById: () => ({}) as unknown,
      createElement: () => ({}) as unknown,
      body: { append() {} },
    },
    fetch: async (url: string) => {
      fetches.push(url);
      return {
        ok: true,
        async json() {
          return { ticket: "ticket_0123456789abcdef" };
        },
      };
    },
    setTimeout(callback: () => void, delay = 0) {
      if (delay <= 60) queueMicrotask(callback);
      return 1;
    },
    clearTimeout() {},
    addEventListener(type: string, listener: () => void) {
      const listeners = lifecycle.get(type) ?? [];
      listeners.push(listener);
      lifecycle.set(type, listeners);
    },
    dispatchEvent(event: { type: string }) {
      for (const listener of lifecycle.get(event.type) ?? []) listener();
      return true;
    },
  } as Record<string, unknown>;
  context.globalThis = context;
  vm.runInNewContext(createBrowserAgentSource(), context);
  return {
    context: context as Record<string, unknown> & {
      __QWEN_LOCAL_CONTROL__?: Record<string, unknown>;
    },
    fetches,
    lifecycle,
    signalRuntimeReady() {
      context.dispatchEvent(new (context.CustomEvent as new (type: string) => { type: string })(
        "qwen-local-runtime-ready",
      ));
    },
    get reloads() {
      return reloads;
    },
  };
};

test("generated agent connects automatically, replays, deduplicates, and reloads", async () => {
  const harness = createAgentHarness();
  let generations = 0;
  harness.context.__QWEN_LOCAL_CONTROL__ = {
    runPrompt() {
      generations += 1;
    },
  };
  harness.signalRuntimeReady();
  const pageshow = harness.lifecycle.get("pageshow")?.[0];
  assert.ok(pageshow);
  pageshow();
  await flush();
  assert.deepEqual(harness.fetches, ["/.local-ticket"]);
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.emit("open");
  const opened = socket.sent as { type: string; eventSeq?: number }[];
  assert.equal(opened[0]?.type, "hello");
  assert.equal(opened[1]?.type, "ready");
  assert.equal(opened[1]?.eventSeq, 1);
  const connected = opened[2] as {
    event: { timestampMs: number; [key: string]: unknown };
    [key: string]: unknown;
  };
  assert.deepEqual({ ...connected, event: { ...connected.event, timestampMs: 0 } }, {
    schemaVersion: 1,
    deviceId: (opened[1] as { deviceId: string }).deviceId,
    tabId: (opened[1] as { tabId: string }).tabId,
    documentId: (opened[1] as { documentId: string }).documentId,
    eventSeq: 2,
    type: "telemetry",
    event: {
      category: "device",
      name: "connected",
      timestampMs: 0,
      metrics: { status: "connected" },
    },
  });

  socket.emit("message", {
    schemaVersion: 1,
    type: "eventAck",
    documentId: (opened[1] as { documentId: string }).documentId,
    status: "accepted",
    acknowledgedSeq: 1,
    expectedSeq: 2,
  });
  const command = {
    schemaVersion: 1,
    type: "command",
    commandId: "command_0123456789abcdef",
    command: "runPrompt",
  };
  socket.emit("message", command);
  await flush();
  socket.emit("message", command);
  await flush();
  assert.equal(generations, 1);
  const beforeGap = socket.sent.length;
  socket.emit("message", {
    schemaVersion: 1,
    type: "eventAck",
    documentId: (opened[1] as { documentId: string }).documentId,
    status: "gap",
    acknowledgedSeq: 1,
    expectedSeq: 2,
  });
  assert.ok(socket.sent.length > beforeGap);

  socket.emit("message", {
    schemaVersion: 1,
    type: "command",
    commandId: "command_1123456789abcdef",
    command: "warmReload",
  });
  await flush();
  assert.equal(harness.reloads, 1);

  socket.emit("message", {
    schemaVersion: 1,
    type: "sequenceSync",
    documentId: (opened[1] as { documentId: string }).documentId,
    expectedSeq: 999,
  });
  assert.equal(socket.closeCode, 1008);
  assert.equal(socket.closeReason, "protocol_error");
});

test("generated agent prevents overlapping reconnects and inherited handlers", async () => {
  const harness = createAgentHarness();
  let inheritedCalls = 0;
  const pageshow = harness.lifecycle.get("pageshow")?.[0];
  assert.ok(pageshow);
  pageshow();
  pageshow();
  await flush();
  assert.equal(FakeWebSocket.instances.length, 1);
  pageshow();
  await flush();
  assert.equal(FakeWebSocket.instances.length, 1);

  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.emit("open");
  harness.context.__QWEN_LOCAL_CONTROL__ = Object.create({
    runPrompt() {
      inheritedCalls += 1;
    },
  }) as Record<string, unknown>;
  harness.signalRuntimeReady();
  const ready = socket.sent[1] as { documentId: string };
  socket.emit("message", {
    schemaVersion: 1,
    type: "command",
    commandId: "command_0123456789abcdef",
    command: "runPrompt",
  });
  await flush();
  assert.equal(inheritedCalls, 0);
  assert.equal(
    (socket.sent.at(-1) as { state: string }).state,
    "failed",
  );
  assert.ok(ready.documentId);
});

test("generated agent does not identify or accept queued work before runtime handlers are ready", async () => {
  const harness = createAgentHarness();
  const pageshow = harness.lifecycle.get("pageshow")?.[0];
  assert.ok(pageshow);
  pageshow();
  await flush();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.emit("open");
  assert.deepEqual(socket.sent, []);

  let generations = 0;
  harness.context.__QWEN_LOCAL_CONTROL__ = {
    runPrompt() { generations += 1; },
  };
  harness.signalRuntimeReady();
  await flush();
  assert.equal((socket.sent[0] as { type: string }).type, "hello");
  assert.equal((socket.sent[1] as { type: string }).type, "ready");

  socket.emit("message", {
    schemaVersion: 1,
    type: "command",
    commandId: "command_0123456789abcdef",
    command: "runPrompt",
    payload: { prompt: "ready now", maxNewTokens: 1 },
  });
  await flush();
  assert.equal(generations, 1);
});

test("generated agent classifies an interrupted active prompt as cancelled", async () => {
  const harness = createAgentHarness();
  let rejectPrompt: ((error: Error) => void) | undefined;
  harness.context.__QWEN_LOCAL_CONTROL__ = {
    runPrompt() {
      return new Promise<void>((_resolve, reject) => { rejectPrompt = reject; });
    },
    async cancelPrompt() {
      const error = new Error("cancelled");
      error.name = "AbortError";
      rejectPrompt?.(error);
    },
  };
  harness.signalRuntimeReady();
  const pageshow = harness.lifecycle.get("pageshow")?.[0];
  assert.ok(pageshow);
  pageshow();
  await flush();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.emit("open");
  socket.emit("message", {
    schemaVersion: 1,
    type: "command",
    commandId: "command_prompt_0123456789",
    command: "runPrompt",
    payload: { prompt: "cancel me", maxNewTokens: 8 },
  });
  await flush();
  socket.emit("message", {
    schemaVersion: 1,
    type: "command",
    commandId: "command_cancel_0123456789",
    command: "cancelPrompt",
  });
  await flush();

  const commandStates = (socket.sent as { commandId?: string; state?: string }[])
    .filter((message) => message.commandId !== undefined);
  assert.equal(
    commandStates.filter((message) => message.commandId === "command_prompt_0123456789").at(-1)?.state,
    "cancelled",
  );
  assert.equal(
    commandStates.filter((message) => message.commandId === "command_cancel_0123456789").at(-1)?.state,
    "completed",
  );
});

test("authenticated in-memory server boundary dispatches only after ticket consumption", () => {
  const identity = {
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    documentId: "document_0123456789abcdef",
  };
  let tokenIndex = 0;
  const authority = new SessionTicketAuthority({
    now: () => 1,
    randomToken: () => `${String(++tokenIndex).padStart(2, "0")}${"B".repeat(41)}`,
  });
  const ticket = authority.issueTicket(identity);
  const auth = consumePhoneTicketProtocols(
    `qwen-control.v1, ${ticket.ticket}`,
    (candidate) => authority.consumeTicket(candidate),
  );
  assert.equal(auth.accepted, true);
  assert.deepEqual(
    consumePhoneTicketProtocols(
      `qwen-control.v1, ${ticket.ticket}`,
      (candidate) => authority.consumeTicket(candidate),
    ),
    { accepted: false },
  );
  if (!auth.accepted) return;
  const sent: { type: string }[] = [];
  const plane = new ControlPlane();
  const connectionId = connectTicketBoundPhone(plane, auth.identity, identity, (message) => {
    sent.push(message);
    return true;
  });
  assert.throws(
    () =>
      connectTicketBoundPhone(
        plane,
        auth.identity,
        { ...identity, documentId: "document_spoofed_0123456789" },
        () => true,
      ),
    /identity/i,
  );
  plane.issueCommand({
    deviceId: identity.deviceId,
    tabId: identity.tabId,
    commandId: "command_0123456789abcdef",
    command: "getState",
  });
  assert.equal(sent.filter((message) => message.type === "command").length, 1);
  for (const [eventSeq, state] of [
    [1, "accepted"],
    [2, "started"],
    [3, "completed"],
  ] as const) {
    plane.receive(connectionId, {
      schemaVersion: 1,
      type: "commandState",
      ...identity,
      eventSeq,
      commandId: "command_0123456789abcdef",
      state,
    });
  }
});
