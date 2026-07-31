import assert from "node:assert/strict";
import vm from "node:vm";
import test from "node:test";

import { createBrowserAgentSource } from "../dev/control/browser-agent-source.js";
import { ControlPlane } from "../dev/control/control-plane.js";
import { PairingAuthority } from "../dev/control/pairing.js";
import {
  authenticatePhoneProtocols,
  connectAuthenticatedPhone,
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

const createAgentHarness = (sessionCapability?: string) => {
  FakeWebSocket.instances = [];
  const localStorage = new FakeStorage();
  const sessionStorage = new FakeStorage();
  if (sessionCapability !== undefined) {
    sessionStorage.setItem("qwen.control.session", sessionCapability);
  }
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
          return url === "/.local-pair"
            ? { sessionCapability: "session_0123456789abcdef" }
            : { ticket: "ticket_0123456789abcdef" };
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
    dispatchEvent() {},
  } as Record<string, unknown>;
  context.globalThis = context;
  vm.runInNewContext(createBrowserAgentSource(), context);
  return {
    context: context as Record<string, unknown> & {
      __QWEN_LOCAL_PAIR__: (code: string) => Promise<void>;
      __QWEN_LOCAL_CONTROL__?: Record<string, unknown>;
    },
    fetches,
    lifecycle,
    get reloads() {
      return reloads;
    },
  };
};

test("generated agent executes pairing, replay, deduplication, reload, and fail-closed behavior", async () => {
  const harness = createAgentHarness();
  await harness.context.__QWEN_LOCAL_PAIR__("pairing-code");
  assert.deepEqual(harness.fetches, ["/.local-pair", "/.local-ticket"]);
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.emit("open");
  const opened = socket.sent as { type: string; eventSeq?: number }[];
  assert.equal(opened[0]?.type, "hello");
  assert.equal(opened[1]?.type, "ready");
  assert.equal(opened[1]?.eventSeq, 1);

  socket.emit("message", {
    schemaVersion: 1,
    type: "eventAck",
    documentId: (opened[1] as { documentId: string }).documentId,
    status: "accepted",
    acknowledgedSeq: 1,
    expectedSeq: 2,
  });
  let generations = 0;
  harness.context.__QWEN_LOCAL_CONTROL__ = {
    runPrompt() {
      generations += 1;
    },
  };
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
  const harness = createAgentHarness("session_0123456789abcdef");
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
  const ready = socket.sent[1] as { documentId: string };
  let inheritedCalls = 0;
  harness.context.__QWEN_LOCAL_CONTROL__ = Object.create({
    runPrompt() {
      inheritedCalls += 1;
    },
  }) as Record<string, unknown>;
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

test("authenticated in-memory server boundary dispatches only after ticket consumption", () => {
  const identity = {
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    documentId: "document_0123456789abcdef",
  };
  const pairingCode = "A".repeat(43);
  let tokenIndex = 0;
  const authority = new PairingAuthority({
    pairingCode,
    now: () => 1,
    randomToken: () => `${String(++tokenIndex).padStart(2, "0")}${"B".repeat(41)}`,
  });
  const session = authority.pair(pairingCode, identity);
  const ticket = authority.issueTicket(session.sessionCapability, identity);
  const auth = authenticatePhoneProtocols(
    `qwen-control.v1, ${ticket.ticket}`,
    (candidate) => authority.consumeTicket(candidate),
  );
  assert.equal(auth.accepted, true);
  assert.deepEqual(
    authenticatePhoneProtocols(
      `qwen-control.v1, ${ticket.ticket}`,
      (candidate) => authority.consumeTicket(candidate),
    ),
    { accepted: false },
  );
  if (!auth.accepted) return;
  const sent: { type: string }[] = [];
  const plane = new ControlPlane();
  const connectionId = connectAuthenticatedPhone(plane, auth.identity, identity, (message) => {
    sent.push(message);
    return true;
  });
  assert.throws(
    () =>
      connectAuthenticatedPhone(
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
