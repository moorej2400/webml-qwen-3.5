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
  type HarnessEvent = { readonly type: string; readonly detail?: unknown };
  const lifecycle = new Map<string, ((event?: HarnessEvent) => void)[]>();
  const fetches: string[] = [];
  const delayedTimers: (() => void)[] = [];
  let reloads = 0;
  let uuid = 0;
  const context = {
    BroadcastChannel: class {
      addEventListener(): void {}
      postMessage(): void {}
    },
    CustomEvent: class {
      readonly detail: unknown;
      constructor(readonly type: string, init?: { readonly detail?: unknown }) {
        this.detail = init?.detail;
      }
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
      else delayedTimers.push(callback);
      return 1;
    },
    clearTimeout() {},
    addEventListener(type: string, listener: (event?: HarnessEvent) => void) {
      const listeners = lifecycle.get(type) ?? [];
      listeners.push(listener);
      lifecycle.set(type, listeners);
    },
    dispatchEvent(event: HarnessEvent) {
      for (const listener of lifecycle.get(event.type) ?? []) listener(event);
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
    signalLoadEvent(detail: Record<string, unknown>) {
      context.dispatchEvent(new (context.CustomEvent as new (
        type: string,
        init: { readonly detail: unknown },
      ) => { type: string; detail: unknown })(
        "qwen-local-runtime-load-event",
        { detail },
      ));
    },
    signalUploadEvent(detail: Record<string, unknown>) {
      context.dispatchEvent(new (context.CustomEvent as new (
        type: string,
        init: { readonly detail: unknown },
      ) => { type: string; detail: unknown })(
        "qwen-local-runtime-upload-event",
        { detail },
      ));
    },
    runDelayedTimers() {
      for (const callback of delayedTimers.splice(0)) callback();
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

test("reconcile settles a conflicting local terminal and reconnect does not replay it", async () => {
  const harness = createAgentHarness();
  let generations = 0;
  harness.context.__QWEN_LOCAL_CONTROL__ = {
    async runPrompt() {
      generations += 1;
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
  const ready = socket.sent.find((message) =>
    (message as { readonly type?: string }).type === "ready"
  ) as { readonly documentId: string } | undefined;
  assert.ok(ready);
  const command = {
    schemaVersion: 1,
    type: "command",
    commandId: "command_prompt_0123456789",
    command: "runPrompt",
    payload: { promptRef: "fixture-1" },
  };
  socket.emit("message", command);
  await flush();
  assert.equal(generations, 1);
  const commandStates = () => (socket.sent as Array<{
    readonly type?: string;
    readonly commandId?: string;
    readonly state?: string;
    readonly eventSeq?: number;
  }>).filter((message) =>
    message.type === "commandState" && message.commandId === command.commandId
  );
  assert.equal(commandStates().at(-1)?.state, "completed");
  const stateCountBeforeReconcile = commandStates().length;

  const reconcile = {
    schemaVersion: 1,
    type: "reconcile",
    commands: [{
      commandId: command.commandId,
      command: "runPrompt",
      state: "timed_out",
      issuedAtMs: 1,
      startedAtMs: 2,
      terminalAtMs: 3,
      reason: "command_timeout",
    }],
  };
  socket.emit("message", reconcile);
  socket.emit("message", reconcile);
  await flush();
  assert.equal(commandStates().length, stateCountBeforeReconcile);

  const highestEventSeq = Math.max(...(socket.sent as Array<{ readonly eventSeq?: number }>)
    .flatMap((message) => message.eventSeq === undefined ? [] : [message.eventSeq]));
  socket.emit("message", {
    schemaVersion: 1,
    type: "eventAck",
    documentId: ready.documentId,
    status: "accepted",
    acknowledgedSeq: highestEventSeq,
    expectedSeq: highestEventSeq + 1,
  });
  socket.emit("close");
  harness.runDelayedTimers();
  await flush();
  const replacement = FakeWebSocket.instances[1];
  assert.ok(replacement);
  replacement.emit("open");
  const replacementStateCount = () => (replacement.sent as Array<{
    readonly type?: string;
    readonly commandId?: string;
  }>).filter((message) =>
    message.type === "commandState" && message.commandId === command.commandId
  ).length;
  socket.emit("message", reconcile);
  replacement.emit("message", reconcile);
  await flush();
  assert.equal(replacementStateCount(), 0);

  replacement.emit("message", command);
  await flush();
  assert.equal(generations, 1);
  const replayedTerminal = (replacement.sent as Array<{
    readonly commandId?: string;
    readonly state?: string;
  }>).filter((message) => message.commandId === command.commandId).at(-1);
  assert.equal(replayedTerminal?.state, "timed_out");
});

test("server terminal settles an in-flight browser command and suppresses its late completion", async () => {
  const harness = createAgentHarness();
  let generations = 0;
  let finishPrompt!: () => void;
  const pendingPrompt = new Promise<void>((resolve) => {
    finishPrompt = resolve;
  });
  harness.context.__QWEN_LOCAL_CONTROL__ = {
    runPrompt() {
      generations += 1;
      return pendingPrompt;
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
  const command = {
    schemaVersion: 1,
    type: "command",
    commandId: "command_prompt_0123456789",
    command: "runPrompt",
  };
  socket.emit("message", command);
  await flush();
  assert.equal(generations, 1);
  const commandStates = () => (socket.sent as Array<{
    readonly type?: string;
    readonly commandId?: string;
    readonly state?: string;
  }>).filter((message) =>
    message.type === "commandState" && message.commandId === command.commandId
  );
  assert.equal(commandStates().at(-1)?.state, "started");
  const beforeReconcile = commandStates().length;

  socket.emit("message", {
    schemaVersion: 1,
    type: "reconcile",
    commands: [{
      commandId: command.commandId,
      command: "runPrompt",
      state: "timed_out",
      issuedAtMs: 1,
      terminalAtMs: 2,
      reason: "command_timeout",
    }],
  });
  await flush();
  assert.equal(commandStates().length, beforeReconcile);

  finishPrompt();
  await flush();
  assert.equal(commandStates().length, beforeReconcile);
  socket.emit("message", command);
  await flush();
  assert.equal(generations, 1);
  assert.equal(commandStates().at(-1)?.state, "timed_out");
});

test("generated agent propagates only allocation diagnostic enum codes", async () => {
  const allowedCodes = [
    "gpu_out_of_memory",
    "gpu_validation",
    "buffer_creation",
    "error_scope",
    "allocation_conflict",
    "unknown",
  ] as const;
  for (const code of allowedCodes) {
    const harness = createAgentHarness();
    harness.context.__QWEN_LOCAL_CONTROL__ = {
      async runPrompt() {
        const error = Object.assign(
          new Error("private allocation text at local-path:<path>/<model-id>.gguf"),
          { code, path: "local-path:<path>/<model-id>.gguf" },
        );
        error.name = "PrivateGpuError";
        error.stack = "private stack local-path:<path>/<model-id>.gguf";
        throw error;
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
    });
    await flush();

    const terminal = (socket.sent as Array<{
      readonly commandId?: string;
      readonly state?: string;
      readonly reason?: string;
    }>).filter((message) =>
      message.commandId === "command_prompt_0123456789" && message.state === "failed"
    ).at(-1);
    assert.equal(terminal?.reason, code);
    const runtimeError = (socket.sent as Array<{
      readonly event?: {
        readonly category?: string;
        readonly name?: string;
        readonly timestampMs?: number;
        readonly metrics?: Record<string, unknown>;
      };
    }>).filter((message) => message.event?.name === "runtime_error").at(-1);
    assert.deepEqual(
      runtimeError?.event === undefined
        ? undefined
        : { ...runtimeError.event, timestampMs: 0 },
      {
        category: "error",
        name: "runtime_error",
        timestampMs: 0,
        metrics: { code },
      },
    );
    assert.doesNotMatch(
      JSON.stringify({ terminal, runtimeError }),
      /private|local|model\.gguf|stack|path|PrivateGpuError/i,
    );
  }
});

test("generated agent maps an unapproved diagnostic code to unknown", async () => {
  const harness = createAgentHarness();
  harness.context.__QWEN_LOCAL_CONTROL__ = {
    async load() {
      throw Object.assign(new Error("private browser error"), {
        code: "private_dynamic_code",
        stack: "private stack",
      });
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
    commandId: "command_load_0123456789",
    command: "load",
  });
  await flush();
  const terminal = (socket.sent as Array<{
    readonly commandId?: string;
    readonly state?: string;
    readonly reason?: string;
  }>).filter((message) =>
    message.commandId === "command_load_0123456789" && message.state === "failed"
  ).at(-1);
  assert.equal(terminal?.reason, "unknown");
  const runtimeError = (socket.sent as Array<{
    readonly event?: {
      readonly category?: string;
      readonly name?: string;
      readonly timestampMs?: number;
      readonly metrics?: Record<string, unknown>;
    };
  }>).filter((message) => message.event?.name === "runtime_error").at(-1);
  assert.deepEqual(
    runtimeError?.event === undefined
      ? undefined
      : { ...runtimeError.event, timestampMs: 0 },
    {
      category: "error",
      name: "runtime_error",
      timestampMs: 0,
      metrics: { code: "unknown" },
    },
  );
  assert.doesNotMatch(
    JSON.stringify({ terminal, runtimeError }),
    /private|dynamic|stack|browser error/i,
  );
});

test("generated agent preserves a safe runtime code without exposing error text", async () => {
  const harness = createAgentHarness();
  harness.context.__QWEN_LOCAL_CONTROL__ = {
    async runPrompt() {
      throw Object.assign(new Error("private browser error at /private/path"), {
        code: "greedy-rolling-embedding-failed",
        stack: "private stack /private/path",
      });
    },
  };
  harness.signalRuntimeReady();
  harness.lifecycle.get("pageshow")?.[0]?.();
  await flush();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.emit("open");
  socket.emit("message", {
    schemaVersion: 1,
    type: "command",
    commandId: "command_runtime_0123456789",
    command: "runPrompt",
  });
  await flush();

  const serialized = JSON.stringify(socket.sent);
  assert.match(serialized, /greedy-rolling-embedding-failed/);
  assert.doesNotMatch(serialized, /private|stack|browser error|\/private\/path/i);
});

test("progress telemetry preserves terminal load telemetry and command capacity", async () => {
  const harness = createAgentHarness();
  let finishLoad!: () => void;
  const loadPending = new Promise<void>((resolve) => { finishLoad = resolve; });
  let loadCalls = 0;
  harness.context.__QWEN_LOCAL_CONTROL__ = {
    load() {
      loadCalls += 1;
      return loadPending;
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
    commandId: "command_load_0123456789",
    command: "load",
  });
  await flush();
  assert.equal(loadCalls, 1);

  for (let index = 0; index < 300; index += 1) {
    harness.signalLoadEvent({
      phase: index % 2 === 0 ? "lock_wait" : "cache_scan",
      completedBytes: 0,
      totalBytes: 0,
    });
  }
  harness.signalLoadEvent({
    phase: "ready",
    completedBytes: 0,
    totalBytes: 0,
  });
  finishLoad();
  await flush();

  const sent = socket.sent as Array<{
    readonly type?: string;
    readonly commandId?: string;
    readonly state?: string;
    readonly event?: { readonly name?: string };
  }>;
  assert.equal(
    sent.filter((message) =>
      message.commandId === "command_load_0123456789" &&
      message.state === "completed"
    ).length,
    1,
  );
  assert.equal(
    sent.filter((message) => message.event?.name === "load_completed").length,
    1,
  );
});

test("upload diagnostics reach the socket with only fixed labels and bounded numeric fields", async () => {
  const harness = createAgentHarness();
  harness.context.__QWEN_LOCAL_CONTROL__ = { async getState() { return {}; } };
  harness.signalRuntimeReady();
  const pageshow = harness.lifecycle.get("pageshow")?.[0];
  assert.ok(pageshow);
  pageshow();
  await flush();
  const socket = FakeWebSocket.instances[0];
  assert.ok(socket);
  socket.emit("open");

  harness.signalUploadEvent({
    stage: "before_write",
    ordinal: 7,
    shardIndex: 13,
    shardCount: 20,
    segmentIndex: 4,
    segmentCount: 18,
    globalOffset: 1_824_496_640,
    byteCount: 8 * 1024 * 1024,
    bufferShardBytes: 128 * 1024 * 1024,
    uploadLaneBytes: 8 * 1024 * 1024,
    retireAfterEachWrite: false,
    tensorName: "private tensor identity",
    url: "https://private.invalid/model",
    path: "/private/local/path",
    prompt: "private prompt",
    response: "private response",
    stack: "private stack",
    secret: "private secret",
  });
  harness.signalUploadEvent({
    stage: "after_write",
    ordinal: 0,
    shardIndex: 20,
    shardCount: 20,
    segmentIndex: 1,
    segmentCount: 0,
    globalOffset: Number.MAX_SAFE_INTEGER - 3,
    byteCount: 8,
    bufferShardBytes: 0,
    uploadLaneBytes: 3,
    retireAfterEachWrite: "false",
  });
  harness.signalUploadEvent({
    stage: "after_retire",
    ordinal: 8,
    shardIndex: 0,
    shardCount: 1,
    segmentIndex: 0,
    segmentCount: 1,
    globalOffset: 0,
    byteCount: 8,
    bufferShardBytes: 64,
    uploadLaneBytes: 4,
    retireAfterEachWrite: false,
  });

  const message = (socket.sent as Array<{
    readonly type?: string;
    readonly eventSeq?: number;
    readonly event?: Record<string, unknown>;
  }>).find((candidate) => candidate.event?.name === "before_write");
  assert.ok(message);
  assert.deepEqual(
    { ...message.event, timestampMs: 0 },
    {
      category: "upload",
      name: "before_write",
      timestampMs: 0,
      metrics: {
        ordinal: 7,
        shardIndex: 13,
        shardCount: 20,
        segmentIndex: 4,
        segmentCount: 18,
        globalOffset: 1_824_496_640,
        byteCount: 8 * 1024 * 1024,
        bufferShardBytes: 128 * 1024 * 1024,
        uploadLaneBytes: 8 * 1024 * 1024,
        retireAfterEachWrite: false,
      },
    },
  );
  assert.doesNotMatch(
    JSON.stringify(message),
    /tensor|private|url|path|prompt|response|stack|secret/i,
  );
  assert.equal(
    (socket.sent as Array<{ readonly event?: { readonly name?: string } }>)
      .filter((candidate) => candidate.event?.name === "after_write").length,
    0,
  );
  assert.equal(
    (socket.sent as Array<{ readonly event?: { readonly name?: string } }>)
      .filter((candidate) => candidate.event?.name === "after_retire").length,
    0,
  );
  const sequences = (socket.sent as Array<{ readonly eventSeq?: number }>)
    .flatMap((candidate) => candidate.eventSeq === undefined ? [] : [candidate.eventSeq]);
  assert.deepEqual(
    sequences,
    Array.from({ length: sequences.length }, (_, index) => index + 1),
  );
});

test("upload diagnostic pressure preserves terminal load and command completion slots", async () => {
  const harness = createAgentHarness();
  let finishLoad!: () => void;
  const pendingLoad = new Promise<void>((resolve) => { finishLoad = resolve; });
  harness.context.__QWEN_LOCAL_CONTROL__ = { load: () => pendingLoad };
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
    commandId: "command_upload_0123456789",
    command: "load",
  });
  await flush();

  for (let ordinal = 1; ordinal <= 300; ordinal += 1) {
    harness.signalUploadEvent({
      stage: "before_write",
      ordinal,
      shardIndex: 0,
      shardCount: 20,
      segmentIndex: 0,
      segmentCount: 1,
      globalOffset: (ordinal - 1) * 8,
      byteCount: 8,
      bufferShardBytes: 128 * 1024 * 1024,
      uploadLaneBytes: 8 * 1024 * 1024,
      retireAfterEachWrite: false,
    });
  }
  harness.signalLoadEvent({ phase: "ready", completedBytes: 0, totalBytes: 0 });
  finishLoad();
  await flush();

  const sent = socket.sent as Array<{
    readonly eventSeq?: number;
    readonly commandId?: string;
    readonly state?: string;
    readonly event?: { readonly name?: string };
  }>;
  assert.ok(sent.some((message) => message.event?.name === "before_write"));
  assert.equal(sent.filter((message) => message.event?.name === "load_completed").length, 1);
  assert.equal(
    sent.filter((message) =>
      message.commandId === "command_upload_0123456789" && message.state === "completed"
    ).length,
    1,
  );
  const sequences = sent.flatMap((message) =>
    message.eventSeq === undefined ? [] : [message.eventSeq]
  );
  assert.deepEqual(
    sequences,
    Array.from({ length: sequences.length }, (_, index) => index + 1),
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
