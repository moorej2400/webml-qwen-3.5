import assert from "node:assert/strict";
import test from "node:test";

import { ControlPlane, type ServerToPhoneMessage } from "../dev/control/control-plane.js";
import { PhoneEventOutbox } from "../dev/control/phone-event-outbox.js";
import { PhoneAgentRuntime, type AgentPlatform } from "../dev/control/phone-agent-runtime.js";

const phoneIdentity = {
  deviceId: "device_0123456789abcdef",
  tabId: "tab_0123456789abcdef",
  documentId: "document_0123456789abcdef",
};

test("terminal state completed while disconnected is reconciled without rerunning prompt", async () => {
  let online = true;
  const delivered: unknown[] = [];
  const platform: AgentPlatform = {
    send(message) {
      if (online) delivered.push(message);
    },
    reload() {},
    setTimer() {
      return 1;
    },
    clearTimer() {},
  };
  let finishPrompt!: () => void;
  let generations = 0;
  const agent = new PhoneAgentRuntime({
    identity: phoneIdentity,
    platform,
    handlers: {
      runPrompt: () =>
        new Promise<void>((resolve) => {
          generations += 1;
          finishPrompt = resolve;
        }),
    },
  });
  const running = agent.receive({
    schemaVersion: 1,
    type: "command",
    commandId: "command_0123456789abcdef",
    command: "runPrompt",
  });
  online = false;
  finishPrompt();
  await running;
  online = true;

  await agent.receive({
    schemaVersion: 1,
    type: "reconcile",
    commands: [
      {
        commandId: "command_0123456789abcdef",
        command: "runPrompt",
        state: "started",
        issuedAtMs: 1,
      },
    ],
  });

  assert.equal(generations, 1);
  assert.equal((delivered.at(-1) as { state: string }).state, "completed");
});

test("server terminal wins a reconcile conflict without replaying or rerunning prompt", async () => {
  const delivered: unknown[] = [];
  let generations = 0;
  const agent = new PhoneAgentRuntime({
    identity: phoneIdentity,
    platform: {
      send(message) {
        delivered.push(message);
      },
      reload() {},
      setTimer() {
        return 1;
      },
      clearTimer() {},
    },
    handlers: {
      runPrompt() {
        generations += 1;
      },
    },
  });
  const command = {
    schemaVersion: 1 as const,
    type: "command" as const,
    commandId: "command_0123456789abcdef",
    command: "runPrompt" as const,
  };
  await agent.receive(command);
  assert.equal(generations, 1);
  assert.equal((delivered.at(-1) as { readonly state?: string }).state, "completed");
  await agent.receive({
    schemaVersion: 1,
    type: "eventAck",
    documentId: phoneIdentity.documentId,
    status: "accepted",
    acknowledgedSeq: 3,
    expectedSeq: 4,
  });
  const beforeReconcile = delivered.length;
  const reconcile = {
    schemaVersion: 1 as const,
    type: "reconcile" as const,
    commands: [{
      commandId: command.commandId,
      command: command.command,
      state: "timed_out" as const,
      issuedAtMs: 1,
      startedAtMs: 2,
      terminalAtMs: 3,
      reason: "command_timeout",
    }],
  };

  await agent.receive(reconcile);
  await agent.receive(reconcile);
  assert.equal(delivered.length, beforeReconcile);
  await agent.receive(command);
  assert.equal(generations, 1);
  assert.equal((delivered.at(-1) as { readonly state?: string }).state, "timed_out");
});

test("server terminal settles an in-flight PhoneAgentRuntime handler", async () => {
  const delivered: unknown[] = [];
  let generations = 0;
  let finishPrompt!: () => void;
  const pendingPrompt = new Promise<void>((resolve) => {
    finishPrompt = resolve;
  });
  const agent = new PhoneAgentRuntime({
    identity: phoneIdentity,
    platform: {
      send(message) {
        delivered.push(message);
      },
      reload() {},
      setTimer() {
        return 1;
      },
      clearTimer() {},
    },
    handlers: {
      runPrompt() {
        generations += 1;
        return pendingPrompt;
      },
    },
  });
  const command = {
    schemaVersion: 1 as const,
    type: "command" as const,
    commandId: "command_0123456789abcdef",
    command: "runPrompt" as const,
  };
  const running = agent.receive(command);
  await Promise.resolve();
  assert.equal(generations, 1);
  const beforeReconcile = delivered.length;

  await agent.receive({
    schemaVersion: 1,
    type: "reconcile",
    commands: [{
      commandId: command.commandId,
      command: command.command,
      state: "timed_out",
      issuedAtMs: 1,
      terminalAtMs: 2,
      reason: "command_timeout",
    }],
  });
  assert.equal(delivered.length, beforeReconcile);

  finishPrompt();
  await running;
  assert.equal(delivered.length, beforeReconcile);
  await agent.receive(command);
  assert.equal(generations, 1);
  assert.equal((delivered.at(-1) as { readonly state?: string }).state, "timed_out");
});

test("sequence sync prunes an accepted frame whose acknowledgement was lost", async () => {
  const delivered: unknown[] = [];
  const agent = new PhoneAgentRuntime({
    identity: phoneIdentity,
    platform: {
      send(message) {
        delivered.push(message);
      },
      reload() {},
      setTimer() {
        return 1;
      },
      clearTimer() {},
    },
    handlers: {},
  });

  agent.reportReady();
  await agent.receive({
    schemaVersion: 1,
    type: "sequenceSync",
    documentId: phoneIdentity.documentId,
    expectedSeq: 2,
  });
  agent.reportTelemetry({ name: "after_sync" });

  assert.deepEqual(
    delivered.map((message) => (message as { eventSeq: number }).eventSeq),
    [1, 2],
  );
});

test("sequence sync rejects spoofed, future, and regressed proof", async () => {
  const agent = new PhoneAgentRuntime({
    identity: phoneIdentity,
    platform: {
      send() {},
      reload() {},
      setTimer() {
        return 1;
      },
      clearTimer() {},
    },
    handlers: {},
  });
  agent.reportReady();

  await assert.rejects(
    agent.receive({
      schemaVersion: 1,
      type: "sequenceSync",
      documentId: "document_spoofed_0123456789",
      expectedSeq: 2,
    }),
    /document/i,
  );
  await assert.rejects(
    agent.receive({
      schemaVersion: 1,
      type: "sequenceSync",
      documentId: phoneIdentity.documentId,
      expectedSeq: 3,
    }),
    /bounds|future|emitted/i,
  );
  await agent.receive({
    schemaVersion: 1,
    type: "sequenceSync",
    documentId: phoneIdentity.documentId,
    expectedSeq: 2,
  });
  await assert.rejects(
    agent.receive({
      schemaVersion: 1,
      type: "sequenceSync",
      documentId: phoneIdentity.documentId,
      expectedSeq: 1,
    }),
    /regression/i,
  );
});

test("lost state frame is replayed after reconnect and prompt runs exactly once", async () => {
  const plane = new ControlPlane();
  let agent: PhoneAgentRuntime | undefined;
  let connectionId = "";
  let online = true;
  let dropSequenceTwo = true;
  let generations = 0;
  let finishPrompt!: () => void;
  let running: Promise<void> | undefined;
  let connecting = false;
  const queuedServerMessages: ServerToPhoneMessage[] = [];
  const deliverServer = (message: ServerToPhoneMessage): boolean => {
    // A WebSocket cannot deliver a phone reply re-entrantly before connect()
    // returns the server-side connection identity.
    if (agent === undefined || connecting) {
      queuedServerMessages.push(message);
      return true;
    }
    const delivery = agent.receive(message);
    if (message.type === "command") running = delivery;
    return true;
  };
  const platform: AgentPlatform = {
    send(message) {
      if (!online) return;
      const event = message as { eventSeq: number };
      if (event.eventSeq === 2 && dropSequenceTwo) {
        dropSequenceTwo = false;
        return;
      }
      plane.receive(connectionId, message);
    },
    reload() {},
    setTimer() {
      return 1;
    },
    clearTimer() {},
  };

  connectionId = plane.connect(phoneIdentity, deliverServer);
  agent = new PhoneAgentRuntime({
    identity: phoneIdentity,
    platform,
    handlers: {
      runPrompt: () =>
        new Promise<void>((resolve) => {
          generations += 1;
          finishPrompt = resolve;
        }),
    },
  });
  for (const message of queuedServerMessages.splice(0)) await agent.receive(message);
  plane.issueCommand({
    deviceId: phoneIdentity.deviceId,
    tabId: phoneIdentity.tabId,
    commandId: "command_0123456789abcdef",
    command: "runPrompt",
  });

  plane.disconnect(connectionId, { kind: "socket_loss" });
  online = false;
  finishPrompt();
  await running;
  online = true;
  connecting = true;
  connectionId = plane.connect(phoneIdentity, deliverServer);
  connecting = false;
  for (const message of queuedServerMessages.splice(0)) await agent.receive(message);

  assert.equal(generations, 1);
  assert.equal(plane.getCommand("command_0123456789abcdef")?.state, "completed");
});

test("queued command dropped before phone receipt is dispatched on reconnect", async () => {
  const plane = new ControlPlane();
  const first = plane.connect(phoneIdentity, () => true);
  plane.issueCommand({
    deviceId: phoneIdentity.deviceId,
    tabId: phoneIdentity.tabId,
    commandId: "command_0123456789abcdef",
    command: "runPrompt",
  });
  plane.disconnect(first, { kind: "socket_loss" });

  let connectionId = "";
  let generations = 0;
  const queued: ServerToPhoneMessage[] = [];
  const agent = new PhoneAgentRuntime({
    identity: phoneIdentity,
    platform: {
      send(message) {
        plane.receive(connectionId, message);
      },
      reload() {},
      setTimer() {
        return 1;
      },
      clearTimer() {},
    },
    handlers: {
      runPrompt() {
        generations += 1;
      },
    },
  });
  connectionId = plane.connect(phoneIdentity, (message) => {
    queued.push(message);
    return true;
  });
  const resent = queued.filter((message) => message.type === "command");
  for (const message of queued.splice(0)) await agent.receive(message);

  assert.equal(resent.length, 1);
  assert.equal(generations, 1);
  assert.equal(plane.getCommand("command_0123456789abcdef")?.state, "completed");
});

test("delivered command is safely resent and deduplicated when receipt evidence is lost", async () => {
  const plane = new ControlPlane();
  let forwardPhoneEvents = false;
  let connectionId = "";
  let generations = 0;
  const agent = new PhoneAgentRuntime({
    identity: phoneIdentity,
    platform: {
      send(message) {
        if (forwardPhoneEvents) plane.receive(connectionId, message);
      },
      reload() {},
      setTimer() {
        return 1;
      },
      clearTimer() {},
    },
    handlers: {
      runPrompt() {
        generations += 1;
      },
    },
  });
  let firstRun: Promise<void> | undefined;
  connectionId = plane.connect(phoneIdentity, (message) => {
    if (message.type === "command") firstRun = agent.receive(message);
    return true;
  });
  plane.issueCommand({
    deviceId: phoneIdentity.deviceId,
    tabId: phoneIdentity.tabId,
    commandId: "command_0123456789abcdef",
    command: "runPrompt",
  });
  await firstRun;
  plane.disconnect(connectionId, { kind: "socket_loss" });

  const queued: ServerToPhoneMessage[] = [];
  forwardPhoneEvents = true;
  connectionId = plane.connect(phoneIdentity, (message) => {
    queued.push(message);
    return true;
  });
  const resent = queued.filter((message) => message.type === "command");
  for (const message of queued.splice(0)) await agent.receive(message);

  assert.equal(resent.length, 1);
  assert.equal(generations, 1);
  assert.equal(plane.getCommand("command_0123456789abcdef")?.state, "completed");
});

test("outbox is bounded and rejects spoofed acknowledgements and unavailable replay", () => {
  const outbox = new PhoneEventOutbox({ documentId: phoneIdentity.documentId, maxEntries: 2 });
  outbox.enqueue({
    schemaVersion: 1,
    type: "ready",
    ...phoneIdentity,
    eventSeq: 1,
  });
  outbox.enqueue({
    schemaVersion: 1,
    type: "ready",
    ...phoneIdentity,
    eventSeq: 2,
  });
  assert.throws(
    () =>
      outbox.enqueue({
        schemaVersion: 1,
        type: "ready",
        ...phoneIdentity,
        eventSeq: 3,
      }),
    /outbox.*limit/i,
  );
  assert.throws(
    () =>
      outbox.acknowledge({
        schemaVersion: 1,
        type: "eventAck",
        documentId: "document_spoofed_0123456789",
        status: "accepted",
        acknowledgedSeq: 1,
        expectedSeq: 2,
      }),
    /document/i,
  );
  outbox.acknowledge({
    schemaVersion: 1,
    type: "eventAck",
    documentId: phoneIdentity.documentId,
    status: "accepted",
    acknowledgedSeq: 2,
    expectedSeq: 3,
  });
  assert.throws(() => outbox.replayFrom(2), /already acknowledged|unavailable/i);
  assert.throws(() => outbox.replayFrom(4), /unavailable/i);
  assert.throws(
    () =>
      outbox.acknowledge({
        schemaVersion: 1,
        type: "eventAck",
        documentId: phoneIdentity.documentId,
        status: "spoofed" as "accepted",
        acknowledgedSeq: 2,
        expectedSeq: 3,
      }),
    /status/i,
  );
});
