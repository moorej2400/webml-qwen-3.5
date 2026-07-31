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
  const deliverServer = (message: ServerToPhoneMessage): void => {
    // A WebSocket cannot deliver a phone reply re-entrantly before connect()
    // returns the server-side connection identity.
    if (agent === undefined || connecting) {
      queuedServerMessages.push(message);
      return;
    }
    const delivery = agent.receive(message);
    if (message.type === "command") running = delivery;
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
