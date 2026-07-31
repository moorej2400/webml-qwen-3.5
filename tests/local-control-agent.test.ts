import assert from "node:assert/strict";
import test from "node:test";

import { PhoneAgentRuntime, type AgentPlatform } from "../dev/control/phone-agent-runtime.js";

const createPlatform = (): AgentPlatform & {
  reloads: number;
  sent: unknown[];
} => ({
  reloads: 0,
  sent: [],
  send(message) {
    this.sent.push(message);
  },
  reload() {
    this.reloads += 1;
  },
  setTimer(callback) {
    callback();
    return 1;
  },
  clearTimer() {},
});

test("warm reload calls location reload only after accepted state is sent", () => {
  const platform = createPlatform();
  const agent = new PhoneAgentRuntime({
    identity: {
      deviceId: "device_0123456789abcdef",
      tabId: "tab_0123456789abcdef",
      documentId: "document_0123456789abcdef",
    },
    platform,
    handlers: {},
  });

  agent.receive({
    schemaVersion: 1,
    type: "command",
    commandId: "command_0123456789abcdef",
    command: "warmReload",
  });

  assert.equal(platform.reloads, 1);
  assert.deepEqual(
    platform.sent.map((message) => (message as { state: string }).state),
    ["accepted", "started"],
  );
});

test("repeated runPrompt command id is reconciled without duplicate generation", async () => {
  const platform = createPlatform();
  let generations = 0;
  const agent = new PhoneAgentRuntime({
    identity: {
      deviceId: "device_0123456789abcdef",
      tabId: "tab_0123456789abcdef",
      documentId: "document_0123456789abcdef",
    },
    platform,
    handlers: {
      async runPrompt() {
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
  await agent.receive(command);

  assert.equal(generations, 1);
  assert.equal(
    platform.sent.filter(
      (message) => (message as { state?: string }).state === "completed",
    ).length,
    2,
  );
});

test("cold app reload reports external fallback instead of inventing control", async () => {
  const platform = createPlatform();
  const agent = new PhoneAgentRuntime({
    identity: {
      deviceId: "device_0123456789abcdef",
      tabId: "tab_0123456789abcdef",
      documentId: "document_0123456789abcdef",
    },
    platform,
    handlers: {},
  });

  await agent.receive({
    schemaVersion: 1,
    type: "command",
    commandId: "command_0123456789abcdef",
    command: "coldAppReload",
  });

  assert.equal(
    (platform.sent.at(-1) as { state: string }).state,
    "indeterminate",
  );
  assert.equal(
    (platform.sent.at(-1) as { reason: string }).reason,
    "external_fallback_required",
  );
});
