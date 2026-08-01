import assert from "node:assert/strict";
import test from "node:test";

import {
  ChatOperationGate,
  emptyChatContextCopy,
  presentChatGenerationFailure,
} from "../src/chat-app-state.js";

test("reports a cancelled prompt whose GPU state requires disposal as recoverable", () => {
  assert.deepEqual(
    presentChatGenerationFailure(new Error("driver cancellation failed"), {
      cancellationRequested: true,
      runtimeFailed: true,
    }),
    {
      kind: "cancelled",
      notice: "Generation stopped. The runtime was disposed to protect GPU state.",
      status: "Ready to reload",
    },
  );
});

test("keeps a successful cancellation ready for another prompt", () => {
  assert.deepEqual(
    presentChatGenerationFailure(new Error("cancelled"), {
      cancellationRequested: true,
      runtimeFailed: false,
    }),
    {
      kind: "cancelled",
      notice: "Generation stopped.",
      status: "Ready for a prompt",
    },
  );
});

test("reports non-cancellation failures as attention items", () => {
  assert.deepEqual(
    presentChatGenerationFailure(new Error("model failed"), {
      cancellationRequested: false,
      runtimeFailed: true,
    }),
    {
      kind: "error",
      notice: "model failed",
      status: "Needs attention",
    },
  );
});

test("prevents a new conversation reset from overlapping a prompt operation", async () => {
  const gate = new ChatOperationGate();
  const release = Promise.resolve();
  const first = gate.run(async () => {
    await release;
    return "first";
  });
  assert.equal(await gate.run(async () => "second"), undefined);
  assert.equal(await first, "first");
  assert.equal(await gate.run(async () => "third"), "third");
});

test("provides the empty context copy after a conversation reset", () => {
  assert.equal(emptyChatContextCopy(), "Prefill a conversation to measure it");
});
