import assert from "node:assert/strict";
import test from "node:test";

import {
  CommandTracker,
  EventSequenceTracker,
  parsePhoneMessage,
  validateProtocolId,
} from "../dev/control/protocol.js";

test("protocol accepts bounded printable identifiers and rejects unsafe values", () => {
  assert.equal(validateProtocolId("device_0123456789abcdef", "deviceId"), "device_0123456789abcdef");
  assert.throws(() => validateProtocolId("short", "deviceId"), /deviceId/);
  assert.throws(() => validateProtocolId("bad\nidentifier_012345", "deviceId"), /deviceId/);
  assert.throws(() => validateProtocolId("x".repeat(129), "deviceId"), /deviceId/);
});

test("phone messages reject unknown schema versions", () => {
  assert.throws(
    () =>
      parsePhoneMessage({
        schemaVersion: 2,
        type: "ready",
        deviceId: "device_0123456789abcdef",
        tabId: "tab_0123456789abcdef",
        documentId: "document_0123456789abcdef",
        eventSeq: 1,
      }),
    /schema version/i,
  );
});

test("command tracker enforces accepted, started, and one terminal state", () => {
  const tracker = new CommandTracker({
    commandId: "command_0123456789abcdef",
    command: "load",
    issuedAtMs: 10,
  });

  tracker.transition("accepted", 11);
  tracker.transition("started", 12);
  tracker.transition("completed", 13);

  assert.equal(tracker.snapshot().state, "completed");
  assert.throws(() => tracker.transition("failed", 14), /terminal/i);
});

test("command tracker rejects skipped and regressed lifecycle states", () => {
  const skipped = new CommandTracker({
    commandId: "command_0123456789abcdef",
    command: "runPrompt",
    issuedAtMs: 10,
  });
  assert.throws(() => skipped.transition("started", 11), /accepted/i);

  const regressed = new CommandTracker({
    commandId: "command_1123456789abcdef",
    command: "runPrompt",
    issuedAtMs: 10,
  });
  regressed.transition("accepted", 11);
  regressed.transition("started", 12);
  assert.throws(() => regressed.transition("accepted", 13), /started/i);
});

test("event sequence tracker detects replay and gaps per document", () => {
  const sequences = new EventSequenceTracker();
  const documentId = "document_0123456789abcdef";

  assert.deepEqual(sequences.accept(documentId, 1), { accepted: true, expected: 1 });
  assert.deepEqual(sequences.accept(documentId, 2), { accepted: true, expected: 2 });
  assert.deepEqual(sequences.accept(documentId, 2), {
    accepted: false,
    expected: 3,
    classification: "replay",
  });
  assert.deepEqual(sequences.accept(documentId, 5), {
    accepted: false,
    expected: 3,
    classification: "gap",
  });
});

test("event sequence tracker bounds and expires document state", () => {
  let nowMs = 0;
  const tracker = new EventSequenceTracker({
    maxDocuments: 2,
    retentionMs: 10,
    now: () => nowMs,
  });
  tracker.accept("document_0123456789abcdef", 1);
  tracker.accept("document_1123456789abcdef", 1);
  assert.throws(
    () => tracker.accept("document_2123456789abcdef", 1),
    /capacity/i,
  );
  nowMs = 11;
  tracker.accept("document_2123456789abcdef", 1);
  assert.equal(tracker.size, 1);
});
