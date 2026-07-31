import assert from "node:assert/strict";
import test from "node:test";

import {
  ControlPlane,
  classifyDisconnect,
  type Clock,
  type PhoneIdentity,
  type ServerToPhoneMessage,
} from "../dev/control/control-plane.js";

class FakeClock implements Clock {
  nowMs = 1_000;
  #nextTimer = 1;
  #timers = new Map<number, { atMs: number; callback: () => void }>();

  now(): number {
    return this.nowMs;
  }

  setTimeout(callback: () => void, delayMs: number): number {
    const id = this.#nextTimer++;
    this.#timers.set(id, { atMs: this.nowMs + delayMs, callback });
    return id;
  }

  clearTimeout(id: number): void {
    this.#timers.delete(id);
  }

  advance(ms: number): void {
    this.nowMs += ms;
    const due = [...this.#timers.entries()]
      .filter(([, timer]) => timer.atMs <= this.nowMs)
      .sort((left, right) => left[1].atMs - right[1].atMs);
    for (const [id, timer] of due) {
      this.#timers.delete(id);
      timer.callback();
    }
  }
}

const identity = (documentId: string, tab = "tab_0123456789abcdef"): PhoneIdentity => ({
  deviceId: "device_0123456789abcdef",
  tabId: tab,
  documentId,
});

const connect = (
  plane: ControlPlane,
  currentIdentity: PhoneIdentity,
): { sent: ServerToPhoneMessage[]; connectionId: string } => {
  const sent: ServerToPhoneMessage[] = [];
  const connectionId = plane.connect(currentIdentity, (message) => {
    sent.push(message);
    return true;
  });
  return { sent, connectionId };
};

test("two tabs are tracked independently for one durable device", () => {
  const plane = new ControlPlane({ clock: new FakeClock() });
  connect(plane, identity("document_0123456789abcdef"));
  connect(plane, identity("document_1123456789abcdef", "tab_1123456789abcdef"));

  assert.equal(plane.getConnectedState().length, 2);
  assert.deepEqual(
    plane.getConnectedState().map((state) => state.tabId).sort(),
    ["tab_0123456789abcdef", "tab_1123456789abcdef"],
  );
});

test("retrying a command id never sends runPrompt twice", () => {
  const plane = new ControlPlane({ clock: new FakeClock() });
  const phone = connect(plane, identity("document_0123456789abcdef"));
  const request = {
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    commandId: "command_0123456789abcdef",
    benchmarkId: "benchmark_0123456789abcdef",
    command: "runPrompt" as const,
    payload: { promptRef: "fixture-1" },
  };

  const first = plane.issueCommand(request);
  const retry = plane.issueCommand(request);

  assert.equal(first.commandId, retry.commandId);
  assert.equal(phone.sent.filter((message) => message.type === "command").length, 1);
});

test("throwing command transport retains retry and timeout ownership", () => {
  const clock = new FakeClock();
  const plane = new ControlPlane({ clock, commandTimeoutMs: 500 });
  let throwing = true;
  let deliveries = 0;
  plane.connect(identity("document_0123456789abcdef"), (message) => {
    if (message.type === "command") {
      if (throwing) throw new Error("transport queue failed");
      deliveries += 1;
    }
    return true;
  });
  const request = {
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    commandId: "command_0123456789abcdef",
    command: "runPrompt" as const,
  };

  assert.doesNotThrow(() => plane.issueCommand(request));
  throwing = false;
  plane.issueCommand(request);
  plane.issueCommand(request);
  assert.equal(deliveries, 1);

  clock.advance(500);
  assert.equal(plane.getCommand(request.commandId)?.state, "timed_out");
});

test("closed transport reports failed queueing and later retry dispatches once", () => {
  const plane = new ControlPlane({ clock: new FakeClock() });
  let open = false;
  let deliveries = 0;
  plane.connect(identity("document_0123456789abcdef"), (message) => {
    if (message.type !== "command") return true;
    if (!open) return false;
    deliveries += 1;
    return true;
  });
  const request = {
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    commandId: "command_0123456789abcdef",
    command: "runPrompt" as const,
  };

  plane.issueCommand(request);
  plane.issueCommand(request);
  assert.equal(deliveries, 0);
  open = true;
  plane.issueCommand(request);
  plane.issueCommand(request);
  assert.equal(deliveries, 1);
});

test("undispatched command is sent once by a later connection", () => {
  const plane = new ControlPlane({ clock: new FakeClock() });
  const first = plane.connect(identity("document_0123456789abcdef"), (message) => {
    if (message.type === "command") throw new Error("transport queue failed");
    return true;
  });
  const request = {
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    commandId: "command_0123456789abcdef",
    command: "runPrompt" as const,
  };
  plane.issueCommand(request);
  plane.disconnect(first, { kind: "socket_loss" });
  let deliveries = 0;
  plane.connect(identity("document_0123456789abcdef"), (message) => {
    if (message.type === "command") deliveries += 1;
    return true;
  });

  assert.equal(deliveries, 1);
  plane.issueCommand(request);
  assert.equal(deliveries, 1);
});

test("timed-out undispatched command is not sent by a later connection", () => {
  const clock = new FakeClock();
  const plane = new ControlPlane({ clock, commandTimeoutMs: 500 });
  const first = plane.connect(identity("document_0123456789abcdef"), () => false);
  plane.issueCommand({
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    commandId: "command_0123456789abcdef",
    command: "runPrompt",
  });
  clock.advance(500);
  plane.disconnect(first, { kind: "socket_loss" });
  let deliveries = 0;
  plane.connect(identity("document_0123456789abcdef"), (message) => {
    if (message.type === "command") deliveries += 1;
    return true;
  });

  assert.equal(plane.getCommand("command_0123456789abcdef")?.state, "timed_out");
  assert.equal(deliveries, 0);
});

test("terminal command is not dispatched by a later connection", () => {
  const plane = new ControlPlane({ clock: new FakeClock() });
  const first = connect(plane, identity("document_0123456789abcdef"));
  plane.issueCommand({
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    commandId: "command_0123456789abcdef",
    command: "runPrompt",
  });
  for (const [eventSeq, state] of [
    [1, "accepted"],
    [2, "started"],
    [3, "completed"],
  ] as const) {
    plane.receive(first.connectionId, {
      schemaVersion: 1,
      type: "commandState",
      ...identity("document_0123456789abcdef"),
      eventSeq,
      commandId: "command_0123456789abcdef",
      state,
    });
  }
  plane.disconnect(first.connectionId, { kind: "socket_loss" });
  const replacement = connect(plane, identity("document_0123456789abcdef"));

  assert.equal(replacement.sent.some((message) => message.type === "command"), false);
});

test("successful reentrant acknowledgement sees command ownership", () => {
  const plane = new ControlPlane({ clock: new FakeClock() });
  let connectionId = "";
  let commandDeliveries = 0;
  connectionId = plane.connect(identity("document_0123456789abcdef"), (message) => {
    if (message.type === "command") {
      commandDeliveries += 1;
      plane.receive(connectionId, {
        schemaVersion: 1,
        type: "commandState",
        ...identity("document_0123456789abcdef"),
        eventSeq: 1,
        commandId: message.commandId,
        state: "accepted",
      });
    }
    return true;
  });

  const snapshot = plane.issueCommand({
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    commandId: "command_0123456789abcdef",
    command: "runPrompt",
  });

  assert.equal(snapshot.state, "accepted");
  assert.equal(commandDeliveries, 1);
});

test("same-document reconnect reconciles command state without reissuing work", () => {
  const clock = new FakeClock();
  const plane = new ControlPlane({ clock });
  const first = connect(plane, identity("document_0123456789abcdef"));
  plane.issueCommand({
    deviceId: identity("").deviceId,
    tabId: identity("").tabId,
    commandId: "command_0123456789abcdef",
    command: "runPrompt",
  });
  plane.receive(first.connectionId, {
    schemaVersion: 1,
    type: "commandState",
    ...identity("document_0123456789abcdef"),
    eventSeq: 1,
    commandId: "command_0123456789abcdef",
    state: "accepted",
  });
  plane.disconnect(first.connectionId, { kind: "socket_loss" });

  const replacement = connect(plane, identity("document_0123456789abcdef"));
  const reconcile = replacement.sent.find((message) => message.type === "reconcile");
  assert.ok(reconcile && reconcile.type === "reconcile");
  assert.equal(reconcile.commands[0]?.state, "accepted");
  assert.equal(replacement.sent.some((message) => message.type === "command"), false);
});

test("warm reload completes only after a replacement document reports ready", () => {
  const clock = new FakeClock();
  const plane = new ControlPlane({ clock, reloadTimeoutMs: 5_000 });
  const original = connect(plane, identity("document_0123456789abcdef"));
  plane.issueCommand({
    deviceId: identity("").deviceId,
    tabId: identity("").tabId,
    commandId: "command_0123456789abcdef",
    command: "warmReload",
  });
  for (const [eventSeq, state] of [
    [1, "accepted"],
    [2, "started"],
  ] as const) {
    plane.receive(original.connectionId, {
      schemaVersion: 1,
      type: "commandState",
      ...identity("document_0123456789abcdef"),
      eventSeq,
      commandId: "command_0123456789abcdef",
      state,
    });
  }
  plane.disconnect(original.connectionId, { kind: "page_lifecycle", lifecycle: "navigation" });

  assert.equal(plane.getCommand("command_0123456789abcdef")?.state, "started");
  const replacement = connect(plane, identity("document_1123456789abcdef"));
  plane.receive(replacement.connectionId, {
    schemaVersion: 1,
    type: "ready",
    ...identity("document_1123456789abcdef"),
    eventSeq: 1,
  });

  assert.equal(plane.getCommand("command_0123456789abcdef")?.state, "completed");
});

test("disconnect before reload start cannot prove replacement completion", () => {
  const plane = new ControlPlane({ clock: new FakeClock(), reloadTimeoutMs: 5_000 });
  const stale = connect(plane, identity("document_0123456789abcdef"));
  plane.disconnect(stale.connectionId, { kind: "socket_loss" });
  const current = connect(plane, identity("document_0123456789abcdef"));
  plane.issueCommand({
    deviceId: identity("").deviceId,
    tabId: identity("").tabId,
    commandId: "command_0123456789abcdef",
    command: "warmReload",
  });
  for (const [eventSeq, state] of [
    [1, "accepted"],
    [2, "started"],
  ] as const) {
    plane.receive(current.connectionId, {
      schemaVersion: 1,
      type: "commandState",
      ...identity("document_0123456789abcdef"),
      eventSeq,
      commandId: "command_0123456789abcdef",
      state,
    });
  }
  const premature = connect(plane, identity("document_1123456789abcdef"));
  plane.receive(premature.connectionId, {
    schemaVersion: 1,
    type: "ready",
    ...identity("document_1123456789abcdef"),
    eventSeq: 1,
  });
  assert.equal(plane.getCommand("command_0123456789abcdef")?.state, "started");

  plane.disconnect(current.connectionId, { kind: "page_lifecycle", lifecycle: "navigation" });
  const replacement = connect(plane, identity("document_2123456789abcdef"));
  plane.receive(replacement.connectionId, {
    schemaVersion: 1,
    type: "ready",
    ...identity("document_2123456789abcdef"),
    eventSeq: 1,
  });
  assert.equal(plane.getCommand("command_0123456789abcdef")?.state, "completed");
});

test("replacement ready cannot complete reload until the old document disconnects", () => {
  const plane = new ControlPlane({ clock: new FakeClock(), reloadTimeoutMs: 5_000 });
  const original = connect(plane, identity("document_0123456789abcdef"));
  plane.issueCommand({
    deviceId: identity("").deviceId,
    tabId: identity("").tabId,
    commandId: "command_0123456789abcdef",
    command: "warmReload",
  });
  plane.receive(original.connectionId, {
    schemaVersion: 1,
    type: "commandState",
    ...identity("document_0123456789abcdef"),
    eventSeq: 1,
    commandId: "command_0123456789abcdef",
    state: "accepted",
  });
  plane.receive(original.connectionId, {
    schemaVersion: 1,
    type: "commandState",
    ...identity("document_0123456789abcdef"),
    eventSeq: 2,
    commandId: "command_0123456789abcdef",
    state: "started",
  });

  const replacement = connect(plane, identity("document_1123456789abcdef"));
  plane.receive(replacement.connectionId, {
    schemaVersion: 1,
    type: "ready",
    ...identity("document_1123456789abcdef"),
    eventSeq: 1,
  });
  assert.equal(plane.getCommand("command_0123456789abcdef")?.state, "started");
});

test("reload timeout is indeterminate when no replacement document can be proven", () => {
  const clock = new FakeClock();
  const plane = new ControlPlane({ clock, reloadTimeoutMs: 500 });
  const phone = connect(plane, identity("document_0123456789abcdef"));
  plane.issueCommand({
    deviceId: identity("").deviceId,
    tabId: identity("").tabId,
    commandId: "command_0123456789abcdef",
    command: "coldAppReload",
  });
  for (const [eventSeq, state] of [
    [1, "accepted"],
    [2, "started"],
  ] as const) {
    plane.receive(phone.connectionId, {
      schemaVersion: 1,
      type: "commandState",
      ...identity("document_0123456789abcdef"),
      eventSeq,
      commandId: "command_0123456789abcdef",
      state,
    });
  }

  clock.advance(501);
  assert.equal(plane.getCommand("command_0123456789abcdef")?.state, "indeterminate");
});

for (const rejectedState of ["completed", "indeterminate"] as const) {
  test(`phone cannot report server-owned reload state ${rejectedState}`, () => {
    const plane = new ControlPlane({ clock: new FakeClock(), reloadTimeoutMs: 5_000 });
    const phone = connect(plane, identity("document_0123456789abcdef"));
    plane.issueCommand({
      deviceId: identity("").deviceId,
      tabId: identity("").tabId,
      commandId: "command_0123456789abcdef",
      command: "warmReload",
    });
    plane.receive(phone.connectionId, {
      schemaVersion: 1,
      type: "commandState",
      ...identity("document_0123456789abcdef"),
      eventSeq: 1,
      commandId: "command_0123456789abcdef",
      state: "accepted",
    });
    plane.receive(phone.connectionId, {
      schemaVersion: 1,
      type: "commandState",
      ...identity("document_0123456789abcdef"),
      eventSeq: 2,
      commandId: "command_0123456789abcdef",
      state: "started",
    });

    assert.throws(
      () =>
        plane.receive(phone.connectionId, {
          schemaVersion: 1,
          type: "commandState",
          ...identity("document_0123456789abcdef"),
          eventSeq: 3,
          commandId: "command_0123456789abcdef",
          state: rejectedState,
        }),
      /server-owned reload state/i,
    );
    assert.equal(plane.getCommand("command_0123456789abcdef")?.state, "started");
  });
}

test("phone-reported reload failure cannot later bypass replacement proof", () => {
  const plane = new ControlPlane({ clock: new FakeClock(), reloadTimeoutMs: 5_000 });
  const original = connect(plane, identity("document_0123456789abcdef"));
  plane.issueCommand({
    deviceId: identity("").deviceId,
    tabId: identity("").tabId,
    commandId: "command_0123456789abcdef",
    command: "warmReload",
  });
  for (const [eventSeq, state] of [
    [1, "accepted"],
    [2, "started"],
    [3, "failed"],
  ] as const) {
    plane.receive(original.connectionId, {
      schemaVersion: 1,
      type: "commandState",
      ...identity("document_0123456789abcdef"),
      eventSeq,
      commandId: "command_0123456789abcdef",
      state,
    });
  }
  plane.disconnect(original.connectionId, { kind: "page_lifecycle", lifecycle: "navigation" });
  const replacement = connect(plane, identity("document_1123456789abcdef"));
  plane.receive(replacement.connectionId, {
    schemaVersion: 1,
    type: "ready",
    ...identity("document_1123456789abcdef"),
    eventSeq: 1,
  });

  assert.equal(plane.getCommand("command_0123456789abcdef")?.state, "failed");
});

test("ordinary command timeout is timed_out", () => {
  const clock = new FakeClock();
  const plane = new ControlPlane({ clock, commandTimeoutMs: 500 });
  connect(plane, identity("document_0123456789abcdef"));
  plane.issueCommand({
    deviceId: identity("").deviceId,
    tabId: identity("").tabId,
    commandId: "command_0123456789abcdef",
    command: "load",
  });
  clock.advance(501);
  assert.equal(plane.getCommand("command_0123456789abcdef")?.state, "timed_out");
});

test("disconnect classifier does not claim a crash without device evidence", () => {
  assert.equal(classifyDisconnect({ kind: "socket_loss" }).classification, "socket_loss");
  assert.equal(
    classifyDisconnect({ kind: "page_lifecycle", lifecycle: "pagehide" }).classification,
    "page_lifecycle",
  );
  assert.equal(classifyDisconnect({ kind: "device_lost" }).classification, "device_lost");
  assert.equal(classifyDisconnect({ kind: "clean_unload" }).classification, "clean_unload");
  assert.equal(classifyDisconnect({ kind: "command_timeout" }).classification, "command_timeout");
  assert.equal(classifyDisconnect({ kind: "socket_loss" }).confirmedCrash, false);
});

test("phone lifecycle evidence refines later socket-loss classification", () => {
  const plane = new ControlPlane({ clock: new FakeClock() });
  const phone = connect(plane, identity("document_0123456789abcdef"));
  plane.receive(phone.connectionId, {
    schemaVersion: 1,
    type: "telemetry",
    ...identity("document_0123456789abcdef"),
    eventSeq: 1,
    event: {
      category: "lifecycle",
      name: "pagehide",
      metrics: { lifecycle: "navigation" },
    },
  });
  plane.disconnect(phone.connectionId);

  assert.equal(plane.getDisconnectEvidence().at(-1)?.classification, "page_lifecycle");
  assert.equal(plane.getDisconnectEvidence().at(-1)?.confirmedCrash, false);
});

test("journal callback receives authenticated correlation, not spoofed event identity", () => {
  const events: Record<string, unknown>[] = [];
  const plane = new ControlPlane({
    clock: new FakeClock(),
    onTelemetry: (event) => {
      events.push(event);
    },
  });
  const phone = connect(plane, identity("document_0123456789abcdef"));
  plane.receive(phone.connectionId, {
    schemaVersion: 1,
    type: "telemetry",
    ...identity("document_0123456789abcdef"),
    eventSeq: 1,
    event: {
      category: "generation",
      name: "token_rate",
      timestampMs: 1_000,
      deviceId: "device_spoofed_0123456789",
      eventSeq: 999,
      metrics: { tokensPerSecond: 31 },
    },
  });

  assert.equal(events[0]?.deviceId, "device_0123456789abcdef");
  assert.equal(events[0]?.tabId, "tab_0123456789abcdef");
  assert.equal(events[0]?.documentId, "document_0123456789abcdef");
  assert.equal(events[0]?.eventSeq, 1);
});

test("socket loss becomes suspected_crash only after reconnect timeout", () => {
  const clock = new FakeClock();
  const plane = new ControlPlane({ clock, suspectedCrashTimeoutMs: 500 });
  const phone = connect(plane, identity("document_0123456789abcdef"));
  plane.disconnect(phone.connectionId, { kind: "socket_loss" });

  assert.equal(plane.getDisconnectEvidence().at(-1)?.classification, "socket_loss");
  clock.advance(501);
  const evidence = plane.getDisconnectEvidence().at(-1);
  assert.equal(evidence?.classification, "suspected_crash");
  assert.deepEqual(evidence?.evidence, ["socket_loss", "reconnect_timeout"]);
  assert.equal(evidence?.confirmedCrash, false);
});

test("reconnect before timeout prevents suspected crash classification", () => {
  const clock = new FakeClock();
  const plane = new ControlPlane({ clock, suspectedCrashTimeoutMs: 500 });
  const phone = connect(plane, identity("document_0123456789abcdef"));
  plane.disconnect(phone.connectionId, { kind: "socket_loss" });
  connect(plane, identity("document_0123456789abcdef"));
  clock.advance(501);

  assert.equal(plane.getDisconnectEvidence().at(-1)?.classification, "socket_loss");
});

test("sequence gaps and replays are rejected without applying state", () => {
  const plane = new ControlPlane({ clock: new FakeClock() });
  const phone = connect(plane, identity("document_0123456789abcdef"));
  plane.issueCommand({
    deviceId: identity("").deviceId,
    tabId: identity("").tabId,
    commandId: "command_0123456789abcdef",
    command: "load",
  });

  const base = {
    schemaVersion: 1 as const,
    type: "commandState" as const,
    ...identity("document_0123456789abcdef"),
    commandId: "command_0123456789abcdef",
    state: "accepted" as const,
  };
  assert.deepEqual(plane.receive(phone.connectionId, { ...base, eventSeq: 2 }), {
    accepted: false,
    classification: "gap",
    expected: 1,
  });
  assert.deepEqual(plane.receive(phone.connectionId, { ...base, eventSeq: 1 }), {
    accepted: true,
    expected: 1,
  });
  assert.deepEqual(plane.receive(phone.connectionId, { ...base, eventSeq: 1 }), {
    accepted: false,
    classification: "replay",
    expected: 2,
  });
});

test("a quiet connected document keeps its next event sequence past retention", () => {
  const clock = new FakeClock();
  const plane = new ControlPlane({ clock, retentionMs: 10 });
  const phone = connect(plane, identity("document_0123456789abcdef"));

  assert.deepEqual(
    plane.receive(phone.connectionId, {
      schemaVersion: 1,
      type: "ready",
      ...identity("document_0123456789abcdef"),
      eventSeq: 1,
    }),
    { accepted: true, expected: 1 },
  );
  clock.advance(11);

  assert.deepEqual(
    plane.receive(phone.connectionId, {
      schemaVersion: 1,
      type: "telemetry",
      ...identity("document_0123456789abcdef"),
      eventSeq: 2,
      event: { category: "generation", name: "quiet_resume" },
    }),
    { accepted: true, expected: 2 },
  );
});

test("closing one of two connections cannot expire their shared document sequence", () => {
  const clock = new FakeClock();
  const plane = new ControlPlane({ clock, retentionMs: 10 });
  const first = connect(plane, identity("document_0123456789abcdef"));
  const second = connect(plane, identity("document_0123456789abcdef"));

  plane.receive(first.connectionId, {
    schemaVersion: 1,
    type: "ready",
    ...identity("document_0123456789abcdef"),
    eventSeq: 1,
  });
  plane.disconnect(first.connectionId, { kind: "socket_loss" });
  clock.advance(11);

  assert.deepEqual(
    plane.receive(second.connectionId, {
      schemaVersion: 1,
      type: "telemetry",
      ...identity("document_0123456789abcdef"),
      eventSeq: 2,
      event: { category: "generation", name: "shared_connection_resume" },
    }),
    { accepted: true, expected: 2 },
  );
});

test("document retention starts when its final connection closes", () => {
  const clock = new FakeClock();
  const plane = new ControlPlane({ clock, retentionMs: 10 });
  const first = connect(plane, identity("document_0123456789abcdef"));
  const second = connect(plane, identity("document_0123456789abcdef"));

  plane.receive(first.connectionId, {
    schemaVersion: 1,
    type: "ready",
    ...identity("document_0123456789abcdef"),
    eventSeq: 1,
  });
  clock.advance(9);
  plane.disconnect(first.connectionId, { kind: "socket_loss" });
  plane.disconnect(second.connectionId, { kind: "socket_loss" });
  clock.advance(9);

  const withinRetention = connect(plane, identity("document_0123456789abcdef"));
  assert.deepEqual(withinRetention.sent[0], {
    schemaVersion: 1,
    type: "sequenceSync",
    documentId: "document_0123456789abcdef",
    expectedSeq: 2,
  });
  plane.disconnect(withinRetention.connectionId, { kind: "socket_loss" });
  clock.advance(11);

  const afterRetention = connect(plane, identity("document_0123456789abcdef"));
  assert.deepEqual(afterRetention.sent[0], {
    schemaVersion: 1,
    type: "sequenceSync",
    documentId: "document_0123456789abcdef",
    expectedSeq: 1,
  });
});

test("sequence capacity never evicts active documents", () => {
  const clock = new FakeClock();
  const plane = new ControlPlane({ clock, retentionMs: 10, maxSequenceDocuments: 2 });
  const first = connect(plane, identity("document_0123456789abcdef"));
  connect(plane, identity("document_1123456789abcdef", "tab_1123456789abcdef"));
  clock.advance(11);

  assert.throws(
    () => connect(plane, identity("document_2123456789abcdef", "tab_2123456789abcdef")),
    /capacity/i,
  );
  assert.equal(plane.getConnectedState().length, 2);

  plane.disconnect(first.connectionId, { kind: "socket_loss" });
  clock.advance(11);
  assert.doesNotThrow(() =>
    connect(plane, identity("document_2123456789abcdef", "tab_2123456789abcdef")),
  );
});

test("control state enforces caps, expiry, timer retirement, and payload retirement", () => {
  const clock = new FakeClock();
  const sentCommands: Extract<ServerToPhoneMessage, { type: "command" }>[] = [];
  const plane = new ControlPlane({
    clock,
    commandTimeoutMs: 10,
    retentionMs: 20,
    maxCommands: 2,
    maxDisconnectRecords: 2,
  });
  const phone = plane.connect(identity("document_0123456789abcdef"), (message) => {
    if (message.type === "command") sentCommands.push(message);
    return true;
  });
  const request = (suffix: string) => ({
    deviceId: identity("").deviceId,
    tabId: identity("").tabId,
    commandId: `command_${suffix.padEnd(16, "0")}`,
    command: "runPrompt" as const,
    payload: { prompt: `private-${suffix}` },
  });
  plane.issueCommand(request("a"));
  plane.receive(phone, {
    schemaVersion: 1,
    type: "commandState",
    ...identity("document_0123456789abcdef"),
    eventSeq: 1,
    commandId: request("a").commandId,
    state: "accepted",
  });
  assert.equal(sentCommands[0]?.payload, undefined);
  assert.doesNotThrow(() => plane.issueCommand(request("a")));
  assert.throws(
    () => plane.issueCommand({ ...request("a"), payload: { prompt: "different" } }),
    /different work/i,
  );
  plane.issueCommand(request("b"));
  assert.throws(() => plane.issueCommand(request("c")), /capacity/i);
  clock.advance(10);
  assert.equal(plane.getControlMetrics().activeTimers, 0);
  clock.advance(21);
  assert.doesNotThrow(() => plane.issueCommand(request("c")));

  for (let index = 0; index < 3; index += 1) {
    const connection = connect(
      plane,
      identity(`document_disconnect_${String(index).padStart(2, "0")}`),
    );
    plane.disconnect(connection.connectionId, { kind: "socket_loss" });
  }
  assert.equal(plane.getDisconnectEvidence().length, 2);
  clock.advance(21);
  assert.equal(plane.getDisconnectEvidence().length, 0);
});
