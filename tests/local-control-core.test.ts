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
  const connectionId = plane.connect(currentIdentity, (message) => sent.push(message));
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
