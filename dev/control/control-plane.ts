import { createHash, randomUUID } from "node:crypto";

import type { SocketDeviceMetadata } from "./device-correlation.js";
import {
  CONTROL_SCHEMA_VERSION,
  CommandTracker,
  EventSequenceTracker,
  parsePhoneMessage,
  validateProtocolId,
  type CommandSnapshot,
  type ControlCommand,
  type PhoneIdentity,
  type PhoneToServerMessage,
  type SequenceResult,
  type ServerToPhoneMessage,
  type TerminalCommandState,
} from "./protocol.js";

export type { PhoneIdentity, ServerToPhoneMessage } from "./protocol.js";

export interface Clock {
  now(): number;
  setTimeout(callback: () => void, delayMs: number): number | NodeJS.Timeout;
  clearTimeout(id: number | NodeJS.Timeout): void;
}

const systemClock: Clock = {
  now: Date.now,
  setTimeout: (callback, delayMs) => setTimeout(callback, delayMs),
  clearTimeout: (id) => clearTimeout(id),
};

export type DisconnectEvidence =
  | { kind: "socket_loss" }
  | { kind: "page_lifecycle"; lifecycle: "navigation" | "pagehide" | "freeze" }
  | { kind: "device_lost" }
  | { kind: "command_timeout" }
  | { kind: "clean_unload" };

export const classifyDisconnect = (
  evidence: DisconnectEvidence,
): {
  classification: DisconnectEvidence["kind"];
  evidence: DisconnectEvidence["kind"][];
  confirmedCrash: false;
} => ({
  classification: evidence.kind,
  evidence: [evidence.kind],
  // None of these browser-visible signals proves that the process crashed.
  confirmedCrash: false,
});

interface Connection {
  connectionId: string;
  identity: PhoneIdentity;
  deviceMetadata?: SocketDeviceMetadata;
  send: (message: ServerToPhoneMessage) => boolean;
  lastEvidence?: DisconnectEvidence;
}

interface StoredCommand {
  tracker: CommandTracker;
  target: Pick<PhoneIdentity, "deviceId" | "tabId">;
  message: Extract<ServerToPhoneMessage, { type: "command" }>;
  dispatched: boolean;
  phoneReceived: boolean;
  payloadFingerprint: string;
  reloadProof?: {
    connectionId: string;
    documentId: string;
    startedAtMs: number;
    disconnectedAtMs?: number;
  };
  timer?: number | NodeJS.Timeout;
  retireAtMs?: number;
}

interface DisconnectRecord {
  atMs: number;
  identity: PhoneIdentity;
  classification: DisconnectEvidence["kind"] | "suspected_crash";
  evidence: string[];
  confirmedCrash: false;
  timer?: number | NodeJS.Timeout;
  reconnectedAtMs?: number;
}

export interface IssueCommandRequest {
  deviceId: string;
  tabId: string;
  commandId: string;
  command: ControlCommand;
  benchmarkId?: string;
  payload?: Record<string, unknown>;
}

export interface ControlPlaneOptions {
  clock?: Clock;
  commandTimeoutMs?: number;
  reloadTimeoutMs?: number;
  suspectedCrashTimeoutMs?: number;
  retentionMs?: number;
  maxCommands?: number;
  maxDisconnectRecords?: number;
  maxSequenceDocuments?: number;
  onTelemetry?: (event: Record<string, unknown>) => void | Promise<void>;
}

const tabKey = (identity: Pick<PhoneIdentity, "deviceId" | "tabId">): string =>
  `${identity.deviceId}\u0000${identity.tabId}`;

const isReload = (command: ControlCommand): boolean =>
  command === "warmReload" || command === "coldAppReload";

const isTerminal = (state: CommandSnapshot["state"]): state is TerminalCommandState =>
  state === "completed" ||
  state === "failed" ||
  state === "cancelled" ||
  state === "timed_out" ||
  state === "indeterminate";

const payloadFingerprint = (payload: Record<string, unknown> | undefined): string =>
  createHash("sha256").update(JSON.stringify(payload ?? null)).digest("hex");

export class ControlPlane {
  readonly #clock: Clock;
  readonly #commandTimeoutMs: number;
  readonly #reloadTimeoutMs: number;
  readonly #suspectedCrashTimeoutMs: number;
  readonly #retentionMs: number;
  readonly #maxCommands: number;
  readonly #maxDisconnectRecords: number;
  readonly #onTelemetry?: ControlPlaneOptions["onTelemetry"];
  readonly #connections = new Map<string, Connection>();
  readonly #connectionByTab = new Map<string, string>();
  readonly #commands = new Map<string, StoredCommand>();
  readonly #sequences: EventSequenceTracker;
  readonly #disconnects: DisconnectRecord[] = [];

  constructor(options: ControlPlaneOptions = {}) {
    this.#clock = options.clock ?? systemClock;
    this.#commandTimeoutMs = options.commandTimeoutMs ?? 120_000;
    this.#reloadTimeoutMs = options.reloadTimeoutMs ?? 30_000;
    this.#suspectedCrashTimeoutMs = options.suspectedCrashTimeoutMs ?? 10_000;
    this.#retentionMs = options.retentionMs ?? 15 * 60_000;
    this.#maxCommands = options.maxCommands ?? 1_024;
    this.#maxDisconnectRecords = options.maxDisconnectRecords ?? 1_024;
    for (const [value, label] of [
      [this.#retentionMs, "retention"],
      [this.#maxCommands, "command capacity"],
      [this.#maxDisconnectRecords, "disconnect capacity"],
    ] as const) {
      if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be positive`);
    }
    this.#sequences = new EventSequenceTracker({
      maxDocuments: options.maxSequenceDocuments ?? 1_024,
      retentionMs: this.#retentionMs,
      now: () => this.#clock.now(),
    });
    this.#onTelemetry = options.onTelemetry;
  }

  connect(
    identityInput: PhoneIdentity,
    send: Connection["send"],
    deviceMetadata?: SocketDeviceMetadata,
  ): string {
    this.#prune();
    const identity = {
      deviceId: validateProtocolId(identityInput.deviceId, "deviceId"),
      tabId: validateProtocolId(identityInput.tabId, "tabId"),
      documentId: validateProtocolId(identityInput.documentId, "documentId"),
    };
    const key = tabKey(identity);
    const connectionId = `connection_${randomUUID()}`;
    const connection = {
      connectionId,
      identity,
      send,
      ...(deviceMetadata === undefined ? {} : { deviceMetadata: structuredClone(deviceMetadata) }),
    };
    // Reserve bounded sequence ownership before publishing the connection.
    const expectedSeq = this.#sequences.acquire(identity.documentId);
    this.#connections.set(connectionId, connection);
    this.#connectionByTab.set(key, connectionId);
    for (const disconnect of this.#disconnects) {
      if (
        disconnect.identity.deviceId === identity.deviceId &&
        disconnect.identity.tabId === identity.tabId &&
        disconnect.classification === "socket_loss"
      ) {
        disconnect.reconnectedAtMs = this.#clock.now();
        if (disconnect.timer !== undefined) {
          this.#clock.clearTimeout(disconnect.timer);
          delete disconnect.timer;
        }
      }
    }

    this.#tryQueue(connection, {
      schemaVersion: CONTROL_SCHEMA_VERSION,
      type: "sequenceSync",
      documentId: identity.documentId,
      expectedSeq,
    });

    const commands = [...this.#commands.values()]
      .filter((stored) => tabKey(stored.target) === key)
      .map((stored) => stored.tracker.snapshot());
    this.#tryQueue(connection, { schemaVersion: CONTROL_SCHEMA_VERSION, type: "reconcile", commands });

    for (const stored of this.#commands.values()) {
      // A successful socket queue does not prove that the phone parsed the frame.
      // Reconnect retries remain safe because the phone deduplicates commandId.
      if (
        tabKey(stored.target) === key &&
        !stored.phoneReceived &&
        !isTerminal(stored.tracker.snapshot().state)
      ) {
        this.#attemptDispatch(stored, connection, true);
      }
    }
    return connectionId;
  }

  disconnect(connectionId: string, evidence?: DisconnectEvidence): void {
    this.#prune();
    const connection = this.#connections.get(connectionId);
    if (connection === undefined) return;
    const resolvedEvidence = evidence ?? connection.lastEvidence ?? { kind: "socket_loss" };
    this.#connections.delete(connectionId);
    this.#sequences.release(connection.identity.documentId);
    if (this.#connectionByTab.get(tabKey(connection.identity)) === connectionId) {
      this.#connectionByTab.delete(tabKey(connection.identity));
    }
    for (const stored of this.#commands.values()) {
      if (
        stored.reloadProof?.connectionId === connectionId &&
        stored.reloadProof.disconnectedAtMs === undefined &&
        stored.tracker.snapshot().state === "started"
      ) {
        stored.reloadProof.disconnectedAtMs = this.#clock.now();
      }
    }
    const disconnectRecord: DisconnectRecord = {
      atMs: this.#clock.now(),
      identity: connection.identity,
      ...classifyDisconnect(resolvedEvidence),
    };
    if (resolvedEvidence.kind === "socket_loss") {
      disconnectRecord.timer = this.#clock.setTimeout(() => {
        delete disconnectRecord.timer;
        if (disconnectRecord.reconnectedAtMs !== undefined) return;
        disconnectRecord.classification = "suspected_crash";
        if (!disconnectRecord.evidence.includes("reconnect_timeout")) {
          disconnectRecord.evidence.push("reconnect_timeout");
        }
      }, this.#suspectedCrashTimeoutMs);
    }
    while (this.#disconnects.length >= this.#maxDisconnectRecords) {
      const removed = this.#disconnects.shift();
      if (removed?.timer !== undefined) this.#clock.clearTimeout(removed.timer);
    }
    this.#disconnects.push(disconnectRecord);
  }

  issueCommand(request: IssueCommandRequest): CommandSnapshot {
    this.#prune();
    const commandId = validateProtocolId(request.commandId, "commandId");
    const deviceId = validateProtocolId(request.deviceId, "deviceId");
    const tabId = validateProtocolId(request.tabId, "tabId");
    const existing = this.#commands.get(commandId);
    if (existing !== undefined) {
      const same =
        existing.message.command === request.command &&
        existing.target.deviceId === deviceId &&
        existing.target.tabId === tabId &&
        existing.payloadFingerprint === payloadFingerprint(request.payload);
      if (!same) throw new Error("commandId is already assigned to different work");
      if (!existing.dispatched && !isTerminal(existing.tracker.snapshot().state)) {
        this.#attemptDispatch(existing);
      }
      return existing.tracker.snapshot();
    }
    if (this.#commands.size >= this.#maxCommands) throw new Error("command capacity reached");

    const tracker = new CommandTracker({
      commandId,
      command: request.command,
      issuedAtMs: this.#clock.now(),
      ...(request.benchmarkId === undefined ? {} : { benchmarkId: request.benchmarkId }),
    });
    // The operator endpoint accepts the command before delivery; phone-reported
    // accepted is therefore an idempotent acknowledgement of the same state.
    tracker.transition("accepted", this.#clock.now());
    const message: StoredCommand["message"] = {
      schemaVersion: CONTROL_SCHEMA_VERSION,
      type: "command",
      commandId,
      command: request.command,
      ...(request.benchmarkId === undefined ? {} : { benchmarkId: request.benchmarkId }),
      ...(request.payload === undefined ? {} : { payload: structuredClone(request.payload) }),
    };
    const target = { deviceId, tabId };
    const stored: StoredCommand = {
      tracker,
      target,
      message,
      dispatched: false,
      phoneReceived: false,
      payloadFingerprint: payloadFingerprint(request.payload),
    };
    // Publish tracker ownership before transport delivery because in-memory
    // tests and future local transports can synchronously acknowledge a send.
    this.#commands.set(commandId, stored);

    const timeoutMs = isReload(request.command) ? this.#reloadTimeoutMs : this.#commandTimeoutMs;
    stored.timer = this.#clock.setTimeout(() => {
      const state = tracker.snapshot().state;
      if (isTerminal(state)) return;
      tracker.forceTerminal(
        isReload(request.command) ? "indeterminate" : "timed_out",
        this.#clock.now(),
        isReload(request.command) ? "replacement_document_not_proven" : "command_timeout",
      );
      if (!isReload(request.command)) {
        this.#promoteSuspectedCrash(target, "command_timeout");
      }
      this.#retireTerminal(stored);
    }, timeoutMs);
    this.#attemptDispatch(stored);
    return tracker.snapshot();
  }

  #tryQueue(connection: Connection, message: ServerToPhoneMessage): boolean {
    try {
      return connection.send(message);
    } catch {
      return false;
    }
  }

  #attemptDispatch(stored: StoredCommand, connection?: Connection, force = false): boolean {
    if (stored.dispatched && !force) return true;
    if (isTerminal(stored.tracker.snapshot().state)) return false;
    const selected = connection ?? (() => {
      const connectionId = this.#connectionByTab.get(tabKey(stored.target));
      return connectionId === undefined ? undefined : this.#connections.get(connectionId);
    })();
    if (selected === undefined || tabKey(selected.identity) !== tabKey(stored.target)) return false;
    if (!this.#tryQueue(selected, stored.message)) return false;
    stored.dispatched = true;
    return true;
  }

  receive(connectionId: string, input: unknown): SequenceResult {
    this.#prune();
    const connection = this.#connections.get(connectionId);
    if (connection === undefined) throw new Error("unknown phone connection");
    const message = parsePhoneMessage(input);
    if (
      message.deviceId !== connection.identity.deviceId ||
      message.tabId !== connection.identity.tabId ||
      message.documentId !== connection.identity.documentId
    ) {
      throw new Error("phone message identity does not match its authenticated connection");
    }
    const sequence = this.#sequences.accept(message.documentId, message.eventSeq);
    if (!sequence.accepted) {
      connection.send({
        schemaVersion: CONTROL_SCHEMA_VERSION,
        type: "eventAck",
        documentId: message.documentId,
        status: sequence.classification,
        acknowledgedSeq: sequence.expected - 1,
        expectedSeq: sequence.expected,
      });
      return sequence;
    }

    if (message.type === "commandState") this.#applyCommandState(message, connection);
    if (message.type === "ready") this.#completeReplacementReload(message);
    if (message.type === "telemetry") {
      const evidence = disconnectEvidenceFromTelemetry(message.event);
      if (evidence !== undefined) connection.lastEvidence = evidence;
      if (this.#onTelemetry !== undefined) {
        // Telemetry storage is intentionally detached from protocol progress.
        void Promise.resolve(
          this.#onTelemetry({
            ...message.event,
            deviceId: connection.identity.deviceId,
            tabId: connection.identity.tabId,
            documentId: connection.identity.documentId,
            eventSeq: message.eventSeq,
            ...(connection.deviceMetadata === undefined
              ? {}
              : { deviceMetadata: connection.deviceMetadata }),
          }),
        ).catch(() => undefined);
      }
    }
    connection.send({
      schemaVersion: CONTROL_SCHEMA_VERSION,
      type: "eventAck",
      documentId: message.documentId,
      status: "accepted",
      acknowledgedSeq: message.eventSeq,
      expectedSeq: message.eventSeq + 1,
    });
    return sequence;
  }

  #applyCommandState(
    message: Extract<PhoneToServerMessage, { type: "commandState" }>,
    connection: Connection,
  ): void {
    const stored = this.#commands.get(message.commandId);
    if (stored === undefined) throw new Error("unknown commandId");
    if (tabKey(stored.target) !== tabKey(message)) throw new Error("command target mismatch");
    if (
      isReload(stored.message.command) &&
      (message.state === "completed" ||
        message.state === "indeterminate" ||
        message.state === "timed_out")
    ) {
      throw new Error(`phone cannot report server-owned reload state ${message.state}`);
    }
    stored.tracker.transition(message.state, this.#clock.now(), message.reason, message.result);
    stored.phoneReceived = true;
    if (stored.message.command === "runPrompt") delete stored.message.payload;
    if (isReload(stored.message.command) && message.state === "started") {
      stored.reloadProof = {
        connectionId: connection.connectionId,
        documentId: message.documentId,
        startedAtMs: this.#clock.now(),
      };
    }
    if (isTerminal(stored.tracker.snapshot().state)) {
      this.#retireTerminal(stored);
    }
  }

  #completeReplacementReload(message: Extract<PhoneToServerMessage, { type: "ready" }>): void {
    for (const stored of this.#commands.values()) {
      const snapshot = stored.tracker.snapshot();
      if (
        isReload(snapshot.command) &&
        snapshot.state === "started" &&
        tabKey(stored.target) === tabKey(message) &&
        stored.reloadProof?.disconnectedAtMs !== undefined &&
        stored.reloadProof.documentId !== message.documentId
      ) {
        stored.tracker.transition("completed", this.#clock.now());
        this.#retireTerminal(stored);
      }
    }
  }

  getCommand(commandId: string): CommandSnapshot | undefined {
    this.#prune();
    return this.#commands.get(commandId)?.tracker.snapshot();
  }

  getConnectedState(): Array<PhoneIdentity & { deviceMetadata?: SocketDeviceMetadata }> {
    this.#prune();
    return [...this.#connections.values()].map(({ identity, deviceMetadata }) =>
      structuredClone({
        ...identity,
        ...(deviceMetadata === undefined ? {} : { deviceMetadata }),
      }),
    );
  }

  getBenchmark(benchmarkId: string): CommandSnapshot[] {
    this.#prune();
    return [...this.#commands.values()]
      .map(({ tracker }) => tracker.snapshot())
      .filter((snapshot) => snapshot.benchmarkId === benchmarkId);
  }

  getDisconnectEvidence(): readonly Readonly<{
    atMs: number;
    identity: PhoneIdentity;
    classification: DisconnectEvidence["kind"] | "suspected_crash";
    evidence: string[];
    confirmedCrash: false;
  }>[] {
    this.#prune();
    return this.#disconnects.map(({ timer: _timer, reconnectedAtMs: _reconnectedAtMs, ...record }) =>
      structuredClone(record),
    );
  }

  getControlMetrics(): Readonly<{
    commands: number;
    disconnectRecords: number;
    sequenceDocuments: number;
    activeTimers: number;
    reloadProofs: number;
  }> {
    this.#prune();
    return Object.freeze({
      commands: this.#commands.size,
      disconnectRecords: this.#disconnects.length,
      sequenceDocuments: this.#sequences.size,
      activeTimers:
        [...this.#commands.values()].filter((stored) => stored.timer !== undefined).length +
        this.#disconnects.filter((record) => record.timer !== undefined).length,
      reloadProofs: [...this.#commands.values()].filter(
        (stored) => stored.reloadProof !== undefined,
      ).length,
    });
  }

  #retireTerminal(stored: StoredCommand): void {
    if (stored.timer !== undefined) {
      this.#clock.clearTimeout(stored.timer);
      delete stored.timer;
    }
    if (stored.message.command === "runPrompt") delete stored.message.payload;
    delete stored.reloadProof;
    stored.retireAtMs = this.#clock.now() + this.#retentionMs;
  }

  #prune(): void {
    const now = this.#clock.now();
    for (const [commandId, stored] of this.#commands) {
      if (stored.retireAtMs !== undefined && stored.retireAtMs < now) {
        if (stored.timer !== undefined) this.#clock.clearTimeout(stored.timer);
        this.#commands.delete(commandId);
      }
    }
    const cutoff = now - this.#retentionMs;
    for (let index = this.#disconnects.length - 1; index >= 0; index -= 1) {
      const record = this.#disconnects[index];
      if (record !== undefined && record.atMs < cutoff) {
        if (record.timer !== undefined) this.#clock.clearTimeout(record.timer);
        this.#disconnects.splice(index, 1);
      }
    }
  }

  #promoteSuspectedCrash(
    target: Pick<PhoneIdentity, "deviceId" | "tabId">,
    evidence: "command_timeout",
  ): void {
    const record = [...this.#disconnects]
      .reverse()
      .find(
        (candidate) =>
          candidate.identity.deviceId === target.deviceId &&
          candidate.identity.tabId === target.tabId &&
          candidate.reconnectedAtMs === undefined &&
          candidate.classification === "socket_loss",
      );
    if (record === undefined) return;
    record.classification = "suspected_crash";
    if (!record.evidence.includes(evidence)) record.evidence.push(evidence);
  }
}

const disconnectEvidenceFromTelemetry = (
  event: Record<string, unknown>,
): DisconnectEvidence | undefined => {
  if (event.category === "device" && event.name === "device_lost") {
    return { kind: "device_lost" };
  }
  if (event.category !== "lifecycle") return undefined;
  if (event.name === "clean_unload") return { kind: "clean_unload" };
  const metrics =
    typeof event.metrics === "object" && event.metrics !== null
      ? (event.metrics as Record<string, unknown>)
      : {};
  const lifecycle = metrics.lifecycle;
  return lifecycle === "navigation" || lifecycle === "freeze" || lifecycle === "pagehide"
    ? { kind: "page_lifecycle", lifecycle }
    : { kind: "page_lifecycle", lifecycle: "pagehide" };
};
