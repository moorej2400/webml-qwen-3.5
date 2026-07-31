import { sanitizeStateResult, type SanitizedStateResult } from "./state-result.js";

export const CONTROL_SCHEMA_VERSION = 1 as const;

export const CONTROL_COMMANDS = [
  "load",
  "dispose",
  "runPrompt",
  "cancelPrompt",
  "getState",
  "warmReload",
  "coldAppReload",
] as const;

export const TERMINAL_COMMAND_STATES = [
  "completed",
  "failed",
  "cancelled",
  "timed_out",
  "indeterminate",
] as const;

export type ControlCommand = (typeof CONTROL_COMMANDS)[number];
export type TerminalCommandState = (typeof TERMINAL_COMMAND_STATES)[number];
export type CommandState = "issued" | "accepted" | "started" | TerminalCommandState;

export interface PhoneIdentity {
  deviceId: string;
  tabId: string;
  documentId: string;
}

export interface CommandStateMessage extends PhoneIdentity {
  schemaVersion: typeof CONTROL_SCHEMA_VERSION;
  type: "commandState";
  eventSeq: number;
  commandId: string;
  state: Exclude<CommandState, "issued">;
  reason?: string;
  result?: SanitizedStateResult;
}

export interface ReadyMessage extends PhoneIdentity {
  schemaVersion: typeof CONTROL_SCHEMA_VERSION;
  type: "ready";
  eventSeq: number;
}

export interface TelemetryMessage extends PhoneIdentity {
  schemaVersion: typeof CONTROL_SCHEMA_VERSION;
  type: "telemetry";
  eventSeq: number;
  event: Record<string, unknown>;
}

export type PhoneToServerMessage = CommandStateMessage | ReadyMessage | TelemetryMessage;

export interface CommandMessage {
  schemaVersion: typeof CONTROL_SCHEMA_VERSION;
  type: "command";
  commandId: string;
  command: ControlCommand;
  benchmarkId?: string;
  payload?: Record<string, unknown>;
}

export interface CommandSnapshot {
  commandId: string;
  command: ControlCommand;
  state: CommandState;
  benchmarkId?: string;
  issuedAtMs: number;
  startedAtMs?: number;
  terminalAtMs?: number;
  reason?: string;
  result?: SanitizedStateResult;
}

export interface ReconcileMessage {
  schemaVersion: typeof CONTROL_SCHEMA_VERSION;
  type: "reconcile";
  commands: CommandSnapshot[];
}

export interface EventAckMessage {
  schemaVersion: typeof CONTROL_SCHEMA_VERSION;
  type: "eventAck";
  documentId: string;
  status: "accepted" | "gap" | "replay";
  acknowledgedSeq: number;
  expectedSeq: number;
}

export interface SequenceSyncMessage {
  schemaVersion: typeof CONTROL_SCHEMA_VERSION;
  type: "sequenceSync";
  documentId: string;
  expectedSeq: number;
}

export type ServerToPhoneMessage =
  | CommandMessage
  | ReconcileMessage
  | EventAckMessage
  | SequenceSyncMessage;

const ID_PATTERN = /^[A-Za-z0-9._:-]{16,128}$/;

export const validateProtocolId = (value: unknown, field: string): string => {
  if (typeof value !== "string" || !ID_PATTERN.test(value)) {
    throw new TypeError(`${field} must be a 16-128 character safe printable identifier`);
  }
  return value;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const parseIdentity = (value: Record<string, unknown>): PhoneIdentity => ({
  deviceId: validateProtocolId(value.deviceId, "deviceId"),
  tabId: validateProtocolId(value.tabId, "tabId"),
  documentId: validateProtocolId(value.documentId, "documentId"),
});

const parseSequence = (value: unknown): number => {
  if (!Number.isSafeInteger(value) || (value as number) < 1) {
    throw new TypeError("eventSeq must be a positive safe integer");
  }
  return value as number;
};

const isTerminal = (state: string): state is TerminalCommandState =>
  (TERMINAL_COMMAND_STATES as readonly string[]).includes(state);

export const parsePhoneMessage = (input: unknown): PhoneToServerMessage => {
  if (!isRecord(input)) {
    throw new TypeError("phone message must be an object");
  }
  if (input.schemaVersion !== CONTROL_SCHEMA_VERSION) {
    throw new TypeError("unsupported control schema version");
  }
  const identity = parseIdentity(input);
  const eventSeq = parseSequence(input.eventSeq);

  if (input.type === "ready") {
    return { schemaVersion: CONTROL_SCHEMA_VERSION, type: "ready", ...identity, eventSeq };
  }
  if (input.type === "commandState") {
    const state = input.state;
    if (
      state !== "accepted" &&
      state !== "started" &&
      (typeof state !== "string" || !isTerminal(state))
    ) {
      throw new TypeError("invalid command state");
    }
    const message: CommandStateMessage = {
      schemaVersion: CONTROL_SCHEMA_VERSION,
      type: "commandState",
      ...identity,
      eventSeq,
      commandId: validateProtocolId(input.commandId, "commandId"),
      state,
    };
    if (typeof input.reason === "string") {
      message.reason = input.reason.slice(0, 128);
    }
    if (input.result !== undefined) message.result = sanitizeStateResult(input.result);
    return message;
  }
  if (input.type === "telemetry" && isRecord(input.event)) {
    return {
      schemaVersion: CONTROL_SCHEMA_VERSION,
      type: "telemetry",
      ...identity,
      eventSeq,
      event: input.event,
    };
  }
  throw new TypeError("unsupported phone message type");
};

export interface CommandTrackerInit {
  commandId: string;
  command: ControlCommand;
  issuedAtMs: number;
  benchmarkId?: string;
}

export class CommandTracker {
  readonly #snapshot: CommandSnapshot;

  constructor(initial: CommandTrackerInit) {
    this.#snapshot = {
      commandId: validateProtocolId(initial.commandId, "commandId"),
      command: initial.command,
      state: "issued",
      issuedAtMs: initial.issuedAtMs,
      ...(initial.benchmarkId === undefined
        ? {}
        : { benchmarkId: validateProtocolId(initial.benchmarkId, "benchmarkId") }),
    };
  }

  transition(
    state: Exclude<CommandState, "issued">,
    atMs: number,
    reason?: string,
    result?: SanitizedStateResult,
  ): void {
    const current = this.#snapshot.state;
    if (isTerminal(current)) {
      if (state === current) return;
      throw new Error(`command already reached terminal state ${current}`);
    }
    if (state === current) {
      if (reason !== undefined) this.#snapshot.reason = reason.slice(0, 128);
      return;
    }
    if (current === "issued" && state !== "accepted") {
      throw new Error("command must be accepted before it can start");
    }
    if (current === "accepted" && state !== "started") {
      throw new Error("command must be started before it can terminate");
    }
    if (current === "started" && (state === "accepted" || state === "started")) {
      throw new Error("command cannot regress after it has started");
    }

    this.#snapshot.state = state;
    if (state === "started") this.#snapshot.startedAtMs = atMs;
    if (isTerminal(state)) {
      this.#snapshot.terminalAtMs = atMs;
      if (reason !== undefined) this.#snapshot.reason = reason.slice(0, 128);
      if (result !== undefined) this.#snapshot.result = sanitizeStateResult(result);
    }
  }

  /**
   * Server-side expiry preserves the public lifecycle even when the phone
   * disappears before it can report the intermediate states.
   */
  forceTerminal(state: TerminalCommandState, atMs: number, reason: string): void {
    if (isTerminal(this.#snapshot.state)) return;
    if (this.#snapshot.state === "issued") this.transition("accepted", atMs);
    if (this.#snapshot.state === "accepted") this.transition("started", atMs);
    this.transition(state, atMs, reason);
  }

  snapshot(): CommandSnapshot {
    return structuredClone(this.#snapshot);
  }
}

export type SequenceResult =
  | { accepted: true; expected: number }
  | { accepted: false; expected: number; classification: "gap" | "replay" };

export interface EventSequenceTrackerOptions {
  maxDocuments?: number;
  retentionMs?: number;
  now?: () => number;
}

export class EventSequenceTracker {
  readonly #documents = new Map<string, { next: number; lastSeenAtMs: number }>();
  readonly #maxDocuments: number;
  readonly #retentionMs: number;
  readonly #now: () => number;

  constructor(options: EventSequenceTrackerOptions = {}) {
    this.#maxDocuments = options.maxDocuments ?? 1_024;
    this.#retentionMs = options.retentionMs ?? 15 * 60_000;
    this.#now = options.now ?? Date.now;
    if (!Number.isSafeInteger(this.#maxDocuments) || this.#maxDocuments < 1) {
      throw new RangeError("sequence document capacity must be positive");
    }
    if (!Number.isSafeInteger(this.#retentionMs) || this.#retentionMs < 1) {
      throw new RangeError("sequence retention must be positive");
    }
  }

  accept(documentId: string, sequence: number): SequenceResult {
    validateProtocolId(documentId, "documentId");
    const record = this.#getOrCreate(documentId);
    const expected = record.next;
    if (sequence < expected) return { accepted: false, expected, classification: "replay" };
    if (sequence > expected) return { accepted: false, expected, classification: "gap" };
    record.next = expected + 1;
    record.lastSeenAtMs = this.#now();
    return { accepted: true, expected };
  }

  expected(documentId: string): number {
    validateProtocolId(documentId, "documentId");
    return this.#getOrCreate(documentId).next;
  }

  get size(): number {
    this.#prune();
    return this.#documents.size;
  }

  #getOrCreate(documentId: string): { next: number; lastSeenAtMs: number } {
    this.#prune();
    const existing = this.#documents.get(documentId);
    if (existing !== undefined) {
      existing.lastSeenAtMs = this.#now();
      return existing;
    }
    if (this.#documents.size >= this.#maxDocuments) {
      throw new Error("sequence document capacity reached");
    }
    const created = { next: 1, lastSeenAtMs: this.#now() };
    this.#documents.set(documentId, created);
    return created;
  }

  #prune(): void {
    const cutoff = this.#now() - this.#retentionMs;
    for (const [documentId, record] of this.#documents) {
      if (record.lastSeenAtMs < cutoff) this.#documents.delete(documentId);
    }
  }
}
