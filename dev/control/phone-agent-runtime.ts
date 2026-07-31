import {
  CONTROL_SCHEMA_VERSION,
  type CommandMessage,
  type CommandState,
  type ControlCommand,
  type PhoneIdentity,
  type ReconcileMessage,
  type ServerToPhoneMessage,
} from "./protocol.js";
import { PhoneEventOutbox } from "./phone-event-outbox.js";
import { sanitizeStateResult, type SanitizedStateResult } from "./state-result.js";

export interface AgentPlatform {
  send(message: unknown): void;
  reload(): void;
  setTimer(callback: () => void, delayMs: number): number;
  clearTimer(id: number): void;
}

type AgentHandler = (payload?: Record<string, unknown>) => unknown | Promise<unknown>;
export type AgentHandlers = Partial<
  Record<Exclude<ControlCommand, "warmReload" | "coldAppReload">, AgentHandler>
>;

export interface PhoneAgentRuntimeOptions {
  identity: PhoneIdentity;
  platform: AgentPlatform;
  handlers: AgentHandlers;
  outboxLimit?: number;
}

interface AgentCommandRecord {
  state: Exclude<CommandState, "issued">;
  reason?: string;
  result?: SanitizedStateResult;
}

export class PhoneAgentRuntime {
  readonly #identity: PhoneIdentity;
  readonly #platform: AgentPlatform;
  readonly #handlers: AgentHandlers;
  readonly #commands = new Map<string, AgentCommandRecord>();
  readonly #outbox: PhoneEventOutbox;
  #eventSeq = 0;

  constructor(options: PhoneAgentRuntimeOptions) {
    this.#identity = options.identity;
    this.#platform = options.platform;
    this.#handlers = options.handlers;
    this.#outbox = new PhoneEventOutbox({
      documentId: options.identity.documentId,
      ...(options.outboxLimit === undefined ? {} : { maxEntries: options.outboxLimit }),
    });
  }

  async receive(message: ServerToPhoneMessage): Promise<void> {
    if (message.type === "eventAck") {
      this.#outbox.acknowledge(message);
      if (message.status === "gap") {
        for (const replay of this.#outbox.replayFrom(message.expectedSeq)) {
          this.#platform.send(replay);
        }
      }
      return;
    }
    if (message.type === "sequenceSync") {
      if (message.documentId !== this.#identity.documentId) {
        throw new Error("sequence sync document mismatch");
      }
      for (const replay of this.#outbox.replayFrom(message.expectedSeq)) {
        this.#platform.send(replay);
      }
      return;
    }
    if (message.type === "reconcile") {
      this.#reconcile(message);
      return;
    }
    const previous = this.#commands.get(message.commandId);
    if (previous !== undefined) {
      // A retry retransmits state only; executing the handler again could
      // generate the same prompt twice after a reconnect race.
      this.#resendOrReport(message.commandId, previous);
      return;
    }

    this.#setAndReport(message.commandId, "accepted");
    this.#setAndReport(message.commandId, "started");

    if (message.command === "warmReload") {
      // `send` queues accepted and started before navigation can close the socket.
      this.#platform.reload();
      return;
    }
    if (message.command === "coldAppReload") {
      this.#setAndReport(message.commandId, "started", "external_fallback_required");
      return;
    }

    const handler = this.#handlers[message.command];
    try {
      if (handler === undefined) {
        throw new Error("handler_unavailable");
      }
      const value = await handler(message.payload);
      this.#setAndReport(
        message.commandId,
        "completed",
        undefined,
        message.command === "getState" ? sanitizeStateResult(value) : undefined,
      );
    } catch (error) {
      this.#setAndReport(
        message.commandId,
        message.command === "cancelPrompt" ? "cancelled" : "failed",
        error instanceof Error ? "handler_failed" : "unknown_failure",
      );
    }
  }

  reportReady(): void {
    this.#emit({
      schemaVersion: CONTROL_SCHEMA_VERSION,
      type: "ready",
      ...this.#identity,
      eventSeq: this.#eventSeq + 1,
    });
  }

  reportTelemetry(event: Record<string, unknown>): void {
    this.#emit({
      schemaVersion: CONTROL_SCHEMA_VERSION,
      type: "telemetry",
      ...this.#identity,
      eventSeq: this.#eventSeq + 1,
      event,
    });
  }

  #setAndReport(
    commandId: string,
    state: AgentCommandRecord["state"],
    reason?: string,
    result?: SanitizedStateResult,
  ): void {
    this.#commands.set(commandId, {
      state,
      ...(reason === undefined ? {} : { reason }),
      ...(result === undefined ? {} : { result }),
    });
    this.#report(commandId, state, reason, result);
  }

  #report(
    commandId: string,
    state: AgentCommandRecord["state"],
    reason?: string,
    result?: SanitizedStateResult,
  ): void {
    this.#emit({
      schemaVersion: CONTROL_SCHEMA_VERSION,
      type: "commandState",
      ...this.#identity,
      eventSeq: this.#eventSeq + 1,
      commandId,
      state,
      ...(reason === undefined ? {} : { reason }),
      ...(result === undefined ? {} : { result }),
    });
  }

  #emit(message: Parameters<PhoneEventOutbox["enqueue"]>[0]): void {
    this.#outbox.enqueue(message);
    this.#eventSeq = message.eventSeq;
    this.#platform.send(message);
  }

  #resendOrReport(commandId: string, record: AgentCommandRecord): void {
    const queued = this.#outbox.latestCommandState(commandId, record.state);
    if (queued !== undefined) {
      this.#platform.send(queued);
      return;
    }
    this.#report(commandId, record.state, record.reason, record.result);
  }

  #reconcile(message: ReconcileMessage): void {
    for (const serverCommand of message.commands) {
      const local = this.#commands.get(serverCommand.commandId);
      if (local !== undefined) {
        if (local.state !== serverCommand.state) this.#resendOrReport(serverCommand.commandId, local);
        continue;
      }
      if (serverCommand.state !== "issued" && serverCommand.state !== "accepted") {
        this.#commands.set(serverCommand.commandId, {
          state: serverCommand.state,
          ...(serverCommand.reason === undefined ? {} : { reason: serverCommand.reason }),
          ...(serverCommand.result === undefined ? {} : { result: serverCommand.result }),
        });
      }
    }
  }
}
