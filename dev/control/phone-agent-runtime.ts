import {
  CONTROL_SCHEMA_VERSION,
  type CommandMessage,
  type CommandState,
  type ControlCommand,
  type PhoneIdentity,
} from "./protocol.js";
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
  #eventSeq = 0;

  constructor(options: PhoneAgentRuntimeOptions) {
    this.#identity = options.identity;
    this.#platform = options.platform;
    this.#handlers = options.handlers;
  }

  async receive(message: CommandMessage): Promise<void> {
    const previous = this.#commands.get(message.commandId);
    if (previous !== undefined) {
      // A retry retransmits state only; executing the handler again could
      // generate the same prompt twice after a reconnect race.
      this.#report(message.commandId, previous.state, previous.reason, previous.result);
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
    this.#platform.send({
      schemaVersion: CONTROL_SCHEMA_VERSION,
      type: "ready",
      ...this.#identity,
      eventSeq: ++this.#eventSeq,
    });
  }

  reportTelemetry(event: Record<string, unknown>): void {
    this.#platform.send({
      schemaVersion: CONTROL_SCHEMA_VERSION,
      type: "telemetry",
      ...this.#identity,
      eventSeq: ++this.#eventSeq,
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
    this.#platform.send({
      schemaVersion: CONTROL_SCHEMA_VERSION,
      type: "commandState",
      ...this.#identity,
      eventSeq: ++this.#eventSeq,
      commandId,
      state,
      ...(reason === undefined ? {} : { reason }),
      ...(result === undefined ? {} : { result }),
    });
  }
}
