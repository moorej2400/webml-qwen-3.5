import {
  CONTROL_SCHEMA_VERSION,
  validateProtocolId,
  type EventAckMessage,
  type PhoneToServerMessage,
} from "./protocol.js";

export interface PhoneEventOutboxOptions {
  documentId: string;
  maxEntries?: number;
}

/**
 * Retains unacknowledged phone events so a lost WSS frame cannot permanently
 * advance the client beyond the server's expected sequence.
 */
export class PhoneEventOutbox {
  readonly #documentId: string;
  readonly #maxEntries: number;
  readonly #events = new Map<number, PhoneToServerMessage>();
  #lastEnqueued = 0;
  #highestAcknowledged = 0;

  constructor(options: PhoneEventOutboxOptions) {
    this.#documentId = validateProtocolId(options.documentId, "documentId");
    this.#maxEntries = options.maxEntries ?? 256;
    if (!Number.isSafeInteger(this.#maxEntries) || this.#maxEntries < 1 || this.#maxEntries > 1_024) {
      throw new RangeError("outbox maxEntries must be between 1 and 1024");
    }
  }

  enqueue(message: PhoneToServerMessage): void {
    if (message.documentId !== this.#documentId) throw new Error("outbox document mismatch");
    if (message.eventSeq !== this.#lastEnqueued + 1) {
      throw new Error("outbox requires a contiguous monotonic event sequence");
    }
    if (this.#events.size >= this.#maxEntries) {
      throw new Error("outbox entry limit reached before server acknowledgement");
    }
    this.#events.set(message.eventSeq, structuredClone(message));
    this.#lastEnqueued = message.eventSeq;
  }

  acknowledge(message: EventAckMessage): void {
    if (message.schemaVersion !== CONTROL_SCHEMA_VERSION || message.type !== "eventAck") {
      throw new Error("invalid event acknowledgement");
    }
    if (message.documentId !== this.#documentId) throw new Error("ack document mismatch");
    if (message.status !== "accepted" && message.status !== "gap" && message.status !== "replay") {
      throw new Error("invalid event acknowledgement status");
    }
    if (
      !Number.isSafeInteger(message.acknowledgedSeq) ||
      !Number.isSafeInteger(message.expectedSeq) ||
      message.acknowledgedSeq < 0 ||
      message.expectedSeq !== message.acknowledgedSeq + 1
    ) {
      throw new Error("invalid event acknowledgement bounds");
    }
    if (message.status === "gap") return;
    if (message.acknowledgedSeq < this.#highestAcknowledged) {
      throw new Error("ack replay or regression rejected");
    }
    if (message.acknowledgedSeq > this.#lastEnqueued) {
      throw new Error("ack exceeds the emitted event sequence");
    }
    this.#highestAcknowledged = message.acknowledgedSeq;
    for (const sequence of this.#events.keys()) {
      if (sequence <= message.acknowledgedSeq) this.#events.delete(sequence);
    }
  }

  replayFrom(expectedSeq: number): PhoneToServerMessage[] {
    if (!Number.isSafeInteger(expectedSeq) || expectedSeq < 1) {
      throw new Error("invalid replay sequence");
    }
    if (expectedSeq <= this.#highestAcknowledged) {
      throw new Error("requested replay was already acknowledged");
    }
    if (this.#events.size === 0) {
      if (expectedSeq === this.#lastEnqueued + 1) return [];
      throw new Error("requested replay is unavailable in the bounded outbox");
    }
    if (!this.#events.has(expectedSeq)) {
      throw new Error("requested replay is unavailable in the bounded outbox");
    }
    return [...this.#events.entries()]
      .filter(([sequence]) => sequence >= expectedSeq)
      .sort(([left], [right]) => left - right)
      .map(([, message]) => structuredClone(message));
  }

  latestCommandState(commandId: string, state: string): PhoneToServerMessage | undefined {
    const matches = [...this.#events.values()].filter(
      (message) =>
        message.type === "commandState" &&
        message.commandId === commandId &&
        message.state === state,
    );
    return matches.length === 0 ? undefined : structuredClone(matches.at(-1));
  }

  get size(): number {
    return this.#events.size;
  }
}
