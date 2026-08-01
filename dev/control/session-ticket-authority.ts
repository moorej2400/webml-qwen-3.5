import { validateProtocolId, type PhoneIdentity } from "./protocol.js";
import { assertHighEntropyCredential } from "./security.js";

interface TicketRecord {
  identity: PhoneIdentity;
  expiresAtMs: number;
}

export interface SessionTicketAuthorityOptions {
  now?: () => number;
  randomToken?: () => string;
  ticketExpiresMs?: number;
  maxPendingTickets?: number;
  maxIssuesPerWindow?: number;
  issueWindowMs?: number;
}

const validateIdentity = (identity: PhoneIdentity): PhoneIdentity => ({
  deviceId: validateProtocolId(identity.deviceId, "deviceId"),
  tabId: validateProtocolId(identity.tabId, "tabId"),
  documentId: validateProtocolId(identity.documentId, "documentId"),
});

/**
 * Issues a one-use WSS bearer only after the HTTPS request has passed the
 * same-origin boundary; no device credential survives in browser storage.
 */
export class SessionTicketAuthority {
  readonly #now: () => number;
  readonly #randomToken: () => string;
  readonly #ticketExpiresMs: number;
  readonly #maxPendingTickets: number;
  readonly #maxIssuesPerWindow: number;
  readonly #issueWindowMs: number;
  readonly #tickets = new Map<string, TicketRecord>();
  readonly #issuedAtMs: number[] = [];

  constructor(options: SessionTicketAuthorityOptions = {}) {
    this.#now = options.now ?? Date.now;
    this.#randomToken =
      options.randomToken ??
      (() => {
        throw new Error("SessionTicketAuthority requires a cryptographic token generator");
      });
    this.#ticketExpiresMs = options.ticketExpiresMs ?? 30_000;
    this.#maxPendingTickets = options.maxPendingTickets ?? 512;
    this.#maxIssuesPerWindow = options.maxIssuesPerWindow ?? 120;
    this.#issueWindowMs = options.issueWindowMs ?? 60_000;
    if (!Number.isSafeInteger(this.#ticketExpiresMs) || this.#ticketExpiresMs < 1) {
      throw new RangeError("ticket lifetime must be positive");
    }
    if (!Number.isSafeInteger(this.#maxPendingTickets) || this.#maxPendingTickets < 1) {
      throw new RangeError("pending ticket cap must be positive");
    }
    if (!Number.isSafeInteger(this.#maxIssuesPerWindow) || this.#maxIssuesPerWindow < 1) {
      throw new RangeError("ticket issue rate cap must be positive");
    }
    if (!Number.isSafeInteger(this.#issueWindowMs) || this.#issueWindowMs < 1) {
      throw new RangeError("ticket issue window must be positive");
    }
  }

  issueTicket(identityInput: PhoneIdentity): { ticket: string; expiresAtMs: number } {
    const identity = validateIdentity(identityInput);
    const now = this.#now();
    this.#prune(now);
    if (this.#tickets.size >= this.#maxPendingTickets) {
      throw new Error("pending ticket capacity exceeded");
    }
    if (this.#issuedAtMs.length >= this.#maxIssuesPerWindow) {
      throw new Error("ticket issuance rate exceeded");
    }
    const ticket = this.#newUniqueToken();
    const expiresAtMs = now + this.#ticketExpiresMs;
    this.#tickets.set(ticket, { identity, expiresAtMs });
    this.#issuedAtMs.push(now);
    return { ticket, expiresAtMs };
  }

  consumeTicket(ticket: string): PhoneIdentity {
    const record = this.#tickets.get(ticket);
    if (record === undefined) throw new Error("ticket was consumed, replayed, or invalid");
    // Consume before expiry validation so a rejected bearer is never reusable.
    this.#tickets.delete(ticket);
    if (this.#now() > record.expiresAtMs) throw new Error("ticket expired");
    return structuredClone(record.identity);
  }

  #newUniqueToken(): string {
    for (let attempt = 0; attempt < 8; attempt += 1) {
      const token = assertHighEntropyCredential(this.#randomToken(), "generated ticket");
      if (!this.#tickets.has(token)) return token;
    }
    throw new Error("cryptographic token generator repeated values");
  }

  /** Bounds unused ticket and issue-time records during reconnect abuse. */
  #prune(now: number): void {
    for (const [ticket, record] of this.#tickets) {
      if (now > record.expiresAtMs) this.#tickets.delete(ticket);
    }
    let firstRecent = 0;
    while (firstRecent < this.#issuedAtMs.length) {
      const issuedAt = this.#issuedAtMs[firstRecent];
      if (issuedAt === undefined || now - issuedAt <= this.#issueWindowMs) break;
      firstRecent += 1;
    }
    if (firstRecent > 0) this.#issuedAtMs.splice(0, firstRecent);
  }
}
