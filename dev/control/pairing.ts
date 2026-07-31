import { timingSafeEqual } from "node:crypto";

import { validateProtocolId, type PhoneIdentity } from "./protocol.js";
import { assertHighEntropyCredential } from "./security.js";

interface SessionRecord {
  deviceId: string;
  expiresAtMs: number;
}

interface TicketRecord {
  identity: PhoneIdentity;
  expiresAtMs: number;
}

export interface PairingAuthorityOptions {
  pairingCode: string;
  now?: () => number;
  randomToken?: () => string;
  pairingExpiresMs?: number;
  sessionExpiresMs?: number;
  ticketExpiresMs?: number;
}

const equalSecret = (left: string, right: string): boolean => {
  const leftBytes = Buffer.from(left);
  const rightBytes = Buffer.from(right);
  return leftBytes.length === rightBytes.length && timingSafeEqual(leftBytes, rightBytes);
};

const validateIdentity = (identity: PhoneIdentity): PhoneIdentity => ({
  deviceId: validateProtocolId(identity.deviceId, "deviceId"),
  tabId: validateProtocolId(identity.tabId, "tabId"),
  documentId: validateProtocolId(identity.documentId, "documentId"),
});

/**
 * Converts one manually entered pairing code into a device session, then
 * converts that session into single-use WSS tickets. The reusable session is
 * never present in the public agent source or WebSocket handshake.
 */
export class PairingAuthority {
  readonly #pairingCode: string;
  readonly #now: () => number;
  readonly #randomToken: () => string;
  readonly #pairingExpiresAtMs: number;
  readonly #sessionExpiresMs: number;
  readonly #ticketExpiresMs: number;
  readonly #sessions = new Map<string, SessionRecord>();
  readonly #tickets = new Map<string, TicketRecord>();
  #pairingConsumed = false;

  constructor(options: PairingAuthorityOptions) {
    this.#pairingCode = assertHighEntropyCredential(options.pairingCode, "pairing code");
    this.#now = options.now ?? Date.now;
    this.#randomToken =
      options.randomToken ??
      (() => {
        throw new Error("PairingAuthority requires a cryptographic token generator");
      });
    this.#pairingExpiresAtMs = this.#now() + (options.pairingExpiresMs ?? 5 * 60_000);
    this.#sessionExpiresMs = options.sessionExpiresMs ?? 8 * 60 * 60_000;
    this.#ticketExpiresMs = options.ticketExpiresMs ?? 30_000;
  }

  pair(
    candidateCode: string,
    identityInput: PhoneIdentity,
  ): { sessionCapability: string; expiresAtMs: number } {
    if (this.#pairingConsumed) throw new Error("pairing code was already consumed");
    if (this.#now() > this.#pairingExpiresAtMs) throw new Error("pairing code expired");
    if (!equalSecret(candidateCode, this.#pairingCode)) throw new Error("invalid pairing code");
    const identity = validateIdentity(identityInput);
    const sessionCapability = this.#newUniqueToken(this.#sessions);
    const expiresAtMs = this.#now() + this.#sessionExpiresMs;
    this.#sessions.set(sessionCapability, { deviceId: identity.deviceId, expiresAtMs });
    this.#pairingConsumed = true;
    return { sessionCapability, expiresAtMs };
  }

  issueTicket(
    sessionCapability: string,
    identityInput: PhoneIdentity,
  ): { ticket: string; expiresAtMs: number } {
    const session = this.#sessions.get(sessionCapability);
    if (session === undefined) throw new Error("invalid session capability");
    if (this.#now() > session.expiresAtMs) {
      this.#sessions.delete(sessionCapability);
      throw new Error("session capability expired");
    }
    const identity = validateIdentity(identityInput);
    if (identity.deviceId !== session.deviceId) throw new Error("session device mismatch");
    const ticket = this.#newUniqueToken(this.#tickets);
    const expiresAtMs = this.#now() + this.#ticketExpiresMs;
    this.#tickets.set(ticket, { identity, expiresAtMs });
    return { ticket, expiresAtMs };
  }

  consumeTicket(ticket: string): PhoneIdentity {
    const record = this.#tickets.get(ticket);
    if (record === undefined) throw new Error("ticket was consumed, replayed, or invalid");
    // Delete before validation so every outcome consumes the bearer.
    this.#tickets.delete(ticket);
    if (this.#now() > record.expiresAtMs) throw new Error("ticket expired");
    return structuredClone(record.identity);
  }

  activeSessionCount(): number {
    return [...this.#sessions.values()].filter((session) => session.expiresAtMs >= this.#now()).length;
  }

  #newUniqueToken<T>(records: Map<string, T>): string {
    for (let attempt = 0; attempt < 8; attempt += 1) {
      const token = assertHighEntropyCredential(this.#randomToken(), "generated capability");
      if (!records.has(token)) return token;
    }
    throw new Error("cryptographic token generator repeated values");
  }
}
