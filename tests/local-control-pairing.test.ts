import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";
import { Readable } from "node:stream";
import test from "node:test";

import { createBrowserAgentSource } from "../dev/control/browser-agent-source.js";
import { PairingAuthority } from "../dev/control/pairing.js";
import { createDevelopmentRequestHandler } from "../dev/control/server.js";

const identity = {
  deviceId: "device_0123456789abcdef",
  tabId: "tab_0123456789abcdef",
  documentId: "document_0123456789abcdef",
};

test("public browser agent contains no reusable or one-time credential", () => {
  const pairingCode = randomBytes(32).toString("base64url");
  const source = createBrowserAgentSource();

  assert.doesNotMatch(source, new RegExp(pairingCode));
  assert.doesNotMatch(source, /PHONE_TOKEN|PAIRING_CODE|const TOKEN/);
  assert.match(source, /__QWEN_LOCAL_PAIR__/);
  assert.match(source, /qwen-control-pairing-form/);
  assert.match(source, /autocomplete = "one-time-code"/);
});

test("injected browser agent retains and replays bounded unacknowledged events", () => {
  const source = createBrowserAgentSource();

  assert.match(source, /const OUTBOX_LIMIT = 256/);
  assert.match(source, /outbox\.set\(eventSeq, frame\)/);
  assert.match(source, /message\.type === "eventAck"/);
  assert.match(source, /message\.type === "sequenceSync"/);
  assert.match(source, /synchronize\(message\)/);
  assert.match(source, /replayFrom\(message\.expectedSeq\)/);
  assert.match(source, /resendState\(command\.commandId, local\)/);
  const candidate = source.indexOf("const eventSeq = sequence + 1");
  const retained = source.indexOf("outbox.set(eventSeq, frame)");
  const committed = source.indexOf("sequence = eventSeq");
  const transmitted = source.indexOf("sendFrame(frame)", committed);
  assert.ok(
    candidate >= 0 &&
      candidate < retained &&
      retained < committed &&
      committed < transmitted,
  );
});

test("pairing code is one-time and creates a short-lived session capability", () => {
  let nowMs = 1_000;
  const pairingCode = randomBytes(32).toString("base64url");
  const authority = new PairingAuthority({
    pairingCode,
    now: () => nowMs,
    randomToken: () => randomBytes(32).toString("base64url"),
    pairingExpiresMs: 5_000,
    sessionExpiresMs: 10_000,
  });

  const session = authority.pair(pairingCode, identity);
  assert.ok(session.sessionCapability.length >= 43);
  assert.throws(() => authority.pair(pairingCode, identity), /consumed/i);
  nowMs += 10_001;
  assert.throws(
    () => authority.issueTicket(session.sessionCapability, identity),
    /expired/i,
  );
});

test("WSS ticket is bound to identity and cannot be replayed", () => {
  const pairingCode = randomBytes(32).toString("base64url");
  const authority = new PairingAuthority({
    pairingCode,
    now: () => 1_000,
    randomToken: () => randomBytes(32).toString("base64url"),
  });
  const session = authority.pair(pairingCode, identity);
  const ticket = authority.issueTicket(session.sessionCapability, identity);

  assert.deepEqual(authority.consumeTicket(ticket.ticket), identity);
  assert.throws(() => authority.consumeTicket(ticket.ticket), /replay|consumed/i);
});

test("invalid LAN pairing attempt receives no session capability", () => {
  const authority = new PairingAuthority({
    pairingCode: randomBytes(32).toString("base64url"),
    now: () => 1_000,
    randomToken: () => randomBytes(32).toString("base64url"),
  });

  assert.throws(() => authority.pair(randomBytes(32).toString("base64url"), identity), /invalid/i);
  assert.equal(authority.activeSessionCount(), 0);
});

test("unauthenticated development responses never disclose reusable authentication", async () => {
  const pairingCode = randomBytes(32).toString("base64url");
  const authority = new PairingAuthority({
    pairingCode,
    now: () => 1_000,
    randomToken: () => randomBytes(32).toString("base64url"),
  });
  const handler = createDevelopmentRequestHandler({
    pairingAuthority: authority,
    html: "<main></main>",
  });
  const invoke = async (
    method: string,
    url: string,
    body?: Record<string, unknown>,
  ): Promise<{ status: number; body: string }> => {
    const request = Readable.from(body === undefined ? [] : [JSON.stringify(body)]);
    Object.assign(request, { method, url, headers: {} });
    let status = 0;
    let responseBody = "";
    const response = {
      writeHead(code: number) {
        status = code;
        return this;
      },
      end(chunk?: string) {
        responseBody += chunk ?? "";
        return this;
      },
    };
    await handler(
      request as unknown as IncomingMessage,
      response as unknown as ServerResponse,
    );
    return { status, body: responseBody };
  };

  const script = await invoke("GET", "/.local-agent.js");
  assert.equal(script.status, 200);
  assert.doesNotMatch(script.body, new RegExp(pairingCode));
  assert.doesNotMatch(script.body, /sessionCapability\s*[:=]\s*["'][A-Za-z0-9_-]{20}/);

  const rejected = await invoke("POST", "/.local-pair", {
    pairingCode: randomBytes(32).toString("base64url"),
    ...identity,
  });
  assert.equal(rejected.status, 401);
  assert.doesNotMatch(rejected.body, /sessionCapability/);
});
