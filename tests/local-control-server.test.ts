import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import { chmod, mkdtemp, mkdir, writeFile } from "node:fs/promises";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  assertHighEntropyCredential,
  authenticatePhoneProtocols,
  createOperatorRequestHandler,
  createOperatorServer,
  listenOperatorServer,
  loadLocalTlsMaterial,
} from "../dev/control/server.js";
import { ControlPlane } from "../dev/control/control-plane.js";
import { PairingAuthority } from "../dev/control/pairing.js";

test("credentials must be high entropy and are not accepted as short secrets", () => {
  assert.throws(() => assertHighEntropyCredential("password", "credential"), /entropy/i);
  assert.doesNotThrow(() => assertHighEntropyCredential(randomBytes(32).toString("base64url"), "credential"));
});

test("phone authentication accepts only the exact token via WebSocket subprotocol", () => {
  const authority = new PairingAuthority({
    pairingCode: randomBytes(32).toString("base64url"),
    now: () => 1_000,
    randomToken: () => randomBytes(32).toString("base64url"),
  });
  const identity = {
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    documentId: "document_0123456789abcdef",
  };
  const pairingCode = randomBytes(32).toString("base64url");
  const ticketAuthority = new PairingAuthority({
    pairingCode,
    now: () => 1_000,
    randomToken: () => randomBytes(32).toString("base64url"),
  });
  const session = ticketAuthority.pair(pairingCode, identity);
  const ticket = ticketAuthority.issueTicket(session.sessionCapability, identity);
  assert.deepEqual(
    authenticatePhoneProtocols(
      `qwen-control.v1, ${ticket.ticket}`,
      (candidate) => ticketAuthority.consumeTicket(candidate),
    ),
    {
    accepted: true,
    protocol: "qwen-control.v1",
      identity,
    },
  );
  assert.deepEqual(authenticatePhoneProtocols("qwen-control.v1, wrong", (candidate) =>
    authority.consumeTicket(candidate),
  ), {
    accepted: false,
  });
});

test("TLS material must resolve under an ignored .local directory and fails closed", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "qwen-control-tls-"));
  await mkdir(path.join(root, ".local", "tls"), { recursive: true });
  await writeFile(path.join(root, ".local", "tls", "cert.pem"), "cert", { mode: 0o600 });
  await writeFile(path.join(root, ".local", "tls", "key.pem"), "key", { mode: 0o600 });
  await chmod(path.join(root, ".local", "tls", "cert.pem"), 0o600);
  await chmod(path.join(root, ".local", "tls", "key.pem"), 0o600);

  const material = await loadLocalTlsMaterial({
    projectRoot: root,
    certPath: ".local/tls/cert.pem",
    keyPath: ".local/tls/key.pem",
  });
  assert.equal(material.cert.toString(), "cert");
  await assert.rejects(
    loadLocalTlsMaterial({
      projectRoot: root,
      certPath: "../cert.pem",
      keyPath: ".local/tls/key.pem",
    }),
    /\.local/,
  );
});

test("operator server binds only to loopback and rejects missing authentication", async () => {
  const token = randomBytes(32).toString("base64url");
  const options = {
    controlPlane: new ControlPlane(),
    operatorToken: token,
  };
  assert.ok(createOperatorServer(options));

  let status = 0;
  const handler = createOperatorRequestHandler(options);
  await handler(
    {
      headers: {},
      method: "GET",
      url: "/v1/state",
    } as IncomingMessage,
    {
      writeHead(code: number) {
        status = code;
        return this;
      },
      end() {
        return this;
      },
    } as unknown as ServerResponse,
  );
  assert.equal(status, 401);

  let boundHost: string | undefined;
  const fakeServer = {
    once() {
      return this;
    },
    listen(_port: number, host: string, callback: () => void) {
      boundHost = host;
      callback();
      return this;
    },
    address() {
      return { address: "127.0.0.1", family: "IPv4", port: 9012 };
    },
    close() {
      return this;
    },
  };
  const address = await listenOperatorServer(fakeServer as unknown as Server, 0);
  assert.equal(boundHost, "127.0.0.1");
  assert.deepEqual(address, { host: "127.0.0.1", port: 9012 });
});

test("operator command query returns sanitized getState result", async () => {
  const token = randomBytes(32).toString("base64url");
  const plane = new ControlPlane();
  const phoneIdentity = {
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    documentId: "document_0123456789abcdef",
  };
  const connectionId = plane.connect(phoneIdentity, () => true);
  plane.issueCommand({
    deviceId: phoneIdentity.deviceId,
    tabId: phoneIdentity.tabId,
    commandId: "command_0123456789abcdef",
    command: "getState",
  });
  plane.receive(connectionId, {
    schemaVersion: 1,
    type: "commandState",
    ...phoneIdentity,
    eventSeq: 1,
    commandId: "command_0123456789abcdef",
    state: "accepted",
  });
  plane.receive(connectionId, {
    schemaVersion: 1,
    type: "commandState",
    ...phoneIdentity,
    eventSeq: 2,
    commandId: "command_0123456789abcdef",
    state: "started",
  });
  plane.receive(connectionId, {
    schemaVersion: 1,
    type: "commandState",
    ...phoneIdentity,
    eventSeq: 3,
    commandId: "command_0123456789abcdef",
    state: "completed",
    result: { modelState: "loaded", prompt: "private" },
  });

  const handler = createOperatorRequestHandler({ controlPlane: plane, operatorToken: token });
  let status = 0;
  let responseBody = "";
  await handler(
    {
      headers: { authorization: `Bearer ${token}` },
      method: "GET",
      url: "/v1/commands/command_0123456789abcdef",
    } as IncomingMessage,
    {
      writeHead(code: number) {
        status = code;
        return this;
      },
      end(chunk?: string) {
        responseBody += chunk ?? "";
        return this;
      },
    } as unknown as ServerResponse,
  );

  assert.equal(status, 200);
  const snapshot = JSON.parse(responseBody) as { result: Record<string, unknown> };
  assert.deepEqual(snapshot.result, { modelState: "loaded" });
});
