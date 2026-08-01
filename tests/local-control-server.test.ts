import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import { chmod, mkdtemp, mkdir, writeFile } from "node:fs/promises";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import test from "node:test";

import {
  assertHighEntropyCredential,
  consumePhoneTicketProtocols,
  createOperatorRequestHandler,
  createOperatorServer,
  isControlUpgradePath,
  isSameOriginRequest,
  listenOperatorServer,
  loadLocalTlsMaterial,
} from "../dev/control/server.js";
import { ControlPlane } from "../dev/control/control-plane.js";
import { SessionTicketAuthority } from "../dev/control/session-ticket-authority.js";
import { CONTROL_COMMANDS } from "../dev/control/protocol.js";

test("credentials must be high entropy and are not accepted as short secrets", () => {
  assert.throws(() => assertHighEntropyCredential("password", "credential"), /entropy/i);
  assert.doesNotThrow(() => assertHighEntropyCredential(randomBytes(32).toString("base64url"), "credential"));
});

test("ticket and WSS origin guard accepts only the exact local HTTPS origin", () => {
  const request = (headers: Record<string, string>): IncomingMessage => ({ headers } as IncomingMessage);
  assert.equal(
    isSameOriginRequest(request({ host: "development.invalid:8443", origin: "https://development.invalid:8443" }), "https://development.invalid:8443"),
    true,
  );
  assert.equal(
    isSameOriginRequest(request({ host: "development.invalid:8443", origin: "https://elsewhere.invalid" }), "https://development.invalid:8443"),
    false,
  );
  assert.equal(
    isSameOriginRequest(request({ host: "development.invalid", origin: "https://development.invalid:8443" }), "https://development.invalid:8443"),
    false,
  );
  for (const host of [
    "development.invalid:8443/path",
    "development.invalid:8443?query",
    "development.invalid:8443#fragment",
    "user@development.invalid:8443",
  ]) {
    assert.equal(
      isSameOriginRequest(request({ host, origin: "https://development.invalid:8443" }), "https://development.invalid:8443"),
      false,
      `reject malformed Host: ${host}`,
    );
  }
  for (const origin of [
    "https://development.invalid:8443/path",
    "https://development.invalid:8443?query",
    "https://development.invalid:8443#fragment",
    "https://user@development.invalid:8443",
  ]) {
    assert.equal(
      isSameOriginRequest(request({ host: "development.invalid:8443", origin }), "https://development.invalid:8443"),
      false,
      `reject non-origin Origin header: ${origin}`,
    );
  }
  assert.equal(
    isSameOriginRequest(request({ host: "DEVELOPMENT.INVALID:8443", origin: "https://development.invalid:8443" }), "https://development.invalid:8443"),
    true,
  );
  assert.equal(
    isSameOriginRequest(request({ host: "[::1]:8443", origin: "https://[::1]:8443" }), "https://[::1]:8443"),
    true,
  );
  assert.equal(
    isSameOriginRequest(request({ host: "development.invalid:8443", origin: "https://development.invalid:8443" }), "https://user@development.invalid:8443"),
    false,
  );
});

test("only the control WebSocket path can reach ticket consumption", () => {
  assert.equal(isControlUpgradePath("/.local-control"), true);
  assert.equal(isControlUpgradePath("/.local-control?reconnect=1"), false);
  assert.equal(isControlUpgradePath("/.local-control#fragment"), false);
  assert.equal(isControlUpgradePath("https://user@development.invalid/.local-control"), false);
  assert.equal(isControlUpgradePath("/.local-control/../.local-control"), false);
  assert.equal(isControlUpgradePath("//development.invalid/.local-control"), false);
  assert.equal(isControlUpgradePath("/.local-control-wrong"), false);
  assert.equal(isControlUpgradePath("/.local-ticket"), false);
});

test("ticket-bound phone connection accepts only one exact ticket via WebSocket subprotocol", () => {
  const authority = new SessionTicketAuthority({
    now: () => 1_000,
    randomToken: () => randomBytes(32).toString("base64url"),
  });
  const identity = {
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    documentId: "document_0123456789abcdef",
  };
  const ticketAuthority = new SessionTicketAuthority({
    now: () => 1_000,
    randomToken: () => randomBytes(32).toString("base64url"),
  });
  const ticket = ticketAuthority.issueTicket(identity);
  assert.deepEqual(
    consumePhoneTicketProtocols(
      `qwen-control.v1, ${ticket.ticket}`,
      (candidate) => ticketAuthority.consumeTicket(candidate),
    ),
    {
    accepted: true,
    protocol: "qwen-control.v1",
      identity,
    },
  );
  assert.deepEqual(consumePhoneTicketProtocols("qwen-control.v1, wrong", (candidate) =>
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

test("operator command endpoint accepts only the explicit command allowlist", async () => {
  const token = randomBytes(32).toString("base64url");
  const issued: string[] = [];
  const handler = createOperatorRequestHandler({
    controlPlane: {
      issueCommand(request: { command: string }) {
        issued.push(request.command);
        return { command: request.command };
      },
    } as unknown as ControlPlane,
    operatorToken: token,
  });
  const invoke = async (command: string): Promise<number> => {
    const request = Readable.from([
      JSON.stringify({
        deviceId: "device_0123456789abcdef",
        tabId: "tab_0123456789abcdef",
        commandId: `command_${String(issued.length).padStart(16, "0")}`,
        command,
      }),
    ]);
    Object.assign(request, {
      headers: { authorization: `Bearer ${token}` },
      method: "POST",
      url: "/v1/commands",
    });
    let status = 0;
    await handler(
      request as unknown as IncomingMessage,
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
    return status;
  };

  for (const command of CONTROL_COMMANDS) assert.equal(await invoke(command), 202);
  for (const command of ["__proto__", "constructor", "unknownCommand"]) {
    assert.equal(await invoke(command), 400);
  }
  assert.deepEqual(issued, [...CONTROL_COMMANDS]);
});

test("operator state exposes bounded control and journal health without raw errors", async () => {
  const token = randomBytes(32).toString("base64url");
  const handler = createOperatorRequestHandler({
    controlPlane: {
      getConnectedState: () => [],
      getDisconnectEvidence: () => [],
      getControlMetrics: () => ({
        commands: 1,
        disconnectRecords: 0,
        sequenceDocuments: 1,
        activeTimers: 1,
        reloadProofs: 0,
      }),
    } as unknown as ControlPlane,
    operatorToken: token,
    journalHealth: () => ({ state: "degraded", writeFailures: 1, lastFailureAtMs: 1234 }),
  });
  let status = 0;
  let body = "";
  await handler(
    {
      headers: { authorization: `Bearer ${token}` },
      method: "GET",
      url: "/v1/state",
    } as IncomingMessage,
    {
      writeHead(code: number) {
        status = code;
        return this;
      },
      end(chunk?: string) {
        body += chunk ?? "";
        return this;
      },
    } as unknown as ServerResponse,
  );

  assert.equal(status, 200);
  assert.deepEqual(JSON.parse(body).metrics.journal, {
    state: "degraded",
    writeFailures: 1,
    lastFailureAtMs: 1234,
  });
  assert.doesNotMatch(body, /private|path|stack|errorMessage/i);
});
