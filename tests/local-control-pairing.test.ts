import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import { mkdtemp, mkdir, symlink, writeFile } from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import test from "node:test";

import { createBrowserAgentSource } from "../dev/control/browser-agent-source.js";
import { SessionTicketAuthority } from "../dev/control/session-ticket-authority.js";
import { createDevelopmentRequestHandler } from "../dev/control/server.js";

const identity = {
  deviceId: "device_0123456789abcdef",
  tabId: "tab_0123456789abcdef",
  documentId: "document_0123456789abcdef",
};
const publicOrigin = "https://development.invalid:8443";

const createAuthority = (): SessionTicketAuthority =>
  new SessionTicketAuthority({
    now: () => 1_000,
    randomToken: () => randomBytes(32).toString("base64url"),
  });

test("local browser agent connects automatically without a reusable credential or a dialog", () => {
  const source = createBrowserAgentSource();

  assert.match(source, /requestJson\("\/.local-ticket", identity\(\)\)/);
  assert.doesNotMatch(source, /pairing|one-time-code|sessionCapability|Bearer /i);
  assert.match(source, /credentials: "omit"/);
  assert.doesNotMatch(source, /dialog|input\.type|password/i);
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
  assert.match(source, /Object\.hasOwn\(handlers, message\.command\)/);
  const candidate = source.indexOf("const eventSeq = sequence + 1");
  const retained = source.indexOf("outbox.set(eventSeq, frame)");
  const committed = source.indexOf("sequence = eventSeq");
  const transmitted = source.indexOf("sendFrame(frame)", committed);
  assert.ok(candidate >= 0 && candidate < retained && retained < committed && committed < transmitted);
});

test("same-origin ticket is short-lived, document-bound, and single use", () => {
  let nowMs = 1_000;
  const authority = new SessionTicketAuthority({
    now: () => nowMs,
    randomToken: () => randomBytes(32).toString("base64url"),
    ticketExpiresMs: 5_000,
  });
  const ticket = authority.issueTicket(identity);

  assert.deepEqual(authority.consumeTicket(ticket.ticket), identity);
  assert.throws(() => authority.consumeTicket(ticket.ticket), /replay|consumed/i);

  const expired = authority.issueTicket(identity);
  nowMs += 5_001;
  assert.throws(() => authority.consumeTicket(expired.ticket), /expired/i);
});

test("ticket authority prunes expired reconnect tickets and fails safely at its pending cap", () => {
  let nowMs = 1_000;
  let tokenNumber = 0;
  const authority = new SessionTicketAuthority({
    now: () => nowMs,
    randomToken: () => `ticket_${String(++tokenNumber).padStart(40, "0")}`,
    ticketExpiresMs: 5_000,
    maxPendingTickets: 2,
  });

  authority.issueTicket(identity);
  authority.issueTicket(identity);
  assert.throws(() => authority.issueTicket(identity), /capacity/i);

  // Failed reconnect storms cannot retain expired records: the next issue prunes them.
  nowMs += 5_001;
  assert.doesNotThrow(() => authority.issueTicket(identity));
  assert.doesNotThrow(() => authority.issueTicket(identity));
  assert.throws(() => authority.issueTicket(identity), /capacity/i);
});

test("ticket authority bounds repeated ticket requests even when each ticket is consumed", () => {
  let nowMs = 1_000;
  const authority = new SessionTicketAuthority({
    now: () => nowMs,
    randomToken: () => randomBytes(32).toString("base64url"),
    maxPendingTickets: 4,
    maxIssuesPerWindow: 2,
    issueWindowMs: 5_000,
  });

  for (let index = 0; index < 2; index += 1) {
    const ticket = authority.issueTicket(identity);
    authority.consumeTicket(ticket.ticket);
  }
  assert.throws(() => authority.issueTicket(identity), /rate/i);
  nowMs += 5_001;
  assert.doesNotThrow(() => authority.issueTicket(identity));
});

test("ticket endpoint admits only an exact same-origin request and exposes no reusable secret", async () => {
  const handler = createDevelopmentRequestHandler({
    ticketAuthority: createAuthority(),
    publicOrigin,
    html: "<main></main>",
  });
  const invoke = async (
    method: string,
    url: string,
    headers: Record<string, string>,
    body?: Record<string, unknown>,
  ): Promise<{ status: number; body: string }> => {
    const request = Readable.from(body === undefined ? [] : [JSON.stringify(body)]);
    Object.assign(request, { method, url, headers });
    let status = 0;
    let responseBody = "";
    await handler(
      request as unknown as IncomingMessage,
      {
        writeHead(code: number) { status = code; return this; },
        end(chunk?: string) { responseBody += chunk ?? ""; return this; },
      } as unknown as ServerResponse,
    );
    return { status, body: responseBody };
  };

  const script = await invoke("GET", "/.local-agent.js", { host: "development.invalid:8443" });
  assert.equal(script.status, 200);
  assert.doesNotMatch(script.body, /pairing|sessionCapability|Bearer /i);

  const ticket = await invoke(
    "POST",
    "/.local-ticket",
    { host: "development.invalid:8443", origin: publicOrigin },
    identity,
  );
  assert.equal(ticket.status, 200);
  assert.match(ticket.body, /"ticket":"[A-Za-z0-9_-]{43,}"/);

  const crossOrigin = await invoke(
    "POST",
    "/.local-ticket",
    { host: "development.invalid:8443", origin: "https://elsewhere.invalid" },
    identity,
  );
  assert.equal(crossOrigin.status, 403);
  assert.doesNotMatch(crossOrigin.body, /ticket|secret|credential/i);
  assert.equal(
    (await invoke("POST", "/.local-pair", { host: "development.invalid:8443", origin: publicOrigin }, identity)).status,
    404,
  );
});

test("development server serves only real JavaScript files under explicit module roots", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "qwen-static-root-"));
  const outside = await mkdtemp(path.join(tmpdir(), "qwen-static-outside-"));
  await mkdir(path.join(root, "src"), { recursive: true });
  await writeFile(path.join(root, "src", "browser.js"), "export const safe = true;\n");
  await writeFile(path.join(root, "src", "private.txt"), "not served\n");
  await writeFile(path.join(outside, "outside.js"), "export const escaped = true;\n");
  await symlink(path.join(outside, "outside.js"), path.join(root, "src", "linked.js"));
  const handler = createDevelopmentRequestHandler({
    ticketAuthority: createAuthority(),
    publicOrigin,
    html: "<main></main>",
    staticModuleRoots: [{ routePrefix: "/assets/src/", directory: path.join(root, "src") }],
  });
  const invoke = async (url: string, requestHeaders: Record<string, string> = {}): Promise<{ status: number; body: string; headers: Record<string, string> }> => {
    let status = 0;
    let responseBody = "";
    let headers: Record<string, string> = {};
    await handler(
      { method: "GET", url, headers: requestHeaders } as IncomingMessage,
      {
        writeHead(code: number, values?: Record<string, string>) { status = code; headers = values ?? {}; return this; },
        end(chunk?: string | Buffer) { responseBody += chunk?.toString() ?? ""; return this; },
      } as unknown as ServerResponse,
    );
    return { status, body: responseBody, headers };
  };

  const served = await invoke("/assets/src/browser.js");
  assert.equal(served.status, 200);
  assert.equal(served.body, "export const safe = true;\n");
  assert.equal(served.headers["content-type"], "text/javascript; charset=utf-8");
  assert.equal(served.headers["cache-control"], "no-store");
  assert.equal((await invoke("/assets/src/private.txt")).status, 404);
  assert.equal((await invoke("/assets/src/linked.js")).status, 404);
  assert.equal((await invoke("/assets/src/%2e%2e%2foutside.js")).status, 404);
});

test("development server serves reviewed public JavaScript and CSS assets only", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "qwen-public-root-"));
  await writeFile(path.join(root, "chat-app.js"), "export const app = true;\n");
  await writeFile(path.join(root, "app.css"), "body { color: red; }\n");
  await writeFile(path.join(root, "model.bin"), Buffer.from("0123456789", "ascii"));
  await writeFile(path.join(root, "private.json"), "{\"no\":true}\n");
  const handler = createDevelopmentRequestHandler({
    ticketAuthority: createAuthority(),
    publicOrigin,
    html: "<main></main>",
    staticAssetRoots: [{
      routePrefix: "/assets/public/",
      directory: root,
      extensions: [".js", ".css", ".bin"],
    }],
  });
  const invoke = async (url: string, requestHeaders: Record<string, string> = {}): Promise<{ status: number; body: string; headers: Record<string, string> }> => {
    let status = 0;
    let body = "";
    let headers: Record<string, string> = {};
    await handler(
      { method: "GET", url, headers: requestHeaders } as IncomingMessage,
      {
        writeHead(code: number, values?: Record<string, string>) { status = code; headers = values ?? {}; return this; },
        end(chunk?: string | Buffer) { body += chunk?.toString() ?? ""; return this; },
      } as unknown as ServerResponse,
    );
    return { status, body, headers };
  };

  const script = await invoke("/assets/public/chat-app.js");
  assert.equal(script.status, 200);
  assert.equal(script.body, "export const app = true;\n");
  assert.equal(script.headers["content-type"], "text/javascript; charset=utf-8");
  const stylesheet = await invoke("/assets/public/app.css");
  assert.equal(stylesheet.status, 200);
  assert.equal(stylesheet.headers["content-type"], "text/css; charset=utf-8");
  const range = await invoke("/assets/public/model.bin", { range: "bytes=2-5" });
  assert.equal(range.status, 206);
  assert.equal(range.body, "2345");
  assert.equal(range.headers["content-range"], "bytes 2-5/10");
  assert.equal((await invoke("/assets/public/private.json")).status, 404);
  assert.equal((await invoke("/assets/public/%2e%2e%2foutside.js")).status, 404);
});

test("development page exposes no-store credential-free runtime configuration", async () => {
  const runtimeConfiguration = {
    manifest: { format: "fixture" },
    packageBaseUrl: `https://huggingface.co/example/package/resolve/${"a".repeat(40)}/`,
    expectedPackageBaseUrl: `https://huggingface.co/example/package/resolve/${"a".repeat(40)}/`,
    expectedManifestSha256: "b".repeat(64),
    compiledTokenizerUrl: `https://huggingface.co/example/package/resolve/${"a".repeat(40)}/tokenizer.bin`,
  };
  const handler = createDevelopmentRequestHandler({
    ticketAuthority: createAuthority(),
    publicOrigin,
    html: "<main></main>",
    runtimeConfiguration,
  });
  let status = 0;
  let body = "";
  let headers: Record<string, string> = {};
  await handler(
    { method: "GET", url: "/.local-runtime-config.json", headers: {} } as IncomingMessage,
    {
      writeHead(code: number, values?: Record<string, string>) { status = code; headers = values ?? {}; return this; },
      end(chunk?: string) { body += chunk ?? ""; return this; },
    } as unknown as ServerResponse,
  );

  assert.equal(status, 200);
  assert.equal(headers["cache-control"], "no-store");
  assert.deepEqual(JSON.parse(body), runtimeConfiguration);
  assert.doesNotMatch(body, /operator|credential|manifestPath/i);

  let page = "";
  await handler(
    { method: "GET", url: "/", headers: {} } as IncomingMessage,
    {
      writeHead() { return this; },
      end(chunk?: string) { page += chunk ?? ""; return this; },
    } as unknown as ServerResponse,
  );
  assert.match(page, /__QWEN35_RUNTIME_CONFIG__/);
  assert.match(page, /\.local-agent\.js/);
});
