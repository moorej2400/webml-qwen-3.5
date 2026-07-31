import { timingSafeEqual } from "node:crypto";
import { lstat, readFile, realpath } from "node:fs/promises";
import http, { type IncomingMessage, type ServerResponse } from "node:http";
import https from "node:https";
import path from "node:path";

import { WebSocketServer, type WebSocket } from "ws";

import { createBrowserAgentSource } from "./browser-agent-source.js";
import {
  ControlPlane,
  type IssueCommandRequest,
  type ServerToPhoneMessage,
} from "./control-plane.js";
import { PairingAuthority } from "./pairing.js";
import {
  CONTROL_COMMANDS,
  CONTROL_SCHEMA_VERSION,
  validateProtocolId,
  type ControlCommand,
  type PhoneIdentity,
} from "./protocol.js";
import { listRunSummaries, type RunJournalHealth } from "./run-journal.js";
import { assertHighEntropyCredential } from "./security.js";

export { assertHighEntropyCredential } from "./security.js";

const MAX_OPERATOR_BODY_BYTES = 65_536;
const CONTROL_COMMAND_SET = new Set<string>(CONTROL_COMMANDS);

const constantTimeEqual = (left: string, right: string): boolean => {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  return leftBuffer.length === rightBuffer.length && timingSafeEqual(leftBuffer, rightBuffer);
};

export const authenticatePhoneProtocols = (
  header: string | string[] | undefined,
  consumeTicket: (ticket: string) => PhoneIdentity,
):
  | { accepted: true; protocol: "qwen-control.v1"; identity: PhoneIdentity }
  | { accepted: false } => {
  const value = Array.isArray(header) ? header.join(",") : header ?? "";
  const protocols = value.split(",").map((entry) => entry.trim());
  if (protocols[0] !== "qwen-control.v1" || protocols[1] === undefined) {
    return { accepted: false };
  }
  try {
    return {
      accepted: true,
      protocol: "qwen-control.v1",
      identity: consumeTicket(protocols[1]),
    };
  } catch {
    return { accepted: false };
  }
};

export const connectAuthenticatedPhone = (
  controlPlane: ControlPlane,
  authenticatedIdentity: PhoneIdentity,
  hello: PhoneIdentity,
  send: (message: ServerToPhoneMessage) => boolean,
): string => {
  if (
    authenticatedIdentity.deviceId !== hello.deviceId ||
    authenticatedIdentity.tabId !== hello.tabId ||
    authenticatedIdentity.documentId !== hello.documentId
  ) {
    throw new Error("hello identity does not match WSS ticket");
  }
  return controlPlane.connect(hello, send);
};

const resolveLocalFile = async (projectRoot: string, relativePath: string): Promise<string> => {
  if (path.isAbsolute(relativePath)) throw new Error("TLS paths must be under .local");
  const localRoot = path.resolve(projectRoot, ".local");
  const requested = path.resolve(projectRoot, relativePath);
  if (!requested.startsWith(`${localRoot}${path.sep}`)) {
    throw new Error("TLS paths must be under .local");
  }
  const info = await lstat(requested);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("TLS material must be a regular file");
  const [realLocalRoot, realRequested] = await Promise.all([realpath(localRoot), realpath(requested)]);
  if (!realRequested.startsWith(`${realLocalRoot}${path.sep}`)) {
    throw new Error("TLS material must resolve under .local");
  }
  return realRequested;
};

export const loadLocalTlsMaterial = async (options: {
  projectRoot: string;
  certPath: string;
  keyPath: string;
}): Promise<{ cert: Buffer; key: Buffer }> => {
  const [certPath, keyPath] = await Promise.all([
    resolveLocalFile(options.projectRoot, options.certPath),
    resolveLocalFile(options.projectRoot, options.keyPath),
  ]);
  const [cert, key] = await Promise.all([readFile(certPath), readFile(keyPath)]);
  if (cert.length === 0 || key.length === 0) throw new Error("TLS material cannot be empty");
  return { cert, key };
};

const isAuthorizedOperator = (request: IncomingMessage, token: string): boolean => {
  const authorization = request.headers.authorization;
  if (typeof authorization !== "string" || !authorization.startsWith("Bearer ")) return false;
  return constantTimeEqual(authorization.slice("Bearer ".length), token);
};

const sendJson = (response: ServerResponse, status: number, body: unknown): void => {
  const encoded = JSON.stringify(body);
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(encoded),
    "cache-control": "no-store",
  });
  response.end(encoded);
};

const readJsonBody = async (request: IncomingMessage): Promise<Record<string, unknown>> => {
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    bytes += buffer.length;
    if (bytes > MAX_OPERATOR_BODY_BYTES) throw new Error("operator request body is too large");
    chunks.push(buffer);
  }
  const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new TypeError("operator request must be a JSON object");
  }
  return parsed as Record<string, unknown>;
};

const parseIssueCommandRequest = (body: Record<string, unknown>): IssueCommandRequest => {
  if (
    !Object.hasOwn(body, "command") ||
    typeof body.command !== "string" ||
    !CONTROL_COMMAND_SET.has(body.command)
  ) {
    throw new Error("operator command is not allowed");
  }
  return { ...body, command: body.command as ControlCommand } as unknown as IssueCommandRequest;
};

export interface OperatorServerOptions {
  controlPlane: ControlPlane;
  operatorToken: string;
  runsDirectory?: string;
  journalHealth?: () => Readonly<RunJournalHealth>;
}

export const createOperatorRequestHandler = (
  options: OperatorServerOptions,
): ((request: IncomingMessage, response: ServerResponse) => Promise<void>) => {
  const token = assertHighEntropyCredential(options.operatorToken, "operator credential");
  return async (request, response) => {
    if (!isAuthorizedOperator(request, token)) {
      sendJson(response, 401, { error: "unauthorized" });
      return;
    }
    const url = new URL(request.url ?? "/", "http://operator.invalid");
    try {
      if (request.method === "GET" && url.pathname === "/v1/state") {
        sendJson(response, 200, {
          devices: options.controlPlane.getConnectedState(),
          disconnectEvidence: options.controlPlane.getDisconnectEvidence(),
          metrics: {
            control: options.controlPlane.getControlMetrics(),
            journal: options.journalHealth?.() ?? { state: "healthy", writeFailures: 0 },
          },
        });
        return;
      }
      if (request.method === "GET" && url.pathname === "/v1/runs") {
        sendJson(
          response,
          200,
          options.runsDirectory === undefined
            ? []
            : await listRunSummaries(options.runsDirectory),
        );
        return;
      }
      if (request.method === "GET" && url.pathname.startsWith("/v1/benchmarks/")) {
        const benchmarkId = validateProtocolId(
          decodeURIComponent(url.pathname.slice("/v1/benchmarks/".length)),
          "benchmarkId",
        );
        sendJson(response, 200, options.controlPlane.getBenchmark(benchmarkId));
        return;
      }
      if (request.method === "GET" && url.pathname.startsWith("/v1/commands/")) {
        const commandId = validateProtocolId(
          decodeURIComponent(url.pathname.slice("/v1/commands/".length)),
          "commandId",
        );
        const command = options.controlPlane.getCommand(commandId);
        sendJson(
          response,
          command === undefined ? 404 : 200,
          command ?? { error: "command_not_found" },
        );
        return;
      }
      if (request.method === "POST" && url.pathname === "/v1/commands") {
        const body = await readJsonBody(request);
        const issued = options.controlPlane.issueCommand(parseIssueCommandRequest(body));
        sendJson(response, 202, issued);
        return;
      }
      sendJson(response, 404, { error: "not_found" });
    } catch {
      // Operator responses never echo parse errors because they can contain input data.
      sendJson(response, 400, { error: "invalid_request" });
    }
  };
};

export const createOperatorServer = (options: OperatorServerOptions): http.Server => {
  const handler = createOperatorRequestHandler(options);
  return http.createServer((request, response) => {
    void handler(request, response);
  });
};

export const listenOperatorServer = async (
  server: http.Server,
  port: number,
): Promise<{ host: "127.0.0.1"; port: number }> => {
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (address === null || typeof address === "string" || address.address !== "127.0.0.1") {
    server.close();
    throw new Error("operator server failed closed because it is not bound to loopback");
  }
  return { host: "127.0.0.1", port: address.port };
};

export interface DevelopmentServerOptions {
  tls: { cert: Buffer; key: Buffer };
  controlPlane: ControlPlane;
  pairingAuthority: PairingAuthority;
  html: string;
}

export const createDevelopmentRequestHandler = (
  options: Pick<DevelopmentServerOptions, "pairingAuthority" | "html">,
): ((request: IncomingMessage, response: ServerResponse) => Promise<void>) => {
  const agentSource = createBrowserAgentSource();
  return async (request, response) => {
    const url = new URL(request.url ?? "/", "https://development.invalid");
    if (url.pathname === "/.local-agent.js") {
      response.writeHead(200, {
        "content-type": "text/javascript; charset=utf-8",
        "cache-control": "no-store",
      });
      response.end(agentSource);
      return;
    }
    if (request.method === "POST" && url.pathname === "/.local-pair") {
      try {
        const body = await readJsonBody(request);
        const identity = parsePhoneIdentity(body);
        const paired = options.pairingAuthority.pair(
          typeof body.pairingCode === "string" ? body.pairingCode : "",
          identity,
        );
        sendJson(response, 200, paired);
      } catch {
        sendJson(response, 401, { error: "pairing_failed" });
      }
      return;
    }
    if (request.method === "POST" && url.pathname === "/.local-ticket") {
      try {
        const authorization = request.headers.authorization;
        if (typeof authorization !== "string" || !authorization.startsWith("Bearer ")) {
          throw new Error("missing session capability");
        }
        const body = await readJsonBody(request);
        const ticket = options.pairingAuthority.issueTicket(
          authorization.slice("Bearer ".length),
          parsePhoneIdentity(body),
        );
        sendJson(response, 200, ticket);
      } catch {
        sendJson(response, 401, { error: "ticket_failed" });
      }
      return;
    }
    if (url.pathname === "/" || url.pathname === "/index.html") {
      const injection = '<script src="/.local-agent.js"></script>';
      const html = options.html.includes("</body>")
        ? options.html.replace("</body>", `${injection}</body>`)
        : `${options.html}${injection}`;
      response.writeHead(200, {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store",
      });
      response.end(html);
      return;
    }
    response.writeHead(404);
    response.end();
  };
};

export const createDevelopmentServer = (options: DevelopmentServerOptions): https.Server => {
  const requestHandler = createDevelopmentRequestHandler(options);
  const server = https.createServer(options.tls, (request, response) => {
    void requestHandler(request, response);
  });
  const websocketServer = new WebSocketServer({
    noServer: true,
    handleProtocols(protocols) {
      return protocols.has("qwen-control.v1") ? "qwen-control.v1" : false;
    },
    maxPayload: 64 * 1024,
  });
  const authenticatedIdentities = new WeakMap<WebSocket, PhoneIdentity>();

  server.on("upgrade", (request, socket, head) => {
    const url = new URL(request.url ?? "/", "https://development.invalid");
    const auth = authenticatePhoneProtocols(
      request.headers["sec-websocket-protocol"],
      (ticket) => options.pairingAuthority.consumeTicket(ticket),
    );
    if (url.pathname !== "/.local-control" || !auth.accepted) {
      socket.write("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
      socket.destroy();
      return;
    }
    websocketServer.handleUpgrade(request, socket, head, (websocket) => {
      authenticatedIdentities.set(websocket, auth.identity);
      websocketServer.emit("connection", websocket, request);
    });
  });

  websocketServer.on("connection", (websocket: WebSocket) => {
    let connectionId: string | undefined;
    websocket.on("message", (data, isBinary) => {
      const byteLength = Array.isArray(data)
        ? data.reduce((total, chunk) => total + chunk.byteLength, 0)
        : data.byteLength;
      if (isBinary || byteLength > 64 * 1024) {
        websocket.close(1008, "invalid_message");
        return;
      }
      try {
        const parsed: unknown = JSON.parse(data.toString());
        if (connectionId === undefined) {
          const hello = parseHello(parsed);
          const authenticatedIdentity = authenticatedIdentities.get(websocket);
          if (authenticatedIdentity === undefined) throw new Error("missing WSS ticket identity");
          connectionId = connectAuthenticatedPhone(
            options.controlPlane,
            authenticatedIdentity,
            hello,
            (message) => {
              if (websocket.readyState !== websocket.OPEN) return false;
              websocket.send(JSON.stringify(message));
              return true;
            },
          );
          return;
        }
        options.controlPlane.receive(connectionId, parsed);
      } catch {
        websocket.close(1008, "invalid_message");
      }
    });
    websocket.on("close", () => {
      if (connectionId !== undefined) {
        options.controlPlane.disconnect(connectionId);
      }
    });
  });

  server.on("close", () => websocketServer.close());
  return server;
};

const parseHello = (input: unknown): PhoneIdentity => {
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    throw new TypeError("hello must be an object");
  }
  const value = input as Record<string, unknown>;
  if (value.schemaVersion !== CONTROL_SCHEMA_VERSION || value.type !== "hello") {
    throw new TypeError("first phone message must be a v1 hello");
  }
  return parsePhoneIdentity(value);
};

const parsePhoneIdentity = (value: Record<string, unknown>): PhoneIdentity => ({
  deviceId: validateProtocolId(value.deviceId, "deviceId"),
  tabId: validateProtocolId(value.tabId, "tabId"),
  documentId: validateProtocolId(value.documentId, "documentId"),
});
