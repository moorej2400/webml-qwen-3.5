import { timingSafeEqual } from "node:crypto";
import { lstat, readFile, realpath } from "node:fs/promises";
import http, { type IncomingMessage, type ServerResponse } from "node:http";
import https from "node:https";
import path from "node:path";

import { WebSocketServer, type WebSocket } from "ws";

import { createBrowserAgentSource } from "./browser-agent-source.js";
import { ControlPlane, type IssueCommandRequest } from "./control-plane.js";
import { CONTROL_SCHEMA_VERSION, validateProtocolId, type PhoneIdentity } from "./protocol.js";
import { listRunSummaries } from "./run-journal.js";
import { assertHighEntropyCredential } from "./security.js";

export { assertHighEntropyCredential } from "./security.js";

const MAX_OPERATOR_BODY_BYTES = 65_536;

const constantTimeEqual = (left: string, right: string): boolean => {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  return leftBuffer.length === rightBuffer.length && timingSafeEqual(leftBuffer, rightBuffer);
};

export const authenticatePhoneProtocols = (
  header: string | string[] | undefined,
  expectedToken: string,
): { accepted: true; protocol: "qwen-control.v1" } | { accepted: false } => {
  assertHighEntropyCredential(expectedToken, "phone credential");
  const value = Array.isArray(header) ? header.join(",") : header ?? "";
  const protocols = value.split(",").map((entry) => entry.trim());
  return protocols[0] === "qwen-control.v1" &&
    protocols[1] !== undefined &&
    constantTimeEqual(protocols[1], expectedToken)
    ? { accepted: true, protocol: "qwen-control.v1" }
    : { accepted: false };
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

export interface OperatorServerOptions {
  controlPlane: ControlPlane;
  operatorToken: string;
  runsDirectory?: string;
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
      if (request.method === "POST" && url.pathname === "/v1/commands") {
        const body = await readJsonBody(request);
        const issued = options.controlPlane.issueCommand(body as unknown as IssueCommandRequest);
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
  phoneToken: string;
  html: string;
}

export const createDevelopmentServer = (options: DevelopmentServerOptions): https.Server => {
  const phoneToken = assertHighEntropyCredential(options.phoneToken, "phone credential");
  const agentSource = createBrowserAgentSource({ phoneToken });
  const server = https.createServer(options.tls, (request, response) => {
    const url = new URL(request.url ?? "/", "https://development.invalid");
    if (url.pathname === "/.local-agent.js") {
      response.writeHead(200, {
        "content-type": "text/javascript; charset=utf-8",
        "cache-control": "no-store",
      });
      response.end(agentSource);
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
  });
  const websocketServer = new WebSocketServer({
    noServer: true,
    handleProtocols(protocols) {
      return protocols.has("qwen-control.v1") ? "qwen-control.v1" : false;
    },
    maxPayload: 64 * 1024,
  });

  server.on("upgrade", (request, socket, head) => {
    const url = new URL(request.url ?? "/", "https://development.invalid");
    const auth = authenticatePhoneProtocols(
      request.headers["sec-websocket-protocol"],
      phoneToken,
    );
    if (url.pathname !== "/.local-control" || !auth.accepted) {
      socket.write("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
      socket.destroy();
      return;
    }
    websocketServer.handleUpgrade(request, socket, head, (websocket) => {
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
          connectionId = options.controlPlane.connect(hello, (message) => {
            if (websocket.readyState === websocket.OPEN) websocket.send(JSON.stringify(message));
          });
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
  return {
    deviceId: validateProtocolId(value.deviceId, "deviceId"),
    tabId: validateProtocolId(value.tabId, "tabId"),
    documentId: validateProtocolId(value.documentId, "documentId"),
  };
};
