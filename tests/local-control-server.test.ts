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

test("credentials must be high entropy and are not accepted as short secrets", () => {
  assert.throws(() => assertHighEntropyCredential("password", "credential"), /entropy/i);
  assert.doesNotThrow(() => assertHighEntropyCredential(randomBytes(32).toString("base64url"), "credential"));
});

test("phone authentication accepts only the exact token via WebSocket subprotocol", () => {
  const token = randomBytes(32).toString("base64url");
  assert.deepEqual(authenticatePhoneProtocols(`qwen-control.v1, ${token}`, token), {
    accepted: true,
    protocol: "qwen-control.v1",
  });
  assert.deepEqual(authenticatePhoneProtocols("qwen-control.v1, wrong", token), {
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
