import { randomBytes } from "node:crypto";
import path from "node:path";

import { loadControlEnvironment } from "./config.js";
import { ControlPlane } from "./control-plane.js";
import { PairingAuthority } from "./pairing.js";
import { RunJournal } from "./run-journal.js";
import {
  createDevelopmentServer,
  createOperatorServer,
  listenOperatorServer,
  loadLocalTlsMaterial,
} from "./server.js";

const projectRoot = process.cwd();
const config = loadControlEnvironment(process.env);
const tls = await loadLocalTlsMaterial({
  projectRoot,
  certPath: config.certPath,
  keyPath: config.keyPath,
});
const runId = `run_${randomBytes(16).toString("hex")}`;
const journal = new RunJournal({
  runsDirectory: path.join(projectRoot, ".local", "runs"),
  runId,
});
const controlPlane = new ControlPlane({
  onTelemetry: (event) => journal.append(event),
});
const pairingAuthority = new PairingAuthority({
  pairingCode: config.pairingCode,
  randomToken: () => randomBytes(32).toString("base64url"),
});
const appServer = createDevelopmentServer({
  tls,
  controlPlane,
  pairingAuthority,
  html: "<!doctype html><meta charset=utf-8><title>Qwen WebGPU development</title><main id=app></main>",
});
const operatorServer = createOperatorServer({
  controlPlane,
  operatorToken: config.operatorToken,
  runsDirectory: path.join(projectRoot, ".local", "runs"),
  journalHealth: () => journal.getHealth(),
});

await Promise.all([
  new Promise<void>((resolve, reject) => {
    appServer.once("error", reject);
    appServer.listen(config.publicPort, config.publicHost, resolve);
  }),
  listenOperatorServer(operatorServer, config.operatorPort),
]);

// Do not print addresses or credentials; shell history and captured logs are not trusted storage.
process.stdout.write("Local control servers are ready.\n");

const shutdown = async (): Promise<void> => {
  await Promise.all([
    new Promise<void>((resolve) => appServer.close(() => resolve())),
    new Promise<void>((resolve) => operatorServer.close(() => resolve())),
    journal.close(),
  ]);
};

process.once("SIGINT", () => {
  void shutdown().finally(() => process.exit(0));
});
process.once("SIGTERM", () => {
  void shutdown().finally(() => process.exit(0));
});
