import { randomBytes } from "node:crypto";
import path from "node:path";

import { loadControlEnvironment } from "./config.js";
import { ControlPlane } from "./control-plane.js";
import { PairingAuthority } from "./pairing.js";
import {
  loadBrowserRuntimeConfiguration,
  loadBrowserRuntimeEnvironment,
} from "./runtime-config.js";
import { RunJournal } from "./run-journal.js";
import {
  createDevelopmentServer,
  createOperatorServer,
  listenOperatorServer,
  loadLocalTlsMaterial,
} from "./server.js";

const projectRoot = process.cwd();
const config = loadControlEnvironment(process.env);
const runtimeConfiguration = await loadBrowserRuntimeConfiguration({
  projectRoot,
  environment: loadBrowserRuntimeEnvironment(process.env),
});
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
  runtimeConfiguration,
  staticModuleRoots: [
    {
      routePrefix: "/assets/dev/browser/",
      directory: path.join(projectRoot, "dev-dist", "dev", "browser"),
    },
    {
      routePrefix: "/assets/src/",
      directory: path.join(projectRoot, "dev-dist", "src"),
    },
  ],
  html: `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen WebGPU text runtime</title>
  <style>
    :root { color-scheme: light dark; font: 16px system-ui, sans-serif; }
    body { margin: 0 auto; max-width: 48rem; padding: 1rem; }
    main { display: grid; gap: 0.75rem; }
    textarea, input, button { box-sizing: border-box; font: inherit; padding: 0.65rem; }
    textarea { min-height: 8rem; width: 100%; }
    .controls { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    output, pre { border: 1px solid currentColor; min-height: 2rem; padding: 0.75rem; white-space: pre-wrap; }
  </style>
  <script type="module" src="/assets/dev/browser/app.js"></script>
</head>
<body>
  <main>
    <h1>Qwen WebGPU text runtime</h1>
    <label for="runtime-prompt">Prompt</label>
    <textarea id="runtime-prompt">Write one short sentence about WebGPU.</textarea>
    <label for="runtime-max-tokens">Maximum new tokens</label>
    <input id="runtime-max-tokens" type="number" min="1" max="16384" value="128">
    <div class="controls">
      <button id="runtime-load" type="button">Load</button>
      <button id="runtime-run" type="button">Run prompt</button>
      <button id="runtime-cancel" type="button">Cancel</button>
      <button id="runtime-dispose" type="button">Dispose</button>
      <button id="runtime-state" type="button">Get state</button>
    </div>
    <output id="runtime-status" aria-live="polite"></output>
    <pre id="runtime-output" aria-live="polite"></pre>
  </main>
</body>
</html>`,
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
