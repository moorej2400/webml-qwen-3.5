import { createHash, randomBytes } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

import { loadControlEnvironment } from "./config.js";
import { ControlPlane } from "./control-plane.js";
import { SessionTicketAuthority } from "./session-ticket-authority.js";
import {
  loadBrowserRuntimeConfiguration,
  loadBrowserRuntimeEnvironment,
} from "./runtime-config.js";
import {
  assertQwen35PackageIdentity,
  type Qwen35BrowserLoadOptions,
  snapshotQwen35Manifest,
} from "../../src/qwen35-model-loader.js";
import { modelCacheKey } from "../../src/opfs-model-cache.js";
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
const publicOrigin = new URL(`https://${config.publicHost}:${config.publicPort}`).origin;

const localPackageDirectory = process.env.QWEN_RUNTIME_LOCAL_PACKAGE_DIR;
const localTokenizerDirectory = process.env.QWEN_RUNTIME_LOCAL_TOKENIZER_DIR;
const localVisionPackageDirectory = process.env.QWEN_RUNTIME_LOCAL_VISION_PACKAGE_DIR;
const localRuntimeConfiguration =
  localPackageDirectory === undefined && localTokenizerDirectory === undefined && localVisionPackageDirectory === undefined
    ? runtimeConfiguration
    : await (async () => {
        if (localPackageDirectory === undefined || localTokenizerDirectory === undefined) {
          throw new Error("Local package mode requires both model and tokenizer directories");
        }
        const packageManifest = JSON.parse(
          await readFile(path.join(localPackageDirectory, "manifest.json"), "utf8"),
        ) as Qwen35BrowserLoadOptions["manifest"];
        const manifest = snapshotQwen35Manifest({
          ...packageManifest,
          shards: packageManifest.shards.map((shard) => ({
            ...shard,
            url: new URL(`/assets/model/${shard.url}`, publicOrigin).href,
          })),
        });
        assertQwen35PackageIdentity(manifest);
        const visionPackagePins = localVisionPackageDirectory === undefined
          ? undefined
          : await (async () => {
              const visionManifestBytes = await readFile(path.join(localVisionPackageDirectory, "manifest.json"));
              const visionLayerIndexBytes = await readFile(path.join(localVisionPackageDirectory, "layer-index.json"));
              return Object.freeze({
                packageBaseUrl: new URL("/assets/vision/", publicOrigin).href,
                expectedPackageBaseUrl: new URL("/assets/vision/", publicOrigin).href,
                expectedManifestSha256: createHash("sha256").update(visionManifestBytes).digest("hex"),
                expectedLayerIndexSha256: createHash("sha256").update(visionLayerIndexBytes).digest("hex"),
                allowInsecureLocalhost: true,
                manifestFile: "manifest.json",
                layerIndexFile: "layer-index.json",
              });
            })();
        return Object.freeze({
          ...runtimeConfiguration,
          manifest,
          expectedManifestSha256: modelCacheKey(manifest),
          compiledTokenizerUrl: new URL("/assets/tokenizer/tokenizer.bin", publicOrigin).href,
          ...(visionPackagePins === undefined ? {} : { visionPackagePins }),
        });
      })();
const publicHtml = (await readFile(path.join(projectRoot, "public-dist", "index.html"), "utf8"))
  .replaceAll('href="./app.css"', 'href="/assets/public/app.css"')
  .replaceAll('src="./chat-app.js"', 'src="/assets/public/chat-app.js"');
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
const ticketAuthority = new SessionTicketAuthority({
  randomToken: () => randomBytes(32).toString("base64url"),
});
const appServer = createDevelopmentServer({
  tls,
  controlPlane,
  ticketAuthority,
  publicOrigin,
  runtimeConfiguration: localRuntimeConfiguration,
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
  staticAssetRoots: [
    {
      routePrefix: "/assets/public/",
      directory: path.join(projectRoot, "public-dist"),
      extensions: [".js", ".css"],
    },
    ...(localPackageDirectory === undefined
      ? []
      : [{
          routePrefix: "/assets/model/",
          directory: localPackageDirectory,
          extensions: [".bin", ".json"],
        }]),
    ...(localTokenizerDirectory === undefined
      ? []
      : [{
          routePrefix: "/assets/tokenizer/",
          directory: localTokenizerDirectory,
          extensions: [".bin"],
        }]),
    ...(localVisionPackageDirectory === undefined
      ? []
      : [{
          routePrefix: "/assets/vision/",
          directory: localVisionPackageDirectory,
          extensions: [".bin", ".json"],
        }]),
  ],
  html: publicHtml,
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
