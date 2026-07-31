import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { loadControlEnvironment, RECOVERY_ORDER } from "../dev/control/config.js";
import {
  loadBrowserRuntimeConfiguration,
  loadBrowserRuntimeEnvironment,
} from "../dev/control/runtime-config.js";
import type { ModelPackageManifest } from "../src/manifest.js";
import { modelCacheKey } from "../src/opfs-model-cache.js";

const validEnvironment = {
  QWEN_CONTROL_TLS_CERT: ".local/tls/cert.pem",
  QWEN_CONTROL_TLS_KEY: ".local/tls/key.pem",
  QWEN_CONTROL_PAIRING_CODE: randomBytes(32).toString("base64url"),
  QWEN_CONTROL_OPERATOR_TOKEN: randomBytes(32).toString("base64url"),
  QWEN_CONTROL_PUBLIC_HOST: "development-host.invalid",
};

test("control environment fails closed without TLS, auth, or an explicit public host", () => {
  assert.throws(() => loadControlEnvironment({}), /TLS certificate/i);
  assert.throws(
    () => loadControlEnvironment({ ...validEnvironment, QWEN_CONTROL_PAIRING_CODE: undefined }),
    /pairing code/i,
  );
  assert.throws(
    () => loadControlEnvironment({ ...validEnvironment, QWEN_CONTROL_PUBLIC_HOST: undefined }),
    /public host/i,
  );
});

test("control environment keeps TLS paths under .local and tokens distinct", () => {
  assert.throws(
    () =>
      loadControlEnvironment({
        ...validEnvironment,
        QWEN_CONTROL_TLS_CERT: "cert.pem",
      }),
    /\.local/,
  );
  assert.throws(
    () =>
      loadControlEnvironment({
        ...validEnvironment,
        QWEN_CONTROL_OPERATOR_TOKEN: validEnvironment.QWEN_CONTROL_PAIRING_CODE,
      }),
    /distinct/i,
  );

  const config = loadControlEnvironment(validEnvironment);
  assert.equal(config.certPath, ".local/tls/cert.pem");
  assert.equal(config.publicHost, "development-host.invalid");
});

test("recovery policy exhausts protocol recovery before external fallback", () => {
  assert.deepEqual(RECOVERY_ORDER, [
    "reconnect",
    "reconcile",
    "dispose",
    "reload",
    "retry",
    "external_fallback",
  ]);
});

const runtimeManifest = (): ModelPackageManifest => ({
  format: "webml-qwen-package",
  version: 1,
  packageKind: "language",
  runtime: { abi: "qwen35-webgpu-v1" },
  source: {
    repository: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF",
    revision: "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
    file: "Qwen_Qwen3.5-4B-Q3_K_L.gguf",
    size: "2665441248",
    sha256: "41c3f1bf47e477693dab332e73347c7138d5e9fbfe74c6d2eaba590be1f3d20a",
  },
  tokenizer: {
    repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
    revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    file: "tokenizer.json",
    size: "12807982",
    sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
  },
  tensorLayout: [
    {
      name: "tensor.weight",
      shape: ["256"],
      ggmlType: 12,
      storageType: "q4-k-144",
      shard: 0,
      shardOffset: "0",
      tensorOffset: "0",
      length: "144",
      quantization: { blockElements: 256, blockBytes: 144 },
    },
  ],
  shards: [
    {
      url: "shards/part-0.bin",
      offset: "0",
      length: "144",
      sha256: "a".repeat(64),
    },
  ],
  excludedTensors: [],
});

test("browser runtime configuration requires local manifest bytes and immutable public pins", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "qwen-runtime-config-"));
  await mkdir(path.join(root, ".local", "model"), { recursive: true });
  const manifest = runtimeManifest();
  await writeFile(
    path.join(root, ".local", "model", "manifest.json"),
    JSON.stringify(manifest),
  );
  const revision = "c".repeat(40);
  const environment = {
    QWEN_RUNTIME_MANIFEST: ".local/model/manifest.json",
    QWEN_RUNTIME_PACKAGE_BASE_URL:
      `https://huggingface.co/example/browser-package/resolve/${revision}/`,
    QWEN_RUNTIME_MANIFEST_SHA256: modelCacheKey(manifest),
    QWEN_RUNTIME_TOKENIZER_URL:
      `https://huggingface.co/example/browser-package/resolve/${revision}/tokenizer.bin`,
  };

  const parsed = loadBrowserRuntimeEnvironment(environment);
  const config = await loadBrowserRuntimeConfiguration({ projectRoot: root, environment: parsed });
  assert.equal(config.packageBaseUrl, environment.QWEN_RUNTIME_PACKAGE_BASE_URL);
  assert.equal(config.expectedPackageBaseUrl, environment.QWEN_RUNTIME_PACKAGE_BASE_URL);
  assert.equal(config.expectedManifestSha256, modelCacheKey(manifest));
  assert.equal(config.compiledTokenizerUrl, environment.QWEN_RUNTIME_TOKENIZER_URL);
  assert.equal(config.manifest.runtime.abi, "qwen35-webgpu-v1");
  assert.equal(Object.isFrozen(config), true);
  assert.equal("manifestPath" in config, false);
});

test("browser runtime configuration rejects mutable, credentialed, or mismatched inputs", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "qwen-runtime-config-reject-"));
  await mkdir(path.join(root, ".local"), { recursive: true });
  const manifest = runtimeManifest();
  await writeFile(path.join(root, ".local", "manifest.json"), JSON.stringify(manifest));
  const base = {
    QWEN_RUNTIME_MANIFEST: ".local/manifest.json",
    QWEN_RUNTIME_PACKAGE_BASE_URL:
      `https://huggingface.co/example/browser-package/resolve/${"c".repeat(40)}/`,
    QWEN_RUNTIME_MANIFEST_SHA256: modelCacheKey(manifest),
    QWEN_RUNTIME_TOKENIZER_URL:
      `https://huggingface.co/example/browser-package/resolve/${"d".repeat(40)}/tokenizer.bin`,
  };

  assert.throws(
    () => loadBrowserRuntimeEnvironment({ ...base, QWEN_RUNTIME_MANIFEST: "manifest.json" }),
    /\.local/,
  );
  assert.throws(
    () => loadBrowserRuntimeEnvironment({
      ...base,
      QWEN_RUNTIME_PACKAGE_BASE_URL: "https://huggingface.co/example/browser-package/resolve/main/",
    }),
    /immutable/i,
  );
  assert.throws(
    () => loadBrowserRuntimeEnvironment({
      ...base,
      QWEN_RUNTIME_TOKENIZER_URL:
        `https://user:secret@huggingface.co/example/browser-package/resolve/${"d".repeat(40)}/tokenizer.bin`,
    }),
    /credential-free/i,
  );
  const environment = loadBrowserRuntimeEnvironment({
    ...base,
    QWEN_RUNTIME_MANIFEST_SHA256: "e".repeat(64),
  });
  await assert.rejects(
    loadBrowserRuntimeConfiguration({ projectRoot: root, environment }),
    /manifest/i,
  );
});
