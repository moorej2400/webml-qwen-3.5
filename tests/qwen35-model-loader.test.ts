import assert from "node:assert/strict";
import test from "node:test";

import type { GpuAllocation } from "../src/gpu-arena.js";
import type { ModelPackageManifest } from "../src/manifest.js";
import { modelCacheKey } from "../src/opfs-model-cache.js";
import {
  assertQwen35PackageIdentity,
  assertQwen35ConvertedPackageTrust,
  buildQwen35PackageDirectory,
  buildQwen35TensorDirectory,
  cleanupQwen35GpuResources,
  createQwen35GpuLedger,
  loadQwen35BrowserResources,
  qwen35AllocatedWeightBytes,
  snapshotQwen35Manifest,
} from "../src/qwen35-model-loader.js";
import type { Qwen35ExecutionDriver } from "../src/qwen35-session.js";

const allocation: GpuAllocation = {
  shards: [],
  logicalBytes: 25n,
  allocatedBytes: 28n,
  destroy() {},
};

const driver: Qwen35ExecutionDriver = {
  async prefill() {},
  async *generate() {},
  async cancel() {},
  async reset() {},
  async dispose() {},
};

function pinnedManifest(): ModelPackageManifest {
  return {
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
        shape: ["256", "2"],
        ggmlType: 12,
        storageType: "q4-k-144",
        shard: 0,
        shardOffset: "0",
        tensorOffset: "0",
        length: "144",
        quantization: { blockElements: 256, blockBytes: 144 },
      },
      {
        name: "tensor.weight",
        shape: ["256", "2"],
        ggmlType: 12,
        storageType: "q4-k-144",
        shard: 1,
        shardOffset: "0",
        tensorOffset: "144",
        length: "144",
        quantization: { blockElements: 256, blockBytes: 144 },
      },
    ],
    shards: [
      { url: "shards/part-0.bin", offset: "0", length: "144", sha256: "a".repeat(64) },
      { url: "shards/part-1.bin", offset: "144", length: "144", sha256: "b".repeat(64) },
    ],
    excludedTensors: [],
  };
}

test("accounts for exact tensor-owned GPU bytes without package padding", () => {
  assert.equal(qwen35AllocatedWeightBytes(buildQwen35PackageDirectory(pinnedManifest())), 288n);
});

test("default ledger accounting allows scheduler scratch while explicit budgets enforce limits", () => {
  const defaultLedger = createQwen35GpuLedger(100n);
  defaultLedger.reserve({ id: "model", category: "model", bytes: 80n });
  defaultLedger.reserve({ id: "state", category: "activation", bytes: 20n });
  const scratch = defaultLedger.reserve({
    id: "scratch",
    category: "scratch",
    bytes: 1n,
  });
  assert.deepEqual(
    {
      currentBytes: defaultLedger.snapshot().currentBytes,
      peakBytes: defaultLedger.snapshot().peakBytes,
    },
    { currentBytes: 101n, peakBytes: 101n },
  );
  defaultLedger.release(scratch);
  assert.deepEqual(
    {
      currentBytes: defaultLedger.snapshot().currentBytes,
      peakBytes: defaultLedger.snapshot().peakBytes,
    },
    { currentBytes: 100n, peakBytes: 101n },
  );

  const budgeted = createQwen35GpuLedger(100n, 100n);
  budgeted.reserve({ id: "model", category: "model", bytes: 80n });
  budgeted.reserve({ id: "state", category: "activation", bytes: 20n });
  assert.throws(
    () => budgeted.reserve({ id: "scratch", category: "scratch", bytes: 1n }),
    /ledger limit/i,
  );
  assert.throws(
    () => createQwen35GpuLedger(100n, BigInt(Number.MAX_SAFE_INTEGER) + 1n),
    { code: "gpu-ledger-limit-unsafe" },
  );
});

test("builds one exact program tensor entry from segmented manifest storage", () => {
  const manifest = {
    tensorLayout: [
      {
        name: "tensor.weight",
        shape: ["256", "2"],
        ggmlType: 12,
        storageType: "q4-k-144",
        shard: 0,
        shardOffset: "0",
        tensorOffset: "0",
        length: "144",
        quantization: { blockElements: 256, blockBytes: 144 },
      },
      {
        name: "tensor.weight",
        shape: ["256", "2"],
        ggmlType: 12,
        storageType: "q4-k-144",
        shard: 1,
        shardOffset: "0",
        tensorOffset: "144",
        length: "144",
        quantization: { blockElements: 256, blockBytes: 144 },
      },
    ],
  } as ModelPackageManifest;

  assert.deepEqual(buildQwen35TensorDirectory(manifest), [
    {
      name: "tensor.weight",
      shape: [256, 2],
      ggmlType: 12,
      storageType: "q4-k-144",
    },
  ]);
});

test("fails closed before package access when no Qwen driver is installed", async () => {
  let manifestAccessed = false;
  const options = {
    get manifest() {
      manifestAccessed = true;
      throw new Error("must not read");
    },
  };

  await assert.rejects(
    loadQwen35BrowserResources(
      new AbortController().signal,
      options,
    ),
    { code: "qwen-execution-driver-not-installed" },
  );
  assert.equal(manifestAccessed, false);
});

test("requires the exact pinned language and tokenizer source identities", () => {
  const identity = {
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
  } as ModelPackageManifest;
  assert.doesNotThrow(() => assertQwen35PackageIdentity(identity));

  const changed = {
    ...identity,
    source: { ...identity.source, revision: "1".repeat(40) },
  };
  assert.throws(
    () => assertQwen35PackageIdentity(changed),
    { code: "model-package-identity-mismatch" },
  );
});

test("passes a frozen split-tensor package directory to the driver boundary", () => {
  const directory = buildQwen35PackageDirectory(pinnedManifest());

  assert.equal(directory.tensors.length, 1);
  assert.deepEqual(directory.tensors[0]?.segments, [
    { shardIndex: 0, shardOffset: "0", tensorOffset: "0", length: "144" },
    { shardIndex: 1, shardOffset: "0", tensorOffset: "144", length: "144" },
  ]);
  assert.equal(Object.isFrozen(directory), true);
  assert.equal(Object.isFrozen(directory.tensors), true);
  assert.equal(Object.isFrozen(directory.tensors[0]?.segments), true);
});

test("binds manifest bytes and package URL to an immutable application pin", () => {
  const manifest = pinnedManifest();
  const immutableBase =
    "https://huggingface.co/example/browser-package/resolve/" +
    `${"c".repeat(40)}/`;
  const expectedManifestSha256 = modelCacheKey(manifest);
  assert.doesNotThrow(() =>
    assertQwen35ConvertedPackageTrust(manifest, {
      packageBaseUrl: immutableBase,
      expectedPackageBaseUrl: immutableBase,
      expectedManifestSha256,
    }),
  );

  const forged = structuredClone(manifest);
  forged.shards[0]!.sha256 = "d".repeat(64);
  assert.throws(
    () =>
      assertQwen35ConvertedPackageTrust(forged, {
        packageBaseUrl: immutableBase,
        expectedPackageBaseUrl: immutableBase,
        expectedManifestSha256,
      }),
    { code: "model-package-manifest-mismatch" },
  );
  assert.throws(
    () =>
      assertQwen35ConvertedPackageTrust(manifest, {
        packageBaseUrl:
          "https://huggingface.co/example/browser-package/resolve/main/",
        expectedPackageBaseUrl:
          "https://huggingface.co/example/browser-package/resolve/main/",
        expectedManifestSha256,
      }),
    { code: "model-package-url-mutable" },
  );
  assert.throws(
    () =>
      assertQwen35ConvertedPackageTrust(manifest, {
        packageBaseUrl: immutableBase.replace("example", "other"),
        expectedPackageBaseUrl: immutableBase,
        expectedManifestSha256,
    }),
    { code: "model-package-url-mismatch" },
  );
  assert.throws(
    () =>
      assertQwen35ConvertedPackageTrust(manifest, {
        packageBaseUrl: immutableBase.replace(
          "huggingface.co/",
          "huggingface.co:443/",
        ),
        expectedPackageBaseUrl: immutableBase,
        expectedManifestSha256,
      }),
    { code: "model-package-url-mismatch" },
  );
});

test("snapshots trusted manifest fields before caller mutation can cross an await", async () => {
  const callerManifest = pinnedManifest();
  const expectedManifestSha256 = modelCacheKey(callerManifest);
  const snapshot = snapshotQwen35Manifest(callerManifest);

  await Promise.resolve();
  callerManifest.shards[0]!.url = "https://example.invalid/forged.bin";
  callerManifest.shards[0]!.length = "288";
  callerManifest.shards[0]!.sha256 = "d".repeat(64);
  callerManifest.source.revision = "e".repeat(40);
  callerManifest.runtime.abi = "forged-runtime";
  callerManifest.tensorLayout[0]!.shardOffset = "72";

  assert.equal(Object.isFrozen(callerManifest), false);
  assert.equal(Object.isFrozen(snapshot), true);
  assert.equal(Object.isFrozen(snapshot.source), true);
  assert.equal(Object.isFrozen(snapshot.runtime), true);
  assert.equal(Object.isFrozen(snapshot.shards), true);
  assert.equal(Object.isFrozen(snapshot.shards[0]), true);
  assert.equal(Object.isFrozen(snapshot.tensorLayout[0]), true);
  assert.equal(modelCacheKey(snapshot), expectedManifestSha256);
  assert.equal(snapshot.source.revision, "4168f45a16a1290d65a4ec0fa312ae917a4c15d6");
  assert.equal(snapshot.runtime.abi, "qwen35-webgpu-v1");

  const directory = buildQwen35PackageDirectory(snapshot);
  assert.deepEqual(directory.shards[0], {
    index: 0,
    url: "shards/part-0.bin",
    offset: "0",
    length: "144",
    sha256: "a".repeat(64),
  });
  assert.deepEqual(directory.tensors[0]?.segments[0], {
    shardIndex: 0,
    shardOffset: "0",
    tensorOffset: "0",
    length: "144",
  });
});

test("GPU cleanup awaits queue completion before destroying owned resources", async () => {
  const events: string[] = [];
  let finishQueue!: () => void;
  const queueGate = new Promise<void>((resolve) => {
    finishQueue = resolve;
  });
  const cleanup = cleanupQwen35GpuResources({
    driver: { ...driver, async dispose() { events.push("driver"); } },
    device: {
      queue: {
        async onSubmittedWorkDone() {
          events.push("queue-start");
          await queueGate;
          events.push("queue-end");
        },
      },
      destroy() { events.push("device"); },
    },
    hybridState: { dispose() { events.push("state"); } },
    weightAllocations: [{ ...allocation, destroy() { events.push("weight"); } }],
    ledger: { assertAllReleased() { events.push("ledger"); } },
  });
  await Promise.resolve();
  assert.deepEqual(events, ["driver", "queue-start"]);

  finishQueue();
  await cleanup;
  assert.deepEqual(events, [
    "driver",
    "queue-start",
    "queue-end",
    "state",
    "weight",
    "device",
    "ledger",
  ]);
});

test("GPU cleanup preserves its first failure after every cleanup step settles", async () => {
  const events: string[] = [];
  await assert.rejects(
    cleanupQwen35GpuResources({
      driver: { ...driver, async dispose() { events.push("driver"); throw new Error("first"); } },
      device: {
        queue: { async onSubmittedWorkDone() { events.push("queue"); throw new Error("second"); } },
        destroy() { events.push("device"); },
      },
      hybridState: { dispose() { events.push("state"); } },
      weightAllocations: [{ ...allocation, destroy() { events.push("weight"); } }],
      ledger: { assertAllReleased() { events.push("ledger"); } },
    }),
    /first/,
  );
  assert.deepEqual(events, ["driver", "queue", "state", "weight", "device", "ledger"]);
});
