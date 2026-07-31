import assert from "node:assert/strict";
import test from "node:test";

import type { GpuAllocation } from "../src/gpu-arena.js";
import type { ModelPackageManifest } from "../src/manifest.js";
import type { ModelCacheStorage } from "../src/opfs-model-cache.js";
import {
  assertQwen35PackageIdentity,
  buildQwen35TensorDirectory,
  loadQwen35BrowserResources,
  qwen35AllocatedWeightBytes,
  streamQwen35CachedWeights,
  type Qwen35ExecutionDriverFactory,
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
  async reset() {},
  async dispose() {},
};

function streamingStorage(
  chunks: readonly Uint8Array[],
  onClose?: () => void,
): ModelCacheStorage {
  return {
    async openAtomicWriter() {
      throw new Error("not used");
    },
    async openRead() {
      return (async function* () {
        try {
          yield* chunks;
        } finally {
          onClose?.();
        }
      })();
    },
    async move() {
      return false;
    },
    async list() {
      return [];
    },
  };
}

test("streams cached weights through bounded awaited upload lanes", async () => {
  const uploaded: Array<{ offset: number; length: number }> = [];
  let activeUploads = 0;
  let peakUploads = 0;
  const factory = {
    async uploadWeightChunk(
      _driver: Qwen35ExecutionDriver,
      input: { byteOffset: number; chunk: Uint8Array },
    ) {
      activeUploads += 1;
      peakUploads = Math.max(peakUploads, activeUploads);
      uploaded.push({ offset: input.byteOffset, length: input.chunk.byteLength });
      await Promise.resolve();
      activeUploads -= 1;
    },
  } as Qwen35ExecutionDriverFactory;

  await streamQwen35CachedWeights({
    storage: streamingStorage([new Uint8Array(25)]),
    cached: {
      cacheKey: "a",
      manifestSha256: "b",
      cacheHit: false,
      shards: [{ storagePath: "blob", byteLength: 25, sha256: "c" }],
    },
    allocations: [allocation],
    driver,
    factory,
    uploadLaneBytes: 8,
    signal: new AbortController().signal,
  });

  assert.deepEqual(uploaded, [
    { offset: 0, length: 8 },
    { offset: 8, length: 8 },
    { offset: 16, length: 8 },
    { offset: 24, length: 1 },
  ]);
  assert.equal(peakUploads, 1);
});

test("accounts for exact aligned GPU shard ownership", () => {
  assert.equal(
    qwen35AllocatedWeightBytes([
      { byteLength: 1 },
      { byteLength: 4 },
      { byteLength: 5 },
    ]),
    16n,
  );
});

test("closes the cached shard stream when upload fails", async () => {
  let closed = false;
  const factory = {
    async uploadWeightChunk() {
      throw new Error("private GPU failure");
    },
  } as unknown as Qwen35ExecutionDriverFactory;

  await assert.rejects(
    streamQwen35CachedWeights({
      storage: streamingStorage([new Uint8Array(25)], () => {
        closed = true;
      }),
      cached: {
        cacheKey: "a",
        manifestSha256: "b",
        cacheHit: false,
        shards: [{ storagePath: "blob", byteLength: 25, sha256: "c" }],
      },
      allocations: [allocation],
      driver,
      factory,
      uploadLaneBytes: 8,
      signal: new AbortController().signal,
    }),
  );
  assert.equal(closed, true);
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
