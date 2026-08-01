import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import { GgmlType } from "../src/gguf.js";
import {
  assertIntegrityValidatedQwen35VisionPackage,
  assertProductionTrustedQwen35VisionPackage,
  createIntegrityValidatedQwen35VisionPackage,
  createProductionQwen35VisionPackage,
  type Qwen35VisionPackagePins,
} from "../src/qwen35-vision-package-loader.js";
import { stringifyManifest, type ModelPackageManifest } from "../src/manifest.js";
import type { RangeFetch } from "../src/http-range-reader.js";

const BASE_URL = "https://huggingface.co/public-fixtures/qwen-vision/resolve/0123456789abcdef0123456789abcdef01234567/";
const SOURCE = Object.freeze({
  repository: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF",
  revision: "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
  file: "mmproj-Qwen_Qwen3.5-4B-bf16.gguf",
  size: "675569216",
  sha256: "463f39bd1c291c1186c319a8c90ff8640aafa678b14cbee2232d695113dfbb66",
});
const TOKENIZER = Object.freeze({
  repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
  revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
  file: "tokenizer.json",
  size: "12807982",
  sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
});
const PROCESSOR = Object.freeze({
  repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
  revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
  file: "preprocessor_config.json",
  size: "390",
  sha256: "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
});

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function bytes(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

function fixture(): {
  readonly manifestBytes: Uint8Array;
  readonly layerIndexBytes: Uint8Array;
  readonly shardBytes: readonly Uint8Array[];
  readonly pins: Qwen35VisionPackagePins;
} {
  const shardBytes = Array.from({ length: 26 }, (_, index) =>
    Uint8Array.of(index, index + 1, index + 2, index + 3),
  );
  const manifest: ModelPackageManifest = {
    format: "webml-qwen-package",
    version: 1,
    packageKind: "vision",
    source: SOURCE,
    runtime: { abi: "qwen35-webgpu-vision-v1" },
    tokenizer: TOKENIZER,
    processor: PROCESSOR,
    processorSettings: {
      processorClass: "Qwen3VLProcessor",
      imageProcessorType: "Qwen2VLImageProcessorFast",
      patchSize: 16,
      temporalPatchSize: 2,
      mergeSize: 2,
      shortestEdge: 65_536,
      longestEdge: 16_777_216,
      imageMean: [0.5, 0.5, 0.5],
      imageStd: [0.5, 0.5, 0.5],
    },
    tensorLayout: shardBytes.map((shard, index) => ({
      name: `v.blk.${index}.weight`,
      shape: ["1"],
      ggmlType: GgmlType.F32,
      storageType: "f32",
      shard: index,
      shardOffset: "0",
      tensorOffset: "0",
      length: String(shard.byteLength),
    })),
    shards: shardBytes.map((shard, index) => ({
      url: `vision-${String(index).padStart(5, "0")}.bin`,
      offset: String(index * shard.byteLength),
      length: String(shard.byteLength),
      sha256: sha256(shard),
    })),
    excludedTensors: [],
  };
  const manifestBytes = bytes(stringifyManifest(manifest));
  const layerIndexBytes = bytes(`${JSON.stringify({
    format: "webml-qwen-vision-layer-index",
    version: 1,
    groups: [
      { layer: "bootstrap", shards: [0, 1] },
      ...Array.from({ length: 24 }, (_, index) => ({
        layer: String(index),
        shards: [index + 2],
      })),
    ],
  })}\n`);
  return {
    manifestBytes,
    layerIndexBytes,
    shardBytes,
    pins: {
      packageBaseUrl: BASE_URL,
      expectedPackageBaseUrl: BASE_URL,
      expectedManifestSha256: sha256(manifestBytes),
      expectedLayerIndexSha256: sha256(layerIndexBytes),
    },
  };
}

function parseRange(value: string): readonly [number, number] {
  const match = /^bytes=(\d+)-(\d+)$/u.exec(value);
  assert.ok(match);
  return [Number(match[1]), Number(match[2])];
}

function rangeFetchFor(
  shardBytes: readonly Uint8Array[],
  requests: string[],
): RangeFetch {
  return async (locator, init) => {
    requests.push(locator);
    const index = Number(/vision-(\d{5})\.bin$/u.exec(locator)?.[1]);
    const source = shardBytes[index];
    assert.ok(source);
    const [start, end] = parseRange(init.range);
    const body = source.subarray(start, end + 1);
    return new Response(body, {
      status: 206,
      headers: {
        "content-length": String(body.byteLength),
        "content-range": `bytes ${start}-${end}/${source.byteLength}`,
      },
    });
  };
}

test("integrity-validates a caller-pinned vision package and streams one addressed layer", async () => {
  const input = fixture();
  const requests: string[] = [];
  const vision = createIntegrityValidatedQwen35VisionPackage({
    manifestBytes: input.manifestBytes,
    layerIndexBytes: input.layerIndexBytes,
    pins: input.pins,
    rangeFetch: rangeFetchFor(input.shardBytes, requests),
    rangeStrategies: [2],
  });

  assert.equal(vision.bootstrap.shards.length, 2);
  assert.equal(vision.layers.length, 24);
  assert.deepEqual(vision.layers.map((layer) => layer.index), Array.from({ length: 24 }, (_, index) => index));
  assert.equal(requests.length, 0);
  const staged: number[] = [];
  const committed: number[] = [];
  await vision.streamLayer(0, {
    async write(chunk) {
      staged.push(...chunk);
    },
    async commit() {
      committed.push(...staged);
    },
    async abort() {
      staged.length = 0;
    },
  });
  assert.deepEqual(committed, [...input.shardBytes[2]!]);
  assert.equal(requests.length, 2);
  assert.ok(requests.every((locator) => locator.endsWith("vision-00002.bin")));
  assert.ok(Object.isFrozen(vision.manifest));
  assert.ok(Object.isFrozen(vision.manifest.processorSettings));
  assert.ok(Object.isFrozen(vision.manifest.shards));
  assert.ok(Object.isFrozen(vision.layers));
  assert.ok(Object.isFrozen(vision.layers[0]!.shards));
});

test("rejects mutable package pins, provenance drift, and unauthenticated layer indexes", () => {
  const input = fixture();
  assert.throws(
    () => createIntegrityValidatedQwen35VisionPackage({
      manifestBytes: input.manifestBytes,
      layerIndexBytes: input.layerIndexBytes,
      pins: { ...input.pins, packageBaseUrl: BASE_URL.replace("resolve", "blob") },
    }),
    { code: "vision-package-base-mismatch" },
  );
  const provenanceDrift = JSON.parse(new TextDecoder().decode(input.manifestBytes)) as ModelPackageManifest;
  provenanceDrift.source = { ...provenanceDrift.source, file: "other.gguf" };
  const changedManifest = bytes(stringifyManifest(provenanceDrift));
  assert.throws(
    () => createIntegrityValidatedQwen35VisionPackage({
      manifestBytes: changedManifest,
      layerIndexBytes: input.layerIndexBytes,
      pins: { ...input.pins, expectedManifestSha256: sha256(changedManifest) },
    }),
    { code: "vision-package-identity-mismatch" },
  );
  assert.throws(
    () => createIntegrityValidatedQwen35VisionPackage({
      manifestBytes: input.manifestBytes,
      layerIndexBytes: bytes("{}"),
      pins: input.pins,
    }),
    { code: "vision-layer-index-hash-mismatch" },
  );
  assert.throws(
    () => createIntegrityValidatedQwen35VisionPackage({
      manifestBytes: new Uint8Array((1024 * 1024) + 1),
      layerIndexBytes: input.layerIndexBytes,
      pins: input.pins,
    }),
    { code: "vision-package-manifest-invalid" },
  );
});

test("owns bounded metadata snapshots and rejects shared mutable backing stores", () => {
  const input = fixture();
  const manifestBytes = input.manifestBytes.slice();
  const layerIndexBytes = input.layerIndexBytes.slice();
  const vision = createIntegrityValidatedQwen35VisionPackage({
    manifestBytes,
    layerIndexBytes,
    pins: input.pins,
    rangeFetch: rangeFetchFor(input.shardBytes, []),
  });
  manifestBytes.fill(0);
  layerIndexBytes.fill(0);
  assert.equal(vision.manifest.source.file, SOURCE.file);
  assert.equal(vision.manifestSha256, input.pins.expectedManifestSha256);
  assert.equal(vision.layerIndexSha256, input.pins.expectedLayerIndexSha256);

  const sharedManifest = new Uint8Array(new SharedArrayBuffer(input.manifestBytes.byteLength));
  sharedManifest.set(input.manifestBytes);
  assert.throws(() => createIntegrityValidatedQwen35VisionPackage({
    manifestBytes: sharedManifest,
    layerIndexBytes: input.layerIndexBytes,
    pins: input.pins,
  }), { code: "vision-package-manifest-invalid" });
});

test("keeps caller-pinned integrity validation separate from production release trust", () => {
  const input = fixture();
  const vision = createIntegrityValidatedQwen35VisionPackage({
    manifestBytes: input.manifestBytes,
    layerIndexBytes: input.layerIndexBytes,
    pins: input.pins,
  });
  assert.equal(assertIntegrityValidatedQwen35VisionPackage(vision), vision);
  assert.throws(
    () => assertProductionTrustedQwen35VisionPackage(vision),
    { code: "vision-package-production-untrusted" },
  );
  assert.throws(
    () => assertIntegrityValidatedQwen35VisionPackage({ ...vision }),
    { code: "vision-package-integrity-unvalidated" },
  );
  assert.throws(() => createProductionQwen35VisionPackage({
    manifestBytes: input.manifestBytes,
    layerIndexBytes: input.layerIndexBytes,
  }), { code: "vision-package-manifest-hash-mismatch" });
});

test("permits a localhost vision package only with explicit development opt-in", () => {
  const input = fixture();
  const localPins = {
    ...input.pins,
    packageBaseUrl: "http://localhost:18082/vision/",
    expectedPackageBaseUrl: "http://localhost:18082/vision/",
  };
  assert.throws(() => createIntegrityValidatedQwen35VisionPackage({
    manifestBytes: input.manifestBytes,
    layerIndexBytes: input.layerIndexBytes,
    pins: localPins,
  }), { code: "vision-package-base-mismatch" });
  assert.doesNotThrow(() => createIntegrityValidatedQwen35VisionPackage({
    manifestBytes: input.manifestBytes,
    layerIndexBytes: input.layerIndexBytes,
    pins: { ...localPins, allowInsecureLocalhost: true },
  }));
});

test("rejects an invalid addressed layer index", () => {
  const input = fixture();
  const index = JSON.parse(new TextDecoder().decode(input.layerIndexBytes)) as {
    groups: Array<{ layer: string; shards: number[] }>;
  };
  index.groups[1]!.shards = [1];
  const invalidIndex = bytes(JSON.stringify(index));
  assert.throws(
    () => createIntegrityValidatedQwen35VisionPackage({
      manifestBytes: input.manifestBytes,
      layerIndexBytes: invalidIndex,
      pins: { ...input.pins, expectedLayerIndexSha256: sha256(invalidIndex) },
    }),
    { code: "vision-layer-index-invalid" },
  );

});

test("corrupt shard bytes abort staged work and never commit", async () => {
  const input = fixture();
  const corrupt = input.shardBytes.map((value) => value.slice());
  corrupt[2]![0] ^= 0xff;
  const vision = createIntegrityValidatedQwen35VisionPackage({
    manifestBytes: input.manifestBytes,
    layerIndexBytes: input.layerIndexBytes,
    pins: input.pins,
    rangeFetch: rangeFetchFor(corrupt, []),
    rangeStrategies: [2],
  });
  const staged: number[] = [];
  let commits = 0;
  let aborts = 0;
  await assert.rejects(vision.streamLayer(0, {
    async write(chunk) {
      staged.push(...chunk);
    },
    async commit() {
      commits += 1;
    },
    async abort() {
      aborts += 1;
      staged.length = 0;
    },
  }), { code: "vision-layer-shard-hash-mismatch" });
  assert.equal(commits, 0);
  assert.equal(aborts, 1);
  assert.deepEqual(staged, []);
});

test("write failure stays primary when transactional abort also fails", async () => {
  const input = fixture();
  const primary = new Error("fixture write failure");
  let commits = 0;
  let aborts = 0;
  const vision = createIntegrityValidatedQwen35VisionPackage({
    manifestBytes: input.manifestBytes,
    layerIndexBytes: input.layerIndexBytes,
    pins: input.pins,
    rangeFetch: rangeFetchFor(input.shardBytes, []),
    rangeStrategies: [2],
  });
  await assert.rejects(vision.streamLayer(0, {
    async write() {
      throw primary;
    },
    async commit() {
      commits += 1;
    },
    async abort() {
      aborts += 1;
      throw new Error("fixture abort failure");
    },
  }), (error: unknown) => error === primary);
  assert.equal(commits, 0);
  assert.equal(aborts, 1);
});

test("multi-shard bootstrap commits one transaction after both hashes pass", async () => {
  const input = fixture();
  const vision = createIntegrityValidatedQwen35VisionPackage({
    manifestBytes: input.manifestBytes,
    layerIndexBytes: input.layerIndexBytes,
    pins: input.pins,
    rangeFetch: rangeFetchFor(input.shardBytes, []),
    rangeStrategies: [2],
  });
  const staged: number[] = [];
  let commits = 0;
  let aborts = 0;
  await vision.streamLayer("bootstrap", {
    async write(chunk) {
      staged.push(...chunk);
    },
    async commit() {
      commits += 1;
    },
    async abort() {
      aborts += 1;
    },
  });
  assert.deepEqual(staged, [
    ...input.shardBytes[0]!,
    ...input.shardBytes[1]!,
  ]);
  assert.equal(commits, 1);
  assert.equal(aborts, 0);
});

test("fetch failure and cancellation each abort exactly once", async () => {
  const input = fixture();
  let fetchAborts = 0;
  const failed = createIntegrityValidatedQwen35VisionPackage({
    manifestBytes: input.manifestBytes,
    layerIndexBytes: input.layerIndexBytes,
    pins: input.pins,
    rangeFetch: async () => {
      throw new TypeError("fixture network failure");
    },
    rangeStrategies: [2],
  });
  await assert.rejects(failed.streamLayer(0, {
    async write() {},
    async commit() {},
    async abort() {
      fetchAborts += 1;
    },
  }));
  assert.equal(fetchAborts, 1);

  const controller = new AbortController();
  let cancellationAborts = 0;
  const cancelled = createIntegrityValidatedQwen35VisionPackage({
    manifestBytes: input.manifestBytes,
    layerIndexBytes: input.layerIndexBytes,
    pins: input.pins,
    rangeFetch: rangeFetchFor(input.shardBytes, []),
    rangeStrategies: [2],
  });
  await assert.rejects(cancelled.streamLayer(0, {
    async write() {
      controller.abort();
    },
    async commit() {},
    async abort() {
      cancellationAborts += 1;
    },
  }, controller.signal), { name: "AbortError" });
  assert.equal(cancellationAborts, 1);
});

test("cancellation after the final verified source chunk aborts instead of committing", async () => {
  const input = fixture();
  const controller = new AbortController();
  let commits = 0;
  let aborts = 0;
  const vision = createIntegrityValidatedQwen35VisionPackage({
    manifestBytes: input.manifestBytes,
    layerIndexBytes: input.layerIndexBytes,
    pins: input.pins,
    rangeFetch: rangeFetchFor(input.shardBytes, []),
    // Layer 0 is a four-byte fixture shard, so its write is also the last
    // range/chunk and exposes the commit-window cancellation boundary.
    rangeStrategies: [4],
  });
  await assert.rejects(vision.streamLayer(0, {
    async write() {
      controller.abort(new DOMException("cancelled", "AbortError"));
    },
    async commit() {
      commits += 1;
    },
    async abort() {
      aborts += 1;
    },
  }, controller.signal), { name: "AbortError" });
  assert.equal(commits, 0);
  assert.equal(aborts, 1);
});
