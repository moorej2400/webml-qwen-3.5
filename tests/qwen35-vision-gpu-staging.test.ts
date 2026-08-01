import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import { AllocationLedger } from "../src/allocation-ledger.js";
import { GgmlType } from "../src/gguf.js";
import {
  stringifyManifest,
  type ModelPackageManifest,
  type TensorLayoutEntry,
} from "../src/manifest.js";
import {
  createIntegrityValidatedQwen35VisionPackage,
  type Qwen35IntegrityValidatedVisionPackage,
} from "../src/qwen35-vision-package-loader.js";
import {
  stageQwen35VisionGpuGroup,
  type Qwen35VisionGpuAllocator,
  type Qwen35VisionGpuBuffer,
  type Qwen35VisionGpuQueue,
} from "../src/qwen35-vision-gpu-staging.js";
import { createQwen35VisionProgram } from "../src/qwen35-vision-program.js";
import type { Qwen35VisionProgram } from "../src/qwen35-vision-program.js";
import type { RangeFetch } from "../src/http-range-reader.js";

const BASE_URL = "https://huggingface.co/public-fixtures/qwen-vision/resolve/0123456789abcdef0123456789abcdef01234567/";
const MIB = 1024 * 1024;

function digest(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function responseForLength(
  byteLength: number,
  range: string,
  corrupt = false,
  chunkPrefixByteLength?: number,
): Response {
  const match = /^bytes=(\d+)-(\d+)$/u.exec(range);
  assert.ok(match);
  const start = Number(match[1]);
  const end = Number(match[2]);
  const body = new Uint8Array(end - start + 1);
  if (corrupt && body.byteLength > 0) body[0] ^= 0xff;
  const responseBody = chunkPrefixByteLength === undefined
    ? body
    : new ReadableStream<Uint8Array>({
      start(controller) {
        // The first incomplete u32 forces the staging sink to bridge a real
        // fetch boundary before it can upload the remaining aligned bytes.
        controller.enqueue(body.subarray(0, chunkPrefixByteLength));
        controller.enqueue(body.subarray(chunkPrefixByteLength));
        controller.close();
      },
    });
  return new Response(responseBody, {
    status: 206,
    headers: {
      "content-length": String(body.byteLength),
      "content-range": `bytes ${start}-${end}/${byteLength}`,
    },
  });
}

function tensorByteLength(shape: readonly number[], ggmlType: GgmlType): number {
  return shape.reduce((total, dimension) => total * dimension, 1) * (ggmlType === GgmlType.F32 ? 4 : 2);
}

function layout(
  name: string,
  shape: readonly number[],
  ggmlType: GgmlType,
  shard: number,
  shardOffset: number,
  tensorOffset = 0,
  length = tensorByteLength(shape, ggmlType),
): TensorLayoutEntry {
  return {
    name,
    shape: shape.map(String),
    ggmlType,
    storageType: ggmlType === GgmlType.F32 ? "f32" : "raw",
    shard,
    shardOffset: String(shardOffset),
    tensorOffset: String(tensorOffset),
    length: String(length),
  };
}

function visionTensorLayout(): TensorLayoutEntry[] {
  const values: TensorLayoutEntry[] = [];
  const add = (name: string, shape: readonly number[], type: GgmlType, shard: number, shardOffset: number) =>
    values.push(layout(name, shape, type, shard, shardOffset));
  for (const [name, shape, type] of [
    ["mm.0.bias", [4_096], GgmlType.F32],
    ["mm.0.weight", [4_096, 4_096], GgmlType.BF16],
    ["mm.2.bias", [2_560], GgmlType.F32],
    ["mm.2.weight", [4_096, 2_560], GgmlType.BF16],
    ["v.patch_embd.bias", [1_024], GgmlType.F32],
    ["v.patch_embd.weight", [16, 16, 3, 1_024], GgmlType.F32],
    ["v.patch_embd.weight.1", [16, 16, 3, 1_024], GgmlType.F32],
  ] as const) {
    add(name, shape, type, 0, values.filter((entry) => entry.shard === 0).reduce((total, entry) => total + Number(entry.length), 0));
  }
  const positionLength = tensorByteLength([1_024, 2_304], GgmlType.F32);
  const firstPositionLength = 6_258_688;
  const bootstrapLength = values.reduce((total, entry) => total + Number(entry.length), 0);
  values.push(layout("v.position_embd.weight", [1_024, 2_304], GgmlType.F32, 0, bootstrapLength, 0, firstPositionLength));
  values.push(layout("v.position_embd.weight", [1_024, 2_304], GgmlType.F32, 1, 0, firstPositionLength, positionLength - firstPositionLength));
  add("v.post_ln.bias", [1_024], GgmlType.F32, 1, positionLength - firstPositionLength);
  add("v.post_ln.weight", [1_024], GgmlType.F32, 1, positionLength - firstPositionLength + 4_096);
  for (let layer = 0; layer < 24; layer += 1) {
    const prefix = `v.blk.${layer}`;
    for (const [name, shape, type] of [
      [`${prefix}.attn_out.bias`, [1_024], GgmlType.F32],
      [`${prefix}.attn_out.weight`, [1_024, 1_024], GgmlType.BF16],
      [`${prefix}.attn_qkv.bias`, [3_072], GgmlType.F32],
      [`${prefix}.attn_qkv.weight`, [1_024, 3_072], GgmlType.BF16],
      [`${prefix}.ffn_down.bias`, [1_024], GgmlType.F32],
      [`${prefix}.ffn_down.weight`, [4_096, 1_024], GgmlType.BF16],
      [`${prefix}.ffn_up.bias`, [4_096], GgmlType.F32],
      [`${prefix}.ffn_up.weight`, [1_024, 4_096], GgmlType.BF16],
      [`${prefix}.ln1.bias`, [1_024], GgmlType.F32],
      [`${prefix}.ln1.weight`, [1_024], GgmlType.F32],
      [`${prefix}.ln2.bias`, [1_024], GgmlType.F32],
      [`${prefix}.ln2.weight`, [1_024], GgmlType.F32],
    ] as const) {
      add(name, shape, type, layer + 2, values.filter((entry) => entry.shard === layer + 2).reduce((total, entry) => total + Number(entry.length), 0));
    }
  }
  return values;
}

const HASH_CHUNK = new Uint8Array(64 * 1024);
const ZERO_HASHES = new Map<number, string>();

function zeroHash(length: number): string {
  const cached = ZERO_HASHES.get(length);
  if (cached !== undefined) return cached;
  const hash = createHash("sha256");
  let remaining = length;
  while (remaining > 0) {
    const part = Math.min(remaining, HASH_CHUNK.byteLength);
    hash.update(HASH_CHUNK.subarray(0, part));
    remaining -= part;
  }
  const result = hash.digest("hex");
  ZERO_HASHES.set(length, result);
  return result;
}

function fixture(input: {
  readonly corruptShard?: number;
  readonly shardLength?: number;
  readonly chunkPrefixByteLength?: number;
} = {}): {
  readonly package_: Qwen35IntegrityValidatedVisionPackage;
  readonly program: Qwen35VisionProgram;
  readonly shardLengths: readonly number[];
} {
  const shardLengths = input.shardLength === undefined
    ? [67_106_816, 3_186_688, ...Array.from({ length: 24 }, () => 25_219_072)]
    : Array.from({ length: 26 }, () => input.shardLength!);
  const manifest: ModelPackageManifest = {
    format: "webml-qwen-package",
    version: 1,
    packageKind: "vision",
    source: {
      repository: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF",
      revision: "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
      file: "mmproj-Qwen_Qwen3.5-4B-bf16.gguf",
      size: "675569216",
      sha256: "463f39bd1c291c1186c319a8c90ff8640aafa678b14cbee2232d695113dfbb66",
    },
    runtime: { abi: "qwen35-webgpu-vision-v1" },
    tokenizer: {
      repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
      revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
      file: "tokenizer.json",
      size: "12807982",
      sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    },
    processor: {
      repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
      revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
      file: "preprocessor_config.json",
      size: "390",
      sha256: "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    },
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
    tensorLayout: visionTensorLayout(),
    shards: shardLengths.map((length, index) => ({
      url: `vision-${String(index).padStart(5, "0")}.bin`,
      offset: String(shardLengths.slice(0, index).reduce((total, size) => total + size, 0)),
      length: String(length),
      sha256: zeroHash(length),
    })),
    excludedTensors: [],
  };
  const manifestBytes = new TextEncoder().encode(stringifyManifest(manifest));
  const layerIndexBytes = new TextEncoder().encode(JSON.stringify({
    format: "webml-qwen-vision-layer-index",
    version: 1,
    groups: [
      { layer: "bootstrap", shards: [0, 1] },
      ...Array.from({ length: 24 }, (_, layer) => ({ layer: String(layer), shards: [layer + 2] })),
    ],
  }));
  const rangeFetch: RangeFetch = async (locator, init) => {
    const shard = Number(/vision-(\d{5})\.bin$/u.exec(locator)?.[1]);
    return responseForLength(
      shardLengths[shard]!,
      init.range,
      shard === input.corruptShard,
      input.chunkPrefixByteLength,
    );
  };
  const package_ = createIntegrityValidatedQwen35VisionPackage({
    manifestBytes,
    layerIndexBytes,
    pins: {
      packageBaseUrl: BASE_URL,
      expectedPackageBaseUrl: BASE_URL,
      expectedManifestSha256: digest(manifestBytes),
      expectedLayerIndexSha256: digest(layerIndexBytes),
    },
    rangeFetch,
    rangeStrategies: [32 * 1024 * 1024],
  });
  return {
    shardLengths,
    package_, program: createQwen35VisionProgram(package_),
  };
}

class BufferFake implements Qwen35VisionGpuBuffer {
  destroyed = 0;
  readonly writes: Array<{ readonly offset: number; readonly bytes: Uint8Array }> = [];

  constructor(readonly size: number, private readonly throwOnDestroy = false) {}

  destroy(): void {
    this.destroyed += 1;
    if (this.throwOnDestroy) throw new Error("destroy failed");
  }
}

function gpuFakes(input: { readonly writeFails?: boolean; readonly destroyFails?: boolean; readonly retire?: () => Promise<void> } = {}): {
  readonly allocator: Qwen35VisionGpuAllocator;
  readonly queue: Qwen35VisionGpuQueue;
  readonly buffers: BufferFake[];
  readonly bufferUsages: number[];
  readonly dataOffsets: number[];
} {
  const buffers: BufferFake[] = [];
  const bufferUsages: number[] = [];
  const dataOffsets: number[] = [];
  const allocator: Qwen35VisionGpuAllocator = {
    createBuffer(descriptor) {
      const buffer = new BufferFake(descriptor.size, input.destroyFails);
      buffers.push(buffer);
      bufferUsages.push(descriptor.usage);
      return buffer;
    },
  };
  const queue: Qwen35VisionGpuQueue = {
    writeBuffer(buffer, offset, data, dataOffset = 0, size = data.byteLength - dataOffset) {
      if (input.writeFails) throw new Error("write failed");
      dataOffsets.push(dataOffset);
      (buffer as BufferFake).writes.push({ offset, bytes: data.slice(dataOffset, dataOffset + size) });
    },
    async onSubmittedWorkDone() {
      await input.retire?.();
    },
  };
  return { allocator, queue, buffers, bufferUsages, dataOffsets };
}

test("bridges one-, two-, and three-byte source remainders without unaligned GPU data offsets", async () => {
  for (const remainder of [1, 2, 3]) {
    const input = fixture({ chunkPrefixByteLength: remainder });
    const fake = gpuFakes();
    const ledger = new AllocationLedger(32n * BigInt(MIB));
    const staged = await stageQwen35VisionGpuGroup({
      package: input.package_, program: input.program, layer: 0, ledger,
      allocator: fake.allocator, queue: fake.queue,
      allocationId: `vision-remainder-${remainder}`, uploadLaneBytes: 32 * MIB,
    });
    const writes = fake.buffers[0]!.writes;
    assert.deepEqual(
      writes.map((write) => write.offset),
      [0, 4],
      `remainder ${remainder} must emit a completed u32 then the aligned suffix`,
    );
    assert.deepEqual(
      writes.map((write) => write.bytes.byteLength),
      [4, input.shardLengths[2]! - 4],
    );
    assert.ok(fake.dataOffsets.every((offset) => offset % 4 === 0));
    assert.ok(writes.every((write) => write.bytes.every((byte) => byte === 0)));
    await staged.destroy();
    assert.equal(ledger.snapshot().currentBytes, 0n);
  }
});

test("requires COPY_DST and STORAGE usage for every vision GPU buffer", async () => {
  const input = fixture();
  const fake = gpuFakes();
  const ledger = new AllocationLedger(32n * BigInt(MIB));
  const requiredUsage = 0x0008 | 0x0080;
  const staged = await stageQwen35VisionGpuGroup({
    package: input.package_, program: input.program, layer: 0, ledger,
    allocator: fake.allocator, queue: fake.queue, allocationId: "vision-usage-default",
  });
  assert.deepEqual(fake.bufferUsages, [requiredUsage]);
  await staged.destroy();

  for (const usage of [0x0008, 0x0080]) {
    await assert.rejects(() => stageQwen35VisionGpuGroup({
      package: input.package_, program: input.program, layer: 0, ledger,
      allocator: fake.allocator, queue: fake.queue,
      allocationId: `vision-usage-${usage}`, bufferUsage: usage,
    }), { code: "vision-stage-options-invalid" });
    assert.equal(ledger.snapshot().currentBytes, 0n);
  }
});

test("stages bootstrap shards transactionally and maps immutable BF16/F32 views only after queue retirement", async () => {
  const input = fixture();
  let release!: () => void;
  const retirement = new Promise<void>((resolve) => { release = resolve; });
  const fake = gpuFakes({ retire: () => retirement });
  const ledger = new AllocationLedger(128n * BigInt(MIB));
  let settled = false;
  const stagedPromise = stageQwen35VisionGpuGroup({
    package: input.package_, program: input.program, layer: "bootstrap", ledger,
    allocator: fake.allocator, queue: fake.queue, allocationId: "vision-bootstrap-test", uploadLaneBytes: 32 * MIB,
  }).then((value) => { settled = true; return value; });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(settled, false);
  assert.equal(fake.buffers.length, 2);
  assert.equal(ledger.snapshot().currentBytes, BigInt(input.shardLengths[0]! + input.shardLengths[1]!));
  release();
  const staged = await stagedPromise;
  assert.equal(staged.shards.length, 2);
  assert.deepEqual(staged.shards.map((entry) => entry.shard), [0, 1]);
  assert.ok(staged.tensors.some((entry) => entry.name === "v.patch_embd.weight"));
  assert.ok(staged.tensors.some((entry) => entry.name === "v.position_embd.weight"));
  assert.equal(staged.tensors.find((entry) => entry.name === "v.patch_embd.weight")!.precision, "f32");
  assert.equal(staged.tensors.find((entry) => entry.name === "v.position_embd.weight")!.precision, "f32");
  assert.equal(staged.tensors.find((entry) => entry.name === "v.patch_embd.weight")!.orientation.kind, "patch-conv3d");
  assert.equal(staged.tensors.find((entry) => entry.name === "v.position_embd.weight")!.orientation.kind, "learned-position-hidden-contiguous");
  assert.equal(staged.tensors.find((entry) => entry.name === "mm.0.bias")!.orientation.kind, "element-contiguous");
  assert.equal(staged.tensors.find((entry) => entry.name === "mm.0.weight")!.orientation.kind, "input-width-contiguous");
  assert.equal(
    staged.tensors.find((entry) => entry.name === "v.patch_embd.weight")!.segments[0]!.bufferOffset,
    input.program.bootstrap.patchEmbedding.temporalWeights[0].segments[0]!.shardOffset,
  );
  assert.equal(staged.tensors.find((entry) => entry.name === "v.position_embd.weight")!.segments[1]!.shard, 1);
  assert.ok(Object.isFrozen(staged.tensors));
  assert.equal(fake.buffers[0]!.writes.reduce((total, write) => total + write.bytes.byteLength, 0), input.shardLengths[0]);
  assert.equal(fake.buffers[1]!.writes.reduce((total, write) => total + write.bytes.byteLength, 0), input.shardLengths[1]);
  await staged.destroy();
  assert.equal(ledger.snapshot().currentBytes, 0n);
  assert.equal(fake.buffers[0]!.destroyed, 1);
  await staged.destroy();
  assert.equal(fake.buffers[0]!.destroyed, 1);
});

test("uses the authenticated loader transaction to roll back GPU buffers after a shard hash failure", async () => {
  const input = fixture({ corruptShard: 2 });
  const fake = gpuFakes();
  const ledger = new AllocationLedger(32n * BigInt(MIB));
  await assert.rejects(() => stageQwen35VisionGpuGroup({
    package: input.package_, program: input.program, layer: 0, ledger,
    allocator: fake.allocator, queue: fake.queue, allocationId: "vision-corrupt-test", uploadLaneBytes: 32 * MIB,
  }), { code: "vision-layer-shard-hash-mismatch" });
  assert.equal(fake.buffers.length, 1);
  assert.equal(fake.buffers[0]!.destroyed, 1);
  assert.equal(ledger.snapshot().currentBytes, 0n);
});

test("rolls back committed buffers when immutable view publication fails after loader commit", async () => {
  const input = fixture();
  let destroyed = 0;
  const buffer: Qwen35VisionGpuBuffer = {
    destroy() { destroyed += 1; },
  };
  // A browser buffer is opaque. This hostile plain-object fake proves that a
  // later publication failure still returns ownership through sink.abort().
  Object.defineProperty(buffer, "publicationGetter", {
    enumerable: true,
    get() { throw new Error("publication failed"); },
  });
  const allocator: Qwen35VisionGpuAllocator = { createBuffer() { return buffer; } };
  const queue: Qwen35VisionGpuQueue = {
    writeBuffer() {},
    async onSubmittedWorkDone() {},
  };
  const ledger = new AllocationLedger(32n * BigInt(MIB));
  await assert.rejects(() => stageQwen35VisionGpuGroup({
    package: input.package_, program: input.program, layer: 0, ledger, allocator, queue,
    allocationId: "vision-publication-failure-test", uploadLaneBytes: 32 * MIB,
  }), /publication failed/u);
  assert.equal(destroyed, 1);
  assert.equal(ledger.snapshot().currentBytes, 0n);
});

test("rejects unauthenticated programs, package mismatches, and unaligned group buffers before publication", async () => {
  const input = fixture();
  const other = fixture();
  const fake = gpuFakes();
  const ledger = new AllocationLedger(32n * BigInt(MIB));
  await assert.rejects(() => stageQwen35VisionGpuGroup({
    package: input.package_, program: other.program, layer: 0, ledger,
    allocator: fake.allocator, queue: fake.queue, allocationId: "vision-mismatch-test", uploadLaneBytes: 32 * MIB,
  }), { code: "vision-program-package-mismatch" });
  await assert.rejects(() => stageQwen35VisionGpuGroup({
    package: input.package_, program: { ...input.program } as Qwen35VisionProgram, layer: 0, ledger,
    allocator: fake.allocator, queue: fake.queue, allocationId: "vision-fake-test", uploadLaneBytes: 32 * MIB,
  }), { code: "vision-program-integrity-unvalidated" });
  assert.equal(ledger.snapshot().currentBytes, 0n);

  await assert.rejects(() => stageQwen35VisionGpuGroup({
    package: input.package_, program: input.program, layer: 0, ledger,
    allocator: fake.allocator, queue: fake.queue, allocationId: "vision-unaligned-test", uploadLaneBytes: 32 * MIB,
    bufferCapBytes: 4,
  }), { code: "vision-stage-shard-unaligned" });
});

test("cancellation, write failure, and cleanup failure release every reservation without replacing the primary failure", async () => {
  const cancelled = fixture();
  const controller = new AbortController();
  controller.abort(new DOMException("cancelled", "AbortError"));
  const cancellationFakes = gpuFakes();
  const cancellationLedger = new AllocationLedger(32n * BigInt(MIB));
  await assert.rejects(() => stageQwen35VisionGpuGroup({
    package: cancelled.package_, program: cancelled.program, layer: 0, ledger: cancellationLedger,
    allocator: cancellationFakes.allocator, queue: cancellationFakes.queue,
    allocationId: "vision-cancel-test", uploadLaneBytes: 32 * MIB, signal: controller.signal,
  }), { name: "AbortError" });
  assert.equal(cancellationFakes.buffers[0]!.destroyed, 1);
  assert.equal(cancellationLedger.snapshot().currentBytes, 0n);

  const failedWrite = fixture();
  const writeFakes = gpuFakes({ writeFails: true });
  const writeLedger = new AllocationLedger(32n * BigInt(MIB));
  await assert.rejects(() => stageQwen35VisionGpuGroup({
    package: failedWrite.package_, program: failedWrite.program, layer: 0, ledger: writeLedger,
    allocator: writeFakes.allocator, queue: writeFakes.queue, allocationId: "vision-write-test", uploadLaneBytes: 32 * MIB,
  }), { code: "vision-stage-upload-failed" });
  assert.equal(writeFakes.buffers[0]!.destroyed, 1);
  assert.equal(writeLedger.snapshot().currentBytes, 0n);

  const cleanupInput = fixture({ corruptShard: 2 });
  const cleanupFakes = gpuFakes({ destroyFails: true, retire: async () => { throw new Error("retire failed"); } });
  const cleanupLedger = new AllocationLedger(32n * BigInt(MIB));
  await assert.rejects(() => stageQwen35VisionGpuGroup({
    package: cleanupInput.package_, program: cleanupInput.program, layer: 0, ledger: cleanupLedger,
    allocator: cleanupFakes.allocator, queue: cleanupFakes.queue,
    allocationId: "vision-cleanup-test", uploadLaneBytes: 32 * MIB,
  }), { code: "vision-layer-shard-hash-mismatch" });
  assert.equal(cleanupFakes.buffers[0]!.destroyed, 1);
  assert.equal(cleanupLedger.snapshot().currentBytes, 0n);
});
