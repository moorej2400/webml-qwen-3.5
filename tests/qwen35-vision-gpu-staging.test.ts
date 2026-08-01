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
  assertAuthenticatedQwen35VisionGpuStagedGroup,
  stageQwen35VisionGpuGroup,
  type Qwen35VisionGpuAllocator,
  type Qwen35VisionGpuBuffer,
  type Qwen35VisionGpuQueue,
} from "../src/qwen35-vision-gpu-staging.js";
import { planQwen35VisionBootstrapFoundationDispatches } from "../src/qwen35-vision-foundation-kernels.js";
import { planQwen35VisionLayerDispatches } from "../src/qwen35-vision-layer-kernels.js";
import { planQwen35VisionMergerDispatches } from "../src/qwen35-vision-merger-kernels.js";
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

test("plans the authenticated bootstrap merger with four-row logical input orientation", async () => {
  const input = fixture();
  const fakes = gpuFakes();
  const ledger = new AllocationLedger(128n * BigInt(MIB));
  const bootstrap = await stageQwen35VisionGpuGroup({ package: input.package_, program: input.program, layer: "bootstrap", ledger, allocator: fakes.allocator, queue: fakes.queue, allocationId: "vision-merger-bootstrap", uploadLaneBytes: 32 * MIB });
  const buffers = Array.from({ length: 8 }, () => ({}));
  const workspace = {
    hidden: { buffer: buffers[0]!, byteLength: 4 * 1_024 * 4 },
    normalized: { buffer: buffers[1]!, byteLength: 4 * 1_024 * 4 },
    intermediate: { buffer: buffers[2]!, byteLength: 4_096 * 4 },
    projected: { buffer: buffers[3]!, byteLength: 2_560 * 4 },
    uniforms: [4, 5, 6, 7].map((index) => ({ buffer: buffers[index]!, byteLength: 16 })),
  };
  const limits = { minStorageBufferOffsetAlignment: 4, minUniformBufferOffsetAlignment: 4, maxStorageBufferBindingSize: 128 * MIB, maxUniformBufferBindingSize: 16, maxComputeWorkgroupsPerDimension: 16_384 };
  const plans = planQwen35VisionMergerDispatches({ bootstrap, workspace, patchCount: 4, limits });
  assert.deepEqual(plans.map((plan) => plan.kernel.id), ["qwen35-vision-layernorm-f32", "qwen35-vision-bf16-linear-f32", "qwen35-vision-exact-gelu-f32", "qwen35-vision-bf16-linear-f32"]);
  assert.deepEqual(plans.map((plan) => plan.uniformWords), [[4, 1_024, new Uint32Array(new Float32Array([0.000001]).buffer)[0], 0], [1, 4_096, 4_096, 0], [1, 0, 0, 0], [1, 4_096, 2_560, 0]]);
  assert.equal(plans[1]!.bindings[0]!.size, 4 * 1_024 * 4, "mm.0 reads four contiguous normalized patch rows without a shuffle buffer");
  assert.equal(plans[3]!.bindings[3]!.size, 2_560 * 4);
  await bootstrap.destroy(); ledger.assertAllReleased();
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
  // Plan only from the branded, hash-validated object. The workspace buffers
  // are small handles because this test verifies binding assembly, not GPU work.
  const workspaceStorage = (byteLength: number) => ({ buffer: new BufferFake(byteLength), byteLength });
  const workspace = {
    patches: workspaceStorage(4 * 1_536 * 4),
    embeddings: workspaceStorage(4 * 1_024 * 4),
    rope: workspaceStorage(4 * 64 * 4),
    patchUniform: workspaceStorage(16),
    positionUniform: workspaceStorage(16),
    ropeUniform: workspaceStorage(16),
  };
  const limits = {
    minStorageBufferOffsetAlignment: 256,
    minUniformBufferOffsetAlignment: 256,
    maxStorageBufferBindingSize: 16 * MIB,
    maxUniformBufferBindingSize: 256,
    maxComputeWorkgroupsPerDimension: 64,
  };
  const plans = planQwen35VisionBootstrapFoundationDispatches({
    bootstrap: staged,
    workspace,
    gridHeight: 2,
    gridWidth: 2,
    limits,
  });
  assert.equal(plans.length, 3);
  assert.deepEqual(plans.map((plan) => plan.workgroups), [
    { x: 16, y: 4, z: 1 },
    { x: 16, y: 4, z: 1 },
    { x: 1, y: 4, z: 1 },
  ]);
  assert.deepEqual(plans.map((plan) => plan.uniformWords), [
    [4, 0, 0, 0],
    [4, 2, 2, 1_564_672],
    [4, 2, 2, 0],
  ]);
  assert.deepEqual(plans[0]!.bindings.map((binding) => binding.size), [
    24_576, 3_145_728, 3_145_728, 4_096, 16_384, 16,
  ]);
  const position = staged.tensors.find((entry) => entry.name === "v.position_embd.weight")!;
  assert.deepEqual(plans[1]!.bindings.slice(1, 3).map((binding) => ({
    buffer: binding.buffer,
    offset: binding.offset,
    size: binding.size,
  })), position.segments.map((segment) => ({
    buffer: segment.buffer,
    offset: segment.bufferOffset,
    size: segment.byteLength,
  })));
  assert.throws(() => planQwen35VisionBootstrapFoundationDispatches({
    bootstrap: staged,
    workspace,
    gridHeight: 2,
    gridWidth: 2,
    limits: { ...limits, maxComputeWorkgroupsPerDimension: 15 },
  }), { code: "vision-foundation-dispatch-invalid" });
  await staged.destroy();
  assert.throws(
    () => assertAuthenticatedQwen35VisionGpuStagedGroup(staged),
    { code: "vision-stage-group-unauthenticated" },
  );
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

test("plans one authenticated staged vision layer with planar QKV and fixed residual order", async () => {
  const input = fixture();
  const fake = gpuFakes();
  const ledger = new AllocationLedger(32n * BigInt(MIB));
  const staged = await stageQwen35VisionGpuGroup({
    package: input.package_, program: input.program, layer: 0, ledger,
    allocator: fake.allocator, queue: fake.queue, allocationId: "vision-layer-plan-test",
  });
  const storage = (byteLength: number) => ({ buffer: new BufferFake(byteLength), byteLength });
  const hidden = 4 * 1_024 * 4;
  const workspace = {
    hidden: storage(hidden), normalized: storage(hidden), qkv: storage(hidden * 3), attention: storage(hidden),
    mlp: storage(4 * 4_096 * 4), rope: storage(4 * 64 * 4), segmentOffsets: storage(8),
    uniforms: Array.from({ length: 10 }, () => storage(16)),
  };
  const limits = {
    minStorageBufferOffsetAlignment: 256, minUniformBufferOffsetAlignment: 256,
    maxStorageBufferBindingSize: 16 * MIB, maxUniformBufferBindingSize: 256,
    maxComputeWorkgroupsPerDimension: 64,
  };
  const plan = planQwen35VisionLayerDispatches({
    layer: 0,
    staged,
    workspace,
    tokenCount: 4,
    segmentCount: 1,
    limits,
  });
  assert.deepEqual(plan.map((entry) => entry.kernel.id), [
    "qwen35-vision-layernorm-f32", "qwen35-vision-qkv-bf16-linear-f32", "qwen35-vision-apply-2d-rope-f32",
    "qwen35-vision-online-attention-f32", "qwen35-vision-bf16-linear-f32", "qwen35-vision-residual-add-f32",
    "qwen35-vision-layernorm-f32", "qwen35-vision-bf16-linear-f32", "qwen35-vision-tanh-gelu-f32",
    "qwen35-vision-bf16-linear-f32", "qwen35-vision-residual-add-f32",
  ]);
  assert.deepEqual(plan.map((entry) => entry.workgroups), [
    { x: 1, y: 4, z: 1 }, { x: 48, y: 4, z: 1 }, { x: 1, y: 4, z: 16 }, { x: 1, y: 4, z: 16 },
    { x: 16, y: 4, z: 1 }, { x: 16, y: 4, z: 1 }, { x: 1, y: 4, z: 1 }, { x: 64, y: 4, z: 1 },
    { x: 64, y: 4, z: 1 }, { x: 16, y: 4, z: 1 }, { x: 16, y: 4, z: 1 },
  ]);
  assert.deepEqual(plan[2]!.bindings.slice(0, 2).map((binding) => binding.offset), [0, hidden]);
  assert.deepEqual(plan[3]!.bindings.slice(0, 3).map((binding) => binding.offset), [0, hidden, hidden * 2]);
  assert.deepEqual(plan.map((entry) => entry.uniformWords), [
    [4, 1_024, 897_988_541, 0], [4, 0, 0, 0], [4, 0, 0, 0], [4, 1, 0, 0], [4, 1_024, 1_024, 0],
    [4_096, 0, 0, 0], [4, 1_024, 897_988_541, 0], [4, 1_024, 4_096, 0], [16_384, 0, 0, 0],
    [4, 4_096, 1_024, 0], [4_096, 0, 0, 0],
  ]);
  assert.throws(() => planQwen35VisionLayerDispatches({
    layer: 0, staged, workspace, tokenCount: 4, segmentCount: 1,
    limits: { ...limits, maxComputeWorkgroupsPerDimension: 15 },
  }), { code: "vision-layer-dispatch-invalid" });
  const aliasedUniforms = [...workspace.uniforms];
  aliasedUniforms[1] = aliasedUniforms[0]!;
  assert.throws(() => planQwen35VisionLayerDispatches({
    layer: 0, staged, tokenCount: 4, segmentCount: 1,
    workspace: { ...workspace, uniforms: aliasedUniforms }, limits,
  }), { code: "vision-layer-uniform-alias-invalid" });
  const boundaryTokens = 16_384;
  const boundaryHidden = boundaryTokens * 1_024 * 4;
  const boundaryMlp = boundaryTokens * 4_096 * 4;
  const boundaryWorkspace = {
    hidden: storage(boundaryHidden), normalized: storage(boundaryHidden), qkv: storage(boundaryHidden * 3),
    attention: storage(boundaryHidden), mlp: storage(boundaryMlp), rope: storage(boundaryTokens * 64 * 4),
    segmentOffsets: storage(8), uniforms: Array.from({ length: 10 }, () => storage(16)),
  };
  const boundaryLimits = {
    ...limits, maxStorageBufferBindingSize: 256 * MIB, maxComputeWorkgroupsPerDimension: boundaryTokens,
  };
  const boundaryPlan = planQwen35VisionLayerDispatches({
    layer: 0, staged, workspace: boundaryWorkspace, tokenCount: boundaryTokens, segmentCount: 1, limits: boundaryLimits,
  });
  assert.equal(boundaryPlan[7]!.bindings[3]!.size, boundaryMlp);
  assert.equal(boundaryMlp, 256 * MIB);
  assert.throws(() => planQwen35VisionLayerDispatches({
    layer: 0, staged, workspace: boundaryWorkspace, tokenCount: boundaryTokens, segmentCount: 1,
    limits: { ...boundaryLimits, maxStorageBufferBindingSize: boundaryMlp - 1 },
  }), { code: "vision-layer-binding-invalid" });
  await staged.destroy();
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
