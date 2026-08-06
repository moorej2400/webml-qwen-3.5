import assert from "node:assert/strict";
import test from "node:test";

import { AllocationLedger } from "../src/allocation-ledger.js";
import {
  GpuArena,
  type GpuAllocationRequest,
  type GpuBufferLike,
} from "../src/gpu-arena.js";
import type {
  CachedModelPackage,
  ModelCacheStorage,
} from "../src/opfs-model-cache.js";
import { QWEN35_PRODUCT_CONTEXT_CAP } from "../src/qwen35-config.js";
import {
  cleanupQwen35GpuResources,
  qwen35AllocatedWeightBytes,
} from "../src/qwen35-model-loader.js";
import {
  allocateQwen35WeightDirectory,
  type Qwen35PackageDirectory,
  type Qwen35PackageTensor,
} from "../src/qwen35-weight-directory.js";
import {
  initializeQwen35WeightExecution,
  uploadQwen35CachedWeights,
  type Qwen35WeightWriteQueue,
} from "../src/qwen35-weight-upload.js";

const SUBJECT_PATH = "../src/qwen35-disk-backed-tied-embedding.js";
const ROW_BYTES = 2_120;
const INPUT_CACHE_ROWS = 2;
const OUTPUT_TILE_ROWS = 2;
const CACHE_BYTES = ROW_BYTES * (INPUT_CACHE_ROWS + OUTPUT_TILE_ROWS * 2);
const EXACT_TIED_BYTES = 526_438_400n;

interface PackedRangeRead {
  readonly storagePath: string;
  readonly offset: number;
  readonly byteLength: number;
  readonly signal: AbortSignal;
}

interface PackedRangeReader {
  read(input: PackedRangeRead): Promise<Uint8Array>;
}

interface StagedPackedRows {
  readonly tensorName: "token_embd.weight";
  readonly storageType: string;
  readonly firstRow: number;
  readonly rowCount: number;
  readonly rowBytes: number;
  readonly buffer: object;
  readonly bufferOffset: number;
  readonly byteLength: number;
}

interface LogitCandidate {
  readonly tokenId: number;
  readonly score: number;
}

interface TiedEmbeddingMetrics {
  readonly logicalTensorBytes: number;
  readonly permanentGpuBytes: number;
  readonly currentCacheGpuBytes: number;
  readonly peakCacheGpuBytes: number;
  readonly diskReadBytes: number;
  readonly maxDiskReadBytes: number;
  readonly inputRowCacheHits: number;
  readonly inputRowCacheMisses: number;
  readonly prefillRowRequests: number;
  readonly decodeRowRequests: number;
  readonly outputTileReads: number;
}

interface DiskBackedTiedEmbeddingStore {
  readonly tensor: Qwen35PackageTensor;
  readonly permanentGpuBytes: 0n;
  readonly cacheGpuBytes: bigint;
  stageInputRow(input: {
    readonly tokenId: number;
    readonly phase: "prefill" | "decode";
    readonly signal: AbortSignal;
  }): Promise<StagedPackedRows>;
  selectTopK(input: {
    readonly phase: "prefill" | "decode";
    readonly topK: number;
    readonly signal: AbortSignal;
    readonly scoreTile: (
      tile: StagedPackedRows,
    ) => Promise<readonly LogitCandidate[]> | readonly LogitCandidate[];
  }): Promise<readonly LogitCandidate[]>;
  selectTopKGpu(input: {
    readonly phase: "prefill" | "decode";
    readonly signal: AbortSignal;
    readonly scoreTile: (
      tile: StagedPackedRows,
      candidateSlot: number,
    ) => Promise<void> | void;
    readonly flush: () => Promise<void> | void;
    readonly finalize: () => Promise<number> | number;
  }): Promise<number>;
  cancel(): Promise<void>;
  dispose(): Promise<void>;
  getMetrics(): TiedEmbeddingMetrics;
}

interface WeightResidencyPlan {
  readonly permanentDirectory: Qwen35PackageDirectory;
  readonly tiedTensor: Qwen35PackageTensor;
  readonly permanentBytes: bigint;
  readonly streamedBytes: bigint;
}

interface SubjectModule {
  planQwen35TiedEmbeddingResidency(
    packageDirectory: Qwen35PackageDirectory,
  ): WeightResidencyPlan;
  createQwen35DiskBackedTiedEmbeddingStore(input: {
    readonly arena: GpuArena;
    readonly queue: Qwen35WeightWriteQueue;
    readonly packageDirectory: Qwen35PackageDirectory;
    readonly cached: CachedModelPackage;
    readonly rangeReader: PackedRangeReader;
    readonly inputRowCapacity: number;
    readonly outputTileRows: number;
    readonly decodableRows: number;
  }): Promise<DiskBackedTiedEmbeddingStore>;
}

async function subject(): Promise<SubjectModule> {
  try {
    return await import(SUBJECT_PATH) as SubjectModule;
  } catch (error) {
    if (
      (error as { readonly code?: unknown }).code === "ERR_MODULE_NOT_FOUND" &&
      (error as Error).message.includes("qwen35-disk-backed-tied-embedding")
    ) {
      assert.fail("disk-backed tied embedding module is not implemented");
    }
    throw error;
  }
}

function exactPackageDirectory(): Qwen35PackageDirectory {
  const shardLengths = [
    134_215_168,
    134_217_248,
    134_216_576,
    134_217_024,
    134_213_440,
    134_216_960,
    134_217_632,
    134_216_800,
    134_217_408,
    134_216_736,
    134_216_808,
    134_216_408,
    134_213_248,
    134_213_184,
    134_217_632,
    134_215_776,
    134_217_200,
    134_217_200,
    134_217_200,
    5_244_880,
  ];
  let packageOffset = 0;
  const shards = shardLengths.map((length, index) => {
    const shard = {
      index,
      url: `shards/model-${index.toString().padStart(5, "0")}.bin`,
      offset: String(packageOffset),
      length: String(length),
      sha256: index.toString(16).padStart(64, "0"),
    };
    packageOffset += length;
    return shard;
  });
  return {
    manifestSha256: "f".repeat(64),
    shards,
    tensors: [
      {
        name: "output_norm.weight",
        shape: [2_560],
        ggmlType: 0,
        storageType: "f32",
        segments: [{
          shardIndex: 15,
          shardOffset: "15663616",
          tensorOffset: "0",
          length: "10240",
        }],
      },
      {
        name: "token_embd.weight",
        shape: [2_560, 248_320],
        ggmlType: 14,
        storageType: "q6-k-212",
        segments: [
          {
            shardIndex: 15,
            shardOffset: "15673856",
            tensorOffset: "0",
            length: "118541920",
          },
          {
            shardIndex: 16,
            shardOffset: "0",
            tensorOffset: "118541920",
            length: "134217200",
          },
          {
            shardIndex: 17,
            shardOffset: "0",
            tensorOffset: "252759120",
            length: "134217200",
          },
          {
            shardIndex: 18,
            shardOffset: "0",
            tensorOffset: "386976320",
            length: "134217200",
          },
          {
            shardIndex: 19,
            shardOffset: "0",
            tensorOffset: "521193520",
            length: "5244880",
          },
        ],
      },
    ],
  };
}

interface SmallFixture {
  readonly packageDirectory: Qwen35PackageDirectory;
  readonly cached: CachedModelPackage;
  readonly bytesByPath: Readonly<Record<string, Uint8Array>>;
  readonly tensorBytes: Uint8Array;
}

function smallFixture(): SmallFixture {
  const scores = [1, 9, 9, 7, 11, 11];
  const tensorBytes = new Uint8Array(ROW_BYTES * scores.length);
  for (const [row, score] of scores.entries()) {
    const bytes = tensorBytes.subarray(row * ROW_BYTES, (row + 1) * ROW_BYTES);
    for (let index = 0; index < bytes.byteLength; index += 1) {
      bytes[index] = (row * 37 + index * 13 + 5) & 0xff;
    }
    bytes[0] = score;
  }
  // The synthetic split cuts through row 2 so range assembly cannot assume
  // converter segments and OPFS files always share packed-row boundaries.
  const firstLength = ROW_BYTES * 2 + 500;
  const first = tensorBytes.slice(0, firstLength);
  const second = tensorBytes.slice(firstLength);
  const norm = Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8);
  return {
    packageDirectory: {
      manifestSha256: "a".repeat(64),
      shards: [
        { index: 0, url: "part-0.bin", offset: "0", length: String(first.byteLength), sha256: "b".repeat(64) },
        { index: 1, url: "part-1.bin", offset: String(first.byteLength), length: String(second.byteLength), sha256: "c".repeat(64) },
        { index: 2, url: "part-2.bin", offset: String(tensorBytes.byteLength), length: String(norm.byteLength), sha256: "d".repeat(64) },
      ],
      tensors: [
        {
          name: "token_embd.weight",
          shape: [2_560, scores.length],
          ggmlType: 14,
          storageType: "q6-k-212",
          segments: [
            { shardIndex: 0, shardOffset: "0", tensorOffset: "0", length: String(first.byteLength) },
            { shardIndex: 1, shardOffset: "0", tensorOffset: String(first.byteLength), length: String(second.byteLength) },
          ],
        },
        {
          name: "output_norm.weight",
          shape: [2],
          ggmlType: 0,
          storageType: "f32",
          segments: [{ shardIndex: 2, shardOffset: "0", tensorOffset: "0", length: String(norm.byteLength) }],
        },
      ],
    },
    cached: {
      cacheKey: "cache",
      manifestSha256: "a".repeat(64),
      cacheHit: true,
      shards: [
        { storagePath: "blobs/part-0.bin", byteLength: first.byteLength, sha256: "b".repeat(64) },
        { storagePath: "blobs/part-1.bin", byteLength: second.byteLength, sha256: "c".repeat(64) },
        { storagePath: "blobs/part-2.bin", byteLength: norm.byteLength, sha256: "d".repeat(64) },
      ],
    },
    bytesByPath: {
      "blobs/part-0.bin": first,
      "blobs/part-1.bin": second,
      "blobs/part-2.bin": norm,
    },
    tensorBytes,
  };
}

function alignedSmallFixture(): SmallFixture {
  const source = smallFixture();
  const first = source.tensorBytes.slice(0, ROW_BYTES * 2);
  const second = source.tensorBytes.slice(ROW_BYTES * 2);
  const norm = source.bytesByPath["blobs/part-2.bin"]!;
  const packageDirectory: Qwen35PackageDirectory = {
    ...source.packageDirectory,
    shards: [
      { index: 0, url: "part-0.bin", offset: "0", length: String(first.byteLength), sha256: "b".repeat(64) },
      { index: 1, url: "part-1.bin", offset: String(first.byteLength), length: String(second.byteLength), sha256: "c".repeat(64) },
      { index: 2, url: "part-2.bin", offset: String(source.tensorBytes.byteLength), length: String(norm.byteLength), sha256: "d".repeat(64) },
    ],
    tensors: source.packageDirectory.tensors.map((tensor) => tensor.name === "token_embd.weight"
      ? {
          ...tensor,
          segments: [
            { shardIndex: 0, shardOffset: "0", tensorOffset: "0", length: String(first.byteLength) },
            { shardIndex: 1, shardOffset: "0", tensorOffset: String(first.byteLength), length: String(second.byteLength) },
          ],
        }
      : tensor),
  };
  return {
    packageDirectory,
    cached: {
      ...source.cached,
      shards: [
        { storagePath: "blobs/part-0.bin", byteLength: first.byteLength, sha256: "b".repeat(64) },
        { storagePath: "blobs/part-1.bin", byteLength: second.byteLength, sha256: "c".repeat(64) },
        { storagePath: "blobs/part-2.bin", byteLength: norm.byteLength, sha256: "d".repeat(64) },
      ],
    },
    bytesByPath: {
      "blobs/part-0.bin": first,
      "blobs/part-1.bin": second,
      "blobs/part-2.bin": norm,
    },
    tensorBytes: source.tensorBytes,
  };
}

class MemoryGpuBuffer implements GpuBufferLike {
  readonly bytes: Uint8Array;
  destroyed = false;

  constructor(byteLength: number, private readonly events: string[]) {
    this.bytes = new Uint8Array(byteLength);
  }

  destroy(): void {
    this.destroyed = true;
    this.events.push("destroy");
  }
}

function gpuFixture(options: { readonly failAllocation?: number } = {}): {
  readonly ledger: AllocationLedger;
  readonly arena: GpuArena;
  readonly queue: Qwen35WeightWriteQueue;
  readonly requests: GpuAllocationRequest[];
  readonly buffers: MemoryGpuBuffer[];
  readonly events: string[];
} {
  const ledger = new AllocationLedger(64n * 1024n * 1024n);
  const requests: GpuAllocationRequest[] = [];
  const buffers: MemoryGpuBuffer[] = [];
  const events: string[] = [];
  let allocation = 0;
  const arena = new GpuArena({
    limits: {
      maxBufferSize: 64 * 1024 * 1024,
      maxStorageBufferBindingSize: 64 * 1024 * 1024,
    },
    pushErrorScope() {},
    async popErrorScope() { return null; },
    createBuffer(descriptor) {
      allocation += 1;
      if (allocation === options.failAllocation) {
        throw new Error("private device allocation failure");
      }
      const buffer = new MemoryGpuBuffer(descriptor.size, events);
      buffers.push(buffer);
      return buffer;
    },
  }, ledger, { bufferShardCapBytes: 64n * 1024n * 1024n });
  const queue: Qwen35WeightWriteQueue = {
    writeBuffer(buffer, bufferOffset, data, dataOffset = 0, size) {
      const byteLength = size ?? data.byteLength - dataOffset;
      (buffer as MemoryGpuBuffer).bytes.set(
        new Uint8Array(data.buffer, data.byteOffset + dataOffset, byteLength),
        bufferOffset,
      );
      events.push(`write:${byteLength}`);
    },
    async onSubmittedWorkDone() {
      events.push("retire");
    },
  };
  const originalAllocate = arena.allocate.bind(arena);
  arena.allocate = async (request) => {
    requests.push(request);
    return originalAllocate(request);
  };
  return { ledger, arena, queue, requests, buffers, events };
}

function rangeReader(
  bytesByPath: Readonly<Record<string, Uint8Array>>,
  reads: Array<Omit<PackedRangeRead, "signal">>,
  options: {
    readonly failWith?: Error;
    readonly blockFirstUntilAbort?: boolean;
  } = {},
): PackedRangeReader {
  let readCount = 0;
  return {
    async read(input) {
      readCount += 1;
      reads.push({
        storagePath: input.storagePath,
        offset: input.offset,
        byteLength: input.byteLength,
      });
      input.signal.throwIfAborted();
      if (options.blockFirstUntilAbort === true && readCount === 1) {
        return await new Promise<Uint8Array>((_resolve, reject) => {
          input.signal.addEventListener("abort", () => reject(input.signal.reason), { once: true });
        });
      }
      if (options.failWith !== undefined) throw options.failWith;
      const source = bytesByPath[input.storagePath];
      if (
        source === undefined ||
        input.offset < 0 ||
        input.byteLength < 1 ||
        input.offset + input.byteLength > source.byteLength
      ) {
        throw new Error("range outside immutable fixture");
      }
      return source.slice(input.offset, input.offset + input.byteLength);
    },
  };
}

async function createStore(input: {
  readonly fixture?: SmallFixture;
  readonly gpu?: ReturnType<typeof gpuFixture>;
  readonly reader?: PackedRangeReader;
  readonly reads?: Array<Omit<PackedRangeRead, "signal">>;
} = {}): Promise<{
  readonly store: DiskBackedTiedEmbeddingStore;
  readonly fixture: SmallFixture;
  readonly gpu: ReturnType<typeof gpuFixture>;
  readonly reads: Array<Omit<PackedRangeRead, "signal">>;
}> {
  const fixture = input.fixture ?? smallFixture();
  const gpu = input.gpu ?? gpuFixture();
  const reads = input.reads ?? [];
  const module = await subject();
  const store = await module.createQwen35DiskBackedTiedEmbeddingStore({
    arena: gpu.arena,
    queue: gpu.queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: input.reader ?? rangeReader(fixture.bytesByPath, reads),
    inputRowCapacity: INPUT_CACHE_ROWS,
    outputTileRows: OUTPUT_TILE_ROWS,
    decodableRows: 6,
  });
  return { store, fixture, gpu, reads };
}

function stableTopK(
  candidates: readonly LogitCandidate[],
  topK: number,
): readonly LogitCandidate[] {
  return [...candidates]
    .filter((candidate) => Number.isFinite(candidate.score))
    .sort((left, right) => right.score - left.score || left.tokenId - right.tokenId)
    .slice(0, topK);
}

function memoryStorage(
  bytesByPath: Readonly<Record<string, Uint8Array>>,
): ModelCacheStorage {
  return {
    async openAtomicWriter() { throw new Error("not used"); },
    async openRead(path) {
      const bytes = bytesByPath[path];
      if (bytes === undefined) return null;
      return (async function* () { yield bytes; })();
    },
    async move() { return false; },
    async list() { return []; },
  };
}

test("partitions the exact five-segment tied tensor out of permanent GPU residency", async () => {
  const module = await subject();
  const packageDirectory = exactPackageDirectory();
  const snapshot = structuredClone(packageDirectory);

  const plan = module.planQwen35TiedEmbeddingResidency(packageDirectory);

  assert.equal(plan.tiedTensor.name, "token_embd.weight");
  assert.equal(plan.tiedTensor.storageType, "q6-k-212");
  assert.equal(plan.tiedTensor.segments.length, 5);
  assert.equal(
    plan.tiedTensor.segments.reduce((sum, segment) => sum + BigInt(segment.length), 0n),
    EXACT_TIED_BYTES,
  );
  assert.equal(plan.streamedBytes, EXACT_TIED_BYTES);
  assert.equal(plan.permanentBytes, 10_240n);
  assert.equal(plan.permanentDirectory.tensors.some(({ name }) => name === "token_embd.weight"), false);
  assert.deepEqual(packageDirectory, snapshot);
  assert.equal(Object.isFrozen(plan), true);
  assert.equal(Object.isFrozen(plan.tiedTensor.segments), true);
  assert.equal(QWEN35_PRODUCT_CONTEXT_CAP, 16_384);
});

test("loader weight accounting excludes the exact disk-backed tied tensor", () => {
  assert.equal(qwen35AllocatedWeightBytes(exactPackageDirectory()), 10_240n);
});

test("allocates and uploads only the permanent partition", async () => {
  const module = await subject();
  const fixture = smallFixture();
  const plan = module.planQwen35TiedEmbeddingResidency(fixture.packageDirectory);
  const gpu = gpuFixture();
  const directory = await allocateQwen35WeightDirectory(gpu.arena, plan.permanentDirectory);

  await uploadQwen35CachedWeights({
    storage: memoryStorage(fixture.bytesByPath),
    cached: fixture.cached,
    directory,
    queue: gpu.queue,
    uploadLaneBytes: 4 * 1024,
    signal: new AbortController().signal,
  });

  assert.deepEqual(gpu.requests.map(({ id, byteLength }) => ({ id, byteLength })), [
    { id: "model-tensor-0", byteLength: 8n },
  ]);
  assert.equal(directory.get("token_embd.weight"), undefined);
  assert.equal(directory.logicalBytes, 8n);
  assert.equal(
    gpu.events.filter((event) => event.startsWith("write:")).reduce(
      (sum, event) => sum + Number(event.slice("write:".length)),
      0,
    ),
    8,
  );
  directory.destroy();
});

test("weight execution partitions before allocation, upload, and driver creation", async () => {
  const fixture = alignedSmallFixture();
  const gpu = gpuFixture();
  const initialized = await initializeQwen35WeightExecution({
    arena: gpu.arena,
    packageDirectory: fixture.packageDirectory,
    storage: memoryStorage(fixture.bytesByPath),
    cached: fixture.cached,
    queue: gpu.queue,
    uploadLaneBytes: 4 * 1024,
    signal: new AbortController().signal,
    async createDriver(directory) {
      assert.equal(directory.get("token_embd.weight"), undefined);
      assert.equal(directory.logicalBytes, 8n);
      return { marker: "resident-only" };
    },
  });
  try {
    assert.equal(initialized.driver.marker, "resident-only");
    assert.deepEqual(gpu.requests.map(({ byteLength }) => byteLength), [8n]);
    assert.equal(
      gpu.events.filter((event) => event.startsWith("write:")).reduce(
        (sum, event) => sum + Number(event.slice("write:".length)),
        0,
      ),
      8,
    );
  } finally {
    initialized.directory.destroy();
  }
});

test("reuses packed input rows across repeated prefill and decode requests", async () => {
  const { store, fixture, gpu, reads } = await createStore();
  const signal = new AbortController().signal;

  const first = await store.stageInputRow({ tokenId: 1, phase: "prefill", signal });
  const crossing = await store.stageInputRow({ tokenId: 2, phase: "prefill", signal });
  const repeatedPrefill = await store.stageInputRow({ tokenId: 1, phase: "prefill", signal });
  const repeatedDecode = await store.stageInputRow({ tokenId: 2, phase: "decode", signal });

  assert.deepEqual(
    new Uint8Array((crossing.buffer as MemoryGpuBuffer).bytes.buffer, crossing.bufferOffset, ROW_BYTES),
    fixture.tensorBytes.subarray(ROW_BYTES * 2, ROW_BYTES * 3),
  );
  assert.equal(first.buffer, repeatedPrefill.buffer);
  assert.equal(first.bufferOffset, repeatedPrefill.bufferOffset);
  assert.equal(crossing.buffer, repeatedDecode.buffer);
  assert.equal(crossing.bufferOffset, repeatedDecode.bufferOffset);
  assert.deepEqual(reads, [
    { storagePath: "blobs/part-0.bin", offset: ROW_BYTES, byteLength: ROW_BYTES },
    { storagePath: "blobs/part-0.bin", offset: ROW_BYTES * 2, byteLength: 500 },
    { storagePath: "blobs/part-1.bin", offset: 0, byteLength: ROW_BYTES - 500 },
  ]);
  assert.deepEqual(store.getMetrics(), {
    logicalTensorBytes: ROW_BYTES * 6,
    permanentGpuBytes: 0,
    currentCacheGpuBytes: CACHE_BYTES,
    peakCacheGpuBytes: CACHE_BYTES,
    diskReadBytes: ROW_BYTES * 2,
    maxDiskReadBytes: ROW_BYTES,
    inputRowCacheHits: 2,
    inputRowCacheMisses: 2,
    prefillRowRequests: 3,
    decodeRowRequests: 1,
    outputTileReads: 0,
  });
  assert.equal(gpu.events.filter((event) => event === `write:${ROW_BYTES}`).length, 2);
  await store.dispose();
});

test("matches full-resident stable greedy and top-k while reading bounded output tiles", async () => {
  const { store, fixture, reads } = await createStore();
  const fullCandidates = Array.from({ length: 6 }, (_, tokenId) => ({
    tokenId,
    score: fixture.tensorBytes[tokenId * ROW_BYTES]!,
  }));
  const scoreTile = (tile: StagedPackedRows): readonly LogitCandidate[] => {
    const buffer = tile.buffer as MemoryGpuBuffer;
    return Array.from({ length: tile.rowCount }, (_, localRow) => ({
      tokenId: tile.firstRow + localRow,
      score: buffer.bytes[tile.bufferOffset + localRow * tile.rowBytes]!,
    }));
  };
  const signal = new AbortController().signal;

  const greedy = await store.selectTopK({ phase: "decode", topK: 1, signal, scoreTile });
  const topThree = await store.selectTopK({ phase: "decode", topK: 3, signal, scoreTile });

  assert.deepEqual(greedy, stableTopK(fullCandidates, 1));
  assert.deepEqual(topThree, stableTopK(fullCandidates, 3));
  assert.deepEqual(topThree, [
    { tokenId: 4, score: 11 },
    { tokenId: 5, score: 11 },
    { tokenId: 1, score: 9 },
  ]);
  assert.equal(Math.max(...reads.map(({ byteLength }) => byteLength)), ROW_BYTES * OUTPUT_TILE_ROWS);
  assert.equal(reads.some(({ byteLength }) => byteLength === fixture.tensorBytes.byteLength), false);
  assert.equal(store.getMetrics().outputTileReads, 6);
  assert.equal(store.getMetrics().maxDiskReadBytes, ROW_BYTES * OUTPUT_TILE_ROWS);
  await store.dispose();
});

test("keeps streamed tile winners on GPU until one final selection", async () => {
  const { store, gpu } = await createStore();
  const slots: number[] = [];
  const flushes: number[] = [];
  const selected = await store.selectTopKGpu({
    phase: "decode",
    signal: new AbortController().signal,
    scoreTile(_tile, candidateSlot) {
      slots.push(candidateSlot);
    },
    flush() {
      flushes.push(slots.length);
    },
    finalize() {
      return 4;
    },
  });

  assert.equal(selected, 4);
  assert.deepEqual(slots, [0, 1, 2]);
  assert.deepEqual(flushes, [2, 3]);
  assert.equal(gpu.events.filter((event) => event === "retire").length, 2);
  await store.dispose();
});

test("uses one immutable tied source for input rows and output tiles", async () => {
  const { store, fixture } = await createStore();
  const signal = new AbortController().signal;
  const input = await store.stageInputRow({ tokenId: 4, phase: "decode", signal });
  const packedInput = (input.buffer as MemoryGpuBuffer).bytes.slice(
    input.bufferOffset,
    input.bufferOffset + input.byteLength,
  );
  let outputRow: Uint8Array | undefined;

  await store.selectTopK({
    phase: "decode",
    topK: 1,
    signal,
    scoreTile(tile) {
      if (tile.firstRow <= 4 && 4 < tile.firstRow + tile.rowCount) {
        const local = 4 - tile.firstRow;
        outputRow = (tile.buffer as MemoryGpuBuffer).bytes.slice(
          tile.bufferOffset + local * tile.rowBytes,
          tile.bufferOffset + (local + 1) * tile.rowBytes,
        );
      }
      return Array.from({ length: tile.rowCount }, (_, localRow) => ({
        tokenId: tile.firstRow + localRow,
        score: localRow,
      }));
    },
  });

  assert.equal(store.tensor.name, "token_embd.weight");
  assert.equal(store.tensor.storageType, "q6-k-212");
  assert.equal(store.permanentGpuBytes, 0n);
  assert.equal(store.cacheGpuBytes, BigInt(CACHE_BYTES));
  assert.deepEqual(packedInput, fixture.tensorBytes.subarray(ROW_BYTES * 4, ROW_BYTES * 5));
  assert.deepEqual(outputRow, packedInput);
  await store.dispose();
});

test("cancels an in-flight range read without publishing a cache hit and remains reusable", async () => {
  const fixture = smallFixture();
  const reads: Array<Omit<PackedRangeRead, "signal">> = [];
  const gpu = gpuFixture();
  const blockedReader = rangeReader(fixture.bytesByPath, reads, { blockFirstUntilAbort: true });
  const { store } = await createStore({ fixture, gpu, reader: blockedReader, reads });
  const pending = store.stageInputRow({
    tokenId: 0,
    phase: "prefill",
    signal: new AbortController().signal,
  });
  await new Promise((resolve) => setImmediate(resolve));

  await store.cancel();
  await assert.rejects(pending, { name: "AbortError" });
  assert.equal(store.getMetrics().inputRowCacheHits, 0);
  assert.equal(store.getMetrics().inputRowCacheMisses, 1);

  const resumed = await store.stageInputRow({
    tokenId: 0,
    phase: "decode",
    signal: new AbortController().signal,
  });
  assert.deepEqual(
    (resumed.buffer as MemoryGpuBuffer).bytes.slice(
      resumed.bufferOffset,
      resumed.bufferOffset + resumed.byteLength,
    ),
    fixture.tensorBytes.subarray(0, ROW_BYTES),
  );
  assert.equal(store.getMetrics().inputRowCacheHits, 0);
  assert.equal(store.getMetrics().inputRowCacheMisses, 2);
  assert.equal(reads.length, 2);
  await store.dispose();
  assert.equal(gpu.ledger.snapshot().currentBytes, 0n);
});

test("dispose retires queue work, releases ledger ownership, and is idempotent", async () => {
  const { store, gpu } = await createStore();
  await store.stageInputRow({
    tokenId: 0,
    phase: "decode",
    signal: new AbortController().signal,
  });

  await store.dispose();
  await store.dispose();

  assert.equal(gpu.ledger.snapshot().currentBytes, 0n);
  assert.equal(gpu.ledger.snapshot().peakBytes, BigInt(CACHE_BYTES));
  assert.equal(
    gpu.events.filter((event) => event === "destroy").length,
    1 + 2,
  );
  assert.notEqual(gpu.events.lastIndexOf("retire"), -1);
  assert.ok(gpu.events.lastIndexOf("retire") < gpu.events.indexOf("destroy"));
  assert.equal(store.getMetrics().currentCacheGpuBytes, 0);
  assert.equal(store.getMetrics().peakCacheGpuBytes, CACHE_BYTES);
  await assert.rejects(
    store.stageInputRow({
      tokenId: 0,
      phase: "decode",
      signal: new AbortController().signal,
    }),
    { code: "tied-embedding-disposed" },
  );
});

test("allocation rollback releases the first cache buffer when the second allocation fails", async () => {
  const fixture = smallFixture();
  const gpu = gpuFixture({ failAllocation: 2 });
  const module = await subject();

  await assert.rejects(
    module.createQwen35DiskBackedTiedEmbeddingStore({
      arena: gpu.arena,
      queue: gpu.queue,
      packageDirectory: fixture.packageDirectory,
      cached: fixture.cached,
      rangeReader: rangeReader(fixture.bytesByPath, []),
      inputRowCapacity: INPUT_CACHE_ROWS,
      outputTileRows: OUTPUT_TILE_ROWS,
      decodableRows: 6,
    }),
    { code: "tied-embedding-cache-allocation-failed" },
  );
  assert.equal(gpu.ledger.snapshot().currentBytes, 0n);
  assert.equal(gpu.buffers[0]?.destroyed, true);
});

test("range failures expose only a stable diagnostic and never private OPFS details", async () => {
  const fixture = smallFixture();
  const privateDetail = "SENSITIVE_STORAGE_MARKER device-origin.invalid";
  const { store } = await createStore({
    fixture,
    reader: rangeReader(fixture.bytesByPath, [], { failWith: new Error(privateDetail) }),
  });

  await assert.rejects(
    store.stageInputRow({
      tokenId: 0,
      phase: "decode",
      signal: new AbortController().signal,
    }),
    (error: unknown) => {
      assert.equal((error as { readonly code?: unknown }).code, "tied-embedding-range-read-failed");
      assert.equal((error as Error).message, "Immutable tied embedding range read failed");
      assert.equal((error as Error).message.includes(privateDetail), false);
      assert.equal("cause" in (error as object), false);
      return true;
    },
  );
  assert.equal(JSON.stringify(store.getMetrics()).includes("SENSITIVE_STORAGE_MARKER"), false);
  assert.equal(JSON.stringify(store.getMetrics()).includes("device-origin.invalid"), false);
  await store.dispose();
});

test("loader cleanup releases the tied store after driver quiescence and before device destruction", async () => {
  const events: string[] = [];
  const cleanupInput = {
    driver: { async dispose() { events.push("driver"); } },
    tiedEmbedding: { async dispose() { events.push("tied"); } },
    device: {
      queue: { async onSubmittedWorkDone() { events.push("queue"); } },
      destroy() { events.push("device"); },
    },
    hybridState: { dispose() { events.push("state"); } },
    weightAllocations: [{
      logicalBytes: 1n,
      allocatedBytes: 1n,
      shards: [],
      destroy() { events.push("weight"); },
    }],
    ledger: { assertAllReleased() { events.push("ledger"); } },
  };

  await cleanupQwen35GpuResources(cleanupInput);

  assert.deepEqual(events, [
    "driver",
    "tied",
    "queue",
    "state",
    "weight",
    "device",
    "ledger",
  ]);
});
