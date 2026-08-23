import assert from "node:assert/strict";
import test from "node:test";

import { AllocationLedger } from "../src/allocation-ledger.js";
import { diagnosticError } from "../src/diagnostics.js";
import {
  GpuArena,
  type GpuBufferLike,
} from "../src/gpu-arena.js";
import { planQwen35HybridState } from "../src/hybrid-state.js";
import type {
  CachedModelPackage,
  ModelCacheStorage,
} from "../src/opfs-model-cache.js";
import { planQwen35ActivationWorkspace } from "../src/qwen35-activation-workspace.js";
import {
  QWEN35_4B_CONFIG,
  QWEN35_PRODUCT_CONTEXT_CAP,
} from "../src/qwen35-config.js";
import { qwen35AllocatedWeightBytes } from "../src/qwen35-model-loader.js";
import { QWEN35_DEFAULT_MAX_VISUAL_TOKENS } from "../src/qwen35-vision-preprocess.js";
import {
  planQwen35TiedEmbeddingResidency,
  type Qwen35PackedRangeReader,
} from "../src/qwen35-disk-backed-tied-embedding.js";
import type {
  Qwen35PackageDirectory,
  Qwen35PackageTensor,
  Qwen35WeightDirectory,
  Qwen35WeightDirectoryView,
} from "../src/qwen35-weight-directory.js";
import {
  initializeQwen35WeightExecution,
  type Qwen35WeightWriteQueue,
} from "../src/qwen35-weight-upload.js";

const SUBJECT_PATH = "../src/qwen35-rolling-layer-weights.js";
const STREAMED_LAYER_ORDER = Object.freeze(
  Array.from({ length: 32 }, (_, layer) => layer),
);
const EXACT_STREAMED_BYTES = 2_408_433_664n;
const EXACT_PERMANENT_BYTES = 10_240n;
const EXACT_MAX_LAYER_BYTES = 94_851_840n;
const EXACT_LANGUAGE_BYTES = 2_408_443_904n;
const EXACT_TIED_CACHE_BYTES = 2_785_280n;
const EXACT_CANDIDATE_SCRATCH_BYTES = 8n;
const EXACT_FULL_16K_LEDGER_BYTES = 688_282_636n;

interface RollingLayerResidency {
  readonly layer: number;
  readonly byteLength: bigint;
  readonly directory: Qwen35PackageDirectory;
}

interface RollingLayerResidencyPlan {
  readonly streamedLayers: readonly number[];
  readonly streamedBytes: bigint;
  readonly permanentBytes: bigint;
  readonly maxLayerBytes: bigint;
  readonly permanentDirectory: Qwen35PackageDirectory;
  readonly layers: readonly RollingLayerResidency[];
}

interface RollingLayerMetrics {
  readonly currentGpuBytes: number;
  readonly peakGpuBytes: number;
  readonly diskReadBytes: number;
  readonly maxDiskReadBytes: number;
  readonly completedLayers: number;
  readonly failedLayers: number;
}

interface RollingLayerMutation {
  markStateMutation(): void;
}

interface RollingLayerStore {
  readonly streamedLayers: readonly number[];
  readonly poisoned: boolean;
  withLayer<T>(input: {
    readonly layer: number;
    readonly phase: "prefill" | "decode";
    readonly signal: AbortSignal;
    readonly execute: (
      weights: Qwen35WeightDirectoryView,
      mutation: RollingLayerMutation,
    ) => Promise<T> | T;
  }): Promise<T>;
  cancel(): Promise<void>;
  dispose(): Promise<void>;
  getMetrics(): RollingLayerMetrics;
}

interface LayerInvocation {
  readonly layer: number;
  readonly kind: "gated-deltanet" | "full-attention";
}

interface SubjectModule {
  planQwen35RollingLayerResidency(
    packageDirectory: Qwen35PackageDirectory,
    streamedLayers?: readonly number[],
  ): RollingLayerResidencyPlan;
  createQwen35RollingLayerStore(input: {
    readonly arena: GpuArena;
    readonly queue: Qwen35WeightWriteQueue;
    readonly packageDirectory: Qwen35PackageDirectory;
    readonly cached: CachedModelPackage;
    readonly rangeReader: Qwen35PackedRangeReader;
    readonly uploadLaneBytes: number;
    readonly readChunkBytes: number;
  }): Promise<RollingLayerStore>;
  executeQwen35RollingLayerSequence(input: {
    readonly invocations: readonly LayerInvocation[];
    readonly permanentWeights: Qwen35WeightDirectoryView;
    readonly rollingStore: RollingLayerStore;
    readonly phase: "prefill" | "decode";
    readonly signal: AbortSignal;
    readonly execute: (input: {
      readonly invocation: LayerInvocation;
      readonly weights: Qwen35WeightDirectoryView;
      readonly mutation: RollingLayerMutation;
      readonly transientWeights: boolean;
    }) => Promise<void> | void;
    readonly poison: () => void;
  }): Promise<void>;
}

interface RollingWeightInitialization<T> {
  readonly directory: Qwen35WeightDirectory;
  readonly rollingStore: RollingLayerStore;
  readonly driver: T;
}

type InitializeRollingWeights = <T>(input: {
  readonly arena: GpuArena;
  readonly packageDirectory: Qwen35PackageDirectory;
  readonly storage: ModelCacheStorage;
  readonly cached: CachedModelPackage;
  readonly queue: Qwen35WeightWriteQueue;
  readonly uploadLaneBytes: number;
  readonly signal: AbortSignal;
  readonly createDriver: (
    directory: Qwen35WeightDirectoryView,
    rollingStore: RollingLayerStore,
  ) => Promise<T>;
}) => Promise<RollingWeightInitialization<T>>;

let subjectPromise: Promise<SubjectModule> | undefined;

async function subject(): Promise<SubjectModule> {
  subjectPromise ??= import(SUBJECT_PATH).then(
    (module) => module as SubjectModule,
    (error: unknown) => {
      if (
        (error as { readonly code?: unknown }).code === "ERR_MODULE_NOT_FOUND" &&
        (error as Error).message.includes("qwen35-rolling-layer-weights")
      ) {
        assert.fail("rolling layer weight module is not implemented");
      }
      throw error;
    },
  );
  return subjectPromise;
}

const LAYER_BYTES = Object.freeze([
  79_368_960, 79_368_960, 79_368_960, 72_726_528,
  73_634_560, 73_634_560, 79_368_960, 66_500_608,
  73_634_560, 73_634_560, 79_368_960, 66_500_608,
  73_634_560, 73_634_560, 79_368_960, 72_726_528,
  73_634_560, 73_634_560, 79_368_960, 66_500_608,
  73_634_560, 73_634_560, 79_368_960, 66_500_608,
  73_634_560, 73_634_560, 79_368_960, 72_726_528,
  79_368_960, 79_368_960, 79_368_960, 72_726_528,
]);

const DELTA_LAYER_ZERO_TENSORS = Object.freeze([
  ["attn_gate.weight", 7_864_320, "q3-k-fused-f32-192", 11, [2_560, 4_096]],
  ["attn_norm.weight", 10_240, "f32", 0, [2_560]],
  ["attn_qkv.weight", 18_350_080, "q5-k-fused-f32-224", 13, [2_560, 8_192]],
  ["ffn_down.weight", 20_643_840, "q5-k-fused-f32-224", 13, [9_216, 2_560]],
  ["ffn_gate.weight", 17_694_720, "q3-k-fused-f32-192", 11, [2_560, 9_216]],
  ["ffn_up.weight", 17_694_720, "q3-k-fused-f32-192", 11, [2_560, 9_216]],
  ["post_attention_norm.weight", 10_240, "f32", 0, [2_560]],
  ["ssm_a", 128, "f32", 0, [32]],
  ["ssm_alpha.weight", 327_680, "f32", 0, [2_560, 32]],
  ["ssm_beta.weight", 327_680, "f32", 0, [2_560, 32]],
  ["ssm_conv1d.weight", 131_072, "f32", 0, [4, 8_192]],
  ["ssm_dt.bias", 128, "f32", 0, [32]],
  ["ssm_norm.weight", 512, "f32", 0, [128]],
  ["ssm_out.weight", 11_796_480, "q8-0-36", 8, [4_096, 2_560]],
] as const);

function exactLanguageDirectory(): Qwen35PackageDirectory {
  const tensors: Qwen35PackageTensor[] = [];
  const shards: Array<Qwen35PackageDirectory["shards"][number]> = [];
  let shardIndex = 0;
  let packageOffset = 0;
  const add = (tensor: Omit<Qwen35PackageTensor, "segments">, bytes: number) => {
    const index = shardIndex;
    shardIndex += 1;
    shards.push({
      index,
      url: `shards/part-${index.toString().padStart(5, "0")}.bin`,
      offset: String(packageOffset),
      length: String(bytes),
      sha256: index.toString(16).padStart(64, "0"),
    });
    packageOffset += bytes;
    tensors.push({
      ...tensor,
      segments: [{
        shardIndex: index,
        shardOffset: "0",
        tensorOffset: "0",
        length: String(bytes),
      }],
    });
  };

  for (const [suffix, bytes, storageType, ggmlType, shape] of DELTA_LAYER_ZERO_TENSORS) {
    add({ name: `blk.0.${suffix}`, shape, storageType, ggmlType }, bytes);
  }
  for (let layer = 1; layer < 32; layer += 1) {
    const bytes = LAYER_BYTES[layer]!;
    add({
      name: `blk.${layer}.fixture.weight`,
      shape: [bytes / 4],
      storageType: "f32",
      ggmlType: 0,
    }, bytes);
  }
  add({
    name: "output_norm.weight",
    shape: [2_560],
    storageType: "f32",
    ggmlType: 0,
  }, 10_240);
  return {
    manifestSha256: "f".repeat(64),
    shards,
    // Reverse physical directory order to prove policy is not derived from it.
    tensors: tensors.reverse(),
  };
}

interface SmallFixture {
  readonly packageDirectory: Qwen35PackageDirectory;
  readonly cached: CachedModelPackage;
  readonly bytesByPath: Readonly<Record<string, Uint8Array>>;
}

function smallDirectory(options: { readonly exactDeltaTails?: boolean } = {}): SmallFixture {
  const tensors: Qwen35PackageTensor[] = [];
  const shards: Array<Qwen35PackageDirectory["shards"][number]> = [];
  const cachedShards: CachedModelPackage["shards"][number][] = [];
  const bytesByPath: Record<string, Uint8Array> = {};
  let index = 0;
  const add = (name: string, bytes: Uint8Array) => {
    const storagePath = `cache/part-${index}.bin`;
    const sha256 = (index + 1).toString(16).padStart(64, "0");
    shards.push({
      index,
      url: `shards/part-${index}.bin`,
      offset: "0",
      length: String(bytes.byteLength),
      sha256,
    });
    cachedShards.push({ storagePath, byteLength: bytes.byteLength, sha256 });
    bytesByPath[storagePath] = bytes;
    tensors.push({
      name,
      shape: [bytes.byteLength / 4],
      ggmlType: 0,
      storageType: "f32",
      segments: [{
        shardIndex: index,
        shardOffset: "0",
        tensorOffset: "0",
        length: String(bytes.byteLength),
      }],
    });
    index += 1;
  };

  for (let layer = 0; layer < 32; layer += 1) {
    if (layer === 0 && options.exactDeltaTails === true) {
      add("blk.0.ssm_a", new Uint8Array(128).fill(0xa1));
      add("blk.0.ssm_dt.bias", new Uint8Array(128).fill(0xd7));
      add("blk.0.fixture.weight", new Uint8Array(256).fill(0x40));
    } else {
      add(`blk.${layer}.fixture.weight`, new Uint8Array(16).fill(layer + 1));
    }
  }
  add("output_norm.weight", Uint8Array.of(1, 2, 3, 4));
  return {
    packageDirectory: {
      manifestSha256: "a".repeat(64),
      shards,
      tensors: tensors.reverse(),
    },
    cached: {
      cacheKey: "fixture-cache",
      manifestSha256: "a".repeat(64),
      cacheHit: true,
      shards: cachedShards,
    },
    bytesByPath,
  };
}

class MemoryBuffer implements GpuBufferLike {
  readonly bytes: Uint8Array;
  destroyed = false;

  constructor(
    byteLength: number,
    readonly ordinal: number,
    private readonly events: string[],
    private readonly live: Set<MemoryBuffer>,
  ) {
    this.bytes = new Uint8Array(byteLength);
    this.live.add(this);
    this.events.push(`allocate:${this.ordinal}:${byteLength}`);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.live.delete(this);
    this.events.push(`destroy:${this.ordinal}`);
  }
}

function gpuFixture(): {
  readonly arena: GpuArena;
  readonly ledger: AllocationLedger;
  readonly queue: Qwen35WeightWriteQueue;
  readonly events: string[];
  readonly live: Set<MemoryBuffer>;
} {
  const events: string[] = [];
  const live = new Set<MemoryBuffer>();
  const ledger = new AllocationLedger(1024n * 1024n);
  let ordinal = 0;
  const arena = new GpuArena({
    limits: {
      maxBufferSize: 1024 * 1024,
      maxStorageBufferBindingSize: 1024 * 1024,
    },
    pushErrorScope() {},
    popErrorScope: async () => null,
    createBuffer({ size }) {
      return new MemoryBuffer(size, ordinal++, events, live);
    },
  }, ledger, { bufferShardCapBytes: 1024n * 1024n });
  const queue: Qwen35WeightWriteQueue = {
    writeBuffer(buffer, bufferOffset, data, dataOffset = 0, size = data.byteLength - dataOffset) {
      const target = buffer as MemoryBuffer;
      target.bytes.set(data.subarray(dataOffset, dataOffset + size), bufferOffset);
      events.push(`write:${target.ordinal}:${size}`);
    },
    async onSubmittedWorkDone() {
      events.push("retire");
    },
  };
  return { arena, ledger, queue, events, live };
}

function rangeReader(
  fixture: SmallFixture,
  events: string[],
  options: { readonly failAfterReads?: number } = {},
): Qwen35PackedRangeReader {
  let completedReads = 0;
  return {
    async read(input) {
      events.push(`read:${input.storagePath}:${input.offset}:${input.byteLength}`);
      input.signal.throwIfAborted();
      if (
        options.failAfterReads !== undefined &&
        completedReads >= options.failAfterReads
      ) {
        throw new Error("sensitive-origin-marker sensitive-path-marker");
      }
      const source = fixture.bytesByPath[input.storagePath];
      assert.ok(source, "range must use an authenticated cached shard");
      const result = source.slice(input.offset, input.offset + input.byteLength);
      completedReads += 1;
      return result;
    },
  };
}

function storage(fixture: SmallFixture): ModelCacheStorage {
  return {
    async openAtomicWriter() {
      throw new Error("writes are outside this immutable fixture");
    },
    async openRead(path) {
      const bytes = fixture.bytesByPath[path];
      if (bytes === undefined) return null;
      return (async function* () { yield bytes; })();
    },
    async readRange(path, offset, byteLength, signal) {
      signal.throwIfAborted();
      const bytes = fixture.bytesByPath[path];
      return bytes?.slice(offset, offset + byteLength) ?? null;
    },
    async move() { return false; },
    async list() { return []; },
  };
}

function emptyPermanentWeights(): Qwen35WeightDirectoryView {
  const byName = new Map<string, never>();
  return Object.freeze({
    size: 0,
    tensors: Object.freeze([]),
    logicalBytes: 0n,
    allocatedBytes: 0n,
    get: (name: string) => byName.get(name),
    entries: () => byName.entries(),
    [Symbol.iterator]: () => byName[Symbol.iterator](),
  }) as Qwen35WeightDirectoryView;
}

function invocations(): readonly LayerInvocation[] {
  const full = new Set([3, 7, 11, 15, 19, 23, 27, 31]);
  return Object.freeze(Array.from({ length: 32 }, (_, layer) => Object.freeze({
    layer,
    kind: full.has(layer) ? "full-attention" as const : "gated-deltanet" as const,
  })));
}

test("plans every transformer layer for streaming with only the output norm permanent", async () => {
  const module = await subject();
  const source = exactLanguageDirectory();
  const originalNames = source.tensors.map((tensor) => tensor.name);
  const plan = module.planQwen35RollingLayerResidency(source);

  assert.deepEqual(plan.streamedLayers, STREAMED_LAYER_ORDER);
  assert.equal(plan.streamedBytes, EXACT_STREAMED_BYTES);
  assert.equal(plan.permanentBytes, EXACT_PERMANENT_BYTES);
  assert.equal(plan.maxLayerBytes, EXACT_MAX_LAYER_BYTES);
  assert.equal(plan.streamedBytes + plan.permanentBytes, EXACT_LANGUAGE_BYTES);
  assert.deepEqual(plan.layers.map((layer) => layer.layer), STREAMED_LAYER_ORDER);
  assert.equal(new Set(plan.streamedLayers).size, 32, "a layer cannot execute twice");
  assert.deepEqual(
    plan.layers.map(({ layer, directory }) => ({
      layer,
      tensorLayers: [...new Set(directory.tensors.map((tensor) =>
        Number(/^blk\.(\d+)\./.exec(tensor.name)?.[1])))],
    })),
    STREAMED_LAYER_ORDER.map((layer) => ({ layer, tensorLayers: [layer] })),
    "each rolling partition must own only its matching transformer layer",
  );
  assert.equal(plan.layers[0]!.byteLength, EXACT_MAX_LAYER_BYTES);
  assert.equal(
    plan.layers[0]!.directory.tensors
      .filter((tensor) => tensor.name === "blk.0.ssm_a" || tensor.name === "blk.0.ssm_dt.bias")
      .reduce((sum, tensor) => sum + BigInt(tensor.segments[0]!.length), 0n),
    256n,
    "the two exact 128-byte recurrent tensors must not gain 256-byte padding",
  );
  assert.deepEqual(
    plan.permanentDirectory.tensors.map(({ name }) => name),
    ["output_norm.weight"],
    "no blk.0 through blk.31 tensor may enter permanent allocation or upload",
  );
  assert.deepEqual(source.tensors.map((tensor) => tensor.name), originalNames);
  assert.equal(qwen35AllocatedWeightBytes(source), EXACT_PERMANENT_BYTES);
  assert.equal(
    qwen35AllocatedWeightBytes(source, "resident"),
    EXACT_LANGUAGE_BYTES,
  );
});

test("keeps a leading layer subset resident while streaming only the requested suffix", async () => {
  const module = await subject();
  const source = exactLanguageDirectory();
  const streamedLayers = STREAMED_LAYER_ORDER.slice(24);

  const plan = module.planQwen35RollingLayerResidency(source, streamedLayers);

  const expectedStreamedBytes = LAYER_BYTES
    .slice(24)
    .reduce((sum, bytes) => sum + BigInt(bytes), 0n);
  assert.deepEqual(plan.streamedLayers, streamedLayers);
  assert.deepEqual(plan.layers.map(({ layer }) => layer), streamedLayers);
  assert.equal(plan.streamedBytes, expectedStreamedBytes);
  assert.equal(plan.permanentBytes, EXACT_LANGUAGE_BYTES - expectedStreamedBytes);
  assert.equal(
    qwen35AllocatedWeightBytes(source, "hybrid", 24),
    EXACT_LANGUAGE_BYTES - expectedStreamedBytes,
  );
  assert.equal(
    plan.permanentDirectory.tensors.some(({ name }) => name.startsWith("blk.23.")),
    true,
  );
  assert.equal(
    plan.permanentDirectory.tensors.some(({ name }) => name.startsWith("blk.24.")),
    false,
  );
});

test("composes all-layer streaming after the tied Q6_K table and preserves 16K vision contracts", async () => {
  const module = await subject();
  const language = exactLanguageDirectory();
  const tiedBytes = 635_699_200;
  const tied: Qwen35PackageTensor = {
    name: "token_embd.weight",
    shape: [2_560, 248_320],
    ggmlType: 14,
    storageType: "q6-k-fused-f32-256",
    segments: [{
      shardIndex: language.shards.length,
      shardOffset: "0",
      tensorOffset: "0",
      length: String(tiedBytes),
    }],
  };
  const packageDirectory: Qwen35PackageDirectory = {
    ...language,
    shards: [...language.shards, {
      index: language.shards.length,
      url: "shards/tied.bin",
      offset: String(EXACT_LANGUAGE_BYTES),
      length: String(tiedBytes),
      sha256: "e".repeat(64),
    }],
    tensors: [...language.tensors, tied],
  };
  const tiedPlan = planQwen35TiedEmbeddingResidency(packageDirectory);
  const rollingPlan = module.planQwen35RollingLayerResidency(
    tiedPlan.permanentDirectory,
  );

  assert.equal(tiedPlan.streamedBytes, 635_699_200n);
  assert.equal(tiedPlan.tiedTensor.storageType, "q6-k-fused-f32-256");
  assert.equal(tiedPlan.tiedTensor.ggmlType, 14);
  assert.equal(rollingPlan.streamedBytes, EXACT_STREAMED_BYTES);
  assert.equal(rollingPlan.permanentBytes, EXACT_PERMANENT_BYTES);
  assert.equal(qwen35AllocatedWeightBytes(packageDirectory), EXACT_PERMANENT_BYTES);
  assert.equal(
    qwen35AllocatedWeightBytes(packageDirectory, "resident"),
    EXACT_LANGUAGE_BYTES + BigInt(tiedBytes),
  );
  const hybridStreamedBytes = LAYER_BYTES
    .slice(24)
    .reduce((sum, bytes) => sum + BigInt(bytes), 0n);
  assert.equal(
    qwen35AllocatedWeightBytes(packageDirectory, "hybrid", 24),
    EXACT_LANGUAGE_BYTES - hybridStreamedBytes + BigInt(tiedBytes),
    "hybrid residency must keep the tied output table on the GPU",
  );
  assert.deepEqual(
    rollingPlan.permanentDirectory.tensors.map(({ name }) => name),
    ["output_norm.weight"],
  );
  assert.equal(QWEN35_PRODUCT_CONTEXT_CAP, 16_384);
  assert.equal(QWEN35_4B_CONFIG.productContextLength, 16_384);
  assert.equal(QWEN35_4B_CONFIG.baseBlockCount, 32);
  assert.equal(QWEN35_4B_CONFIG.linearAttentionLayerCount, 24);
  assert.equal(QWEN35_4B_CONFIG.fullAttentionLayerCount, 8);
  assert.equal(QWEN35_DEFAULT_MAX_VISUAL_TOKENS, 1_024);
  const hybridStateBytes = planQwen35HybridState(QWEN35_PRODUCT_CONTEXT_CAP).totalBytes;
  const workspaceBytes = planQwen35ActivationWorkspace().totalBytes;
  assert.equal(hybridStateBytes, 590_348_288n);
  assert.equal(workspaceBytes, 286_980n);
  assert.equal(
    2_560n * BigInt(64 + 1_024),
    EXACT_TIED_CACHE_BYTES,
    "64 input rows plus one 1,024-row Q6_K output tile stay bounded",
  );
  assert.equal(
    rollingPlan.permanentBytes + rollingPlan.maxLayerBytes +
      hybridStateBytes + EXACT_TIED_CACHE_BYTES + workspaceBytes +
      EXACT_CANDIDATE_SCRATCH_BYTES,
    EXACT_FULL_16K_LEDGER_BYTES,
    "the projected ledger owns one rolling layer, the 16K state, and bounded tied caches",
  );
});

test("initial allocation and upload exclude every blk.0 through blk.31 tensor", async () => {
  await subject();
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  const initialize = initializeQwen35WeightExecution as unknown as InitializeRollingWeights;
  let callbackDirectory: Qwen35WeightDirectoryView | undefined;
  let callbackStore: RollingLayerStore | undefined;
  const initialized = await initialize({
    arena: gpu.arena,
    packageDirectory: fixture.packageDirectory,
    storage: storage(fixture),
    cached: fixture.cached,
    queue: gpu.queue,
    uploadLaneBytes: 8,
    signal: new AbortController().signal,
    async createDriver(directory, rollingStore) {
      callbackDirectory = directory;
      callbackStore = rollingStore;
      return Object.freeze({ ready: true });
    },
  });

  assert.equal(callbackDirectory, initialized.directory.view);
  assert.equal(callbackStore, initialized.rollingStore);
  assert.equal(initialized.directory.size, 1, "only output_norm remains resident");
  assert.equal(initialized.directory.logicalBytes, 4n);
  assert.equal(
    initialized.directory.tensors.some((tensor) => {
      const match = /^blk\.(\d+)\./.exec(tensor.name);
      return match !== null && Number(match[1]) >= 0 && Number(match[1]) < 32;
    }),
    false,
  );
  assert.deepEqual(
    initialized.directory.tensors.map(({ name }) => name),
    ["output_norm.weight"],
  );
  assert.equal(
    gpu.events.filter((event) => event.startsWith("write:"))
      .reduce((sum, event) => sum + Number(event.split(":").at(-1)), 0),
    4,
  );
  assert.equal(gpu.ledger.snapshot().currentBytes, 4n);
  await initialized.rollingStore.dispose();
  initialized.directory.destroy();
  assert.equal(gpu.ledger.snapshot().currentBytes, 0n);
});

test("resident policy keeps transformer layers in the uploaded directory", async () => {
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  let callbackDirectory: Qwen35WeightDirectoryView | undefined;
  let callbackStore: RollingLayerStore | undefined;
  const initialized = await initializeQwen35WeightExecution({
    arena: gpu.arena,
    packageDirectory: fixture.packageDirectory,
    storage: storage(fixture),
    cached: fixture.cached,
    queue: gpu.queue,
    uploadLaneBytes: 8,
    residencyPolicy: "resident",
    signal: new AbortController().signal,
    async createDriver(directory, rollingStore) {
      callbackDirectory = directory;
      callbackStore = rollingStore;
      return Object.freeze({ ready: true });
    },
  });

  assert.equal(callbackDirectory, initialized.directory.view);
  assert.equal(callbackStore, undefined);
  assert.equal(initialized.rollingStore, undefined);
  assert.equal(initialized.directory.size, 33);
  assert.equal(
    initialized.directory.tensors.some((tensor) => tensor.name.startsWith("blk.")),
    true,
  );
  initialized.directory.destroy();
  assert.equal(gpu.ledger.snapshot().currentBytes, 0n);
});

test("hybrid policy keeps the requested leading layers resident and streams the suffix", async () => {
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  let callbackDirectory: Qwen35WeightDirectoryView | undefined;
  let callbackStore: RollingLayerStore | undefined;
  const initialized = await initializeQwen35WeightExecution({
    arena: gpu.arena,
    packageDirectory: fixture.packageDirectory,
    storage: storage(fixture),
    cached: fixture.cached,
    queue: gpu.queue,
    uploadLaneBytes: 8,
    residencyPolicy: "hybrid",
    residentLayerCount: 24,
    signal: new AbortController().signal,
    async createDriver(directory, rollingStore) {
      callbackDirectory = directory;
      callbackStore = rollingStore;
      return Object.freeze({ ready: true });
    },
  });

  assert.equal(callbackDirectory, initialized.directory.view);
  assert.equal(callbackStore, initialized.rollingStore);
  assert.deepEqual(initialized.rollingStore?.streamedLayers, STREAMED_LAYER_ORDER.slice(24));
  assert.equal(initialized.directory.tensors.some(({ name }) => name.startsWith("blk.23.")), true);
  assert.equal(initialized.directory.tensors.some(({ name }) => name.startsWith("blk.24.")), false);
  await initialized.rollingStore?.dispose();
  initialized.directory.destroy();
  assert.equal(gpu.ledger.snapshot().currentBytes, 0n);
});

test("uses bounded authenticated reads and retires once before destroy", async () => {
  const module = await subject();
  const fixture = smallDirectory({ exactDeltaTails: true });
  const gpu = gpuFixture();
  const store = await module.createQwen35RollingLayerStore({
    arena: gpu.arena,
    queue: gpu.queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: rangeReader(fixture, gpu.events),
    uploadLaneBytes: 64,
    readChunkBytes: 64,
  });

  await store.withLayer({
    layer: 0,
    phase: "decode",
    signal: new AbortController().signal,
    execute(weights, mutation) {
      assert.equal(
        ((weights.get("blk.0.ssm_a")!.physicalRows[0]!.buffer as MemoryBuffer).bytes[0]),
        0xa1,
      );
      assert.equal(
        ((weights.get("blk.0.ssm_dt.bias")!.physicalRows[0]!.buffer as MemoryBuffer).bytes[0]),
        0xd7,
      );
      mutation.markStateMutation();
      gpu.events.push("execute:0");
    },
  });

  const execute = gpu.events.indexOf("execute:0");
  const writes = gpu.events
    .map((event, index) => event.startsWith("write:") ? index : -1)
    .filter((index) => index >= 0);
  const retires = gpu.events
    .map((event, index) => event === "retire" ? index : -1)
    .filter((index) => index >= 0);
  const destroys = gpu.events
    .map((event, index) => event.startsWith("destroy:") ? index : -1)
    .filter((index) => index >= 0);
  assert.ok(Math.max(...writes) < execute);
  assert.equal(
    retires.some((index) => index < execute),
    true,
    "rolling uploads must retire a full copy window before accepting more bytes",
  );
  assert.ok(retires.some((index) => index > execute && index < Math.min(...destroys)));
  assert.ok(retires.length > 1);
  assert.ok(Math.min(...destroys) > execute);
  assert.equal(gpu.live.size, 0);
  assert.equal(gpu.ledger.snapshot().currentBytes, 0n);
  assert.equal(store.getMetrics().peakGpuBytes, 512);
  assert.equal(store.getMetrics().maxDiskReadBytes, 64);
  assert.ok(
    gpu.events.filter((event) => event.startsWith("read:"))
      .every((event) => Number(event.split(":").at(-1)) <= 64),
  );
  await store.dispose();
});

test("uploads a complete single-segment read without a second CPU staging copy", async () => {
  const module = await subject();
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  const reads: Uint8Array[] = [];
  const writes: Uint8Array[] = [];
  const queue: Qwen35WeightWriteQueue = {
    writeBuffer(buffer, bufferOffset, data, dataOffset, size) {
      writes.push(data);
      gpu.queue.writeBuffer(buffer, bufferOffset, data, dataOffset, size);
    },
    onSubmittedWorkDone: () => gpu.queue.onSubmittedWorkDone(),
  };
  const reader: Qwen35PackedRangeReader = {
    async read(input) {
      input.signal.throwIfAborted();
      const source = fixture.bytesByPath[input.storagePath];
      assert.ok(source);
      const bytes = source.slice(input.offset, input.offset + input.byteLength);
      reads.push(bytes);
      return bytes;
    },
  };
  const store = await module.createQwen35RollingLayerStore({
    arena: gpu.arena,
    queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: reader,
    uploadLaneBytes: 16,
    readChunkBytes: 16,
  });

  await store.withLayer({
    layer: 0,
    phase: "decode",
    signal: new AbortController().signal,
    execute() {},
  });

  assert.equal(reads.length, 1);
  assert.equal(writes.length, 1);
  assert.equal(
    writes[0],
    reads[0],
    "queue.writeBuffer must receive the bounded OPFS range directly",
  );
  await store.dispose();
});

test("paces rolling writes so queued upload bytes stay within the upload lane", async () => {
  const module = await subject();
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  let outstandingBytes = 0;
  let maxOutstandingBytes = 0;
  let retirements = 0;
  const queue: Qwen35WeightWriteQueue = {
    writeBuffer(buffer, bufferOffset, data, dataOffset = 0, size = data.byteLength - dataOffset) {
      gpu.queue.writeBuffer(buffer, bufferOffset, data, dataOffset, size);
      outstandingBytes += size;
      maxOutstandingBytes = Math.max(maxOutstandingBytes, outstandingBytes);
    },
    async onSubmittedWorkDone() {
      retirements += 1;
      outstandingBytes = 0;
      await gpu.queue.onSubmittedWorkDone();
    },
  };
  const store = await module.createQwen35RollingLayerStore({
    arena: gpu.arena,
    queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: rangeReader(fixture, gpu.events),
    uploadLaneBytes: 8,
    readChunkBytes: 64,
  });

  await store.withLayer({
    layer: 0,
    phase: "decode",
    signal: new AbortController().signal,
    execute() {},
  });

  assert.ok(maxOutstandingBytes <= 8);
  assert.ok(retirements >= 2, "a multi-write layer must retire intermediate upload windows");
  await store.dispose();
});

test("allows a wider upload window without increasing authenticated read size", async () => {
  const module = await subject();
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  let outstandingBytes = 0;
  let maxOutstandingBytes = 0;
  let retirements = 0;
  const queue: Qwen35WeightWriteQueue = {
    writeBuffer(buffer, bufferOffset, data, dataOffset = 0, size = data.byteLength - dataOffset) {
      gpu.queue.writeBuffer(buffer, bufferOffset, data, dataOffset, size);
      outstandingBytes += size;
      maxOutstandingBytes = Math.max(maxOutstandingBytes, outstandingBytes);
    },
    async onSubmittedWorkDone() {
      retirements += 1;
      outstandingBytes = 0;
      await gpu.queue.onSubmittedWorkDone();
    },
  };
  const store = await module.createQwen35RollingLayerStore({
    arena: gpu.arena,
    queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: rangeReader(fixture, gpu.events),
    uploadLaneBytes: 64 * 1024 * 1024,
    readChunkBytes: 8,
  });

  await store.withLayer({
    layer: 0,
    phase: "decode",
    signal: new AbortController().signal,
    execute() {},
  });

  assert.equal(store.getMetrics().maxDiskReadBytes, 8);
  assert.equal(maxOutstandingBytes, 16);
  assert.equal(retirements, 1, "the two read chunks should share one upload window");
  await store.dispose();
});

test("streams all 32 layers once in static order through one bounded arena and matches the resident oracle", async () => {
  const module = await subject();
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  const store = await module.createQwen35RollingLayerStore({
    arena: gpu.arena,
    queue: gpu.queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: rangeReader(fixture, gpu.events),
    uploadLaneBytes: 8,
    readChunkBytes: 8,
  });
  assert.deepEqual(store.streamedLayers, STREAMED_LAYER_ORDER);
  const sequence = invocations();
  const rollingOrder: number[] = [];
  const oracleState = { recurrent: new Map<number, number>(), kv: new Map<number, number>() };
  const rollingState = { recurrent: new Map<number, number>(), kv: new Map<number, number>() };
  let oracleValue = 7;
  let rollingValue = 7;
  const apply = (
    state: typeof oracleState,
    invocation: LayerInvocation,
    value: number,
    phase: "prefill" | "decode",
  ): number => {
    const target = invocation.kind === "full-attention" ? state.kv : state.recurrent;
    target.set(
      invocation.layer,
      (target.get(invocation.layer) ?? 0) + value + invocation.layer + (phase === "prefill" ? 3 : 11),
    );
    return (value * 33 + invocation.layer * 17 + (phase === "prefill" ? 5 : 13)) % 65_521;
  };

  for (const phase of ["prefill", "decode"] as const) {
    for (const invocation of sequence) {
      oracleValue = apply(oracleState, invocation, oracleValue, phase);
    }
    await module.executeQwen35RollingLayerSequence({
      invocations: sequence,
      permanentWeights: emptyPermanentWeights(),
      rollingStore: store,
      phase,
      signal: new AbortController().signal,
      poison: () => assert.fail("valid execution must not poison state"),
      execute({ invocation, weights, mutation }) {
        rollingOrder.push(invocation.layer);
        const tensor = weights.get(`blk.${invocation.layer}.fixture.weight`);
        assert.ok(tensor, `blk.${invocation.layer} must use its staged rolling directory`);
        assert.equal(
          ((tensor.physicalRows[0]!.buffer as MemoryBuffer).bytes[0]),
          invocation.layer + 1,
        );
        assert.equal(
          [...gpu.live].reduce((sum, buffer) => sum + buffer.bytes.byteLength, 0),
          16,
          "only the current synthetic layer may be live while its kernels execute",
        );
        mutation.markStateMutation();
        rollingValue = apply(rollingState, invocation, rollingValue, phase);
      },
    });
  }

  assert.deepEqual(rollingOrder, [...sequence.map(({ layer }) => layer), ...sequence.map(({ layer }) => layer)]);
  assert.equal(new Set(rollingOrder.slice(0, 32)).size, 32, "prefill must not execute a layer twice");
  assert.equal(new Set(rollingOrder.slice(32)).size, 32, "decode must not execute a layer twice");
  assert.equal(rollingValue, oracleValue);
  assert.deepEqual(rollingState, oracleState);
  assert.equal(store.getMetrics().completedLayers, STREAMED_LAYER_ORDER.length * 2);
  assert.equal(store.getMetrics().peakGpuBytes, 16);
  assert.ok(store.getMetrics().maxDiskReadBytes <= 8);
  assert.equal(gpu.live.size, 0);
  assert.equal(gpu.ledger.snapshot().currentBytes, 0n);
  await store.dispose();
  gpu.ledger.assertAllReleased();
});

test("marks only the streamed suffix as transient in hybrid execution", async () => {
  const module = await subject();
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  const streamedLayers = STREAMED_LAYER_ORDER.slice(24);
  const store = await module.createQwen35RollingLayerStore({
    arena: gpu.arena,
    queue: gpu.queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: rangeReader(fixture, gpu.events),
    uploadLaneBytes: 64,
    readChunkBytes: 64,
    streamedLayers,
  });
  const transient: number[] = [];
  const permanent: number[] = [];

  await module.executeQwen35RollingLayerSequence({
    invocations: invocations(),
    permanentWeights: emptyPermanentWeights(),
    rollingStore: store,
    phase: "decode",
    signal: new AbortController().signal,
    poison: () => assert.fail("valid hybrid execution must not poison state"),
    execute({ invocation, transientWeights }) {
      (transientWeights ? transient : permanent).push(invocation.layer);
    },
  });

  assert.deepEqual(permanent, STREAMED_LAYER_ORDER.slice(0, 24));
  assert.deepEqual(transient, streamedLayers);
  await store.dispose();
});

test("does not retry after mutation failure and permits cancellation before mutation", async () => {
  const module = await subject();
  const fixture = smallDirectory();
  const firstGpu = gpuFixture();
  const poisoned = await module.createQwen35RollingLayerStore({
    arena: firstGpu.arena,
    queue: firstGpu.queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: rangeReader(fixture, firstGpu.events),
    uploadLaneBytes: 8,
    readChunkBytes: 8,
  });
  let executions = 0;
  await assert.rejects(
    poisoned.withLayer({
      layer: 0,
      phase: "decode",
      signal: new AbortController().signal,
      execute(_weights, mutation) {
        executions += 1;
        mutation.markStateMutation();
        throw new Error("driver failed after recurrent state dispatch");
      },
    }),
    (error: unknown) => {
      assert.equal((error as { readonly code?: unknown }).code, "rolling-layer-execution-failed");
      assert.doesNotMatch((error as Error).message, /recurrent state dispatch/);
      return true;
    },
  );
  assert.equal(poisoned.poisoned, true);
  await assert.rejects(poisoned.withLayer({
    layer: 0,
    phase: "decode",
    signal: new AbortController().signal,
    execute() { executions += 1; },
  }), (error: unknown) => {
    assert.equal((error as { readonly code?: unknown }).code, "rolling-layer-poisoned");
    return true;
  });
  assert.equal(executions, 1, "a retry must never execute the mutated token twice");
  assert.equal(firstGpu.ledger.snapshot().currentBytes, 0n);
  await poisoned.dispose();

  const secondGpu = gpuFixture();
  const cancellable = await module.createQwen35RollingLayerStore({
    arena: secondGpu.arena,
    queue: secondGpu.queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: rangeReader(fixture, secondGpu.events),
    uploadLaneBytes: 8,
    readChunkBytes: 8,
  });
  const cancelled = new AbortController();
  cancelled.abort();
  await assert.rejects(cancellable.withLayer({
    layer: 0,
    phase: "prefill",
    signal: cancelled.signal,
    execute() { assert.fail("pre-mutation cancellation must not execute"); },
  }), (error: unknown) => error instanceof DOMException && error.name === "AbortError");
  assert.equal(cancellable.poisoned, false);
  await cancellable.withLayer({
    layer: 0,
    phase: "prefill",
    signal: new AbortController().signal,
    execute(_weights, mutation) { mutation.markStateMutation(); },
  });
  assert.equal(cancellable.getMetrics().completedLayers, 1);
  await cancellable.dispose();
});

test("preserves runtime-owned execution diagnostics without exposing arbitrary errors", async () => {
  const module = await subject();
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  const store = await module.createQwen35RollingLayerStore({
    arena: gpu.arena,
    queue: gpu.queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: rangeReader(fixture, gpu.events),
    uploadLaneBytes: 8,
    readChunkBytes: 8,
  });

  await assert.rejects(store.withLayer({
    layer: 0,
    phase: "decode",
    signal: new AbortController().signal,
    execute() {
      throw diagnosticError("webgpu-compile-portable", "Kernel compilation failed");
    },
  }), { code: "webgpu-compile-portable" });
  await store.dispose();
});

test("preserves a safe GPU allocation diagnostic from the rolling arena", async () => {
  const module = await subject();
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  const arena = {
    async allocate() {
      throw Object.assign(new Error("private Safari validation text"), {
        code: "gpu_validation",
      });
    },
  } as unknown as GpuArena;
  const store = await module.createQwen35RollingLayerStore({
    arena,
    queue: gpu.queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: rangeReader(fixture, gpu.events),
    uploadLaneBytes: 8,
    readChunkBytes: 8,
  });

  await assert.rejects(store.withLayer({
    layer: 0,
    phase: "prefill",
    signal: new AbortController().signal,
    execute() { assert.fail("allocation failure must prevent execution"); },
  }), (error: unknown) => {
    assert.equal((error as { readonly code?: unknown }).code, "gpu_validation");
    assert.doesNotMatch((error as Error).message, /private|Safari|validation text/i);
    return true;
  });
});

test("names rolling allocations outside the permanent tensor namespace", async () => {
  const module = await subject();
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  const allocationIds: string[] = [];
  const arena = {
    allocate(request: Parameters<GpuArena["allocate"]>[0]) {
      allocationIds.push(request.id);
      return gpu.arena.allocate(request);
    },
  } as GpuArena;
  const store = await module.createQwen35RollingLayerStore({
    arena,
    queue: gpu.queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: rangeReader(fixture, gpu.events),
    uploadLaneBytes: 8,
    readChunkBytes: 8,
  });

  await store.withLayer({
    layer: 0,
    phase: "prefill",
    signal: new AbortController().signal,
    execute() {},
  });
  assert.ok(allocationIds.length > 0);
  assert.ok(allocationIds.every((id) => id.startsWith("rolling-layer-0-")));
});

test("poisons a token when a later layer fails after earlier state mutation", async () => {
  const module = await subject();
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  const reader = rangeReader(fixture, gpu.events, { failAfterReads: 2 });
  const store = await module.createQwen35RollingLayerStore({
    arena: gpu.arena,
    queue: gpu.queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: reader,
    uploadLaneBytes: 8,
    readChunkBytes: 8,
  });
  let poisoned = 0;
  const executed: number[] = [];
  await assert.rejects(module.executeQwen35RollingLayerSequence({
    invocations: invocations().slice(0, 2),
    permanentWeights: emptyPermanentWeights(),
    rollingStore: store,
    phase: "decode",
    signal: new AbortController().signal,
    poison: () => { poisoned += 1; },
    execute({ invocation, mutation }) {
      executed.push(invocation.layer);
      mutation.markStateMutation();
    },
  }), (error: unknown) => {
    assert.equal((error as { readonly code?: unknown }).code, "rolling-layer-range-read-failed");
    return true;
  });
  assert.deepEqual(executed, [0]);
  assert.equal(poisoned, 1);
  assert.equal(gpu.ledger.snapshot().currentBytes, 0n);
  await store.dispose();
});

test("rolls back every buffer and redacts cache failures", async () => {
  const module = await subject();
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  const store = await module.createQwen35RollingLayerStore({
    arena: gpu.arena,
    queue: gpu.queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: rangeReader(fixture, gpu.events, { failAfterReads: 1 }),
    uploadLaneBytes: 8,
    readChunkBytes: 8,
  });

  await assert.rejects(store.withLayer({
    layer: 0,
    phase: "prefill",
    signal: new AbortController().signal,
    execute() { assert.fail("failed upload must not execute"); },
  }), (error: unknown) => {
    assert.equal((error as { readonly code?: unknown }).code, "rolling-layer-range-read-failed");
    assert.doesNotMatch(
      (error as Error).message,
      /sensitive-origin-marker|sensitive-path-marker/,
    );
    return true;
  });
  assert.equal(store.poisoned, false);
  const lastWrite = gpu.events.findLastIndex((event) => event.startsWith("write:"));
  const firstDestroy = gpu.events.findIndex((event) => event.startsWith("destroy:"));
  assert.ok(lastWrite >= 0, "the rollback fixture must fail after one accepted GPU write");
  assert.ok(
    gpu.events.some((event, index) => event === "retire" && index > lastWrite && index < firstDestroy),
    "accepted writes must retire before rollback destroys their buffers",
  );
  assert.equal(gpu.live.size, 0);
  assert.equal(gpu.ledger.snapshot().currentBytes, 0n);
  assert.equal(store.getMetrics().failedLayers, 1);
  await store.dispose();
  await store.dispose();
  gpu.ledger.assertAllReleased();
});

test("poisons the store when accepted upload work cannot retire", async () => {
  const module = await subject();
  const fixture = smallDirectory();
  const gpu = gpuFixture();
  const queue: Qwen35WeightWriteQueue = {
    writeBuffer: (...args) => gpu.queue.writeBuffer(...args),
    async onSubmittedWorkDone() {
      gpu.events.push("retire-failed");
      throw new Error("sensitive-queue-marker");
    },
  };
  const store = await module.createQwen35RollingLayerStore({
    arena: gpu.arena,
    queue,
    packageDirectory: fixture.packageDirectory,
    cached: fixture.cached,
    rangeReader: rangeReader(fixture, gpu.events),
    uploadLaneBytes: 16,
    readChunkBytes: 16,
  });
  let executed = false;

  await assert.rejects(store.withLayer({
    layer: 0,
    phase: "decode",
    signal: new AbortController().signal,
    execute() { executed = true; },
  }), (error: unknown) => {
    assert.equal((error as { readonly code?: unknown }).code, "rolling-layer-retirement-failed");
    assert.doesNotMatch((error as Error).message, /sensitive-queue-marker/);
    return true;
  });
  assert.equal(executed, true, "queue retirement follows the submitted layer execution");
  assert.equal(
    store.poisoned,
    true,
    "the device cannot safely reuse a layer store after queue retirement fails",
  );
  await assert.rejects(store.withLayer({
    layer: 0,
    phase: "decode",
    signal: new AbortController().signal,
    execute() { assert.fail("a poisoned store must not execute"); },
  }), { code: "rolling-layer-poisoned" });
  assert.equal(gpu.ledger.snapshot().currentBytes, 0n);
  await store.dispose();
});
