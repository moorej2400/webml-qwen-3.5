import {
  AllocationLedger,
  type AllocationHandle,
} from "./allocation-ledger.js";
import { diagnosticError } from "./diagnostics.js";
import type { RangeChunkMetadata } from "./http-range-reader.js";
import {
  assertIntegrityValidatedQwen35VisionPackage,
  type Qwen35VisionLayerSink,
  type Qwen35IntegrityValidatedVisionPackage,
} from "./qwen35-vision-package-loader.js";
import type {
  Qwen35VisionProgram,
  Qwen35VisionTensor,
} from "./qwen35-vision-program.js";
import { assertIntegrityValidatedQwen35VisionProgram } from "./qwen35-vision-program.js";

const GPU_BUFFER_USAGE_COPY_DST = 0x0008;
const GPU_BUFFER_USAGE_STORAGE = 0x0080;
const DEFAULT_BUFFER_CAP_BYTES = 256 * 1024 * 1024;
const DEFAULT_UPLOAD_LANE_BYTES = 32 * 1024 * 1024;

export interface Qwen35VisionGpuBuffer {
  destroy(): void;
}

export interface Qwen35VisionGpuAllocator {
  createBuffer(descriptor: {
    readonly size: number;
    readonly usage: number;
    readonly label: string;
  }): Qwen35VisionGpuBuffer;
}

export interface Qwen35VisionGpuQueue {
  writeBuffer(
    buffer: Qwen35VisionGpuBuffer,
    bufferOffset: number,
    data: Uint8Array,
    dataOffset?: number,
    size?: number,
  ): void;
  onSubmittedWorkDone(): Promise<void>;
}

export interface Qwen35VisionGpuShardPlan {
  readonly shard: number;
  readonly byteLength: number;
}

export interface Qwen35VisionGpuStagedShard {
  readonly shard: number;
  readonly byteLength: number;
  readonly buffer: Qwen35VisionGpuBuffer;
}

export type Qwen35VisionGpuTensorOrientation =
  | Readonly<{
      readonly kind: "element-contiguous";
      readonly contiguousDimension: "element";
    }>
  | Readonly<{
      readonly kind: "input-width-contiguous";
      readonly manifestShape: readonly ["input-width", "output-rows"];
      readonly contiguousDimension: "input-width";
    }>
  | Readonly<{
      readonly kind: "patch-conv3d";
      readonly manifestShape: readonly [
        "kernel-width",
        "kernel-height",
        "input-channel",
        "output-channel",
      ];
      readonly temporalSliceOrder: readonly [
        "v.patch_embd.weight",
        "v.patch_embd.weight.1",
      ];
      readonly accumulation: "sum-slice-0-then-slice-1-then-add-one-bias";
    }>
  | Readonly<{
      readonly kind: "learned-position-hidden-contiguous";
      readonly manifestShape: readonly ["hidden-width", "position-count"];
      readonly tableShape: readonly [48, 48, 1_024];
      readonly contiguousDimension: "hidden-width";
      readonly interpolation: "bilinear";
      readonly alignCorners: true;
    }>;

export interface Qwen35VisionGpuTensorSegment {
  readonly shard: number;
  readonly buffer: Qwen35VisionGpuBuffer;
  readonly bufferOffset: number;
  readonly tensorOffset: number;
  readonly byteLength: number;
}

/** Immutable view of one authenticated tensor in only the loaded group. */
export interface Qwen35VisionGpuTensorView {
  readonly name: string;
  readonly shape: readonly number[];
  readonly precision: "f32" | "bf16";
  readonly storageType: "f32" | "raw";
  readonly orientation: Qwen35VisionGpuTensorOrientation;
  readonly segments: readonly Qwen35VisionGpuTensorSegment[];
}

export interface Qwen35VisionGpuStagedGroup {
  readonly layer: number | "bootstrap";
  readonly shards: readonly Qwen35VisionGpuStagedShard[];
  readonly tensors: readonly Qwen35VisionGpuTensorView[];
  /** Idempotent asynchronous ownership release for all buffers in this group. */
  destroy(): Promise<void>;
}

// Tensor views become executable bindings only after the authenticated loader
// committed every shard hash and GPU uploads retired. Structural lookalikes
// must not point foundation kernels at arbitrary buffers.
const AUTHENTICATED_STAGED_GROUPS = new WeakSet<Qwen35VisionGpuStagedGroup>();

export function assertAuthenticatedQwen35VisionGpuStagedGroup(
  value: unknown,
): Qwen35VisionGpuStagedGroup {
  if (
    typeof value !== "object" || value === null ||
    !AUTHENTICATED_STAGED_GROUPS.has(value as Qwen35VisionGpuStagedGroup)
  ) {
    fail("vision-stage-group-unauthenticated", "Vision GPU staged group was not authenticated");
  }
  return value as Qwen35VisionGpuStagedGroup;
}

interface CreateQwen35VisionGpuLayerSinkOptions {
  readonly shardPlan: readonly Qwen35VisionGpuShardPlan[];
  readonly ledger: AllocationLedger;
  readonly allocator: Qwen35VisionGpuAllocator;
  readonly queue: Qwen35VisionGpuQueue;
  readonly allocationId: string;
  readonly uploadLaneBytes?: number;
  readonly bufferCapBytes?: number;
  readonly bufferUsage?: number;
  readonly signal?: AbortSignal;
}

interface Qwen35VisionGpuLayerSink extends Qwen35VisionLayerSink {
  /** Available only after the loader has verified every shard hash and committed. */
  stagedShards(): readonly Qwen35VisionGpuStagedShard[];
}

export interface StageQwen35VisionGpuGroupOptions {
  /** Low-level staging accepts integrity-validated fixtures; production loads only through the fixed bootstrap. */
  readonly package: Qwen35IntegrityValidatedVisionPackage;
  readonly program: Qwen35VisionProgram;
  readonly layer: number | "bootstrap";
  readonly ledger: AllocationLedger;
  readonly allocator: Qwen35VisionGpuAllocator;
  readonly queue: Qwen35VisionGpuQueue;
  readonly allocationId: string;
  readonly uploadLaneBytes?: number;
  readonly bufferCapBytes?: number;
  readonly bufferUsage?: number;
  readonly signal?: AbortSignal;
}

interface MutableShard {
  readonly shard: number;
  readonly byteLength: number;
  readonly buffer: Qwen35VisionGpuBuffer;
  sourceCursor: number;
  gpuCursor: number;
  pending: Uint8Array;
}

function safePositiveInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function fail(code: string, message: string): never {
  throw diagnosticError(code, message);
}

function freeze<T>(value: T): T {
  if (
    typeof value !== "object" || value === null || Object.isFrozen(value) ||
    ArrayBuffer.isView(value) || value instanceof ArrayBuffer ||
    !Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype
  ) return value;
  for (const child of Object.values(value)) freeze(child);
  return Object.freeze(value);
}

function validateSinkOptions(input: CreateQwen35VisionGpuLayerSinkOptions): {
  readonly uploadLaneBytes: number;
  readonly bufferCapBytes: number;
  readonly bufferUsage: number;
  readonly signal: AbortSignal;
} {
  const uploadLaneBytes = input.uploadLaneBytes ?? DEFAULT_UPLOAD_LANE_BYTES;
  const bufferCapBytes = input.bufferCapBytes ?? DEFAULT_BUFFER_CAP_BYTES;
  const bufferUsage = input.bufferUsage ?? (GPU_BUFFER_USAGE_COPY_DST | GPU_BUFFER_USAGE_STORAGE);
  if (
    input.allocationId.length === 0 ||
    !safePositiveInteger(uploadLaneBytes) || uploadLaneBytes % 4 !== 0 ||
    !safePositiveInteger(bufferCapBytes) || bufferCapBytes % 4 !== 0 ||
    !safePositiveInteger(bufferUsage) ||
    (bufferUsage & (GPU_BUFFER_USAGE_COPY_DST | GPU_BUFFER_USAGE_STORAGE)) !==
      (GPU_BUFFER_USAGE_COPY_DST | GPU_BUFFER_USAGE_STORAGE) ||
    input.shardPlan.length === 0
  ) {
    fail("vision-stage-options-invalid", "Vision GPU staging options are invalid");
  }
  return {
    uploadLaneBytes,
    bufferCapBytes,
    bufferUsage,
    signal: input.signal ?? new AbortController().signal,
  };
}

/**
 * One loader sink owns provisional GPU buffers until the package loader calls
 * commit after its SHA-256 checks. This prevents corrupt bytes from escaping
 * into a reusable vision layer.
 */
class TransactionalVisionGpuLayerSink implements Qwen35VisionGpuLayerSink {
  readonly #queue: Qwen35VisionGpuQueue;
  readonly #ledger: AllocationLedger;
  readonly #ledgerHandle: AllocationHandle;
  readonly #uploadLaneBytes: number;
  readonly #signal: AbortSignal;
  readonly #shards: MutableShard[];
  #currentShard = 0;
  #outstandingUploadBytes = 0;
  #committed = false;
  #closed = false;
  #cleanupPromise: Promise<void> | null = null;

  constructor(input: {
    readonly queue: Qwen35VisionGpuQueue;
    readonly ledger: AllocationLedger;
    readonly ledgerHandle: AllocationHandle;
    readonly uploadLaneBytes: number;
    readonly signal: AbortSignal;
    readonly shards: MutableShard[];
  }) {
    this.#queue = input.queue;
    this.#ledger = input.ledger;
    this.#ledgerHandle = input.ledgerHandle;
    this.#uploadLaneBytes = input.uploadLaneBytes;
    this.#signal = input.signal;
    this.#shards = input.shards;
  }

  stagedShards(): readonly Qwen35VisionGpuStagedShard[] {
    if (!this.#committed || this.#closed) {
      fail("vision-stage-unpublished", "Vision GPU buffers are not published");
    }
    return freeze(this.#shards.map((shard) => ({
      shard: shard.shard,
      byteLength: shard.byteLength,
      buffer: shard.buffer,
    })));
  }

  async write(chunk: Uint8Array, metadata: RangeChunkMetadata): Promise<void> {
    this.#assertStaging();
    this.#signal.throwIfAborted();
    const shard = this.#current();
    if (
      !(chunk instanceof Uint8Array) || chunk.byteLength === 0 ||
      !Number.isSafeInteger(metadata.absoluteOffset) || metadata.absoluteOffset < 0 ||
      !Number.isSafeInteger(metadata.rangeOffset) || metadata.rangeOffset < 0 ||
      !Number.isSafeInteger(metadata.rangeLength) || metadata.rangeLength < 1 ||
      metadata.rangeOffset + chunk.byteLength > metadata.rangeLength
    ) {
      fail("vision-stage-stream-metadata-invalid", "Vision GPU stream metadata is invalid");
    }
    // HttpRangeReader restarts absolute offsets for each source shard. The
    // explicit per-shard cursor therefore detects both a missing byte and a
    // replay at the bootstrap boundary where two source streams meet.
    if (metadata.absoluteOffset !== shard.sourceCursor) {
      fail("vision-stage-stream-gap", "Vision GPU stream is not contiguous");
    }
    if (chunk.byteLength > shard.byteLength - shard.sourceCursor) {
      fail("vision-stage-stream-overrun", "Vision GPU stream exceeds the addressed shard");
    }
    await this.#append(shard, chunk);
    shard.sourceCursor += chunk.byteLength;
    if (shard.sourceCursor === shard.byteLength) {
      this.#currentShard += 1;
    }
  }

  async commit(): Promise<void> {
    this.#assertStaging();
    this.#signal.throwIfAborted();
    if (
      this.#currentShard !== this.#shards.length ||
      this.#shards.some((shard) =>
        shard.sourceCursor !== shard.byteLength || shard.pending.byteLength !== 0 || shard.gpuCursor !== shard.byteLength,
      )
    ) {
      fail("vision-stage-stream-incomplete", "Vision GPU stream did not cover every shard exactly");
    }
    await this.#retireOutstanding();
    this.#signal.throwIfAborted();
    this.#committed = true;
  }

  async abort(): Promise<void> {
    if (this.#cleanupPromise !== null) {
      return this.#cleanupPromise;
    }
    this.#closed = true;
    this.#cleanupPromise = this.#cleanup();
    return this.#cleanupPromise;
  }

  #current(): MutableShard {
    const shard = this.#shards[this.#currentShard];
    if (shard === undefined) {
      fail("vision-stage-stream-overrun", "Vision GPU stream exceeds the addressed group");
    }
    return shard;
  }

  #assertStaging(): void {
    if (this.#closed || this.#committed) {
      fail("vision-stage-closed", "Vision GPU staging transaction is closed");
    }
  }

  async #append(shard: MutableShard, bytes: Uint8Array): Promise<void> {
    let offset = 0;
    if (shard.pending.byteLength > 0) {
      const needed = 4 - shard.pending.byteLength;
      const copied = Math.min(needed, bytes.byteLength);
      const combined = new Uint8Array(shard.pending.byteLength + copied);
      combined.set(shard.pending);
      combined.set(bytes.subarray(0, copied), shard.pending.byteLength);
      offset = copied;
      if (combined.byteLength < 4) {
        shard.pending = combined;
        return;
      }
      await this.#submit(shard, combined, 0, 4);
      shard.pending = new Uint8Array(0);
    }

    const directBytes = Math.floor((bytes.byteLength - offset) / 4) * 4;
    if (directBytes > 0) {
      // The source chunk can start after a 1–3 byte bridge. A subarray keeps
      // the large aligned suffix zero-copy while dataOffset remains WebGPU-u32 aligned.
      await this.#submit(shard, bytes.subarray(offset, offset + directBytes), 0, directBytes);
      offset += directBytes;
    }
    if (offset < bytes.byteLength) {
      // WebGPU accepts u32-aligned writes only. This copies at most three bytes
      // while every aligned portion is handed to GPUQueue directly from fetch.
      shard.pending = bytes.slice(offset);
    }
  }

  async #submit(
    shard: MutableShard,
    data: Uint8Array,
    dataOffset: number,
    byteLength: number,
  ): Promise<void> {
    if (
      shard.gpuCursor % 4 !== 0 || dataOffset % 4 !== 0 ||
      byteLength < 1 || byteLength % 4 !== 0 ||
      shard.gpuCursor + byteLength > shard.byteLength
    ) {
      fail("vision-stage-write-unaligned", "Vision GPU upload is not u32 aligned");
    }
    let sourceOffset = dataOffset;
    let remaining = byteLength;
    while (remaining > 0) {
      if (
        this.#outstandingUploadBytes > 0 &&
        this.#outstandingUploadBytes + remaining > this.#uploadLaneBytes
      ) {
        await this.#retireOutstanding();
        this.#signal.throwIfAborted();
      }
      const submittedBytes = Math.min(remaining, this.#uploadLaneBytes);
      try {
        this.#queue.writeBuffer(shard.buffer, shard.gpuCursor, data, sourceOffset, submittedBytes);
      } catch {
        fail("vision-stage-upload-failed", "Vision GPU upload failed");
      }
      shard.gpuCursor += submittedBytes;
      this.#outstandingUploadBytes += submittedBytes;
      sourceOffset += submittedBytes;
      remaining -= submittedBytes;
    }
  }

  async #retireOutstanding(): Promise<void> {
    if (this.#outstandingUploadBytes === 0) return;
    try {
      await this.#queue.onSubmittedWorkDone();
    } catch {
      fail("vision-stage-queue-retirement-failed", "Vision GPU upload retirement failed");
    }
    this.#outstandingUploadBytes = 0;
  }

  async #cleanup(): Promise<void> {
    let failed = false;
    // GPUQueue can retain source data after writeBuffer returns. Retire it
    // before destruction so a failed hash or cancellation cannot hand a live
    // buffer to a later layer allocation.
    try {
      await this.#queue.onSubmittedWorkDone();
    } catch {
      failed = true;
    }
    for (const shard of this.#shards) {
      try {
        shard.buffer.destroy();
      } catch {
        failed = true;
      }
    }
    try {
      this.#ledger.release(this.#ledgerHandle);
    } catch {
      failed = true;
    }
    if (failed) {
      fail("vision-stage-cleanup-failed", "Vision GPU staging cleanup did not complete");
    }
  }
}

/** Creates the exact Qwen35VisionLayerSink consumed by the authenticated loader. */
function createQwen35VisionGpuLayerSink(
  input: CreateQwen35VisionGpuLayerSinkOptions,
): Qwen35VisionGpuLayerSink {
  const options = validateSinkOptions(input);
  const seen = new Set<number>();
  let totalBytes = 0;
  for (const shard of input.shardPlan) {
    if (
      !Number.isSafeInteger(shard.shard) || shard.shard < 0 || seen.has(shard.shard) ||
      !safePositiveInteger(shard.byteLength) || shard.byteLength % 4 !== 0 ||
      shard.byteLength > options.bufferCapBytes || totalBytes > Number.MAX_SAFE_INTEGER - shard.byteLength
    ) {
      fail("vision-stage-shard-unaligned", "Vision GPU shard plan is not aligned or conservative");
    }
    seen.add(shard.shard);
    totalBytes += shard.byteLength;
  }
  const ledgerHandle = input.ledger.reserve({
    id: input.allocationId,
    category: "model",
    bytes: BigInt(totalBytes),
  });
  const shards: MutableShard[] = [];
  try {
    for (const [ordinal, plan] of input.shardPlan.entries()) {
      const buffer = input.allocator.createBuffer({
        size: plan.byteLength,
        usage: options.bufferUsage,
        // Group-local labels avoid tensor names, prompts, and external URLs.
        label: `qwen35-vision:group:${ordinal}`,
      });
      shards.push({
        shard: plan.shard,
        byteLength: plan.byteLength,
        buffer,
        sourceCursor: 0,
        gpuCursor: 0,
        pending: new Uint8Array(0),
      });
    }
  } catch {
    let cleanupFailed = false;
    for (const shard of shards) {
      try {
        shard.buffer.destroy();
      } catch {
        cleanupFailed = true;
      }
    }
    try {
      input.ledger.release(ledgerHandle);
    } catch {
      cleanupFailed = true;
    }
    if (cleanupFailed) {
      fail("vision-stage-allocation-rollback-failed", "Vision GPU allocation rollback did not complete");
    }
    fail("vision-stage-allocation-failed", "Vision GPU allocation failed");
  }
  return new TransactionalVisionGpuLayerSink({
    queue: input.queue,
    ledger: input.ledger,
    ledgerHandle,
    uploadLaneBytes: options.uploadLaneBytes,
    signal: options.signal,
    shards,
  });
}

function addTensor(
  tensors: Map<string, Qwen35VisionTensor>,
  tensor: Qwen35VisionTensor,
): void {
  const existing = tensors.get(tensor.name);
  if (existing !== undefined && !sameTensorDescriptor(existing, tensor)) {
    fail("vision-stage-program-invalid", "Vision program has conflicting tensor descriptors");
  }
  tensors.set(tensor.name, tensor);
}

function sameTensorDescriptor(
  left: Qwen35VisionTensor,
  right: Qwen35VisionTensor,
): boolean {
  return left.name === right.name &&
    left.precision === right.precision &&
    left.storageType === right.storageType &&
    left.shape.length === right.shape.length &&
    left.shape.every((value, index) => value === right.shape[index]) &&
    left.segments.length === right.segments.length &&
    left.segments.every((segment, index) => {
      const other = right.segments[index];
      return other !== undefined &&
        segment.shard === other.shard &&
        segment.shardOffset === other.shardOffset &&
        segment.tensorOffset === other.tensorOffset &&
        segment.byteLength === other.byteLength;
    });
}

function tensorsForGroup(
  program: Qwen35VisionProgram,
  layer: number | "bootstrap",
  allowedShards: ReadonlySet<number>,
): readonly Qwen35VisionTensor[] {
  const tensors = new Map<string, Qwen35VisionTensor>();
  if (layer === "bootstrap") {
    const bootstrap = program.bootstrap;
    for (const tensor of [
      ...bootstrap.patchEmbedding.temporalWeights,
      bootstrap.patchEmbedding.bias,
      bootstrap.positionEmbedding,
      bootstrap.postLayerNorm.weight,
      bootstrap.postLayerNorm.bias,
      bootstrap.merger.input.weight,
      bootstrap.merger.input.bias,
      bootstrap.merger.output.weight,
      bootstrap.merger.output.bias,
    ]) addTensor(tensors, tensor);
  } else {
    const selected = program.layers[layer];
    if (selected === undefined || selected.layer !== layer) {
      fail("vision-stage-program-invalid", "Vision program does not contain the requested layer");
    }
    for (const tensor of [
      selected.normalization.preAttention.weight,
      selected.normalization.preAttention.bias,
      selected.normalization.preMlp.weight,
      selected.normalization.preMlp.bias,
      selected.attention.qkv.weight,
      selected.attention.qkv.bias,
      selected.attention.output.weight,
      selected.attention.output.bias,
      selected.mlp.up.weight,
      selected.mlp.up.bias,
      selected.mlp.down.weight,
      selected.mlp.down.bias,
    ]) addTensor(tensors, tensor);
  }
  for (const tensor of tensors.values()) {
    if (
      tensor.segments.length === 0 ||
      tensor.segments.some((segment) =>
        !allowedShards.has(segment.shard) ||
        !safePositiveInteger(segment.byteLength) || segment.byteLength % 4 !== 0 ||
        !Number.isSafeInteger(segment.shardOffset) || segment.shardOffset < 0 || segment.shardOffset % 4 !== 0 ||
        !Number.isSafeInteger(segment.tensorOffset) || segment.tensorOffset < 0 || segment.tensorOffset % 4 !== 0,
      )
    ) {
      fail("vision-stage-program-invalid", "Vision program tensor does not belong to the requested group");
    }
  }
  return [...tensors.values()];
}

function orientationFor(
  program: Qwen35VisionProgram,
  tensor: Qwen35VisionTensor,
): Qwen35VisionGpuTensorOrientation {
  if (tensor.name === "v.patch_embd.weight" || tensor.name === "v.patch_embd.weight.1") {
    if (!sameNumbers(tensor.shape, [16, 16, 3, 1_024]) || tensor.precision !== "f32" || tensor.storageType !== "f32") {
      fail("vision-stage-program-invalid", "Vision patch tensor has an invalid layout");
    }
    const orientation = program.architecture.tensorOrientation.patchConv3d;
    return freeze({
      kind: "patch-conv3d" as const,
      manifestShape: orientation.manifestShape,
      temporalSliceOrder: orientation.temporalSliceOrder,
      accumulation: orientation.accumulation,
    });
  }
  if (tensor.name === "v.position_embd.weight") {
    if (
      !sameNumbers(tensor.shape, [1_024, 2_304]) || tensor.precision !== "f32" ||
      tensor.storageType !== "f32" || tensor.segments.length !== 2
    ) {
      fail("vision-stage-program-invalid", "Vision position tensor has an invalid layout");
    }
    const position = program.architecture.learnedPosition;
    return freeze({
      kind: "learned-position-hidden-contiguous" as const,
      manifestShape: ["hidden-width", "position-count"] as const,
      tableShape: position.tableShape,
      contiguousDimension: "hidden-width" as const,
      interpolation: position.interpolation,
      alignCorners: position.alignCorners,
    });
  }
  if (tensor.shape.length === 1) {
    if (tensor.precision !== "f32" || tensor.storageType !== "f32") {
      fail("vision-stage-program-invalid", "Vision vector tensor has an invalid precision");
    }
    return freeze({ kind: "element-contiguous" as const, contiguousDimension: "element" as const });
  }
  if (tensor.shape.length !== 2 || tensor.precision !== "bf16" || tensor.storageType !== "raw") {
    fail("vision-stage-program-invalid", "Vision linear tensor has an invalid layout");
  }
  const orientation = program.architecture.tensorOrientation.linear;
  return freeze({
    kind: "input-width-contiguous" as const,
    manifestShape: orientation.manifestShape,
    contiguousDimension: orientation.contiguousDimension,
  });
}

function sameNumbers(actual: readonly number[], expected: readonly number[]): boolean {
  return actual.length === expected.length && actual.every((value, index) => value === expected[index]);
}

function createViews(
  program: Qwen35VisionProgram,
  tensors: readonly Qwen35VisionTensor[],
  shards: readonly Qwen35VisionGpuStagedShard[],
): readonly Qwen35VisionGpuTensorView[] {
  const byShard = new Map(shards.map((shard) => [shard.shard, shard]));
  return freeze(tensors.map((tensor) => {
    const segments = tensor.segments.map((segment) => {
      const shard = byShard.get(segment.shard);
      if (
        shard === undefined ||
        segment.shardOffset + segment.byteLength > shard.byteLength
      ) {
        fail("vision-stage-program-invalid", "Vision program tensor exceeds its loaded shard");
      }
      return freeze({
        shard: segment.shard,
        buffer: shard.buffer,
        bufferOffset: segment.shardOffset,
        tensorOffset: segment.tensorOffset,
        byteLength: segment.byteLength,
      });
    });
    return freeze({
      name: tensor.name,
      shape: freeze([...tensor.shape]),
      precision: tensor.precision,
      storageType: tensor.storageType,
      orientation: orientationFor(program, tensor),
      segments: freeze(segments),
    });
  }));
}

/**
 * Streams one authenticated Qwen3.5 vision group into provisional GPU buffers.
 * The returned views cannot reference another layer, and exist only after both
 * SHA-256 verification and GPUQueue retirement complete.
 */
export async function stageQwen35VisionGpuGroup(
  input: StageQwen35VisionGpuGroupOptions,
): Promise<Qwen35VisionGpuStagedGroup> {
  const package_ = assertIntegrityValidatedQwen35VisionPackage(input.package);
  const program = assertIntegrityValidatedQwen35VisionProgram(input.program, package_);
  const selected = input.layer === "bootstrap"
    ? package_.bootstrap
    : package_.layers[input.layer];
  if (
    selected === undefined ||
    (input.layer !== "bootstrap" && (!Number.isSafeInteger(input.layer) || input.layer < 0))
  ) {
    fail("vision-stage-layer-invalid", "Vision GPU staging layer is invalid");
  }
  const shardPlan = selected.shards.map((shard) => ({
    shard: shard.index,
    byteLength: shard.byteLength,
  }));
  const allowedShards = new Set(shardPlan.map((shard) => shard.shard));
  const tensors = tensorsForGroup(program, input.layer, allowedShards);
  const sink = createQwen35VisionGpuLayerSink({
    shardPlan,
    ledger: input.ledger,
    allocator: input.allocator,
    queue: input.queue,
    allocationId: input.allocationId,
    ...(input.uploadLaneBytes === undefined ? {} : { uploadLaneBytes: input.uploadLaneBytes }),
    ...(input.bufferCapBytes === undefined ? {} : { bufferCapBytes: input.bufferCapBytes }),
    ...(input.bufferUsage === undefined ? {} : { bufferUsage: input.bufferUsage }),
    ...(input.signal === undefined ? {} : { signal: input.signal }),
  });
  try {
    await package_.streamLayer(input.layer, sink, input.signal);
    const shards = sink.stagedShards();
    const views = createViews(program, tensors, shards);
    let stagedGroup!: Qwen35VisionGpuStagedGroup;
    stagedGroup = freeze({
      layer: input.layer,
      shards,
      tensors: views,
      destroy: async () => {
        // Retire publication before async cleanup: a rejected destroy cannot
        // leave callers able to bind buffers that it has begun to release.
        AUTHENTICATED_STAGED_GROUPS.delete(stagedGroup);
        await sink.abort();
      },
    });
    AUTHENTICATED_STAGED_GROUPS.add(stagedGroup);
    return stagedGroup;
  } catch (error) {
    try {
      // Loader failures already call abort. A post-commit view failure does not,
      // so this idempotent call closes both paths without replacing the cause.
      await sink.abort();
    } catch {
      // The package, hash, view, or cancellation error is the useful primary error.
    }
    throw error;
  }
}
