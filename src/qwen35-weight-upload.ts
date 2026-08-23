import {
  ALLOCATION_DIAGNOSTIC_CODES,
  RuntimeDiagnosticError,
  diagnosticError,
} from "./diagnostics.js";
import type {
  CachedModelPackage,
  ModelCacheStorage,
} from "./opfs-model-cache.js";
import {
  allocateQwen35WeightDirectory,
  type Qwen35PackageDirectory,
  type Qwen35TensorWeight,
  type Qwen35WeightArena,
  type Qwen35WeightDirectory,
  type Qwen35WeightDirectoryView,
} from "./qwen35-weight-directory.js";
import {
  createQwen35ModelCacheRangeReader,
} from "./qwen35-disk-backed-tied-embedding.js";
import {
  createQwen35RollingLayerStore,
  hasQwen35RollingLayerSet,
  qwen35HybridStreamedLayers,
  qwen35PermanentWeightPackage,
  QWEN35_ROLLING_LAYER_ORDER,
  type Qwen35RollingLayerStore,
} from "./qwen35-rolling-layer-weights.js";

export interface Qwen35WeightWriteQueue {
  writeBuffer(
    buffer: object,
    bufferOffset: number,
    data: Uint8Array,
    dataOffset?: number,
    size?: number,
  ): void;
  onSubmittedWorkDone(): Promise<void>;
}

export type Qwen35UploadRetirementPolicy = "window" | "per-write";
export type Qwen35WeightResidencyPolicy = "rolling" | "resident" | "hybrid" | "auto";

const RETRYABLE_RESIDENT_FAILURES = new Set<string>(
  ALLOCATION_DIAGNOSTIC_CODES,
);

function isRetryableResidentFailure(error: unknown): boolean {
  return (
    error instanceof RuntimeDiagnosticError &&
    RETRYABLE_RESIDENT_FAILURES.has(error.code)
  );
}

interface UploadSegment {
  readonly tensor: Qwen35TensorWeight;
  readonly sourceStart: number;
  readonly sourceEnd: number;
  readonly targetStart: number;
  consumed: number;
  pending: Uint8Array;
}

interface UploadWindow {
  outstandingBytes: number;
}

async function retireUploadWindow(input: {
  readonly queue: Qwen35WeightWriteQueue;
  readonly window: UploadWindow;
}): Promise<void> {
  if (input.window.outstandingBytes === 0) return;
  await input.queue.onSubmittedWorkDone();
  input.window.outstandingBytes = 0;
}

function safeByteNumber(value: string, label: string): number {
  const parsed = BigInt(value);
  if (parsed < 0n || parsed > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw diagnosticError(
      "model-weight-size-unsafe",
      `${label} cannot be represented exactly`,
    );
  }
  return Number(parsed);
}

function uploadSegments(
  packageDirectory: Qwen35PackageDirectory,
  directory: Qwen35WeightDirectory,
): readonly (readonly UploadSegment[])[] {
  const byShard: UploadSegment[][] = packageDirectory.shards.map(() => []);
  for (const packageTensor of packageDirectory.tensors) {
    const tensor = directory.get(packageTensor.name);
    if (tensor === undefined) {
      throw diagnosticError(
        "model-weight-directory-incomplete",
        "The allocated Qwen3.5 weight directory is incomplete",
      );
    }
    for (const segment of packageTensor.segments) {
      const sourceStart = safeByteNumber(segment.shardOffset, "Shard offset");
      const length = safeByteNumber(segment.length, "Segment length");
      const targetStart = safeByteNumber(segment.tensorOffset, "Tensor offset");
      if (
        segment.shardIndex < 0 ||
        segment.shardIndex >= byShard.length ||
        length <= 0 ||
        sourceStart % 4 !== 0 ||
        targetStart % 4 !== 0 ||
        length % 4 !== 0
      ) {
        throw diagnosticError(
          "model-weight-segment-invalid",
          "A Qwen3.5 upload segment is not u32 aligned",
        );
      }
      byShard[segment.shardIndex]!.push({
        tensor,
        sourceStart,
        sourceEnd: sourceStart + length,
        targetStart,
        consumed: 0,
        pending: new Uint8Array(0),
      });
    }
  }
  for (const segments of byShard) {
    segments.sort((left, right) => left.sourceStart - right.sourceStart);
    for (let index = 1; index < segments.length; index += 1) {
      if (segments[index]!.sourceStart < segments[index - 1]!.sourceEnd) {
        throw diagnosticError(
          "model-weight-segment-overlap",
          "Qwen3.5 upload segments overlap inside a package shard",
        );
      }
    }
  }
  return Object.freeze(
    byShard.map((segments) => Object.freeze(segments)),
  );
}

async function writePhysicalRange(input: {
  readonly queue: Qwen35WeightWriteQueue;
  readonly tensor: Qwen35TensorWeight;
  readonly tensorByteOffset: number;
  readonly data: Uint8Array;
  readonly dataOffset: number;
  readonly byteLength: number;
  readonly uploadLaneBytes: number;
  readonly uploadRetirementPolicy: Qwen35UploadRetirementPolicy;
  readonly window: UploadWindow;
  readonly signal: AbortSignal;
  readonly onBytesUploaded?: (byteLength: number) => void;
}): Promise<void> {
  let tensorOffset = input.tensorByteOffset;
  let dataOffset = input.dataOffset;
  let remaining = input.byteLength;
  while (remaining > 0) {
    input.signal.throwIfAborted();
    const view = input.tensor.physicalRows.find(
      (candidate) =>
        tensorOffset >= candidate.tensorByteOffset &&
        tensorOffset < candidate.tensorByteOffset + candidate.byteLength,
    );
    if (view === undefined) {
      throw diagnosticError(
        "model-weight-physical-range-missing",
        "The Qwen3.5 weight upload has no physical GPU destination",
      );
    }
    const withinView = tensorOffset - view.tensorByteOffset;
    const byteLength = Math.min(
      remaining,
      view.byteLength - withinView,
      input.uploadLaneBytes,
    );
    if (
      tensorOffset % 4 !== 0 ||
      dataOffset % 4 !== 0 ||
      byteLength <= 0 ||
      byteLength % 4 !== 0
    ) {
      throw diagnosticError(
        "model-weight-write-unaligned",
        "A Qwen3.5 GPU write is not u32 aligned",
      );
    }
    if (
      input.window.outstandingBytes > 0 &&
      input.window.outstandingBytes + byteLength > input.uploadLaneBytes
    ) {
      // GPUQueue copies writeBuffer data into queue-owned staging. Retire that
      // copy window throughout the model upload so it cannot grow to model size.
      await retireUploadWindow({
        queue: input.queue,
        window: input.window,
      });
      input.signal.throwIfAborted();
    }
    input.queue.writeBuffer(
      view.buffer,
      view.bufferByteOffset + withinView,
      input.data,
      dataOffset,
      byteLength,
    );
    input.onBytesUploaded?.(byteLength);
    input.window.outstandingBytes += byteLength;
    if (input.uploadRetirementPolicy === "per-write") {
      await retireUploadWindow({
        queue: input.queue,
        window: input.window,
      });
      input.signal.throwIfAborted();
    }
    tensorOffset += byteLength;
    dataOffset += byteLength;
    remaining -= byteLength;
  }
}

async function appendSegmentBytes(input: {
  readonly segment: UploadSegment;
  readonly sourceOffset: number;
  readonly bytes: Uint8Array;
  readonly queue: Qwen35WeightWriteQueue;
  readonly uploadLaneBytes: number;
  readonly uploadRetirementPolicy: Qwen35UploadRetirementPolicy;
  readonly window: UploadWindow;
  readonly signal: AbortSignal;
  readonly onBytesUploaded?: (byteLength: number) => void;
}): Promise<void> {
  const expectedSource = input.segment.sourceStart + input.segment.consumed;
  if (input.sourceOffset !== expectedSource) {
    throw diagnosticError(
      "model-weight-stream-gap",
      "The cached Qwen3.5 shard did not provide a contiguous tensor segment",
    );
  }
  const targetOffset = input.segment.targetStart + input.segment.consumed;
  let consumed = 0;
  if (input.segment.pending.byteLength > 0) {
    const required = 4 - input.segment.pending.byteLength;
    const copied = Math.min(required, input.bytes.byteLength);
    const combined = new Uint8Array(input.segment.pending.byteLength + copied);
    combined.set(input.segment.pending);
    combined.set(input.bytes.subarray(0, copied), input.segment.pending.byteLength);
    consumed += copied;
    if (combined.byteLength === 4) {
      await writePhysicalRange({
        queue: input.queue,
        tensor: input.segment.tensor,
        tensorByteOffset: targetOffset - input.segment.pending.byteLength,
        data: combined,
        dataOffset: 0,
        byteLength: 4,
        uploadLaneBytes: input.uploadLaneBytes,
        uploadRetirementPolicy: input.uploadRetirementPolicy,
        window: input.window,
        signal: input.signal,
        ...(input.onBytesUploaded === undefined
          ? {}
          : { onBytesUploaded: input.onBytesUploaded }),
      });
      input.segment.pending = new Uint8Array(0);
    } else {
      input.segment.pending = combined;
    }
  }

  const directBytes =
    Math.floor((input.bytes.byteLength - consumed) / 4) * 4;
  if (directBytes > 0) {
    await writePhysicalRange({
      queue: input.queue,
      tensor: input.segment.tensor,
      tensorByteOffset: targetOffset + consumed,
      data: input.bytes.subarray(consumed),
      dataOffset: 0,
      byteLength: directBytes,
      uploadLaneBytes: input.uploadLaneBytes,
      uploadRetirementPolicy: input.uploadRetirementPolicy,
      window: input.window,
      signal: input.signal,
      ...(input.onBytesUploaded === undefined
        ? {}
        : { onBytesUploaded: input.onBytesUploaded }),
    });
    consumed += directBytes;
  }
  if (consumed < input.bytes.byteLength) {
    // Only the incomplete u32 is copied; the OPFS chunk itself is never cloned.
    input.segment.pending = input.bytes.slice(consumed);
  }
  input.segment.consumed += input.bytes.byteLength;
}

/** Scatters authenticated package bytes into tensor-owned physical buffers. */
export async function uploadQwen35CachedWeights(input: {
  readonly storage: ModelCacheStorage;
  readonly cached: CachedModelPackage;
  readonly directory: Qwen35WeightDirectory;
  readonly queue: Qwen35WeightWriteQueue;
  readonly uploadLaneBytes: number;
  readonly uploadRetirementPolicy?: Qwen35UploadRetirementPolicy;
  readonly signal: AbortSignal;
  readonly onProgress?: (completedBytes: number) => void;
}): Promise<void> {
  const packageDirectory = Object.freeze({
    manifestSha256: input.directory.manifestSha256,
    shards: input.directory.packageShards,
    tensors: Object.freeze(
      input.directory.tensors.map((tensor) => Object.freeze({
        name: tensor.name,
        shape: tensor.shape,
        ggmlType: tensor.ggmlType,
        storageType: tensor.storageType,
        segments: tensor.segments,
      })),
    ),
  });
  if (
    !Number.isSafeInteger(input.uploadLaneBytes) ||
    input.uploadLaneBytes < 4 ||
    input.uploadLaneBytes % 4 !== 0
  ) {
    throw diagnosticError(
      "model-upload-lane-invalid",
      "The Qwen3.5 upload lane must be a positive u32 multiple",
    );
  }
  const uploadRetirementPolicy = input.uploadRetirementPolicy ?? "window";
  if (uploadRetirementPolicy !== "window" && uploadRetirementPolicy !== "per-write") {
    throw diagnosticError(
      "model-upload-retirement-policy-invalid",
      "The Qwen3.5 upload retirement policy is invalid",
    );
  }
  if (
    input.cached.manifestSha256 !== packageDirectory.manifestSha256 ||
    input.cached.shards.length !== packageDirectory.shards.length
  ) {
    throw diagnosticError(
      "model-cache-package-mismatch",
      "The authenticated cache does not match the Qwen3.5 package directory",
    );
  }
  const segmentsByShard = uploadSegments(packageDirectory, input.directory);
  const window: UploadWindow = { outstandingBytes: 0 };
  let completedBytes = 0;
  const onBytesUploaded = (byteLength: number): void => {
    completedBytes += byteLength;
    input.onProgress?.(completedBytes);
  };
  try {
    for (const [shardIndex, cachedShard] of input.cached.shards.entries()) {
      const packageShard = packageDirectory.shards[shardIndex]!;
      const expectedBytes = safeByteNumber(packageShard.length, "Package shard length");
      if (
        cachedShard.byteLength !== expectedBytes ||
        cachedShard.sha256 !== packageShard.sha256
      ) {
        throw diagnosticError(
          "model-cache-shard-identity-mismatch",
          "An authenticated cache shard does not match the package directory",
        );
      }
      const segments = segmentsByShard[shardIndex]!;
      // The cache already authenticated every immutable shard. A shard used
      // only by token_embd stays in OPFS and must not be streamed during the
      // permanent-weight upload.
      if (segments.length === 0) continue;
      const stream = await input.storage.openRead(cachedShard.storagePath);
      if (stream === null) {
        throw diagnosticError(
          "model-cache-shard-missing",
          "Authenticated model cache shard is unavailable",
        );
      }
      let shardOffset = 0;
      let segmentIndex = 0;
      for await (const chunk of stream) {
        input.signal.throwIfAborted();
        const chunkStart = shardOffset;
        const chunkEnd = chunkStart + chunk.byteLength;
        if (chunkEnd > expectedBytes) {
          throw diagnosticError(
            "model-cache-shard-size-invalid",
            "Authenticated model cache shard has an invalid length",
          );
        }
        while (
          segmentIndex < segments.length &&
          segments[segmentIndex]!.sourceEnd <= chunkStart
        ) {
          segmentIndex += 1;
        }
        let scan = segmentIndex;
        while (scan < segments.length && segments[scan]!.sourceStart < chunkEnd) {
          const segment = segments[scan]!;
          const intersectionStart = Math.max(chunkStart, segment.sourceStart);
          const intersectionEnd = Math.min(chunkEnd, segment.sourceEnd);
          if (intersectionStart < intersectionEnd) {
            await appendSegmentBytes({
              segment,
              sourceOffset: intersectionStart,
              bytes: chunk.subarray(
                intersectionStart - chunkStart,
                intersectionEnd - chunkStart,
              ),
              queue: input.queue,
              uploadLaneBytes: input.uploadLaneBytes,
              uploadRetirementPolicy,
              window,
              signal: input.signal,
              onBytesUploaded,
            });
          }
          if (segment.sourceEnd <= chunkEnd) {
            segmentIndex = scan + 1;
          }
          scan += 1;
        }
        shardOffset = chunkEnd;
      }
      if (shardOffset !== expectedBytes) {
        throw diagnosticError(
          "model-cache-shard-size-invalid",
          "Authenticated model cache shard has an invalid length",
        );
      }
      for (const segment of segments) {
        if (
          segment.consumed !== segment.sourceEnd - segment.sourceStart ||
          segment.pending.byteLength !== 0
        ) {
          throw diagnosticError(
            "model-weight-segment-incomplete",
            "The cached shard did not fill a complete Qwen3.5 tensor segment",
          );
        }
      }
    }
    input.signal.throwIfAborted();
  } catch (error) {
    // Writes accepted before cancellation still own their destinations until
    // GPUQueue retirement; rollback may destroy buffers only after this await.
    await retireUploadWindow({ queue: input.queue, window });
    throw error;
  }
  await retireUploadWindow({ queue: input.queue, window });
  input.signal.throwIfAborted();
}

async function initializeQwen35WeightExecutionAttempt<T>(input: {
  readonly arena: Qwen35WeightArena;
  readonly packageDirectory: Qwen35PackageDirectory;
  readonly storage: ModelCacheStorage;
  readonly cached: CachedModelPackage;
  readonly queue: Qwen35WeightWriteQueue;
  readonly uploadLaneBytes: number;
  readonly uploadRetirementPolicy?: Qwen35UploadRetirementPolicy;
  readonly residencyPolicy: "rolling" | "resident" | "hybrid";
  /** Required only by the local hybrid-residency experiment. */
  readonly residentLayerCount?: number;
  readonly signal: AbortSignal;
  readonly onWeightsAllocated?: (completedBytes: number) => void;
  readonly onWeightsUploaded?: (completedBytes: number) => void;
  readonly onDriverInitialize?: () => void;
  readonly createDriver: (
    directory: Qwen35WeightDirectoryView,
    rollingStore?: Qwen35RollingLayerStore,
  ) => Promise<T>;
}): Promise<{
  readonly directory: Qwen35WeightDirectory;
  readonly rollingStore?: Qwen35RollingLayerStore;
  readonly driver: T;
}> {
  const streamedLayers = input.residencyPolicy === "resident"
    ? undefined
    : input.residencyPolicy === "rolling"
      ? QWEN35_ROLLING_LAYER_ORDER
      : qwen35HybridStreamedLayers(input.residentLayerCount);
  const residentPackage = streamedLayers === undefined
    ? input.packageDirectory
    : qwen35PermanentWeightPackage(
        input.packageDirectory,
        streamedLayers,
        input.residencyPolicy === "hybrid" ? "resident" : "streamed",
      );
  const usesRollingLayers =
    streamedLayers !== undefined &&
    streamedLayers.length > 0 &&
    hasQwen35RollingLayerSet(input.packageDirectory);
  const directory = await allocateQwen35WeightDirectory(
    input.arena,
    residentPackage,
    {
      ...(input.onWeightsAllocated === undefined
        ? {}
        : { onProgress: input.onWeightsAllocated }),
    },
  );
  let rollingStore: Qwen35RollingLayerStore | null = null;
  try {
    input.onWeightsUploaded?.(0);
    await uploadQwen35CachedWeights({
      storage: input.storage,
      cached: input.cached,
      directory,
      queue: input.queue,
      uploadLaneBytes: input.uploadLaneBytes,
      ...(input.uploadRetirementPolicy === undefined
        ? {}
        : { uploadRetirementPolicy: input.uploadRetirementPolicy }),
      signal: input.signal,
      ...(input.onWeightsUploaded === undefined
        ? {}
        : { onProgress: input.onWeightsUploaded }),
    });
    input.signal.throwIfAborted();
    if (usesRollingLayers) {
      rollingStore = await createQwen35RollingLayerStore({
        arena: input.arena,
        queue: input.queue,
        packageDirectory: input.packageDirectory,
        cached: input.cached,
        rangeReader: createQwen35ModelCacheRangeReader(input.storage),
        uploadLaneBytes: input.uploadLaneBytes,
        readChunkBytes: Math.min(input.uploadLaneBytes, 32 * 1024 * 1024),
        streamedLayers,
      });
    }
    input.signal.throwIfAborted();
    input.onDriverInitialize?.();
    const driver = await input.createDriver(
      directory.view,
      rollingStore ?? undefined,
    );
    return Object.freeze({
      directory,
      ...(rollingStore === null ? {} : { rollingStore }),
      driver,
    });
  } catch (error) {
    let rollbackFailed = false;
    try {
      // Cancellation does not cancel writes already accepted by GPUQueue.
      // Retire them before buffer destruction releases ledger ownership.
      await input.queue.onSubmittedWorkDone();
    } catch {
      rollbackFailed = true;
    }
    try {
      await rollingStore?.dispose();
    } catch {
      rollbackFailed = true;
    }
    try {
      directory.destroy();
    } catch {
      rollbackFailed = true;
    }
    if (rollbackFailed) {
      throw diagnosticError(
        "model-weight-upload-rollback-failed",
        "Qwen3.5 weight upload rollback did not complete",
      );
    }
    throw error;
  }
}

/** Transfers weight ownership to a driver only after all queued writes settle. */
export async function initializeQwen35WeightExecution<T>(input: {
  readonly arena: Qwen35WeightArena;
  readonly packageDirectory: Qwen35PackageDirectory;
  readonly storage: ModelCacheStorage;
  readonly cached: CachedModelPackage;
  readonly queue: Qwen35WeightWriteQueue;
  readonly uploadLaneBytes: number;
  readonly uploadRetirementPolicy?: Qwen35UploadRetirementPolicy;
  readonly residencyPolicy?: Qwen35WeightResidencyPolicy;
  /** Local-only hybrid-residency experiment. */
  readonly residentLayerCount?: number;
  readonly signal: AbortSignal;
  readonly onWeightsAllocated?: (completedBytes: number) => void;
  readonly onWeightsUploaded?: (completedBytes: number) => void;
  readonly onDriverInitialize?: () => void;
  readonly createDriver: (
    directory: Qwen35WeightDirectoryView,
    rollingStore?: Qwen35RollingLayerStore,
  ) => Promise<T>;
}): Promise<{
  readonly directory: Qwen35WeightDirectory;
  readonly rollingStore?: Qwen35RollingLayerStore;
  readonly driver: T;
}> {
  const policy = input.residencyPolicy ?? "rolling";
  if (
    policy !== "rolling" &&
    policy !== "resident" &&
    policy !== "hybrid" &&
    policy !== "auto"
  ) {
    throw diagnosticError(
      "model-residency-policy-invalid",
      "The Qwen3.5 weight residency policy is invalid",
    );
  }
  if (policy !== "auto") {
    return initializeQwen35WeightExecutionAttempt({
      ...input,
      residencyPolicy: policy,
    });
  }
  try {
    return await initializeQwen35WeightExecutionAttempt({
      ...input,
      residencyPolicy: "resident",
    });
  } catch (error) {
    if (!isRetryableResidentFailure(error)) throw error;
    // A failed resident allocation can be retried only after the attempt has
    // completed its transactional rollback. The rolling path preserves the
    // proven iPhone fallback without exposing a half-owned GPU directory.
    return initializeQwen35WeightExecutionAttempt({
      ...input,
      residencyPolicy: "rolling",
    });
  }
}
