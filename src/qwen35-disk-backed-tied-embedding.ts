import { diagnosticError } from "./diagnostics.js";
import type { GpuAllocation, GpuArena } from "./gpu-arena.js";
import type {
  CachedModelPackage,
  ModelCacheStorage,
} from "./opfs-model-cache.js";
import {
  qwen35TensorWeightBytes,
  type Qwen35PackageDirectory,
  type Qwen35PackageSegment,
  type Qwen35PackageTensor,
} from "./qwen35-weight-directory.js";
import type { Qwen35WeightWriteQueue } from "./qwen35-weight-upload.js";

const TIED_TENSOR_NAME = "token_embd.weight";
const QWEN35_HIDDEN_SIZE = 2_560;
const Q6_K_GGML_TYPE = 14;
const Q6_K_STORAGE_TYPE = "q6-k-212";
const Q6_K_BLOCK_ELEMENTS = 256;
const Q6_K_BLOCK_BYTES = 212;
const GPU_STORAGE_AND_COPY_DST = 0x0080 | 0x0008;

/** Distinguishes the exact Q6_K product tensor from small generic test fixtures. */
export function hasQwen35DiskBackedTiedEmbedding(
  packageDirectory: Qwen35PackageDirectory,
): boolean {
  return packageDirectory.tensors.some((tensor) =>
    tensor.name === TIED_TENSOR_NAME &&
    tensor.ggmlType === Q6_K_GGML_TYPE &&
    tensor.storageType === Q6_K_STORAGE_TYPE &&
    tensor.shape.length === 2 &&
    tensor.shape[0] === QWEN35_HIDDEN_SIZE
  );
}

export interface Qwen35PackedRangeRead {
  readonly storagePath: string;
  readonly offset: number;
  readonly byteLength: number;
  readonly signal: AbortSignal;
}

export interface Qwen35PackedRangeReader {
  read(input: Qwen35PackedRangeRead): Promise<Uint8Array>;
}

/** Adapts immutable cache storage to bounded tensor reads. */
export function createQwen35ModelCacheRangeReader(
  storage: ModelCacheStorage,
): Qwen35PackedRangeReader {
  return Object.freeze({
    async read(input: Qwen35PackedRangeRead): Promise<Uint8Array> {
      input.signal.throwIfAborted();
      if (storage.readRange !== undefined) {
        const bytes = await storage.readRange(
          input.storagePath,
          input.offset,
          input.byteLength,
          input.signal,
        );
        if (bytes === null || bytes.byteLength !== input.byteLength) {
          throw new Error("immutable cache range is unavailable");
        }
        return bytes;
      }

      // Injected in-memory test stores predate random reads. This compatibility
      // path stops after the requested range; BrowserOpfsStorage never uses it.
      const stream = await storage.openRead(input.storagePath);
      if (stream === null) throw new Error("immutable cache file is unavailable");
      const output = new Uint8Array(input.byteLength);
      const end = input.offset + input.byteLength;
      let sourceOffset = 0;
      let copied = 0;
      for await (const chunk of stream) {
        input.signal.throwIfAborted();
        const chunkEnd = sourceOffset + chunk.byteLength;
        if (chunkEnd > input.offset && sourceOffset < end) {
          const start = Math.max(input.offset, sourceOffset);
          const stop = Math.min(end, chunkEnd);
          output.set(
            chunk.subarray(start - sourceOffset, stop - sourceOffset),
            start - input.offset,
          );
          copied += stop - start;
        }
        sourceOffset = chunkEnd;
        if (sourceOffset >= end) break;
      }
      if (copied !== input.byteLength) {
        throw new Error("immutable cache range is incomplete");
      }
      return output;
    },
  });
}

export interface Qwen35StagedPackedRows {
  readonly tensorName: "token_embd.weight";
  readonly storageType: string;
  readonly firstRow: number;
  readonly rowCount: number;
  readonly rowBytes: number;
  readonly buffer: object;
  readonly bufferOffset: number;
  readonly byteLength: number;
}

export interface Qwen35LogitCandidate {
  readonly tokenId: number;
  readonly score: number;
}

export interface Qwen35TiedEmbeddingMetrics {
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

export interface Qwen35DiskBackedTiedEmbeddingStore {
  readonly tensor: Qwen35PackageTensor;
  readonly permanentGpuBytes: 0n;
  readonly cacheGpuBytes: bigint;
  stageInputRow(input: {
    readonly tokenId: number;
    readonly phase: "prefill" | "decode";
    readonly signal: AbortSignal;
  }): Promise<Qwen35StagedPackedRows>;
  selectTopK(input: {
    readonly phase: "prefill" | "decode";
    readonly topK: number;
    readonly signal: AbortSignal;
    readonly scoreTile: (
      tile: Qwen35StagedPackedRows,
    ) => Promise<readonly Qwen35LogitCandidate[]> | readonly Qwen35LogitCandidate[];
  }): Promise<readonly Qwen35LogitCandidate[]>;
  cancel(): Promise<void>;
  dispose(): Promise<void>;
  getMetrics(): Qwen35TiedEmbeddingMetrics;
}

export interface Qwen35TiedEmbeddingResidencyPlan {
  readonly permanentDirectory: Qwen35PackageDirectory;
  readonly tiedTensor: Qwen35PackageTensor;
  readonly permanentBytes: bigint;
  readonly streamedBytes: bigint;
}

interface TiedGeometry {
  readonly tensor: Qwen35PackageTensor;
  readonly rowBytes: number;
  readonly rowCount: number;
  readonly logicalBytes: number;
  readonly segments: readonly {
    readonly source: Qwen35PackageSegment;
    readonly tensorStart: number;
    readonly tensorEnd: number;
    readonly shardOffset: number;
  }[];
}

const abortError = (): DOMException =>
  new DOMException("Operation cancelled", "AbortError");

const safeByteNumber = (value: string, label: string): number => {
  if (!/^(?:0|[1-9][0-9]*)$/.test(value)) {
    throw diagnosticError(
      "tied-embedding-package-invalid",
      `${label} is invalid`,
    );
  }
  const parsed = BigInt(value);
  if (parsed > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw diagnosticError(
      "tied-embedding-package-invalid",
      `${label} is unsafe`,
    );
  }
  return Number(parsed);
};

const positiveInteger = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw diagnosticError(
      "tied-embedding-configuration-invalid",
      `${label} must be a positive safe integer`,
    );
  }
  return value;
};

const cloneSegment = (
  segment: Qwen35PackageSegment,
): Qwen35PackageSegment => Object.freeze({ ...segment });

const cloneTensor = (tensor: Qwen35PackageTensor): Qwen35PackageTensor =>
  Object.freeze({
    ...tensor,
    shape: Object.freeze([...tensor.shape]),
    segments: Object.freeze(tensor.segments.map(cloneSegment)),
  });

const cloneDirectory = (
  directory: Qwen35PackageDirectory,
  tensors: readonly Qwen35PackageTensor[],
): Qwen35PackageDirectory => Object.freeze({
  manifestSha256: directory.manifestSha256,
  shards: Object.freeze(directory.shards.map((shard) => Object.freeze({ ...shard }))),
  tensors: Object.freeze(tensors.map(cloneTensor)),
});

const tiedGeometry = (tensor: Qwen35PackageTensor): TiedGeometry => {
  if (
    tensor.name !== TIED_TENSOR_NAME ||
    tensor.ggmlType !== Q6_K_GGML_TYPE ||
    tensor.storageType !== Q6_K_STORAGE_TYPE ||
    tensor.shape.length !== 2 ||
    tensor.shape[0] !== QWEN35_HIDDEN_SIZE ||
    !Number.isSafeInteger(tensor.shape[1]) ||
    tensor.shape[1]! < 1
  ) {
    throw diagnosticError(
      "tied-embedding-package-invalid",
      "The Qwen3.5 tied embedding tensor is invalid",
    );
  }
  const rowBytes = (QWEN35_HIDDEN_SIZE / Q6_K_BLOCK_ELEMENTS) * Q6_K_BLOCK_BYTES;
  const rowCount = tensor.shape[1]!;
  const logicalBytes = rowBytes * rowCount;
  if (!Number.isSafeInteger(logicalBytes)) {
    throw diagnosticError(
      "tied-embedding-package-invalid",
      "The Qwen3.5 tied embedding size is unsafe",
    );
  }
  const segments = tensor.segments.map((source) => {
    const tensorStart = safeByteNumber(source.tensorOffset, "Tied tensor offset");
    const byteLength = safeByteNumber(source.length, "Tied segment length");
    const shardOffset = safeByteNumber(source.shardOffset, "Tied shard offset");
    if (
      !Number.isSafeInteger(source.shardIndex) ||
      source.shardIndex < 0 ||
      byteLength < 1 ||
      !Number.isSafeInteger(tensorStart + byteLength)
    ) {
      throw diagnosticError(
        "tied-embedding-package-invalid",
        "The Qwen3.5 tied embedding segments are invalid",
      );
    }
    return Object.freeze({
      source,
      tensorStart,
      tensorEnd: tensorStart + byteLength,
      shardOffset,
    });
  }).sort((left, right) => left.tensorStart - right.tensorStart);
  let covered = 0;
  for (const segment of segments) {
    if (segment.tensorStart !== covered) {
      throw diagnosticError(
        "tied-embedding-package-invalid",
        "The Qwen3.5 tied embedding segments are not contiguous",
      );
    }
    covered = segment.tensorEnd;
  }
  if (covered !== logicalBytes) {
    throw diagnosticError(
      "tied-embedding-package-invalid",
      "The Qwen3.5 tied embedding segments are incomplete",
    );
  }
  return Object.freeze({
    tensor: cloneTensor(tensor),
    rowBytes,
    rowCount,
    logicalBytes,
    segments: Object.freeze(segments),
  });
};

/** Keeps the exact mixed-quant tied tensor in its authenticated package shards. */
export function planQwen35TiedEmbeddingResidency(
  packageDirectory: Qwen35PackageDirectory,
): Qwen35TiedEmbeddingResidencyPlan {
  const matches = packageDirectory.tensors.filter(
    (tensor) => tensor.name === TIED_TENSOR_NAME,
  );
  if (matches.length !== 1) {
    throw diagnosticError(
      "tied-embedding-package-invalid",
      "The Qwen3.5 package must contain one tied embedding tensor",
    );
  }
  const geometry = tiedGeometry(matches[0]!);
  const permanentDirectory = cloneDirectory(
    packageDirectory,
    packageDirectory.tensors.filter((tensor) => tensor.name !== TIED_TENSOR_NAME),
  );
  const permanentBytes = qwen35TensorWeightBytes(permanentDirectory);
  return Object.freeze({
    permanentDirectory,
    tiedTensor: geometry.tensor,
    permanentBytes,
    streamedBytes: BigInt(geometry.logicalBytes),
  });
}

const stableTopK = (
  candidates: readonly Qwen35LogitCandidate[],
  topK: number,
): readonly Qwen35LogitCandidate[] => Object.freeze(
  [...candidates]
    .filter((candidate) => Number.isFinite(candidate.score))
    .sort((left, right) => right.score - left.score || left.tokenId - right.tokenId)
    .slice(0, topK)
    .map((candidate) => Object.freeze({ ...candidate })),
);

const destroyAllocations = (allocations: readonly GpuAllocation[]): boolean => {
  let failed = false;
  for (let index = allocations.length - 1; index >= 0; index -= 1) {
    try {
      allocations[index]!.destroy();
    } catch {
      failed = true;
    }
  }
  return failed;
};

/**
 * Creates the two fixed cache buffers used by the model-specific tied path.
 * The authenticated OPFS shards remain the only complete tensor owner.
 */
export async function createQwen35DiskBackedTiedEmbeddingStore(input: {
  readonly arena: GpuArena;
  readonly queue: Qwen35WeightWriteQueue;
  readonly packageDirectory: Qwen35PackageDirectory;
  readonly cached: CachedModelPackage;
  readonly rangeReader: Qwen35PackedRangeReader;
  readonly inputRowCapacity: number;
  readonly outputTileRows: number;
  readonly decodableRows: number;
}): Promise<Qwen35DiskBackedTiedEmbeddingStore> {
  const residency = planQwen35TiedEmbeddingResidency(input.packageDirectory);
  const geometry = tiedGeometry(residency.tiedTensor);
  const inputRowCapacity = positiveInteger(
    input.inputRowCapacity,
    "Input row cache capacity",
  );
  const outputTileRows = positiveInteger(
    input.outputTileRows,
    "Output tile rows",
  );
  const decodableRows = positiveInteger(
    input.decodableRows,
    "Decodable rows",
  );
  if (
    decodableRows > geometry.rowCount ||
    inputRowCapacity > geometry.rowCount ||
    outputTileRows > decodableRows ||
    input.cached.manifestSha256 !== input.packageDirectory.manifestSha256 ||
    input.cached.shards.length !== input.packageDirectory.shards.length ||
    typeof input.rangeReader?.read !== "function"
  ) {
    throw diagnosticError(
      "tied-embedding-configuration-invalid",
      "The disk-backed tied embedding configuration is invalid",
    );
  }
  for (const [index, shard] of input.packageDirectory.shards.entries()) {
    const cached = input.cached.shards[index];
    if (
      cached === undefined ||
      cached.byteLength !== safeByteNumber(shard.length, "Package shard length") ||
      cached.sha256 !== shard.sha256 ||
      typeof cached.storagePath !== "string" ||
      cached.storagePath.length === 0
    ) {
      throw diagnosticError(
        "tied-embedding-package-invalid",
        "The authenticated tied embedding cache is invalid",
      );
    }
  }
  for (const segment of geometry.segments) {
    const shard = input.packageDirectory.shards[segment.source.shardIndex];
    if (
      shard === undefined ||
      segment.shardOffset + (segment.tensorEnd - segment.tensorStart) >
        safeByteNumber(shard.length, "Package shard length")
    ) {
      throw diagnosticError(
        "tied-embedding-package-invalid",
        "A tied embedding segment exceeds its authenticated shard",
      );
    }
  }

  const inputBytes = geometry.rowBytes * inputRowCapacity;
  const outputBytes = geometry.rowBytes * outputTileRows;
  const allocations: GpuAllocation[] = [];
  try {
    allocations.push(await input.arena.allocate({
      id: "tied-embedding-input-cache",
      category: "scratch",
      byteLength: BigInt(inputBytes),
      usage: GPU_STORAGE_AND_COPY_DST,
      alignment: 4,
      requiredShardQuantumBytes: BigInt(geometry.rowBytes),
    }));
    allocations.push(await input.arena.allocate({
      id: "tied-embedding-output-tile",
      category: "scratch",
      byteLength: BigInt(outputBytes),
      usage: GPU_STORAGE_AND_COPY_DST,
      alignment: 4,
      requiredShardQuantumBytes: BigInt(geometry.rowBytes),
    }));
    if (
      allocations.some((allocation) =>
        allocation.shards.length !== 1 ||
        allocation.shards[0]!.logicalByteOffset !== 0n ||
        allocation.shards[0]!.logicalByteLength !== allocation.logicalBytes
      )
    ) {
      throw new Error("cache allocation was split");
    }
  } catch {
    destroyAllocations(allocations);
    throw diagnosticError(
      "tied-embedding-cache-allocation-failed",
      "Disk-backed tied embedding cache allocation failed",
    );
  }

  const inputAllocation = allocations[0]!;
  const outputAllocation = allocations[1]!;
  const inputBuffer = inputAllocation.shards[0]!.buffer as object;
  const outputBuffer = outputAllocation.shards[0]!.buffer as object;
  const cacheGpuBytes = inputAllocation.allocatedBytes + outputAllocation.allocatedBytes;
  const slots = Array.from({ length: inputRowCapacity }, () => ({
    tokenId: null as number | null,
    stamp: 0,
  }));
  let stamp = 0;
  let diskReadBytes = 0;
  let maxDiskReadBytes = 0;
  let inputRowCacheHits = 0;
  let inputRowCacheMisses = 0;
  let prefillRowRequests = 0;
  let decodeRowRequests = 0;
  let outputTileReads = 0;
  let queueDirty = false;
  let accepting = true;
  let disposed = false;
  let disposePromise: Promise<void> | null = null;
  let operationTail: Promise<void> = Promise.resolve();
  const controllers = new Set<AbortController>();
  const operations = new Set<Promise<unknown>>();

  const retireQueue = async (): Promise<void> => {
    if (!queueDirty) return;
    try {
      await input.queue.onSubmittedWorkDone();
      queueDirty = false;
    } catch {
      throw diagnosticError(
        "tied-embedding-queue-retirement-failed",
        "Disk-backed tied embedding queue retirement failed",
      );
    }
  };

  const readTensorRange = async (
    tensorOffset: number,
    byteLength: number,
    signal: AbortSignal,
  ): Promise<Uint8Array> => {
    const tensorEnd = tensorOffset + byteLength;
    if (
      !Number.isSafeInteger(tensorOffset) ||
      tensorOffset < 0 ||
      !Number.isSafeInteger(byteLength) ||
      byteLength < 1 ||
      !Number.isSafeInteger(tensorEnd) ||
      tensorEnd > geometry.logicalBytes
    ) {
      throw diagnosticError(
        "tied-embedding-range-invalid",
        "Tied embedding logical range is invalid",
      );
    }
    const result = new Uint8Array(byteLength);
    let copied = 0;
    try {
      for (const segment of geometry.segments) {
        if (segment.tensorEnd <= tensorOffset) continue;
        if (segment.tensorStart >= tensorEnd) break;
        signal.throwIfAborted();
        const start = Math.max(tensorOffset, segment.tensorStart);
        const end = Math.min(tensorEnd, segment.tensorEnd);
        const rangeBytes = end - start;
        const cached = input.cached.shards[segment.source.shardIndex]!;
        const bytes = await input.rangeReader.read({
          storagePath: cached.storagePath,
          offset: segment.shardOffset + start - segment.tensorStart,
          byteLength: rangeBytes,
          signal,
        });
        signal.throwIfAborted();
        if (!(bytes instanceof Uint8Array) || bytes.byteLength !== rangeBytes) {
          throw new Error("range length mismatch");
        }
        result.set(bytes, start - tensorOffset);
        copied += rangeBytes;
      }
      if (copied !== byteLength) throw new Error("range coverage mismatch");
    } catch {
      if (signal.aborted) throw abortError();
      throw diagnosticError(
        "tied-embedding-range-read-failed",
        "Immutable tied embedding range read failed",
      );
    }
    diskReadBytes += byteLength;
    maxDiskReadBytes = Math.max(maxDiskReadBytes, byteLength);
    return result;
  };

  const schedule = <T>(
    externalSignal: AbortSignal,
    operation: (signal: AbortSignal) => Promise<T>,
  ): Promise<T> => {
    if (!accepting) {
      return Promise.reject(diagnosticError(
        "tied-embedding-disposed",
        "Disk-backed tied embedding store is disposed",
      ));
    }
    const controller = new AbortController();
    const abort = (): void => controller.abort(abortError());
    if (externalSignal.aborted) abort();
    else externalSignal.addEventListener("abort", abort, { once: true });
    controllers.add(controller);
    const pending = operationTail.then(async () => {
      controller.signal.throwIfAborted();
      return operation(controller.signal);
    });
    operationTail = pending.then(() => undefined, () => undefined);
    operations.add(pending);
    const release = (): void => {
      externalSignal.removeEventListener("abort", abort);
      controllers.delete(controller);
      operations.delete(pending);
    };
    void pending.then(release, release);
    return pending;
  };

  const stageInputRow = (request: {
    readonly tokenId: number;
    readonly phase: "prefill" | "decode";
    readonly signal: AbortSignal;
  }): Promise<Qwen35StagedPackedRows> => schedule(request.signal, async (signal) => {
    if (
      !Number.isSafeInteger(request.tokenId) ||
      request.tokenId < 0 ||
      request.tokenId >= geometry.rowCount ||
      (request.phase !== "prefill" && request.phase !== "decode")
    ) {
      throw diagnosticError(
        "tied-embedding-row-request-invalid",
        "Tied embedding row request is invalid",
      );
    }
    if (request.phase === "prefill") prefillRowRequests += 1;
    else decodeRowRequests += 1;
    const hitIndex = slots.findIndex((slot) => slot.tokenId === request.tokenId);
    if (hitIndex >= 0) {
      inputRowCacheHits += 1;
      slots[hitIndex]!.stamp = stamp += 1;
      return Object.freeze({
        tensorName: TIED_TENSOR_NAME,
        storageType: geometry.tensor.storageType,
        firstRow: request.tokenId,
        rowCount: 1,
        rowBytes: geometry.rowBytes,
        buffer: inputBuffer,
        bufferOffset: hitIndex * geometry.rowBytes,
        byteLength: geometry.rowBytes,
      });
    }
    inputRowCacheMisses += 1;
    const bytes = await readTensorRange(
      request.tokenId * geometry.rowBytes,
      geometry.rowBytes,
      signal,
    );
    signal.throwIfAborted();
    let slotIndex = slots.findIndex((slot) => slot.tokenId === null);
    if (slotIndex < 0) {
      slotIndex = slots.reduce(
        (oldest, slot, index) => slot.stamp < slots[oldest]!.stamp ? index : oldest,
        0,
      );
      // The consumer may still reference the old row. Fence before its slot is
      // overwritten; free slots require no such serialization.
      await retireQueue();
      signal.throwIfAborted();
    }
    try {
      input.queue.writeBuffer(
        inputBuffer,
        slotIndex * geometry.rowBytes,
        bytes,
        0,
        bytes.byteLength,
      );
      queueDirty = true;
    } catch {
      throw diagnosticError(
        "tied-embedding-cache-upload-failed",
        "Disk-backed tied embedding cache upload failed",
      );
    }
    slots[slotIndex]!.tokenId = request.tokenId;
    slots[slotIndex]!.stamp = stamp += 1;
    return Object.freeze({
      tensorName: TIED_TENSOR_NAME,
      storageType: geometry.tensor.storageType,
      firstRow: request.tokenId,
      rowCount: 1,
      rowBytes: geometry.rowBytes,
      buffer: inputBuffer,
      bufferOffset: slotIndex * geometry.rowBytes,
      byteLength: geometry.rowBytes,
    });
  });

  const selectTopK = (request: {
    readonly phase: "prefill" | "decode";
    readonly topK: number;
    readonly signal: AbortSignal;
    readonly scoreTile: (
      tile: Qwen35StagedPackedRows,
    ) => Promise<readonly Qwen35LogitCandidate[]> | readonly Qwen35LogitCandidate[];
  }): Promise<readonly Qwen35LogitCandidate[]> => schedule(request.signal, async (signal) => {
    if (
      (request.phase !== "prefill" && request.phase !== "decode") ||
      !Number.isSafeInteger(request.topK) ||
      request.topK < 1 ||
      request.topK > decodableRows ||
      typeof request.scoreTile !== "function"
    ) {
      throw diagnosticError(
        "tied-embedding-top-k-request-invalid",
        "Tied embedding top-k request is invalid",
      );
    }
    let winners: readonly Qwen35LogitCandidate[] = Object.freeze([]);
    for (let firstRow = 0; firstRow < decodableRows; firstRow += outputTileRows) {
      signal.throwIfAborted();
      const rowCount = Math.min(outputTileRows, decodableRows - firstRow);
      const bytes = await readTensorRange(
        firstRow * geometry.rowBytes,
        rowCount * geometry.rowBytes,
        signal,
      );
      outputTileReads += 1;
      try {
        input.queue.writeBuffer(outputBuffer, 0, bytes, 0, bytes.byteLength);
        queueDirty = true;
      } catch {
        throw diagnosticError(
          "tied-embedding-cache-upload-failed",
          "Disk-backed tied embedding cache upload failed",
        );
      }
      const tile = Object.freeze({
        tensorName: TIED_TENSOR_NAME,
        storageType: geometry.tensor.storageType,
        firstRow,
        rowCount,
        rowBytes: geometry.rowBytes,
        buffer: outputBuffer,
        bufferOffset: 0,
        byteLength: bytes.byteLength,
      });
      let candidates: readonly Qwen35LogitCandidate[];
      try {
        candidates = await request.scoreTile(tile);
      } catch {
        if (signal.aborted) throw abortError();
        throw diagnosticError(
          "tied-embedding-scoring-failed",
          "Disk-backed tied embedding tile scoring failed",
        );
      }
      if (!Array.isArray(candidates)) {
        throw diagnosticError(
          "tied-embedding-candidates-invalid",
          "Tied embedding tile candidates are invalid",
        );
      }
      for (const candidate of candidates) {
        if (
          typeof candidate !== "object" ||
          candidate === null ||
          !Number.isSafeInteger(candidate.tokenId) ||
          candidate.tokenId < firstRow ||
          candidate.tokenId >= firstRow + rowCount ||
          typeof candidate.score !== "number"
        ) {
          throw diagnosticError(
            "tied-embedding-candidates-invalid",
            "Tied embedding tile candidates are invalid",
          );
        }
      }
      winners = stableTopK([...winners, ...candidates], request.topK);
      // The tile buffer has one owner. It cannot be overwritten until every
      // consumer dispatch accepted by scoreTile has retired.
      await retireQueue();
      signal.throwIfAborted();
    }
    return winners;
  });

  const getMetrics = (): Qwen35TiedEmbeddingMetrics => Object.freeze({
    logicalTensorBytes: geometry.logicalBytes,
    permanentGpuBytes: 0,
    currentCacheGpuBytes: disposed ? 0 : Number(cacheGpuBytes),
    peakCacheGpuBytes: Number(cacheGpuBytes),
    diskReadBytes,
    maxDiskReadBytes,
    inputRowCacheHits,
    inputRowCacheMisses,
    prefillRowRequests,
    decodeRowRequests,
    outputTileReads,
  });

  const cancel = async (): Promise<void> => {
    for (const controller of controllers) controller.abort(abortError());
    await Promise.allSettled([...operations]);
  };

  const dispose = (): Promise<void> => {
    disposePromise ??= (async () => {
      accepting = false;
      await cancel();
      let failed = false;
      try {
        // Always fence: consumers can enqueue reads after the store's last
        // write, so a local dirty flag is not sufficient disposal evidence.
        await input.queue.onSubmittedWorkDone();
        queueDirty = false;
      } catch {
        failed = true;
      }
      failed = destroyAllocations(allocations) || failed;
      disposed = true;
      if (failed) {
        throw diagnosticError(
          "tied-embedding-cleanup-failed",
          "Disk-backed tied embedding cleanup failed",
        );
      }
    })();
    return disposePromise;
  };

  return Object.freeze({
    tensor: geometry.tensor,
    permanentGpuBytes: 0n as const,
    cacheGpuBytes,
    stageInputRow,
    selectTopK,
    cancel,
    dispose,
    getMetrics,
  });
}
