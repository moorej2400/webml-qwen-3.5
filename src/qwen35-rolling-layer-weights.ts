import {
  ALLOCATION_DIAGNOSTIC_CODES,
  RuntimeDiagnosticError,
  allocationDiagnosticError,
  diagnosticError,
} from "./diagnostics.js";
import type { CachedModelPackage } from "./opfs-model-cache.js";
import {
  hasQwen35DiskBackedTiedEmbedding,
  planQwen35TiedEmbeddingResidency,
  type Qwen35PackedRangeReader,
} from "./qwen35-disk-backed-tied-embedding.js";
import {
  allocateQwen35WeightDirectory,
  qwen35TensorWeightBytes,
  type Qwen35PackageDirectory,
  type Qwen35PackageSegment,
  type Qwen35PackageTensor,
  type Qwen35TensorWeight,
  type Qwen35WeightDirectory,
  type Qwen35WeightDirectoryView,
  type Qwen35WeightArena,
} from "./qwen35-weight-directory.js";
import type { Qwen35WeightWriteQueue } from "./qwen35-weight-upload.js";

const MODEL_LAYER_COUNT = 32;
/** The model order used by both full rolling and measured partial residency. */
export const QWEN35_ROLLING_LAYER_ORDER = Object.freeze(
  Array.from({ length: MODEL_LAYER_COUNT }, (_, layer) => layer),
);

const MAX_READ_CHUNK_BYTES = 32 * 1024 * 1024;
const MAX_UPLOAD_WINDOW_BYTES = 64 * 1024 * 1024;

export interface Qwen35RollingLayerResidency {
  readonly layer: number;
  readonly byteLength: bigint;
  readonly directory: Qwen35PackageDirectory;
}

export interface Qwen35RollingLayerResidencyPlan {
  readonly streamedLayers: readonly number[];
  readonly streamedBytes: bigint;
  readonly permanentBytes: bigint;
  readonly maxLayerBytes: bigint;
  readonly permanentDirectory: Qwen35PackageDirectory;
  readonly layers: readonly Qwen35RollingLayerResidency[];
}

export interface Qwen35RollingLayerMetrics {
  readonly currentGpuBytes: number;
  readonly peakGpuBytes: number;
  readonly diskReadBytes: number;
  readonly maxDiskReadBytes: number;
  readonly completedLayers: number;
  readonly failedLayers: number;
}

export interface Qwen35RollingLayerMutation {
  markStateMutation(): void;
}

export interface Qwen35RollingLayerStore {
  readonly streamedLayers: readonly number[];
  readonly poisoned: boolean;
  withLayer<T>(input: {
    readonly layer: number;
    readonly phase: "prefill" | "decode";
    readonly signal: AbortSignal;
    readonly execute: (
      weights: Qwen35WeightDirectoryView,
      mutation: Qwen35RollingLayerMutation,
    ) => Promise<T> | T;
  }): Promise<T>;
  cancel(): Promise<void>;
  dispose(): Promise<void>;
  getMetrics(): Qwen35RollingLayerMetrics;
}

export interface Qwen35RollingLayerInvocation {
  readonly layer: number;
  readonly kind: "gated-deltanet" | "full-attention";
}

const abortError = (): DOMException =>
  new DOMException("Operation cancelled", "AbortError");

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw diagnosticError(
      "rolling-layer-configuration-invalid",
      `${label} must be a positive safe integer`,
    );
  }
  return value;
}

function safeNumber(value: bigint, label: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw diagnosticError(
      "rolling-layer-package-invalid",
      `${label} cannot be represented exactly`,
    );
  }
  return Number(value);
}

function safeDecimal(value: string, label: string): number {
  if (!/^(?:0|[1-9][0-9]*)$/.test(value)) {
    throw diagnosticError(
      "rolling-layer-package-invalid",
      `${label} is invalid`,
    );
  }
  return safeNumber(BigInt(value), label);
}

function layerFromTensorName(name: string): number | null {
  const match = /^blk\.(0|[1-9][0-9]*)\./.exec(name);
  if (match === null) return null;
  const layer = Number(match[1]);
  return Number.isSafeInteger(layer) ? layer : null;
}

/**
 * A partial-residency experiment may stream a suffix of the static program.
 * Require model order here so an accidental reordering cannot change recurrent
 * execution or make two owners claim the same transformer layer.
 */
function normalizedStreamedLayers(
  streamedLayers: readonly number[] | undefined,
): readonly number[] {
  const selected = streamedLayers ?? QWEN35_ROLLING_LAYER_ORDER;
  if (
    selected.some((layer, index) =>
      !Number.isSafeInteger(layer) ||
      layer < 0 ||
      layer >= MODEL_LAYER_COUNT ||
      (index > 0 && selected[index - 1]! >= layer))
  ) {
    throw diagnosticError(
      "rolling-layer-configuration-invalid",
      "Streamed layers must be an ordered unique Qwen3.5 layer subset",
    );
  }
  return Object.freeze([...selected]);
}

/**
 * Turns the local hybrid experiment's resident-prefix count into the exact
 * streamed suffix. Keeping this mapping beside the layer-order invariant makes
 * loaders and ownership code use one validated partition.
 */
export function qwen35HybridStreamedLayers(
  residentLayerCount: number | undefined,
): readonly number[] {
  if (
    !Number.isSafeInteger(residentLayerCount) ||
    residentLayerCount === undefined ||
    residentLayerCount < 1 ||
    residentLayerCount >= MODEL_LAYER_COUNT
  ) {
    throw diagnosticError(
      "model-hybrid-resident-layer-count-invalid",
      "Hybrid residency requires between one and thirty-one resident layers",
    );
  }
  return QWEN35_ROLLING_LAYER_ORDER.slice(residentLayerCount);
}

function cloneSegment(segment: Qwen35PackageSegment): Qwen35PackageSegment {
  return Object.freeze({ ...segment });
}

function cloneTensor(tensor: Qwen35PackageTensor): Qwen35PackageTensor {
  return Object.freeze({
    ...tensor,
    shape: Object.freeze([...tensor.shape]),
    segments: Object.freeze(tensor.segments.map(cloneSegment)),
  });
}

function cloneDirectory(
  source: Qwen35PackageDirectory,
  tensors: readonly Qwen35PackageTensor[],
): Qwen35PackageDirectory {
  return Object.freeze({
    manifestSha256: source.manifestSha256,
    shards: Object.freeze(source.shards.map((shard) => Object.freeze({ ...shard }))),
    tensors: Object.freeze(tensors.map(cloneTensor)),
  });
}

/** True only when every model layer is represented, so small unit fixtures keep the resident path. */
export function hasQwen35RollingLayerSet(
  packageDirectory: Qwen35PackageDirectory,
): boolean {
  const layers = new Set<number>();
  let hasOutputNorm = false;
  for (const tensor of packageDirectory.tensors) {
    const layer = layerFromTensorName(tensor.name);
    if (layer !== null) layers.add(layer);
    if (tensor.name === "output_norm.weight") hasOutputNorm = true;
  }
  return hasOutputNorm && Array.from(
    { length: MODEL_LAYER_COUNT },
    (_, layer) => layer,
  ).every((layer) => layers.has(layer));
}

/**
 * Splits any ordered transformer-layer subset without changing tensor order or
 * bytes. Layers outside the subset remain owned by the permanent directory.
 */
export function planQwen35RollingLayerResidency(
  packageDirectory: Qwen35PackageDirectory,
  streamedLayers?: readonly number[],
): Qwen35RollingLayerResidencyPlan {
  const selectedLayers = normalizedStreamedLayers(streamedLayers);
  const streamedLayerSet = new Set<number>(selectedLayers);
  const byLayer = new Map<number, Qwen35PackageTensor[]>();
  for (const tensor of packageDirectory.tensors) {
    const layer = layerFromTensorName(tensor.name);
    if (layer === null || !streamedLayerSet.has(layer)) continue;
    const tensors = byLayer.get(layer) ?? [];
    tensors.push(tensor);
    byLayer.set(layer, tensors);
  }
  const layers = selectedLayers.map((layer) => {
    const tensors = byLayer.get(layer);
    if (tensors === undefined || tensors.length === 0) {
      throw diagnosticError(
        "rolling-layer-package-invalid",
        "The Qwen3.5 rolling layer package is incomplete",
      );
    }
    const directory = cloneDirectory(packageDirectory, tensors);
    return Object.freeze({
      layer,
      byteLength: qwen35TensorWeightBytes(directory),
      directory,
    });
  });
  const permanentDirectory = cloneDirectory(
    packageDirectory,
    packageDirectory.tensors.filter((tensor) => {
      const layer = layerFromTensorName(tensor.name);
      return layer === null || !streamedLayerSet.has(layer);
    }),
  );
  const streamedBytes = layers.reduce(
    (sum, layer) => sum + layer.byteLength,
    0n,
  );
  const maxLayerBytes = layers.reduce(
    (largest, layer) => layer.byteLength > largest ? layer.byteLength : largest,
    0n,
  );
  return Object.freeze({
    streamedLayers: selectedLayers,
    streamedBytes,
    permanentBytes: qwen35TensorWeightBytes(permanentDirectory),
    maxLayerBytes,
    permanentDirectory,
    layers: Object.freeze(layers),
  });
}

/** Applies tied-table and rolling-layer exclusions in their ownership order. */
export function qwen35PermanentWeightPackage(
  packageDirectory: Qwen35PackageDirectory,
  streamedLayers?: readonly number[],
  tiedEmbeddingResidency: "streamed" | "resident" = "streamed",
): Qwen35PackageDirectory {
  // Hybrid decode keeps the tied table resident because streaming it costs a
  // complete 526 MB vocabulary scan for every token, independent of layer I/O.
  const withoutTied =
    tiedEmbeddingResidency === "streamed" &&
      hasQwen35DiskBackedTiedEmbedding(packageDirectory)
    ? planQwen35TiedEmbeddingResidency(packageDirectory).permanentDirectory
    : packageDirectory;
  return hasQwen35RollingLayerSet(withoutTied)
    ? planQwen35RollingLayerResidency(withoutTied, streamedLayers).permanentDirectory
    : withoutTied;
}

function validateCache(
  packageDirectory: Qwen35PackageDirectory,
  cached: CachedModelPackage,
): void {
  if (
    cached.manifestSha256 !== packageDirectory.manifestSha256 ||
    cached.shards.length !== packageDirectory.shards.length
  ) {
    throw diagnosticError(
      "rolling-layer-package-invalid",
      "The authenticated rolling layer cache is invalid",
    );
  }
  for (const [index, shard] of packageDirectory.shards.entries()) {
    const cachedShard = cached.shards[index];
    if (
      cachedShard === undefined ||
      cachedShard.byteLength !== safeDecimal(shard.length, "Shard length") ||
      cachedShard.sha256 !== shard.sha256 ||
      typeof cachedShard.storagePath !== "string" ||
      cachedShard.storagePath.length === 0
    ) {
      throw diagnosticError(
        "rolling-layer-package-invalid",
        "The authenticated rolling layer cache is invalid",
      );
    }
  }
}

function orderedSegments(
  tensor: Qwen35PackageTensor,
): readonly Qwen35PackageSegment[] {
  return Object.freeze([...tensor.segments].sort((left, right) =>
    safeDecimal(left.tensorOffset, "Tensor offset") -
      safeDecimal(right.tensorOffset, "Tensor offset")));
}

function findWeight(
  directory: Qwen35WeightDirectory,
  name: string,
): Qwen35TensorWeight {
  const weight = directory.get(name);
  if (weight === undefined) {
    throw diagnosticError(
      "rolling-layer-directory-invalid",
      "A staged rolling layer weight is unavailable",
    );
  }
  return weight;
}

/** Allocates one layer at a time; exact tensor allocations avoid hidden binding padding. */
export async function createQwen35RollingLayerStore(input: {
  readonly arena: Qwen35WeightArena;
  readonly queue: Qwen35WeightWriteQueue;
  readonly packageDirectory: Qwen35PackageDirectory;
  readonly cached: CachedModelPackage;
  readonly rangeReader: Qwen35PackedRangeReader;
  readonly uploadLaneBytes: number;
  readonly readChunkBytes: number;
  /** Omitted keeps the known all-layer rolling fallback. */
  readonly streamedLayers?: readonly number[];
}): Promise<Qwen35RollingLayerStore> {
  const plan = planQwen35RollingLayerResidency(
    input.packageDirectory,
    input.streamedLayers,
  );
  validateCache(input.packageDirectory, input.cached);
  const uploadLaneBytes = positiveInteger(input.uploadLaneBytes, "Upload lane bytes");
  const readChunkBytes = positiveInteger(input.readChunkBytes, "Read chunk bytes");
  if (
    uploadLaneBytes > MAX_UPLOAD_WINDOW_BYTES ||
    readChunkBytes > MAX_READ_CHUNK_BYTES ||
    typeof input.rangeReader?.read !== "function"
  ) {
    throw diagnosticError(
      "rolling-layer-configuration-invalid",
      "The rolling layer I/O configuration is invalid",
    );
  }
  const layers = new Map(plan.layers.map((layer) => [layer.layer, layer] as const));
  let accepting = true;
  let poisoned = false;
  let currentGpuBytes = 0;
  let peakGpuBytes = 0;
  let diskReadBytes = 0;
  let maxDiskReadBytes = 0;
  let completedLayers = 0;
  let failedLayers = 0;
  let operationTail: Promise<void> = Promise.resolve();
  let disposePromise: Promise<void> | null = null;
  const uploadWindow = { outstandingBytes: 0 };
  const controllers = new Set<AbortController>();
  const operations = new Set<Promise<unknown>>();

  const retireUploadWindow = async (): Promise<void> => {
    if (uploadWindow.outstandingBytes === 0) return;
    await input.queue.onSubmittedWorkDone();
    uploadWindow.outstandingBytes = 0;
  };

  const schedule = <T>(
    externalSignal: AbortSignal,
    operation: (signal: AbortSignal) => Promise<T>,
  ): Promise<T> => {
    if (!accepting) {
      return Promise.reject(diagnosticError(
        poisoned ? "rolling-layer-poisoned" : "rolling-layer-disposed",
        poisoned
          ? "The rolling layer state must be disposed"
          : "The rolling layer store is disposed",
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

  const readRange = async (
    tensor: Qwen35PackageTensor,
    tensorOffset: number,
    byteLength: number,
    signal: AbortSignal,
  ): Promise<Uint8Array> => {
    const end = tensorOffset + byteLength;
    try {
      const segments = orderedSegments(tensor);
      const intersections = segments.flatMap((segment) => {
        const segmentStart = safeDecimal(segment.tensorOffset, "Tensor offset");
        const segmentBytes = safeDecimal(segment.length, "Segment length");
        const segmentEnd = segmentStart + segmentBytes;
        if (segmentEnd <= tensorOffset || segmentStart >= end) return [];
        return [{
          segment,
          segmentStart,
          start: Math.max(tensorOffset, segmentStart),
          stop: Math.min(end, segmentEnd),
        }];
      });
      const readIntersection = async (
        intersection: typeof intersections[number],
      ): Promise<Uint8Array> => {
        signal.throwIfAborted();
        const rangeBytes = intersection.stop - intersection.start;
        const shard = input.packageDirectory.shards[intersection.segment.shardIndex];
        const cached = input.cached.shards[intersection.segment.shardIndex];
        if (shard === undefined || cached === undefined || rangeBytes < 1) {
          throw new Error("invalid segment");
        }
        const sourceOffset = safeDecimal(
          intersection.segment.shardOffset,
          "Shard offset",
        ) + intersection.start - intersection.segmentStart;
        if (sourceOffset + rangeBytes > safeDecimal(shard.length, "Shard length")) {
          throw new Error("segment exceeds shard");
        }
        const bytes = await input.rangeReader.read({
          storagePath: cached.storagePath,
          offset: sourceOffset,
          byteLength: rangeBytes,
          signal,
        });
        signal.throwIfAborted();
        if (!(bytes instanceof Uint8Array) || bytes.byteLength !== rangeBytes) {
          throw new Error("range length mismatch");
        }
        diskReadBytes += rangeBytes;
        maxDiskReadBytes = Math.max(maxDiskReadBytes, rangeBytes);
        return bytes;
      };

      // Normal converted tensors are row-contiguous inside one immutable shard.
      // Return that authenticated OPFS buffer directly so a 32 MiB read does not
      // coexist with another 32 MiB assembly buffer before queue.writeBuffer.
      const direct = intersections[0];
      if (
        intersections.length === 1 &&
        direct !== undefined &&
        direct.start === tensorOffset &&
        direct.stop === end
      ) {
        return await readIntersection(direct);
      }

      const output = new Uint8Array(byteLength);
      let copied = 0;
      for (const intersection of intersections) {
        const bytes = await readIntersection(intersection);
        output.set(bytes, intersection.start - tensorOffset);
        copied += bytes.byteLength;
      }
      if (copied !== byteLength) throw new Error("range coverage mismatch");
      return output;
    } catch {
      if (signal.aborted) throw abortError();
      throw diagnosticError(
        "rolling-layer-range-read-failed",
        "An immutable rolling layer range read failed",
      );
    }
  };

  const uploadLayer = async (
    layer: Qwen35RollingLayerResidency,
    directory: Qwen35WeightDirectory,
    signal: AbortSignal,
  ): Promise<void> => {
    for (const tensor of layer.directory.tensors) {
      const weight = findWeight(directory, tensor.name);
      for (const view of weight.physicalRows) {
        let localOffset = 0;
        while (localOffset < view.byteLength) {
          signal.throwIfAborted();
          const byteLength = Math.min(
            readChunkBytes,
            uploadLaneBytes,
            view.byteLength - localOffset,
          );
          const bytes = await readRange(
            tensor,
            view.tensorByteOffset + localOffset,
            byteLength,
            signal,
          );
          signal.throwIfAborted();
          if (
            uploadWindow.outstandingBytes > 0 &&
            uploadWindow.outstandingBytes + bytes.byteLength > uploadLaneBytes
          ) {
            // GPUQueue.writeBuffer copies into queue-owned staging. Retire the
            // bounded copy window before accepting another lane of bytes.
            await retireUploadWindow();
            signal.throwIfAborted();
          }
          try {
            input.queue.writeBuffer(
              view.buffer,
              view.bufferByteOffset + localOffset,
              bytes,
              0,
              bytes.byteLength,
            );
          } catch {
            throw diagnosticError(
              "rolling-layer-upload-failed",
              "A rolling layer GPU upload failed",
            );
          }
          uploadWindow.outstandingBytes += bytes.byteLength;
          localOffset += byteLength;
        }
      }
    }
  };

  const withLayer = <T>(request: {
    readonly layer: number;
    readonly phase: "prefill" | "decode";
    readonly signal: AbortSignal;
    readonly execute: (
      weights: Qwen35WeightDirectoryView,
      mutation: Qwen35RollingLayerMutation,
    ) => Promise<T> | T;
  }): Promise<T> => schedule(request.signal, async (signal) => {
    if (poisoned) {
      throw diagnosticError(
        "rolling-layer-poisoned",
        "The rolling layer state must be disposed",
      );
    }
    const layer = layers.get(request.layer);
    if (
      layer === undefined ||
      (request.phase !== "prefill" && request.phase !== "decode") ||
      typeof request.execute !== "function"
    ) {
      throw diagnosticError(
        "rolling-layer-request-invalid",
        "The rolling layer request is invalid",
      );
    }
    let directory: Qwen35WeightDirectory | null = null;
    let mutationMarked = false;
    let primaryError: unknown;
    let result: T | undefined;
    let unsafeCleanup = false;
    let phase: "allocate" | "upload" | "execute" | "retire" = "allocate";
    try {
      const layerArena: Qwen35WeightArena = {
        allocate: (allocationRequest) => input.arena.allocate({
          ...allocationRequest,
          // The permanent directory uses model-tensor-N. A layer prefix keeps
          // this rolling allocation from colliding in the shared ledger.
          id: `rolling-layer-${request.layer}-${allocationRequest.id}`,
        }),
      };
      directory = await allocateQwen35WeightDirectory(
        layerArena,
        layer.directory,
      );
      currentGpuBytes = safeNumber(directory.allocatedBytes, "Layer GPU bytes");
      peakGpuBytes = Math.max(peakGpuBytes, currentGpuBytes);
      phase = "upload";
      // GPUQueue.writeBuffer snapshots its CPU source synchronously. Queue order
      // keeps these writes before layer dispatch; the ownership fence below is
      // the only completion wait needed before buffer destruction.
      await uploadLayer(layer, directory, signal);
      signal.throwIfAborted();
      phase = "execute";
      result = await request.execute(directory.view, Object.freeze({
        markStateMutation(): void {
          mutationMarked = true;
        },
      }));
    } catch (error) {
      primaryError = error;
    }

    // The callback may have submitted work that reads every staged tensor.
    // Fence even when upload already retired, then release ledger ownership.
    if (directory !== null) {
      // Preserve an execution failure as the primary classification. When the
      // callback succeeded, a failed ownership fence is a retirement failure:
      // the store must be poisoned because buffer lifetime is now uncertain.
      if (primaryError === undefined) phase = "retire";
      try {
        await input.queue.onSubmittedWorkDone();
        uploadWindow.outstandingBytes = 0;
      } catch (error) {
        unsafeCleanup = true;
        primaryError ??= error;
      }
      try {
        directory.destroy();
      } catch (error) {
        unsafeCleanup = true;
        primaryError ??= error;
      }
      currentGpuBytes = 0;
    }

    if (unsafeCleanup) {
      poisoned = true;
      accepting = false;
    }

    if (primaryError !== undefined) {
      failedLayers += 1;
      if (mutationMarked) {
        poisoned = true;
        accepting = false;
      }
      if (
        primaryError instanceof DOMException &&
        primaryError.name === "AbortError"
      ) {
        throw abortError();
      }
      const code = (primaryError as { readonly code?: unknown }).code;
      if (code === "rolling-layer-range-read-failed") throw primaryError;
      // Preserve only runtime-owned diagnostic codes. Arbitrary callback errors
      // remain wrapped so browser/compiler text cannot cross the control boundary.
      if (phase === "execute" && primaryError instanceof RuntimeDiagnosticError) {
        throw primaryError;
      }
      if (
        phase === "allocate" &&
        typeof code === "string" &&
        (ALLOCATION_DIAGNOSTIC_CODES as readonly string[]).includes(code)
      ) {
        // Keep the arena's fixed code, but replace all browser-provided text.
        throw allocationDiagnosticError(code);
      }
      throw diagnosticError(
        phase === "execute"
          ? "rolling-layer-execution-failed"
          : phase === "retire"
            ? "rolling-layer-retirement-failed"
          : phase === "upload"
            ? "rolling-layer-upload-failed"
            : "rolling-layer-allocation-failed",
        phase === "execute"
          ? "Rolling layer execution failed"
          : phase === "retire"
            ? "Rolling layer retirement failed"
          : phase === "upload"
            ? "Rolling layer upload failed"
            : "Rolling layer allocation failed",
      );
    }
    completedLayers += 1;
    return result as T;
  });

  const cancel = async (): Promise<void> => {
    for (const controller of controllers) controller.abort(abortError());
    await Promise.allSettled([...operations]);
  };

  const dispose = (): Promise<void> => {
    disposePromise ??= (async () => {
      accepting = false;
      await cancel();
    })();
    return disposePromise;
  };

  return Object.freeze({
    streamedLayers: plan.streamedLayers,
    get poisoned(): boolean { return poisoned; },
    withLayer,
    cancel,
    dispose,
    getMetrics(): Qwen35RollingLayerMetrics {
      return Object.freeze({
        currentGpuBytes,
        peakGpuBytes,
        diskReadBytes,
        maxDiskReadBytes,
        completedLayers,
        failedLayers,
      });
    },
  });
}

const FULL_ATTENTION_LAYERS = new Set([3, 7, 11, 15, 19, 23, 27, 31]);

/** Executes model order, even though the disk-residency policy is grouped by size. */
export async function executeQwen35RollingLayerSequence<
  TInvocation extends Qwen35RollingLayerInvocation,
>(input: {
  readonly invocations: readonly TInvocation[];
  readonly permanentWeights: Qwen35WeightDirectoryView;
  readonly rollingStore: Qwen35RollingLayerStore;
  readonly phase: "prefill" | "decode";
  readonly signal: AbortSignal;
  readonly execute: (input: {
    readonly invocation: TInvocation;
    readonly weights: Qwen35WeightDirectoryView;
    readonly mutation: Qwen35RollingLayerMutation;
    /** True only while bindings reference buffers destroyed after this callback. */
    readonly transientWeights: boolean;
  }) => Promise<void> | void;
  readonly poison: () => void;
}): Promise<void> {
  if (
    input.invocations.length < 1 ||
    input.invocations.length > MODEL_LAYER_COUNT ||
    input.invocations.some((invocation, index) =>
      invocation.layer !== index ||
      invocation.kind !== (FULL_ATTENTION_LAYERS.has(index)
        ? "full-attention"
        : "gated-deltanet")) ||
    (input.phase !== "prefill" && input.phase !== "decode") ||
    typeof input.execute !== "function" ||
    typeof input.poison !== "function"
  ) {
    throw diagnosticError(
      "rolling-layer-sequence-invalid",
      "The Qwen3.5 rolling layer sequence is invalid",
    );
  }
  const streamed = new Set(input.rollingStore.streamedLayers);
  let tokenMutated = false;
  try {
    for (const invocation of input.invocations) {
      input.signal.throwIfAborted();
      if (streamed.has(invocation.layer)) {
        await input.rollingStore.withLayer({
          layer: invocation.layer,
          phase: input.phase,
          signal: input.signal,
          execute: (weights, layerMutation) => input.execute({
            invocation,
            weights,
            transientWeights: true,
            mutation: Object.freeze({
              markStateMutation(): void {
                tokenMutated = true;
                layerMutation.markStateMutation();
              },
            }),
          }),
        });
      } else {
        await input.execute({
          invocation,
          weights: input.permanentWeights,
          transientWeights: false,
          mutation: Object.freeze({
            markStateMutation(): void { tokenMutated = true; },
          }),
        });
      }
    }
  } catch (error) {
    if (tokenMutated || input.rollingStore.poisoned) input.poison();
    throw error;
  }
}
