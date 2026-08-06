import { diagnosticError } from "./diagnostics.js";
import type {
  GpuAllocation,
  GpuAllocationRequest,
} from "./gpu-arena.js";

const GPU_STORAGE_AND_COPY_DST = 0x0080 | 0x0008;

export interface Qwen35PackageSegment {
  readonly shardIndex: number;
  readonly shardOffset: string;
  readonly tensorOffset: string;
  readonly length: string;
}

export interface Qwen35PackageTensor {
  readonly name: string;
  readonly shape: readonly number[];
  readonly ggmlType: number;
  readonly storageType: string;
  readonly segments: readonly Qwen35PackageSegment[];
}

export interface Qwen35PackageDirectory {
  readonly manifestSha256: string;
  readonly shards: readonly {
    readonly index: number;
    readonly url: string;
    readonly offset: string;
    readonly length: string;
    readonly sha256: string;
  }[];
  readonly tensors: readonly Qwen35PackageTensor[];
}

export interface Qwen35WeightArena {
  allocate(request: GpuAllocationRequest): Promise<GpuAllocation>;
}

/** A contiguous set of complete packed rows in one physical GPU buffer. */
export interface Qwen35PhysicalRowView {
  /** Opaque bindable handle; destruction remains with the loader owner. */
  readonly buffer: object;
  readonly firstRow: number;
  readonly rowCount: number;
  readonly tensorByteOffset: number;
  readonly bufferByteOffset: number;
  readonly byteLength: number;
}

export interface Qwen35TensorWeightView {
  readonly name: string;
  readonly shape: readonly number[];
  readonly ggmlType: number;
  readonly storageType: string;
  readonly rowBytes: number;
  readonly rowCount: number;
  readonly logicalBytes: bigint;
  readonly physicalRows: readonly Qwen35PhysicalRowView[];
}

export interface Qwen35TensorWeight extends Qwen35TensorWeightView {
  readonly allocation: GpuAllocation;
  readonly segments: readonly Qwen35PackageSegment[];
  readonly view: Qwen35TensorWeightView;
}

export interface Qwen35WeightDirectoryView
  extends Iterable<readonly [string, Qwen35TensorWeightView]> {
  readonly size: number;
  readonly tensors: readonly Qwen35TensorWeightView[];
  readonly logicalBytes: bigint;
  readonly allocatedBytes: bigint;
  get(name: string): Qwen35TensorWeightView | undefined;
  entries(): MapIterator<[string, Qwen35TensorWeightView]>;
}

/** Model-specific weight ownership; package shards never become GPU objects. */
export interface Qwen35WeightDirectory
  extends Iterable<readonly [string, Qwen35TensorWeight]> {
  readonly size: number;
  readonly manifestSha256: string;
  readonly packageShards: Qwen35PackageDirectory["shards"];
  readonly tensors: readonly Qwen35TensorWeight[];
  readonly allocations: readonly GpuAllocation[];
  readonly logicalBytes: bigint;
  readonly allocatedBytes: bigint;
  readonly view: Qwen35WeightDirectoryView;
  get(name: string): Qwen35TensorWeight | undefined;
  entries(): MapIterator<[string, Qwen35TensorWeight]>;
  destroy(): void;
}

const PACKED_LAYOUTS = new Map<string, {
  readonly valuesPerBlock: number;
  readonly bytesPerBlock: number;
}>([
  ["f32", { valuesPerBlock: 1, bytesPerBlock: 4 }],
  ["q8-0-36", { valuesPerBlock: 32, bytesPerBlock: 36 }],
  ["q3-k-112", { valuesPerBlock: 256, bytesPerBlock: 112 }],
  ["q4-k-144", { valuesPerBlock: 256, bytesPerBlock: 144 }],
  ["q5-k-176", { valuesPerBlock: 256, bytesPerBlock: 176 }],
  ["q6-k-212", { valuesPerBlock: 256, bytesPerBlock: 212 }],
]);

function safePositiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw diagnosticError(
      "model-weight-shape-invalid",
      `${label} must be a positive safe integer`,
    );
  }
  return value;
}

function safeByteNumber(value: bigint, label: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw diagnosticError(
      "model-weight-size-unsafe",
      `${label} cannot be represented exactly`,
    );
  }
  return Number(value);
}

function packedShape(tensor: Qwen35PackageTensor): {
  readonly rowBytes: number;
  readonly rowCount: number;
  readonly logicalBytes: bigint;
} {
  const layout = PACKED_LAYOUTS.get(tensor.storageType);
  if (layout === undefined || tensor.shape.length === 0) {
    throw diagnosticError(
      "model-weight-layout-unsupported",
      "The Qwen3.5 weight has an unsupported packed layout",
    );
  }
  const width = safePositiveInteger(tensor.shape[0]!, "Tensor row width");
  if (width % layout.valuesPerBlock !== 0) {
    throw diagnosticError(
      "model-weight-row-invalid",
      "The Qwen3.5 weight row splits a packed block",
    );
  }
  const rowBytes = safePositiveInteger(
    (width / layout.valuesPerBlock) * layout.bytesPerBlock,
    "Packed row byte length",
  );
  let rowCount = 1;
  for (const dimension of tensor.shape.slice(1)) {
    const safeDimension = safePositiveInteger(dimension, "Tensor dimension");
    rowCount = safePositiveInteger(
      rowCount * safeDimension,
      "Tensor row count",
    );
  }
  const logicalBytes = BigInt(rowBytes) * BigInt(rowCount);
  safeByteNumber(logicalBytes, "Tensor byte length");
  return { rowBytes, rowCount, logicalBytes };
}

function validateSegments(
  tensor: Qwen35PackageTensor,
  logicalBytes: bigint,
  rowBytes: number,
): readonly Qwen35PackageSegment[] {
  const segments = [...tensor.segments].sort((left, right) => {
    const leftOffset = BigInt(left.tensorOffset);
    const rightOffset = BigInt(right.tensorOffset);
    return leftOffset < rightOffset ? -1 : leftOffset > rightOffset ? 1 : 0;
  });
  let covered = 0n;
  for (const segment of segments) {
    const offset = BigInt(segment.tensorOffset);
    const length = BigInt(segment.length);
    if (
      offset !== covered ||
      length <= 0n ||
      offset % BigInt(rowBytes) !== 0n ||
      length % BigInt(rowBytes) !== 0n
    ) {
      throw diagnosticError(
        "model-weight-segments-invalid",
        "The Qwen3.5 weight segments do not form exact contiguous storage",
      );
    }
    covered += length;
  }
  if (covered !== logicalBytes) {
    throw diagnosticError(
      "model-weight-segments-invalid",
      "The Qwen3.5 weight segments do not cover its packed shape",
    );
  }
  return Object.freeze(segments.map((segment) => Object.freeze({ ...segment })));
}

function physicalRowViews(
  allocation: GpuAllocation,
  rowBytes: number,
  logicalBytes: bigint,
): readonly Qwen35PhysicalRowView[] {
  if (allocation.logicalBytes !== logicalBytes) {
    throw diagnosticError(
      "model-weight-allocation-invalid",
      "The GPU allocation does not match the Qwen3.5 weight",
    );
  }
  const quantum = BigInt(rowBytes);
  let covered = 0n;
  const views: Qwen35PhysicalRowView[] = [];
  for (const shard of allocation.shards) {
    if (
      shard.logicalByteOffset !== covered ||
      shard.logicalByteOffset % quantum !== 0n ||
      shard.logicalByteLength <= 0n ||
      shard.logicalByteLength % quantum !== 0n ||
      shard.allocatedByteLength < shard.logicalByteLength
    ) {
      throw diagnosticError(
        "model-weight-row-split-invalid",
        "A physical GPU buffer splits a packed Qwen3.5 row",
      );
    }
    views.push(Object.freeze({
      buffer: shard.buffer as object,
      firstRow: safeByteNumber(shard.logicalByteOffset / quantum, "First row"),
      rowCount: safeByteNumber(shard.logicalByteLength / quantum, "Row count"),
      tensorByteOffset: safeByteNumber(shard.logicalByteOffset, "Tensor byte offset"),
      bufferByteOffset: 0,
      byteLength: safeByteNumber(shard.logicalByteLength, "Physical row bytes"),
    }));
    covered += shard.logicalByteLength;
  }
  if (covered !== logicalBytes) {
    throw diagnosticError(
      "model-weight-allocation-invalid",
      "The GPU buffers do not cover the complete Qwen3.5 weight",
    );
  }
  return Object.freeze(views);
}

function destroyReverse(allocations: readonly GpuAllocation[]): unknown {
  let firstError: unknown;
  for (let index = allocations.length - 1; index >= 0; index -= 1) {
    try {
      allocations[index]!.destroy();
    } catch (error) {
      firstError ??= error;
    }
  }
  return firstError;
}

/** Allocates each tensor independently so every physical split is row-safe. */
export async function allocateQwen35WeightDirectory(
  arena: Qwen35WeightArena,
  packageDirectory: Qwen35PackageDirectory,
  options: {
    readonly onProgress?: (completedBytes: number) => void;
  } = {},
): Promise<Qwen35WeightDirectory> {
  const allocations: GpuAllocation[] = [];
  const tensors: Qwen35TensorWeight[] = [];
  const byName = new Map<string, Qwen35TensorWeight>();
  let completedBytes = 0;
  try {
    for (const [index, tensor] of packageDirectory.tensors.entries()) {
      if (byName.has(tensor.name)) {
        throw diagnosticError(
          "model-weight-name-duplicate",
          "The Qwen3.5 weight directory contains a duplicate tensor",
        );
      }
      const shape = packedShape(tensor);
      const segments = validateSegments(
        tensor,
        shape.logicalBytes,
        shape.rowBytes,
      );
      const allocation = await arena.allocate({
        id: `model-tensor-${index}`,
        category: "model",
        byteLength: shape.logicalBytes,
        usage: GPU_STORAGE_AND_COPY_DST,
        alignment: 4,
        // GpuArena may reduce its physical cap, but it must never split a row.
        requiredShardQuantumBytes: BigInt(shape.rowBytes),
      });
      allocations.push(allocation);
      completedBytes += safeByteNumber(shape.logicalBytes, "Allocated weight bytes");
      options.onProgress?.(completedBytes);
      const physicalRows = physicalRowViews(
        allocation,
        shape.rowBytes,
        shape.logicalBytes,
      );
      const tensorShape = Object.freeze([...tensor.shape]);
      const view: Qwen35TensorWeightView = Object.freeze({
        name: tensor.name,
        shape: tensorShape,
        ggmlType: tensor.ggmlType,
        storageType: tensor.storageType,
        rowBytes: shape.rowBytes,
        rowCount: shape.rowCount,
        logicalBytes: shape.logicalBytes,
        physicalRows,
      });
      const weight = Object.freeze({
        ...view,
        allocation,
        segments,
        view,
      });
      tensors.push(weight);
      byName.set(weight.name, weight);
    }
  } catch (error) {
    const rollbackError = destroyReverse(allocations);
    if (rollbackError !== undefined) {
      throw diagnosticError(
        "model-weight-allocation-rollback-failed",
        "Qwen3.5 weight allocation rollback did not complete",
      );
    }
    throw error;
  }

  const ownedAllocations = Object.freeze([...allocations]);
  const ownedTensors = Object.freeze([...tensors]);
  const ownedPackageShards = Object.freeze(
    packageDirectory.shards.map((shard) => Object.freeze({ ...shard })),
  );
  const viewTensors = Object.freeze(ownedTensors.map((tensor) => tensor.view));
  const viewByName = new Map(
    viewTensors.map((tensor) => [tensor.name, tensor] as const),
  );
  const logicalBytes = tensors.reduce(
    (sum, tensor) => sum + tensor.logicalBytes,
    0n,
  );
  const allocatedBytes = allocations.reduce(
    (sum, allocation) => sum + allocation.allocatedBytes,
    0n,
  );
  const view: Qwen35WeightDirectoryView = Object.freeze({
    size: viewByName.size,
    tensors: viewTensors,
    logicalBytes,
    allocatedBytes,
    get: (name: string) => viewByName.get(name),
    entries: () => viewByName.entries(),
    [Symbol.iterator]: () => viewByName[Symbol.iterator](),
  });
  let destroyed = false;
  return Object.freeze({
    size: byName.size,
    manifestSha256: packageDirectory.manifestSha256,
    packageShards: ownedPackageShards,
    tensors: ownedTensors,
    allocations: ownedAllocations,
    logicalBytes,
    allocatedBytes,
    view,
    get: (name: string) => byName.get(name),
    entries: () => byName.entries(),
    [Symbol.iterator]: () => byName[Symbol.iterator](),
    destroy() {
      if (destroyed) return;
      destroyed = true;
      const error = destroyReverse(ownedAllocations);
      if (error !== undefined) throw error;
    },
  });
}

export function qwen35TensorWeightBytes(
  packageDirectory: Qwen35PackageDirectory,
): bigint {
  return packageDirectory.tensors.reduce(
    (sum, tensor) => sum + packedShape(tensor).logicalBytes,
    0n,
  );
}
