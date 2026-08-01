import {
  GgmlType,
  ggmlTensorByteLength,
  ggmlTypeLayout,
  type GgufTensorInfo,
  type ParsedGguf,
  type RandomAccessReader,
} from "./gguf.js";
import {
  MAX_EXCLUDED_TENSORS,
  MAX_SHARDS,
  MAX_TENSOR_SEGMENTS,
  validateModelPackageManifest,
  type ImmutableArtifactIdentity,
  type ModelPackageManifest,
  type PackageKind,
  type TensorStorageType,
  type VisionProcessorSettings,
} from "./manifest.js";
import {
  repackNativeQ6K,
  repackNativeQ8_0,
} from "./mixed-quant.js";
import { repackNativeQ3K } from "./q3k.js";
import {
  MTP_EXCLUSION_REASON,
  WEBGPU_LANGUAGE_TENSOR_LAYOUTS,
  isMtpTensorName,
  type WebGpuTensorTransform,
} from "./tensor-policy.js";

export { MTP_EXCLUSION_REASON } from "./tensor-policy.js";

export const MAX_STREAM_READ_BYTES = 8 * 1024 * 1024;

export interface ConverterOptions {
  readonly maxShardBytes: bigint;
  readonly tensorAlignment: number;
  /** Optional stricter limits; values cannot exceed the manifest bounds. */
  readonly planningLimits?: {
    readonly maxShards: number;
    readonly maxTensorSegments: number;
    readonly maxExcludedTensors: number;
  };
}

export interface TensorTypeInventory {
  readonly ggmlType: GgmlType;
  readonly tensorCount: number;
  readonly sourceBytes: bigint;
}

export interface PlannedSegment {
  readonly tensorName: string;
  readonly dimensions: readonly bigint[];
  readonly ggmlType: GgmlType;
  readonly storageType: TensorStorageType;
  readonly transform: WebGpuTensorTransform;
  readonly blockElements: number;
  readonly shard: number;
  readonly shardOffset: bigint;
  readonly tensorOffset: bigint;
  readonly sourceOffset: bigint;
  readonly sourceLength: bigint;
  readonly outputLength: bigint;
  readonly blockCount: bigint;
  readonly sourceBlockBytes: number;
  readonly outputBlockBytes: number;
}

export interface ConversionPlan {
  readonly inventory: readonly TensorTypeInventory[];
  readonly excludedTensors: readonly {
    name: string;
    reason: typeof MTP_EXCLUSION_REASON;
  }[];
  readonly shards: readonly { index: number; length: bigint }[];
  readonly segments: readonly PlannedSegment[];
}

export interface RandomAccessWriter {
  write(offset: bigint, bytes: Uint8Array): Promise<void>;
}

interface TypeLayout {
  readonly sourceBytes: number;
  readonly outputBytes: number;
  readonly blockElements: number;
  readonly storageType: TensorStorageType;
  readonly transform: PlannedSegment["transform"];
}

function nativeCopyLayout(type: GgmlType): TypeLayout {
  const layout = ggmlTypeLayout(type);
  return {
    sourceBytes: Number(layout.blockBytes),
    outputBytes: Number(layout.blockBytes),
    blockElements: Number(layout.blockElements),
    storageType: "raw",
    transform: "copy",
  };
}

const TYPE_LAYOUTS = new Map<GgmlType, TypeLayout>([
  [GgmlType.F16, nativeCopyLayout(GgmlType.F16)],
  [GgmlType.BF16, nativeCopyLayout(GgmlType.BF16)],
  [GgmlType.I8, nativeCopyLayout(GgmlType.I8)],
  [GgmlType.I16, nativeCopyLayout(GgmlType.I16)],
  [GgmlType.I32, nativeCopyLayout(GgmlType.I32)],
  [GgmlType.I64, nativeCopyLayout(GgmlType.I64)],
  [GgmlType.F64, nativeCopyLayout(GgmlType.F64)],
  [GgmlType.Q4_0, nativeCopyLayout(GgmlType.Q4_0)],
  [GgmlType.Q4_1, nativeCopyLayout(GgmlType.Q4_1)],
  [GgmlType.Q5_0, nativeCopyLayout(GgmlType.Q5_0)],
  [GgmlType.Q5_1, nativeCopyLayout(GgmlType.Q5_1)],
  [GgmlType.Q8_1, nativeCopyLayout(GgmlType.Q8_1)],
  [GgmlType.Q2_K, nativeCopyLayout(GgmlType.Q2_K)],
  [GgmlType.Q8_K, nativeCopyLayout(GgmlType.Q8_K)],
]);
for (const policy of WEBGPU_LANGUAGE_TENSOR_LAYOUTS) {
  TYPE_LAYOUTS.set(policy.ggmlType, {
    sourceBytes: policy.sourceBlockBytes,
    outputBytes: policy.outputBlockBytes,
    blockElements: policy.blockElements,
    storageType: policy.storageType,
    transform: policy.transform,
  });
}

function tensorLayout(tensor: GgufTensorInfo): {
  layout: TypeLayout;
  blockCount: bigint;
  sourceBytes: bigint;
} {
  const layout = TYPE_LAYOUTS.get(tensor.type);
  if (layout === undefined) {
    throw new Error(
      `Unsupported GGML tensor type ${tensor.type} for ${tensor.name}`,
    );
  }
  const sourceBytes = ggmlTensorByteLength(tensor.type, tensor.dimensions);
  const blockCount = sourceBytes / BigInt(layout.sourceBytes);
  return {
    layout,
    blockCount,
    sourceBytes,
  };
}

function align(value: bigint, alignment: number): bigint {
  const boundary = BigInt(alignment);
  return ((value + boundary - 1n) / boundary) * boundary;
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function planningLimit(
  value: number,
  manifestBound: number,
  label: string,
): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > manifestBound) {
    throw new Error(
      `${label} planning limit must be positive and cannot exceed the manifest bound`,
    );
  }
  return value;
}

/**
 * Creates a byte-range plan from the GGUF directory. Source payloads are not
 * read during planning, and every segment contains complete contiguous rows.
 */
export function planConversion(
  gguf: ParsedGguf,
  options: ConverterOptions,
): ConversionPlan {
  if (options.maxShardBytes < 1n) {
    throw new Error("maxShardBytes must be greater than zero");
  }
  if (
    !Number.isSafeInteger(options.tensorAlignment) ||
    options.tensorAlignment < 1 ||
    options.tensorAlignment > 1024 * 1024 ||
    (options.tensorAlignment & (options.tensorAlignment - 1)) !== 0
  ) {
    throw new Error(
      "tensorAlignment must be a positive power of two within the configured bound",
    );
  }
  const requestedLimits = options.planningLimits ?? {
    maxShards: MAX_SHARDS,
    maxTensorSegments: MAX_TENSOR_SEGMENTS,
    maxExcludedTensors: MAX_EXCLUDED_TENSORS,
  };
  const limits = {
    maxShards: planningLimit(
      requestedLimits.maxShards,
      MAX_SHARDS,
      "Shard count",
    ),
    maxTensorSegments: planningLimit(
      requestedLimits.maxTensorSegments,
      MAX_TENSOR_SEGMENTS,
      "Tensor segment count",
    ),
    maxExcludedTensors: planningLimit(
      requestedLimits.maxExcludedTensors,
      MAX_EXCLUDED_TENSORS,
      "Excluded tensor count",
    ),
  };

  const inventoryByType = new Map<
    GgmlType,
    { tensorCount: number; sourceBytes: bigint }
  >();
  const included: Array<{
    tensor: GgufTensorInfo;
    layout: TypeLayout;
    blockCount: bigint;
  }> = [];
  const excludedTensors: Array<{
    name: string;
    reason: typeof MTP_EXCLUSION_REASON;
  }> = [];

  for (const tensor of gguf.tensors) {
    const { layout, blockCount, sourceBytes } = tensorLayout(tensor);
    const inventory = inventoryByType.get(tensor.type) ?? {
      tensorCount: 0,
      sourceBytes: 0n,
    };
    inventory.tensorCount += 1;
    inventory.sourceBytes += sourceBytes;
    inventoryByType.set(tensor.type, inventory);

    if (isMtpTensorName(tensor.name)) {
      if (excludedTensors.length >= limits.maxExcludedTensors) {
        throw new Error(
          "Conversion plan excluded tensor count exceeds the configured bound",
        );
      }
      excludedTensors.push({
        name: tensor.name,
        reason: MTP_EXCLUSION_REASON,
      });
    } else {
      // Every included tensor needs at least one segment, so this check avoids
      // building an input list that cannot fit in a valid manifest.
      if (included.length >= limits.maxTensorSegments) {
        throw new Error(
          "Conversion plan tensor segment count exceeds the configured bound",
        );
      }
      included.push({ tensor, layout, blockCount });
    }
  }

  // Name ordering makes output independent of upstream directory enumeration.
  included.sort((left, right) =>
    compareText(left.tensor.name, right.tensor.name),
  );
  excludedTensors.sort((left, right) => compareText(left.name, right.name));
  const shards: Array<{ index: number; length: bigint }> = [];
  const segments: PlannedSegment[] = [];

  for (const { tensor, layout, blockCount } of included) {
    const rowBlocks =
      tensor.dimensions[0]! / BigInt(layout.blockElements);
    const rowSourceBytes = rowBlocks * BigInt(layout.sourceBytes);
    const rowOutputBytes = rowBlocks * BigInt(layout.outputBytes);
    if (rowOutputBytes > options.maxShardBytes) {
      throw new Error(
        `Shard size cannot hold one complete ${rowOutputBytes}-byte row for ${tensor.name}`,
      );
    }
    const rowCount = blockCount / rowBlocks;
    let consumedRows = 0n;
    while (consumedRows < rowCount) {
      let shard = shards.at(-1);
      if (shard === undefined) {
        if (shards.length >= limits.maxShards) {
          throw new Error(
            "Conversion plan shard count exceeds the configured bound",
          );
        }
        shard = { index: 0, length: 0n };
        shards.push(shard);
      }
      const alignedOffset = align(shard.length, options.tensorAlignment);
      const available = options.maxShardBytes - alignedOffset;
      const fittingRows = available / rowOutputBytes;
      // A non-aligned shard limit can place alignedOffset past the limit;
      // negative BigInt division is not zero and must not create a segment.
      if (available <= 0n || fittingRows <= 0n) {
        if (shards.length >= limits.maxShards) {
          throw new Error(
            "Conversion plan shard count exceeds the configured bound",
          );
        }
        shard = { index: shards.length, length: 0n };
        shards.push(shard);
        continue;
      }

      const segmentRows =
        rowCount - consumedRows < fittingRows
          ? rowCount - consumedRows
          : fittingRows;
      if (segmentRows <= 0n) {
        throw new Error(`Planner produced an empty segment for ${tensor.name}`);
      }
      const segmentBlocks = segmentRows * rowBlocks;
      const sourceLength = segmentRows * rowSourceBytes;
      const outputLength = segmentRows * rowOutputBytes;
      if (sourceLength <= 0n || outputLength <= 0n) {
        throw new Error(`Planner produced an empty byte range for ${tensor.name}`);
      }
      if (segments.length >= limits.maxTensorSegments) {
        throw new Error(
          "Conversion plan tensor segment count exceeds the configured bound",
        );
      }
      segments.push({
        tensorName: tensor.name,
        dimensions: tensor.dimensions,
        ggmlType: tensor.type,
        storageType: layout.storageType,
        transform: layout.transform,
        blockElements: layout.blockElements,
        shard: shard.index,
        shardOffset: alignedOffset,
        tensorOffset: consumedRows * rowOutputBytes,
        sourceOffset:
          gguf.dataOffset +
          tensor.offset +
          consumedRows * rowSourceBytes,
        sourceLength,
        outputLength,
        blockCount: segmentBlocks,
        sourceBlockBytes: layout.sourceBytes,
        outputBlockBytes: layout.outputBytes,
      });
      shard.length = alignedOffset + outputLength;
      consumedRows += segmentRows;
    }
  }

  if (segments.length === 0) {
    throw new Error("Conversion plan contains no included tensors");
  }

  return {
    inventory: [...inventoryByType.entries()]
      .sort(([left], [right]) => left - right)
      .map(([ggmlType, value]) => ({ ggmlType, ...value })),
    excludedTensors,
    shards,
    segments,
  };
}

export interface ExecuteConversionOptions {
  readonly maxBlocksPerRead?: number;
}

/**
 * Executes planned byte ranges in bounded chunks. Quantized chunks are copied
 * or padded block-for-block and never expanded to floating-point tensor storage.
 */
export async function executeConversionPlan(
  plan: ConversionPlan,
  reader: RandomAccessReader,
  writers: readonly RandomAccessWriter[],
  options: ExecuteConversionOptions = {},
): Promise<void> {
  if (writers.length !== plan.shards.length) {
    throw new Error("Writer count must equal planned shard count");
  }
  const maxBlocksPerRead = options.maxBlocksPerRead ?? 4096;
  if (!Number.isSafeInteger(maxBlocksPerRead) || maxBlocksPerRead < 1) {
    throw new Error("maxBlocksPerRead must be a positive safe integer");
  }
  const writtenThrough = new Array<bigint>(writers.length).fill(0n);

  for (const segment of plan.segments) {
    // Division proves the multiplication is bounded without first overflowing
    // a caller-controlled Number.
    if (
      maxBlocksPerRead >
      Math.floor(MAX_STREAM_READ_BYTES / segment.sourceBlockBytes)
    ) {
      throw new Error(
        `maxBlocksPerRead exceeds the streaming read ceiling for ${segment.tensorName}`,
      );
    }
    const writer = writers[segment.shard]!;
    if (segment.shardOffset > writtenThrough[segment.shard]!) {
      const paddingLength = segment.shardOffset - writtenThrough[segment.shard]!;
      if (paddingLength > BigInt(Number.MAX_SAFE_INTEGER)) {
        throw new Error("Planned padding is too large to materialize");
      }
      await writer.write(
        writtenThrough[segment.shard]!,
        new Uint8Array(Number(paddingLength)),
      );
    }

    let completedBlocks = 0n;
    while (completedBlocks < segment.blockCount) {
      const remaining = segment.blockCount - completedBlocks;
      const chunkBlocks =
        remaining < BigInt(maxBlocksPerRead)
          ? Number(remaining)
          : maxBlocksPerRead;
      const sourceLength = chunkBlocks * segment.sourceBlockBytes;
      const sourceOffset =
        segment.sourceOffset +
        completedBlocks * BigInt(segment.sourceBlockBytes);
      if (sourceOffset + BigInt(sourceLength) > reader.size) {
        throw new Error(`Source range for ${segment.tensorName} is truncated`);
      }
      const source = await reader.read(sourceOffset, sourceLength);
      if (source.byteLength !== sourceLength) {
        throw new Error(`Source reader returned a short read for ${segment.tensorName}`);
      }
      let output: Uint8Array;
      switch (segment.transform) {
        case "copy":
          output = source;
          break;
        case "q8-0-34-to-36":
          output = repackNativeQ8_0(source);
          break;
        case "q3-k-110-to-112":
          output = repackNativeQ3K(source);
          break;
        case "q6-k-210-to-212":
          output = repackNativeQ6K(source);
          break;
      }
      await writer.write(
        segment.shardOffset +
          completedBlocks * BigInt(segment.outputBlockBytes),
        output,
      );
      completedBlocks += BigInt(chunkBlocks);
    }
    writtenThrough[segment.shard] =
      segment.shardOffset + segment.outputLength;
  }
}

export interface ManifestPlanOptions {
  readonly packageKind: PackageKind;
  readonly source: ImmutableArtifactIdentity;
  readonly runtimeAbi: string;
  readonly tokenizer: ImmutableArtifactIdentity;
  readonly processor?: ImmutableArtifactIdentity;
  readonly processorSettings?: VisionProcessorSettings;
  readonly shards: readonly { url: string; sha256: string }[];
}

export function createManifestFromPlan(
  plan: ConversionPlan,
  options: ManifestPlanOptions,
): ModelPackageManifest {
  if (options.shards.length !== plan.shards.length) {
    throw new Error("Shard artifact count must equal planned shard count");
  }
  let packageOffset = 0n;
  const shards = plan.shards.map((shard, index) => {
    const artifact = options.shards[index]!;
    const result = {
      url: artifact.url,
      offset: packageOffset.toString(),
      length: shard.length.toString(),
      sha256: artifact.sha256,
    };
    packageOffset += shard.length;
    return result;
  });
  const manifest: ModelPackageManifest = {
    format: "webml-qwen-package",
    version: 1,
    packageKind: options.packageKind,
    source: options.source,
    runtime: { abi: options.runtimeAbi },
    tokenizer: options.tokenizer,
    ...(options.processor === undefined ? {} : { processor: options.processor }),
    ...(options.processorSettings === undefined
      ? {}
      : { processorSettings: options.processorSettings }),
    tensorLayout: plan.segments.map((segment) => ({
      name: segment.tensorName,
      shape: segment.dimensions.map(String),
      ggmlType: segment.ggmlType,
      storageType: segment.storageType,
      shard: segment.shard,
      shardOffset: segment.shardOffset.toString(),
      tensorOffset: segment.tensorOffset.toString(),
      length: segment.outputLength.toString(),
      ...(segment.storageType === "raw" || segment.storageType === "f32"
        ? {}
        : {
            quantization: {
              blockElements:
                segment.blockElements,
              blockBytes: segment.outputBlockBytes,
            },
          }),
    })),
    shards,
    excludedTensors: [...plan.excludedTensors],
  };
  return validateModelPackageManifest(manifest);
}
