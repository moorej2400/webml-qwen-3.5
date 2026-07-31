import {
  GgmlType,
  ggmlTensorByteLength,
  ggmlTypeLayout,
  type GgufTensorInfo,
  type ParsedGguf,
  type RandomAccessReader,
} from "./gguf.js";
import {
  validateModelPackageManifest,
  type ImmutableArtifactIdentity,
  type ModelPackageManifest,
  type PackageKind,
} from "./manifest.js";
import { repackNativeQ3K } from "./q3k.js";
import {
  MTP_EXCLUSION_REASON,
  isMtpTensorName,
} from "./tensor-policy.js";

export { MTP_EXCLUSION_REASON } from "./tensor-policy.js";

export const MAX_STREAM_READ_BYTES = 8 * 1024 * 1024;

export interface ConverterOptions {
  readonly maxShardBytes: bigint;
  readonly tensorAlignment: number;
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
  readonly transform: "copy" | "q3-k-110-to-112";
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
  readonly transform: PlannedSegment["transform"];
}

function nativeCopyLayout(type: GgmlType): TypeLayout {
  const layout = ggmlTypeLayout(type);
  return {
    sourceBytes: Number(layout.blockBytes),
    outputBytes: Number(layout.blockBytes),
    transform: "copy",
  };
}

const nativeQ3K = ggmlTypeLayout(GgmlType.Q3_K);
const TYPE_LAYOUTS = new Map<GgmlType, TypeLayout>([
  [GgmlType.F32, nativeCopyLayout(GgmlType.F32)],
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
  [GgmlType.Q8_0, nativeCopyLayout(GgmlType.Q8_0)],
  [GgmlType.Q8_1, nativeCopyLayout(GgmlType.Q8_1)],
  [GgmlType.Q2_K, nativeCopyLayout(GgmlType.Q2_K)],
  [
    GgmlType.Q3_K,
    {
      sourceBytes: Number(nativeQ3K.blockBytes),
      outputBytes: 112,
      transform: "q3-k-110-to-112",
    },
  ],
  [GgmlType.Q4_K, nativeCopyLayout(GgmlType.Q4_K)],
  [GgmlType.Q5_K, nativeCopyLayout(GgmlType.Q5_K)],
  [GgmlType.Q6_K, nativeCopyLayout(GgmlType.Q6_K)],
  [GgmlType.Q8_K, nativeCopyLayout(GgmlType.Q8_K)],
]);

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

/**
 * Creates a byte-range plan from the GGUF directory. Source payloads are not
 * read during planning, and every segment boundary is a complete type block.
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
      excludedTensors.push({
        name: tensor.name,
        reason: MTP_EXCLUSION_REASON,
      });
    } else {
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
    if (BigInt(layout.outputBytes) > options.maxShardBytes) {
      throw new Error(
        `Shard size cannot hold one ${layout.outputBytes}-byte block for ${tensor.name}`,
      );
    }
    let consumedBlocks = 0n;
    while (consumedBlocks < blockCount) {
      let shard = shards.at(-1);
      if (shard === undefined) {
        shard = { index: 0, length: 0n };
        shards.push(shard);
      }
      const alignedOffset = align(shard.length, options.tensorAlignment);
      const available = options.maxShardBytes - alignedOffset;
      const fittingBlocks = available / BigInt(layout.outputBytes);
      // A non-aligned shard limit can place alignedOffset past the limit;
      // negative BigInt division is not zero and must not create a segment.
      if (available <= 0n || fittingBlocks <= 0n) {
        shard = { index: shards.length, length: 0n };
        shards.push(shard);
        continue;
      }

      const segmentBlocks =
        blockCount - consumedBlocks < fittingBlocks
          ? blockCount - consumedBlocks
          : fittingBlocks;
      if (segmentBlocks <= 0n) {
        throw new Error(`Planner produced an empty segment for ${tensor.name}`);
      }
      const sourceLength = segmentBlocks * BigInt(layout.sourceBytes);
      const outputLength = segmentBlocks * BigInt(layout.outputBytes);
      if (sourceLength <= 0n || outputLength <= 0n) {
        throw new Error(`Planner produced an empty byte range for ${tensor.name}`);
      }
      segments.push({
        tensorName: tensor.name,
        dimensions: tensor.dimensions,
        ggmlType: tensor.type,
        transform: layout.transform,
        shard: shard.index,
        shardOffset: alignedOffset,
        tensorOffset: consumedBlocks * BigInt(layout.outputBytes),
        sourceOffset:
          gguf.dataOffset +
          tensor.offset +
          consumedBlocks * BigInt(layout.sourceBytes),
        sourceLength,
        outputLength,
        blockCount: segmentBlocks,
        sourceBlockBytes: layout.sourceBytes,
        outputBlockBytes: layout.outputBytes,
      });
      shard.length = alignedOffset + outputLength;
      consumedBlocks += segmentBlocks;
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
 * Executes planned byte ranges in bounded chunks. Q3_K chunks are repacked
 * block-for-block and never expanded to floating-point tensor storage.
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
      const output =
        segment.transform === "q3-k-110-to-112"
          ? repackNativeQ3K(source)
          : source;
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
    tensorLayout: plan.segments.map((segment) => ({
      name: segment.tensorName,
      shape: segment.dimensions.map(String),
      ggmlType: segment.ggmlType,
      storageType:
        segment.transform === "q3-k-110-to-112" ? "q3-k-112" : "raw",
      shard: segment.shard,
      shardOffset: segment.shardOffset.toString(),
      tensorOffset: segment.tensorOffset.toString(),
      length: segment.outputLength.toString(),
      ...(segment.transform === "q3-k-110-to-112"
        ? { quantization: { blockElements: 256, blockBytes: 112 } }
        : {}),
    })),
    shards,
    excludedTensors: [...plan.excludedTensors],
  };
  return validateModelPackageManifest(manifest);
}
