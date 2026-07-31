/**
 * Provides one direct packed-weight GEMV family for each pinned language type.
 *
 * Shaders reconstruct scalar weights in registers and never allocate an
 * expanded floating-point weight matrix.
 */
import { GgmlType } from "./gguf.js";
import type { KernelDefinition } from "./kernel-registry.js";
import {
  K_QUANT_ELEMENTS_PER_BLOCK,
  Q8_0_ELEMENTS_PER_BLOCK,
  WEBGPU_Q4_K_BLOCK_BYTES,
  WEBGPU_Q5_K_BLOCK_BYTES,
  WEBGPU_Q6_K_BLOCK_BYTES,
  WEBGPU_Q8_0_BLOCK_BYTES,
  dequantizeQ4KBlock,
  dequantizeQ5KBlock,
  dequantizeQ6KBlock,
  dequantizeQ8_0Block,
  unpackWebGpuQ4KBlock,
  unpackWebGpuQ5KBlock,
  unpackWebGpuQ6KBlock,
  unpackWebGpuQ8_0Block,
} from "./mixed-quant.js";
import {
  Q3_K_ELEMENTS_PER_BLOCK,
  WEBGPU_Q3_K_BLOCK_BYTES,
  dequantizeQ3KBlock,
  unpackWebGpuQ3KBlock,
} from "./q3k.js";

export type GemvLayout =
  | "f32"
  | "q8-0-36"
  | "q3-k-112"
  | "q4-k-144"
  | "q5-k-176"
  | "q6-k-212";

export interface GemvKernelAbi {
  readonly layout: GemvLayout;
  readonly phase: "shared";
  readonly profile: "portable-f32";
  readonly wordsPerBlock: number;
  readonly bytesPerBlock: number;
  readonly valuesPerBlock: number;
  readonly workgroupSize: 1;
  readonly uniformWords: 5;
  readonly bindings: {
    readonly packedWeights: 0;
    readonly activation: 1;
    readonly output: 2;
    readonly uniforms: 3;
  };
}

export interface LanguageGemvKernel {
  readonly id: string;
  readonly operation: "gemv";
  readonly ggmlType: GgmlType;
  readonly layout: GemvLayout;
  readonly phase: "shared";
  readonly profile: "portable-f32";
  readonly abi: GemvKernelAbi;
  readonly source: string;
}

interface KernelShape {
  readonly ggmlType: GgmlType;
  readonly layout: GemvLayout;
  readonly bytesPerBlock: number;
  readonly valuesPerBlock: number;
  readonly weightValue: string;
}

const WGSL_SCALE_MIN = /* wgsl */ `
fn scale_component(block_word: u32, group: u32, minimum: bool) -> u32 {
  if (group < 4u) {
    let index = select(group, group + 4u, minimum);
    return packed_byte(block_word, 4u + index) & 0x3fu;
  }
  if (!minimum) {
    return (packed_byte(block_word, 4u + group + 4u) & 0x0fu) |
      ((packed_byte(block_word, 4u + group - 4u) >> 6u) << 4u);
  }
  return (packed_byte(block_word, 4u + group + 4u) >> 4u) |
    ((packed_byte(block_word, 4u + group) >> 6u) << 4u);
}
`;

const WGSL_SHAPES: readonly KernelShape[] = [
  {
    ggmlType: GgmlType.F32,
    layout: "f32",
    bytesPerBlock: 4,
    valuesPerBlock: 1,
    weightValue: `
fn weight_value(block_word: u32, element: u32) -> f32 {
  return bitcast<f32>(packed_weights[block_word + element]);
}`,
  },
  {
    ggmlType: GgmlType.Q8_0,
    layout: "q8-0-36",
    bytesPerBlock: WEBGPU_Q8_0_BLOCK_BYTES,
    valuesPerBlock: Q8_0_ELEMENTS_PER_BLOCK,
    weightValue: `
fn weight_value(block_word: u32, element: u32) -> f32 {
  let delta = unpack2x16float(packed_weights[block_word]).x;
  let raw = packed_byte(block_word, 4u + element);
  let quant = select(i32(raw), i32(raw) - 256, raw >= 128u);
  return delta * f32(quant);
}`,
  },
  {
    ggmlType: GgmlType.Q3_K,
    layout: "q3-k-112",
    bytesPerBlock: WEBGPU_Q3_K_BLOCK_BYTES,
    valuesPerBlock: Q3_K_ELEMENTS_PER_BLOCK,
    weightValue: `
fn q3_scale(block_word: u32, element: u32) -> i32 {
  let scale_index = element / 16u;
  var low: u32;
  if (scale_index < 8u) {
    low = packed_byte(block_word, 4u + scale_index) & 0x0fu;
  } else {
    low = packed_byte(block_word, 4u + scale_index - 8u) >> 4u;
  }
  let high = (packed_byte(block_word, 12u + scale_index % 4u) >>
    (2u * (scale_index / 4u))) & 0x03u;
  return i32(low | (high << 4u)) - 32;
}

fn weight_value(block_word: u32, element: u32) -> f32 {
  let group = element / 128u;
  let within_group = element % 128u;
  let subgroup = within_group / 16u;
  let plane = subgroup / 2u;
  let half = subgroup % 2u;
  let lane = element % 16u;
  let qbyte = packed_byte(
    block_word,
    16u + group * 32u + half * 16u + lane,
  );
  let low = (qbyte >> (plane * 2u)) & 0x03u;
  let hbyte = packed_byte(block_word, 80u + half * 16u + lane);
  let high = select(4i, 0i, (hbyte & (1u << (group * 4u + plane))) != 0u);
  let delta = unpack2x16float(packed_weights[block_word]).x;
  return delta * f32(q3_scale(block_word, element) * (i32(low) - high));
}`,
  },
  {
    ggmlType: GgmlType.Q4_K,
    layout: "q4-k-144",
    bytesPerBlock: WEBGPU_Q4_K_BLOCK_BYTES,
    valuesPerBlock: K_QUANT_ELEMENTS_PER_BLOCK,
    weightValue: `${WGSL_SCALE_MIN}
fn weight_value(block_word: u32, element: u32) -> f32 {
  let chunk = element / 64u;
  let upper = (element % 64u) >= 32u;
  let group = chunk * 2u + select(0u, 1u, upper);
  let packed = packed_byte(block_word, 16u + chunk * 32u + element % 32u);
  let quant = select(packed & 0x0fu, packed >> 4u, upper);
  let deltas = unpack2x16float(packed_weights[block_word]);
  return deltas.x * f32(scale_component(block_word, group, false) * quant) -
    deltas.y * f32(scale_component(block_word, group, true));
}`,
  },
  {
    ggmlType: GgmlType.Q5_K,
    layout: "q5-k-176",
    bytesPerBlock: WEBGPU_Q5_K_BLOCK_BYTES,
    valuesPerBlock: K_QUANT_ELEMENTS_PER_BLOCK,
    weightValue: `${WGSL_SCALE_MIN}
fn weight_value(block_word: u32, element: u32) -> f32 {
  let chunk = element / 64u;
  let upper = (element % 64u) >= 32u;
  let group = chunk * 2u + select(0u, 1u, upper);
  let packed = packed_byte(block_word, 48u + chunk * 32u + element % 32u);
  let low = select(packed & 0x0fu, packed >> 4u, upper);
  let mask = 1u << (chunk * 2u + select(0u, 1u, upper));
  let high = select(0u, 16u, (
    packed_byte(block_word, 16u + element % 32u) & mask
  ) != 0u);
  let deltas = unpack2x16float(packed_weights[block_word]);
  return deltas.x * f32(
    scale_component(block_word, group, false) * (low + high)
  ) - deltas.y * f32(scale_component(block_word, group, true));
}`,
  },
  {
    ggmlType: GgmlType.Q6_K,
    layout: "q6-k-212",
    bytesPerBlock: WEBGPU_Q6_K_BLOCK_BYTES,
    valuesPerBlock: K_QUANT_ELEMENTS_PER_BLOCK,
    weightValue: `
fn signed_byte(raw: u32) -> i32 {
  return select(i32(raw), i32(raw) - 256, raw >= 128u);
}

fn weight_value(block_word: u32, element: u32) -> f32 {
  let half = element / 128u;
  let section = (element % 128u) / 32u;
  let lane = element % 32u;
  let ql_offset = half * 64u + lane + select(0u, 32u, section % 2u == 1u);
  let ql = packed_byte(block_word, ql_offset);
  let low = select(ql & 0x0fu, ql >> 4u, section >= 2u);
  let qh = packed_byte(block_word, 128u + half * 32u + lane);
  let quant = i32(low | (((qh >> (section * 2u)) & 3u) << 4u)) - 32;
  let scale_index = half * 8u + section * 2u + lane / 16u;
  let scale = signed_byte(packed_byte(block_word, 192u + scale_index));
  let delta = unpack2x16float(packed_weights[block_word + 52u]).x;
  return delta * f32(scale * quant);
}`,
  },
] as const;

function shaderSource(shape: KernelShape): string {
  return /* wgsl */ `
const WORDS_PER_BLOCK: u32 = ${shape.bytesPerBlock / 4}u;
const VALUES_PER_BLOCK: u32 = ${shape.valuesPerBlock}u;

struct GemvUniforms {
  local_rows: u32,
  columns: u32,
  blocks_per_row: u32,
  weight_word_offset: u32,
  output_row_offset: u32,
}

@group(0) @binding(0) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(1) var<storage, read> activation: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: GemvUniforms;

fn packed_byte(block_word: u32, byte_offset: u32) -> u32 {
  let word = packed_weights[block_word + byte_offset / 4u];
  return (word >> ((byte_offset % 4u) * 8u)) & 0xffu;
}

${shape.weightValue}

@compute @workgroup_size(1)
fn packed_gemv(
  @builtin(global_invocation_id) invocation: vec3<u32>,
  @builtin(num_workgroups) grid: vec3<u32>,
) {
  // Reject surplus 2D invocations before flattening can wrap u32.
  if (invocation.y > (0xffffffffu - invocation.x) / grid.x) {
    return;
  }
  let row = invocation.y * grid.x + invocation.x;
  if (row >= params.local_rows) {
    return;
  }
  var sum = 0.0f;
  for (var block = 0u; block < params.blocks_per_row; block += 1u) {
    let block_word = params.weight_word_offset +
      (row * params.blocks_per_row + block) * WORDS_PER_BLOCK;
    for (var element = 0u; element < VALUES_PER_BLOCK; element += 1u) {
      sum += weight_value(block_word, element) *
        activation[block * VALUES_PER_BLOCK + element];
    }
  }
  output[params.output_row_offset + row] = sum;
}
`;
}

function abi(shape: KernelShape): GemvKernelAbi {
  return Object.freeze({
    layout: shape.layout,
    phase: "shared",
    profile: "portable-f32",
    wordsPerBlock: shape.bytesPerBlock / 4,
    bytesPerBlock: shape.bytesPerBlock,
    valuesPerBlock: shape.valuesPerBlock,
    workgroupSize: 1,
    uniformWords: 5,
    bindings: Object.freeze({
      packedWeights: 0,
      activation: 1,
      output: 2,
      uniforms: 3,
    }),
  });
}

export const LANGUAGE_GEMV_KERNELS: readonly LanguageGemvKernel[] =
  Object.freeze(
    WGSL_SHAPES.map((shape) =>
      Object.freeze({
        id: `${shape.layout}-gemv-shared-portable-f32`,
        operation: "gemv" as const,
        ggmlType: shape.ggmlType,
        layout: shape.layout,
        phase: "shared" as const,
        profile: "portable-f32" as const,
        abi: abi(shape),
        source: shaderSource(shape),
      }),
    ),
  );

export interface LanguageGemvRegistryDefinition extends KernelDefinition {
  readonly ggmlType: GgmlType;
  readonly abi: Omit<GemvKernelAbi, "phase" | "profile"> & {
    readonly phase: "prefill" | "decode";
    readonly profile: "portable-f32";
  };
}

/**
 * The same row kernel is valid in both phases, but registry keys keep the
 * caller's phase explicit so a future specialized kernel cannot be selected
 * by accident.
 */
export function languageGemvRegistryDefinitions(input: {
  readonly phase: "prefill" | "decode";
  readonly profile: "portable-f32";
}): readonly LanguageGemvRegistryDefinition[] {
  if (input.phase !== "prefill" && input.phase !== "decode") {
    throw new Error("Unsupported GEMV kernel phase");
  }
  if (input.profile !== "portable-f32") {
    throw new Error("Unsupported GEMV kernel profile");
  }
  return Object.freeze(
    LANGUAGE_GEMV_KERNELS.map((kernel) =>
      Object.freeze({
        id: `${kernel.layout}-gemv-${input.phase}-${input.profile}`,
        ggmlType: kernel.ggmlType,
        key: Object.freeze({
          operation: "gemv",
          layout: kernel.layout,
          phase: input.phase,
          profile: input.profile,
        }),
        source: kernel.source,
        abi: Object.freeze({
          ...kernel.abi,
          phase: input.phase,
          profile: input.profile,
        }),
      }),
    ),
  );
}

function kernelFor(layout: GemvLayout): LanguageGemvKernel {
  const kernel = LANGUAGE_GEMV_KERNELS.find(
    (candidate) => candidate.layout === layout,
  );
  if (kernel === undefined) {
    throw new Error(`Unsupported GEMV layout`);
  }
  return kernel;
}

function requireU32(value: number, label: string, allowZero = false): void {
  if (
    !Number.isSafeInteger(value) ||
    value < (allowZero ? 0 : 1) ||
    value > 0xffff_ffff
  ) {
    throw new Error(`${label} must fit ${allowZero ? "a non-negative" : "a positive"} u32`);
  }
}

export interface GemvDispatchPlan {
  readonly workgroups: { readonly x: number; readonly y: number; readonly z: 1 };
  readonly uniforms: {
    readonly localRows: number;
    readonly columns: number;
    readonly blocksPerRow: number;
    readonly weightWordOffset: number;
    readonly outputRowOffset: number;
  };
}

export function planGemvDispatch(input: {
  readonly layout: GemvLayout;
  readonly ggmlType?: GgmlType;
  readonly localRows: number;
  readonly columns: number;
  readonly packedByteOffset?: number;
  readonly outputRowOffset?: number;
  readonly maxWorkgroupsPerDimension?: number;
}): GemvDispatchPlan {
  const kernel = kernelFor(input.layout);
  if (input.ggmlType !== undefined && input.ggmlType !== kernel.ggmlType) {
    throw new Error("GGML tensor type does not match the GEMV layout");
  }
  requireU32(input.localRows, "GEMV localRows");
  requireU32(input.columns, "GEMV columns");
  if (input.columns % kernel.abi.valuesPerBlock !== 0) {
    throw new Error(
      `GEMV columns must be a multiple of ${kernel.abi.valuesPerBlock}`,
    );
  }
  const packedByteOffset = input.packedByteOffset ?? 0;
  requireU32(packedByteOffset, "GEMV packed byte offset", true);
  if (packedByteOffset % 4 !== 0) {
    throw new Error("GEMV packed byte offset must be u32 aligned");
  }
  // Shard placement aligns tensor starts to u32, not to each quant block size.
  // Block indexing is relative to this base word inside the bound shard.
  const outputRowOffset = input.outputRowOffset ?? 0;
  requireU32(outputRowOffset, "GEMV output row offset", true);
  if (BigInt(outputRowOffset) + BigInt(input.localRows) > 0x1_0000_0000n) {
    throw new Error("GEMV output rows exceed u32 shader addressing");
  }
  const maxWorkgroups = input.maxWorkgroupsPerDimension ?? 65_535;
  requireU32(maxWorkgroups, "GEMV maximum workgroups per dimension");
  const x = Math.min(input.localRows, maxWorkgroups);
  const y = Math.ceil(input.localRows / x);
  // The generated shader rejects surplus grid cells before it flattens x/y.
  if (y > maxWorkgroups) {
    throw new Error("GEMV dispatch exceeds the two-dimensional device limit");
  }
  const blocksPerRow = input.columns / kernel.abi.valuesPerBlock;
  const weightWordOffset = packedByteOffset / 4;
  const endWord =
    BigInt(weightWordOffset) +
    BigInt(input.localRows) *
      BigInt(blocksPerRow) *
      BigInt(kernel.abi.wordsPerBlock);
  if (endWord > 0x1_0000_0000n) {
    throw new Error("GEMV packed extent exceeds u32 shader addressing");
  }
  return {
    workgroups: { x, y, z: 1 },
    uniforms: {
      localRows: input.localRows,
      columns: input.columns,
      blocksPerRow,
      weightWordOffset,
      outputRowOffset,
    },
  };
}

export interface LogicalMatrixShard {
  readonly logicalByteOffset: bigint;
  readonly logicalByteLength: bigint;
}

export function planMatrixShardDispatch(input: {
  readonly layout: GemvLayout;
  readonly rows: number;
  readonly columns: number;
  readonly shards: readonly LogicalMatrixShard[];
  readonly maxWorkgroupsPerDimension?: number;
}): readonly (GemvDispatchPlan & { readonly shardIndex: number })[] {
  const kernel = kernelFor(input.layout);
  requireU32(input.rows, "GEMV matrix rows");
  const probe = planGemvDispatch({
    layout: input.layout,
    localRows: 1,
    columns: input.columns,
  });
  const rowBytes =
    BigInt(probe.uniforms.blocksPerRow) * BigInt(kernel.abi.bytesPerBlock);
  const expectedBytes = BigInt(input.rows) * rowBytes;
  let nextByte = 0n;
  let nextRow = 0;
  const plans: Array<GemvDispatchPlan & { readonly shardIndex: number }> = [];
  for (const [shardIndex, shard] of input.shards.entries()) {
    if (
      shard.logicalByteOffset !== nextByte ||
      shard.logicalByteLength <= 0n ||
      shard.logicalByteOffset % rowBytes !== 0n ||
      shard.logicalByteLength % rowBytes !== 0n
    ) {
      throw new Error("GEMV matrix shards must be contiguous whole rows");
    }
    const rows = shard.logicalByteLength / rowBytes;
    if (rows > BigInt(0xffff_ffff)) {
      throw new Error("GEMV shard row count exceeds u32");
    }
    const plan = planGemvDispatch({
      layout: input.layout,
      localRows: Number(rows),
      columns: input.columns,
      outputRowOffset: nextRow,
      ...(input.maxWorkgroupsPerDimension === undefined
        ? {}
        : { maxWorkgroupsPerDimension: input.maxWorkgroupsPerDimension }),
    });
    plans.push(Object.freeze({ shardIndex, ...plan }));
    nextByte += shard.logicalByteLength;
    nextRow += Number(rows);
  }
  if (nextByte !== expectedBytes || nextRow !== input.rows) {
    throw new Error("GEMV shards do not cover the complete matrix");
  }
  return Object.freeze(plans);
}

function blockValues(
  layout: GemvLayout,
  bytes: Uint8Array,
): Float32Array {
  switch (layout) {
    case "f32":
      return Float32Array.of(
        new DataView(bytes.buffer, bytes.byteOffset, 4).getFloat32(0, true),
      );
    case "q8-0-36":
      return dequantizeQ8_0Block(unpackWebGpuQ8_0Block(bytes));
    case "q3-k-112":
      return dequantizeQ3KBlock(unpackWebGpuQ3KBlock(bytes));
    case "q4-k-144":
      return dequantizeQ4KBlock(unpackWebGpuQ4KBlock(bytes));
    case "q5-k-176":
      return dequantizeQ5KBlock(unpackWebGpuQ5KBlock(bytes));
    case "q6-k-212":
      return dequantizeQ6KBlock(unpackWebGpuQ6KBlock(bytes));
  }
}

export function gemvCpu(
  layout: GemvLayout,
  packedWeights: Uint8Array,
  activation: Float32Array,
  shape: {
    readonly rows: number;
    readonly columns: number;
    readonly packedByteOffset?: number;
  },
): Float32Array {
  const kernel = kernelFor(layout);
  const plan = planGemvDispatch({
    layout,
    localRows: shape.rows,
    columns: shape.columns,
    ...(shape.packedByteOffset === undefined
      ? {}
      : { packedByteOffset: shape.packedByteOffset }),
  });
  if (activation.length !== shape.columns) {
    throw new Error(`GEMV activation length must equal ${shape.columns}`);
  }
  const packedOffset = shape.packedByteOffset ?? 0;
  const required =
    packedOffset +
    shape.rows * plan.uniforms.blocksPerRow * kernel.abi.bytesPerBlock;
  if (!Number.isSafeInteger(required) || packedWeights.byteLength < required) {
    throw new Error("GEMV packed weights do not cover the matrix extent");
  }
  const output = new Float32Array(shape.rows);
  for (let row = 0; row < shape.rows; row += 1) {
    let sum = 0;
    for (let block = 0; block < plan.uniforms.blocksPerRow; block += 1) {
      const offset =
        packedOffset +
        (row * plan.uniforms.blocksPerRow + block) *
          kernel.abi.bytesPerBlock;
      const values = blockValues(
        layout,
        packedWeights.subarray(offset, offset + kernel.abi.bytesPerBlock),
      );
      for (let element = 0; element < values.length; element += 1) {
        sum +=
          values[element]! *
          activation[block * kernel.abi.valuesPerBlock + element]!;
      }
    }
    output[row] = sum;
  }
  return output;
}
