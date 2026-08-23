import { GgmlType, type GgmlType as GgmlTypeValue } from "./gguf.js";
import {
  BROWSER_Q3_K_BLOCK_BYTES,
  BROWSER_Q3_FUSED_K_BLOCK_BYTES,
  BROWSER_Q4_K_BLOCK_BYTES,
  BROWSER_Q5_K_BLOCK_BYTES,
  BROWSER_Q6_K_BLOCK_BYTES,
  dequantizeBrowserQ3KBlock,
  dequantizeBrowserQ3KFusedBlock,
  dequantizeBrowserQ4KBlock,
  dequantizeBrowserQ5KBlock,
  dequantizeBrowserQ6KBlock,
} from "./browser-quant.js";
import {
  packedWeightDecoder,
  type GemvLayout,
} from "./mixed-gemv.js";
import type { KernelDefinition } from "./kernel-registry.js";
import {
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
  dequantizeQ3KBlock,
  unpackWebGpuQ3KBlock,
} from "./q3k.js";

interface PackedLayout {
  readonly ggmlType: GgmlTypeValue;
  readonly storageType: GemvLayout;
  readonly valuesPerBlock: number;
  readonly bytesPerBlock: number;
}

const PACKED_LAYOUTS: readonly PackedLayout[] = Object.freeze([
  Object.freeze({
    ggmlType: GgmlType.F32,
    storageType: "f32",
    valuesPerBlock: 1,
    bytesPerBlock: 4,
  }),
  Object.freeze({
    ggmlType: GgmlType.Q8_0,
    storageType: "q8-0-36",
    valuesPerBlock: 32,
    bytesPerBlock: 36,
  }),
  Object.freeze({
    ggmlType: GgmlType.Q3_K,
    storageType: "q3-k-112",
    valuesPerBlock: 256,
    bytesPerBlock: 112,
  }),
  Object.freeze({
    ggmlType: GgmlType.Q3_K,
    storageType: "q3-k-nibble-148",
    valuesPerBlock: 256,
    bytesPerBlock: BROWSER_Q3_K_BLOCK_BYTES,
  }),
  Object.freeze({
    ggmlType: GgmlType.Q3_K,
    storageType: "q3-k-fused-f32-192",
    valuesPerBlock: 256,
    bytesPerBlock: BROWSER_Q3_FUSED_K_BLOCK_BYTES,
  }),
  Object.freeze({
    ggmlType: GgmlType.Q4_K,
    storageType: "q4-k-144",
    valuesPerBlock: 256,
    bytesPerBlock: 144,
  }),
  Object.freeze({
    ggmlType: GgmlType.Q4_K,
    storageType: "q4-k-fused-f32-192",
    valuesPerBlock: 256,
    bytesPerBlock: BROWSER_Q4_K_BLOCK_BYTES,
  }),
  Object.freeze({
    ggmlType: GgmlType.Q5_K,
    storageType: "q5-k-176",
    valuesPerBlock: 256,
    bytesPerBlock: 176,
  }),
  Object.freeze({
    ggmlType: GgmlType.Q5_K,
    storageType: "q5-k-fused-f32-224",
    valuesPerBlock: 256,
    bytesPerBlock: BROWSER_Q5_K_BLOCK_BYTES,
  }),
  Object.freeze({
    ggmlType: GgmlType.Q6_K,
    storageType: "q6-k-212",
    valuesPerBlock: 256,
    bytesPerBlock: 212,
  }),
  Object.freeze({
    ggmlType: GgmlType.Q6_K,
    storageType: "q6-k-fused-f32-256",
    valuesPerBlock: 256,
    bytesPerBlock: BROWSER_Q6_K_BLOCK_BYTES,
  }),
]);

function layout(storageType: GemvLayout): PackedLayout {
  const found = PACKED_LAYOUTS.find((item) => item.storageType === storageType);
  if (found === undefined) {
    throw new Error("Unsupported packed embedding layout");
  }
  return found;
}

function blockValues(storageType: GemvLayout, bytes: Uint8Array): Float32Array {
  switch (storageType) {
    case "f32":
      return Float32Array.of(
        new DataView(bytes.buffer, bytes.byteOffset, 4).getFloat32(0, true),
      );
    case "q8-0-36":
      return dequantizeQ8_0Block(unpackWebGpuQ8_0Block(bytes));
    case "q3-k-112":
      return dequantizeQ3KBlock(unpackWebGpuQ3KBlock(bytes));
    case "q3-k-nibble-148":
      return dequantizeBrowserQ3KBlock(bytes);
    case "q3-k-fused-f32-192":
      return dequantizeBrowserQ3KFusedBlock(bytes);
    case "q4-k-144":
      return dequantizeQ4KBlock(unpackWebGpuQ4KBlock(bytes));
    case "q4-k-fused-f32-192":
      return dequantizeBrowserQ4KBlock(bytes);
    case "q5-k-176":
      return dequantizeQ5KBlock(unpackWebGpuQ5KBlock(bytes));
    case "q5-k-fused-f32-224":
      return dequantizeBrowserQ5KBlock(bytes);
    case "q6-k-212":
      return dequantizeQ6KBlock(unpackWebGpuQ6KBlock(bytes));
    case "q6-k-fused-f32-256":
      return dequantizeBrowserQ6KBlock(bytes);
  }
}

function positiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${label} must be a positive safe integer`);
  }
}

function rowGeometry(
  storageType: GemvLayout,
  embeddingLength: number,
): { layout: PackedLayout; blocksPerRow: number; rowBytes: number } {
  positiveInteger(embeddingLength, "Embedding length");
  const selected = layout(storageType);
  if (embeddingLength % selected.valuesPerBlock !== 0) {
    throw new Error("Embedding row must contain complete packed blocks");
  }
  const blocksPerRow = embeddingLength / selected.valuesPerBlock;
  return {
    layout: selected,
    blocksPerRow,
    rowBytes: blocksPerRow * selected.bytesPerBlock,
  };
}

/**
 * Correctness reference that decodes one token row. The full vocabulary table
 * stays packed and is never materialized as a floating-point matrix.
 */
export function embeddingCpu(
  storageType: GemvLayout,
  packedTable: Uint8Array,
  tokenId: number,
  shape: {
    readonly vocabSize: number;
    readonly embeddingLength: number;
  },
): Float32Array {
  positiveInteger(shape.vocabSize, "Embedding vocabulary size");
  if (
    !Number.isSafeInteger(tokenId) ||
    tokenId < 0 ||
    tokenId >= shape.vocabSize
  ) {
    throw new Error("Embedding token is outside vocabulary bounds");
  }
  const geometry = rowGeometry(storageType, shape.embeddingLength);
  const expectedBytes = geometry.rowBytes * shape.vocabSize;
  if (packedTable.byteLength !== expectedBytes) {
    throw new Error("Packed embedding table extent does not match its shape");
  }
  const rowOffset = tokenId * geometry.rowBytes;
  const output = new Float32Array(shape.embeddingLength);
  for (let block = 0; block < geometry.blocksPerRow; block += 1) {
    const byteOffset = rowOffset + block * geometry.layout.bytesPerBlock;
    output.set(
      blockValues(
        storageType,
        packedTable.subarray(
          byteOffset,
          byteOffset + geometry.layout.bytesPerBlock,
        ),
      ),
      block * geometry.layout.valuesPerBlock,
    );
  }
  return output;
}

export interface PackedEmbeddingRowPlan {
  readonly storageType: GemvLayout;
  readonly packedByteOffset: number;
  readonly packedByteLength: number;
  readonly outputElements: number;
  readonly workgroups: { readonly x: number; readonly y: 1; readonly z: 1 };
}

export interface PackedEmbeddingKernelAbi {
  readonly storageType: GemvLayout;
  readonly phase: "shared";
  readonly profile: "portable-f32";
  readonly workgroupSize: 256;
  readonly valuesPerBlock: number;
  readonly bytesPerBlock: number;
  readonly bindings: {
    readonly packedTable: 0;
    readonly output: 1;
    readonly uniforms: 2;
  };
  readonly uniformWords: 4;
}

export interface PackedEmbeddingKernel {
  readonly id: string;
  readonly operation: "embedding-row";
  readonly ggmlType: GgmlTypeValue;
  readonly storageType: GemvLayout;
  readonly phase: "shared";
  readonly profile: "portable-f32";
  readonly abi: PackedEmbeddingKernelAbi;
  readonly source: string;
}

function embeddingShader(storageType: GemvLayout): string {
  const decoder = packedWeightDecoder(storageType);
  return /* wgsl */ `
const WORDS_PER_BLOCK: u32 = ${decoder.bytesPerBlock / 4}u;
const VALUES_PER_BLOCK: u32 = ${decoder.valuesPerBlock}u;
struct Params {
  row_word_offset: u32,
  output_elements: u32,
  blocks_per_row: u32,
  pad: u32,
}
@group(0) @binding(0) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
fn packed_byte(block_word: u32, byte_offset: u32) -> u32 {
  let word = packed_weights[block_word + byte_offset / 4u];
  return (word >> ((byte_offset % 4u) * 8u)) & 0xffu;
}
${decoder.source}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= params.output_elements) { return; }
  let block = index / VALUES_PER_BLOCK;
  if (block >= params.blocks_per_row) { return; }
  let block_word = params.row_word_offset + block * WORDS_PER_BLOCK;
  output_values[index] = weight_value(block_word, index % VALUES_PER_BLOCK);
}`;
}

export const PACKED_EMBEDDING_KERNELS: readonly PackedEmbeddingKernel[] =
  Object.freeze(
    PACKED_LAYOUTS.map((item) =>
      Object.freeze({
        id: `${item.storageType}-embedding-row-shared-portable-f32`,
        operation: "embedding-row" as const,
        ggmlType: item.ggmlType,
        storageType: item.storageType,
        phase: "shared" as const,
        profile: "portable-f32" as const,
        source: embeddingShader(item.storageType),
        abi: Object.freeze({
          storageType: item.storageType,
          phase: "shared" as const,
          profile: "portable-f32" as const,
          workgroupSize: 256 as const,
          valuesPerBlock: item.valuesPerBlock,
          bytesPerBlock: item.bytesPerBlock,
          bindings: Object.freeze({
            packedTable: 0 as const,
            output: 1 as const,
            uniforms: 2 as const,
          }),
          uniformWords: 4 as const,
        }),
      }),
    ),
  );

export interface QwenEmbeddingRegistryDefinition extends KernelDefinition {
  readonly ggmlType: GgmlTypeValue;
  readonly abi: Omit<PackedEmbeddingKernelAbi, "phase"> & {
    readonly phase: "prefill" | "decode";
  };
}

export function qwenEmbeddingRegistryDefinitions(input: {
  readonly phase: "prefill" | "decode";
  readonly profile: "portable-f32";
}): readonly QwenEmbeddingRegistryDefinition[] {
  if (input.phase !== "prefill" && input.phase !== "decode") {
    throw new Error("Unsupported embedding kernel phase");
  }
  if (input.profile !== "portable-f32") {
    throw new Error("Unsupported embedding kernel profile");
  }
  return Object.freeze(
    PACKED_EMBEDDING_KERNELS.map((kernel) =>
      Object.freeze({
        id: `${kernel.storageType}-embedding-row-${input.phase}-${input.profile}`,
        ggmlType: kernel.ggmlType,
        key: Object.freeze({
          operation: kernel.operation,
          layout: kernel.storageType,
          phase: input.phase,
          profile: input.profile,
        }),
        source: kernel.source,
        abi: Object.freeze({ ...kernel.abi, phase: input.phase }),
      }),
    ),
  );
}

export function planPackedEmbeddingRow(input: {
  readonly ggmlType: GgmlTypeValue;
  readonly storageType: GemvLayout;
  readonly tokenId: number;
  readonly vocabSize: number;
  readonly embeddingLength: number;
  /** Live device limit used to reject an invalid 1D dispatch before encoding. */
  readonly maxComputeWorkgroupsPerDimension: number;
}): PackedEmbeddingRowPlan {
  if (
    !Number.isSafeInteger(input.vocabSize) ||
    input.vocabSize < 1 ||
    input.vocabSize > 0xffff_ffff
  ) {
    throw new Error("Embedding vocabulary size must fit a positive u32");
  }
  if (
    !Number.isSafeInteger(input.embeddingLength) ||
    input.embeddingLength < 1 ||
    input.embeddingLength > 0xffff_ffff
  ) {
    throw new Error("Embedding length must fit a positive u32");
  }
  if (
    !Number.isSafeInteger(input.maxComputeWorkgroupsPerDimension) ||
    input.maxComputeWorkgroupsPerDimension < 1 ||
    input.maxComputeWorkgroupsPerDimension > 0xffff_ffff
  ) {
    throw new Error("Embedding device limit must fit a positive u32");
  }
  if (
    !Number.isSafeInteger(input.tokenId) ||
    input.tokenId < 0 ||
    input.tokenId >= input.vocabSize
  ) {
    throw new Error("Embedding token is outside vocabulary bounds");
  }
  const geometry = rowGeometry(input.storageType, input.embeddingLength);
  if (geometry.layout.ggmlType !== input.ggmlType) {
    throw new Error("Embedding GGML type does not match its packed layout");
  }
  const packedByteOffset = input.tokenId * geometry.rowBytes;
  if (
    !Number.isSafeInteger(packedByteOffset) ||
    packedByteOffset > 0xffff_ffff
  ) {
    throw new Error("Embedding row offset exceeds u32 shader addressing");
  }
  if (
    geometry.blocksPerRow > 0xffff_ffff ||
    geometry.rowBytes > 0xffff_ffff
  ) {
    throw new Error("Embedding row geometry exceeds u32 shader addressing");
  }
  const workgroups = Math.ceil(input.embeddingLength / 256);
  if (workgroups > input.maxComputeWorkgroupsPerDimension) {
    throw new Error("Embedding workgroups exceed the device limit");
  }
  return Object.freeze({
    storageType: input.storageType,
    packedByteOffset,
    packedByteLength: geometry.rowBytes,
    outputElements: input.embeddingLength,
    workgroups: Object.freeze({
      x: workgroups,
      y: 1,
      z: 1,
    }),
  });
}
