import {
  Q3_K_ELEMENTS_PER_BLOCK,
  WEBGPU_Q3_K_BLOCK_BYTES,
  dequantizeQ3KBlock,
  unpackWebGpuQ3KBlock,
} from "./q3k.js";

export const Q3K_GEMV_ABI = {
  wordsPerBlock: 28,
  valuesPerBlock: Q3_K_ELEMENTS_PER_BLOCK,
  workgroupSize: 1,
  bindings: {
    packedWeights: 0,
    activation: 1,
    output: 2,
    uniforms: 3,
  },
} as const;

/**
 * Correctness-first packed Q3_K GEMV. Each invocation owns one row and decodes
 * the 28-word block directly into scalar registers; no expanded FP16 matrix is
 * allocated in storage or workgroup memory.
 */
export const Q3K_GEMV_WGSL = /* wgsl */ `
const WORDS_PER_BLOCK: u32 = 28u;
const VALUES_PER_BLOCK: u32 = 256u;

struct GemvUniforms {
  rows: u32,
  columns: u32,
  blocks_per_row: u32,
  weight_word_offset: u32,
}

@group(0) @binding(0) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(1) var<storage, read> activation: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: GemvUniforms;

fn packed_byte(block_word: u32, byte_offset: u32) -> u32 {
  let word = packed_weights[block_word + byte_offset / 4u];
  return (word >> ((byte_offset % 4u) * 8u)) & 0xffu;
}

fn signed_scale(block_word: u32, element: u32) -> i32 {
  let scale_index = element / 16u;
  var low: u32;
  if (scale_index < 8u) {
    low = packed_byte(block_word, 4u + scale_index) & 0x0fu;
  } else {
    low = packed_byte(block_word, 4u + scale_index - 8u) >> 4u;
  }
  let high_byte = packed_byte(
    block_word,
    12u + scale_index % 4u,
  );
  let high = (
    high_byte >> (2u * (scale_index / 4u))
  ) & 0x03u;
  return i32(low | (high << 4u)) - 32;
}

fn signed_quant(block_word: u32, element: u32) -> i32 {
  let group = element / 128u;
  let within_group = element % 128u;
  let subgroup = within_group / 16u;
  let plane = subgroup / 2u;
  let half = subgroup % 2u;
  let lane = element % 16u;
  let q_byte = packed_byte(
    block_word,
    16u + group * 32u + half * 16u + lane,
  );
  let low = (q_byte >> (plane * 2u)) & 0x03u;
  let high_byte = packed_byte(block_word, 80u + half * 16u + lane);
  let high_mask = 1u << (group * 4u + plane);
  let high = select(4i, 0i, (high_byte & high_mask) != 0u);
  return i32(low) - high;
}

fn q3_value(block_word: u32, element: u32) -> f32 {
  let delta = unpack2x16float(packed_weights[block_word]).x;
  return delta * f32(
    signed_scale(block_word, element) *
    signed_quant(block_word, element)
  );
}

@compute @workgroup_size(1)
fn q3k_gemv(
  @builtin(global_invocation_id) invocation: vec3<u32>,
  @builtin(num_workgroups) grid: vec3<u32>,
) {
  let row = invocation.y * grid.x + invocation.x;
  if (row >= params.rows) {
    return;
  }
  var sum = 0.0f;
  for (var block = 0u; block < params.blocks_per_row; block += 1u) {
    let block_word = params.weight_word_offset +
      (row * params.blocks_per_row + block) * WORDS_PER_BLOCK;
    for (var element = 0u; element < VALUES_PER_BLOCK; element += 1u) {
      let activation_index = block * VALUES_PER_BLOCK + element;
      sum += q3_value(block_word, element) * activation[activation_index];
    }
  }
  output[row] = sum;
}
`;

export interface Q3KGemvShape {
  readonly rows: number;
  readonly columns: number;
  readonly packedByteOffset?: number;
  readonly maxWorkgroupsPerDimension?: number;
}

function requireU32(value: number, label: string, allowZero = false): void {
  const minimum = allowZero ? 0 : 1;
  if (
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > 0xffff_ffff
  ) {
    throw new Error(
      `${label} must be a${allowZero ? " non-negative" : " positive"} u32 integer`,
    );
  }
}

export function planQ3KGemvDispatch(shape: Q3KGemvShape): {
  readonly workgroups: {
    readonly x: number;
    readonly y: number;
    readonly z: 1;
  };
  readonly uniforms: {
    readonly rows: number;
    readonly columns: number;
    readonly blocksPerRow: number;
    readonly weightWordOffset: number;
  };
} {
  requireU32(shape.rows, "Q3_K GEMV rows");
  requireU32(shape.columns, "Q3_K GEMV columns");
  if (shape.columns % Q3_K_ELEMENTS_PER_BLOCK !== 0) {
    throw new Error("Q3_K GEMV columns must be a multiple of 256");
  }
  const packedByteOffset = shape.packedByteOffset ?? 0;
  requireU32(packedByteOffset, "Q3_K packed byte offset", true);
  if (packedByteOffset % 4 !== 0) {
    throw new Error("Q3_K packed byte offset must be u32 aligned");
  }
  if (packedByteOffset % WEBGPU_Q3_K_BLOCK_BYTES !== 0) {
    throw new Error("Q3_K packed byte offset must begin on a 112-byte block");
  }
  const blocksPerRow = shape.columns / Q3_K_ELEMENTS_PER_BLOCK;
  const weightWordOffset = packedByteOffset / 4;
  requireU32(weightWordOffset, "Q3_K packed word offset", true);
  const maxWorkgroupsPerDimension =
    shape.maxWorkgroupsPerDimension ?? 65_535;
  requireU32(
    maxWorkgroupsPerDimension,
    "Q3_K max workgroups per dimension",
  );
  const workgroupsX = Math.min(shape.rows, maxWorkgroupsPerDimension);
  const workgroupsY = Math.ceil(shape.rows / workgroupsX);
  if (workgroupsY > maxWorkgroupsPerDimension) {
    throw new Error(
      "Q3_K row dispatch exceeds the two-dimensional device limit",
    );
  }

  const lastWord =
    BigInt(weightWordOffset) +
    BigInt(shape.rows) * BigInt(blocksPerRow) *
      BigInt(Q3K_GEMV_ABI.wordsPerBlock);
  if (lastWord > 0x1_0000_0000n) {
    throw new Error("Q3_K packed extent exceeds u32 shader addressing");
  }

  return {
    workgroups: { x: workgroupsX, y: workgroupsY, z: 1 },
    uniforms: {
      rows: shape.rows,
      columns: shape.columns,
      blocksPerRow,
      weightWordOffset,
    },
  };
}

/**
 * Independent matrix-vector reference over the aligned packed blocks. It uses
 * the established Q3_K block decoder but does not share shader index helpers.
 */
export function q3kGemvCpu(
  packedWeights: Uint8Array,
  activation: Float32Array,
  shape: Q3KGemvShape,
): Float32Array {
  const plan = planQ3KGemvDispatch(shape);
  if (activation.length !== shape.columns) {
    throw new Error(
      `Q3_K GEMV activation length must equal ${shape.columns}`,
    );
  }
  const packedByteOffset = shape.packedByteOffset ?? 0;
  const requiredBytes =
    packedByteOffset +
    shape.rows *
      plan.uniforms.blocksPerRow *
      WEBGPU_Q3_K_BLOCK_BYTES;
  if (
    !Number.isSafeInteger(requiredBytes) ||
    packedWeights.byteLength < requiredBytes
  ) {
    throw new Error("Q3_K GEMV packed weights do not cover the matrix extent");
  }

  const output = new Float32Array(shape.rows);
  for (let row = 0; row < shape.rows; row += 1) {
    let sum = 0;
    for (
      let blockIndex = 0;
      blockIndex < plan.uniforms.blocksPerRow;
      blockIndex += 1
    ) {
      const offset =
        packedByteOffset +
        (row * plan.uniforms.blocksPerRow + blockIndex) *
          WEBGPU_Q3_K_BLOCK_BYTES;
      const values = dequantizeQ3KBlock(
        unpackWebGpuQ3KBlock(
          packedWeights.subarray(offset, offset + WEBGPU_Q3_K_BLOCK_BYTES),
        ),
      );
      const activationOffset = blockIndex * Q3_K_ELEMENTS_PER_BLOCK;
      for (let element = 0; element < values.length; element += 1) {
        sum += values[element]! * activation[activationOffset + element]!;
      }
    }
    output[row] = sum;
  }
  return output;
}
