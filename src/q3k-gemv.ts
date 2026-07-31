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
  uniformWords: 5,
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
  if (row >= params.local_rows) {
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
  output[params.output_row_offset + row] = sum;
}
`;

export interface Q3KGemvDispatchShape {
  readonly localRows: number;
  readonly columns: number;
  readonly packedByteOffset?: number;
  readonly outputRowOffset?: number;
  readonly maxWorkgroupsPerDimension?: number;
}

export interface Q3KMatrixShape {
  readonly rows: number;
  readonly columns: number;
  readonly packedByteOffset?: number;
}

export interface Q3KLogicalShard {
  readonly logicalByteOffset: bigint;
  readonly logicalByteLength: bigint;
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

export interface Q3KGemvDispatchPlan {
  readonly workgroups: {
    readonly x: number;
    readonly y: number;
    readonly z: 1;
  };
  readonly uniforms: {
    readonly localRows: number;
    readonly columns: number;
    readonly blocksPerRow: number;
    readonly weightWordOffset: number;
    readonly outputRowOffset: number;
  };
}

export function planQ3KGemvDispatch(
  shape: Q3KGemvDispatchShape,
): Q3KGemvDispatchPlan {
  requireU32(shape.localRows, "Q3_K GEMV localRows");
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
  const outputRowOffset = shape.outputRowOffset ?? 0;
  requireU32(outputRowOffset, "Q3_K output row offset", true);
  if (
    BigInt(outputRowOffset) + BigInt(shape.localRows) >
    0x1_0000_0000n
  ) {
    throw new Error("Q3_K output rows exceed u32 shader addressing");
  }
  const maxWorkgroupsPerDimension =
    shape.maxWorkgroupsPerDimension ?? 65_535;
  requireU32(
    maxWorkgroupsPerDimension,
    "Q3_K max workgroups per dimension",
  );
  const workgroupsX = Math.min(
    shape.localRows,
    maxWorkgroupsPerDimension,
  );
  const workgroupsY = Math.ceil(shape.localRows / workgroupsX);
  if (workgroupsY > maxWorkgroupsPerDimension) {
    throw new Error(
      "Q3_K row dispatch exceeds the two-dimensional device limit",
    );
  }

  const lastWord =
    BigInt(weightWordOffset) +
    BigInt(shape.localRows) * BigInt(blocksPerRow) *
      BigInt(Q3K_GEMV_ABI.wordsPerBlock);
  if (lastWord > 0x1_0000_0000n) {
    throw new Error("Q3_K packed extent exceeds u32 shader addressing");
  }

  return {
    workgroups: { x: workgroupsX, y: workgroupsY, z: 1 },
    uniforms: {
      localRows: shape.localRows,
      columns: shape.columns,
      blocksPerRow,
      weightWordOffset,
      outputRowOffset,
    },
  };
}

export function planQ3KMatrixShardDispatch(input: {
  readonly rows: number;
  readonly columns: number;
  readonly shards: readonly Q3KLogicalShard[];
  readonly maxWorkgroupsPerDimension?: number;
}): readonly (Q3KGemvDispatchPlan & { readonly shardIndex: number })[] {
  requireU32(input.rows, "Q3_K matrix rows");
  requireU32(input.columns, "Q3_K matrix columns");
  if (input.columns % Q3_K_ELEMENTS_PER_BLOCK !== 0) {
    throw new Error("Q3_K matrix columns must be a multiple of 256");
  }
  const blocksPerRow = input.columns / Q3_K_ELEMENTS_PER_BLOCK;
  const rowBytes =
    BigInt(blocksPerRow) * BigInt(WEBGPU_Q3_K_BLOCK_BYTES);
  const expectedBytes = BigInt(input.rows) * rowBytes;
  let nextByteOffset = 0n;
  let nextOutputRow = 0;
  const plans: Array<
    Q3KGemvDispatchPlan & { readonly shardIndex: number }
  > = [];

  for (const [shardIndex, shard] of input.shards.entries()) {
    // Each shard is bound as its own packed-weight buffer. Therefore its
    // shader-local weight offset is zero, while outputRowOffset preserves the
    // matrix-global output position.
    if (
      shard.logicalByteOffset !== nextByteOffset ||
      shard.logicalByteLength <= 0n ||
      shard.logicalByteOffset % BigInt(WEBGPU_Q3_K_BLOCK_BYTES) !== 0n ||
      shard.logicalByteLength % BigInt(WEBGPU_Q3_K_BLOCK_BYTES) !== 0n ||
      shard.logicalByteOffset % rowBytes !== 0n ||
      shard.logicalByteLength % rowBytes !== 0n
    ) {
      throw new Error(
        "Q3_K matrix shards must be contiguous whole rows and blocks",
      );
    }
    const localRowsBig = shard.logicalByteLength / rowBytes;
    if (localRowsBig > BigInt(0xffff_ffff)) {
      throw new Error("Q3_K matrix shard row count exceeds u32");
    }
    const plan = planQ3KGemvDispatch({
      localRows: Number(localRowsBig),
      columns: input.columns,
      outputRowOffset: nextOutputRow,
      ...(input.maxWorkgroupsPerDimension === undefined
        ? {}
        : {
            maxWorkgroupsPerDimension:
              input.maxWorkgroupsPerDimension,
          }),
    });
    plans.push(Object.freeze({ shardIndex, ...plan }));
    nextByteOffset += shard.logicalByteLength;
    nextOutputRow += Number(localRowsBig);
  }
  if (nextByteOffset !== expectedBytes || nextOutputRow !== input.rows) {
    throw new Error("Q3_K matrix shards do not cover the complete matrix");
  }
  return Object.freeze(plans);
}

/**
 * Independent matrix-vector reference over the aligned packed blocks. It uses
 * the established Q3_K block decoder but does not share shader index helpers.
 */
export function q3kGemvCpu(
  packedWeights: Uint8Array,
  activation: Float32Array,
  shape: Q3KMatrixShape,
): Float32Array {
  const plan = planQ3KGemvDispatch({
    localRows: shape.rows,
    columns: shape.columns,
    ...(shape.packedByteOffset === undefined
      ? {}
      : { packedByteOffset: shape.packedByteOffset }),
  });
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
