/**
 * Provides one direct packed-weight GEMV family for each pinned language type.
 *
 * Shaders reconstruct scalar weights in registers and never allocate an
 * expanded floating-point weight matrix.
 */
import { GgmlType } from "./gguf.js";
import type { KernelDefinition } from "./kernel-registry.js";
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
  | "q3-k-nibble-148"
  | "q3-k-fused-f32-192"
  | "q4-k-144"
  | "q4-k-fused-f32-192"
  | "q5-k-176"
  | "q5-k-fused-f32-224"
  | "q6-k-212"
  | "q6-k-fused-f32-256";

export type GemvKernelProfile = "portable-f32" | "mobile-f16-subgroup";

export interface GemvKernelAbi {
  readonly layout: GemvLayout;
  readonly phase: "shared";
  readonly profile: GemvKernelProfile;
  readonly wordsPerBlock: number;
  readonly bytesPerBlock: number;
  readonly valuesPerBlock: number;
  readonly workgroupSize: 32 | 64 | 128;
  readonly rowsPerWorkgroup: 1 | 2 | 4 | 6 | 8 | 16;
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
  readonly profile: GemvKernelProfile;
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
    ggmlType: GgmlType.Q3_K,
    layout: "q3-k-fused-f32-192",
    bytesPerBlock: BROWSER_Q3_FUSED_K_BLOCK_BYTES,
    valuesPerBlock: K_QUANT_ELEMENTS_PER_BLOCK,
    weightValue: `
fn weight_value(block_word: u32, element: u32) -> f32 {
  let scale_index = element / 16u;
  let factor = bitcast<f32>(packed_weights[block_word + scale_index]);
  let packed = packed_byte(block_word, 64u + element / 2u);
  let quant = i32((packed >> ((element % 2u) * 4u)) & 15u) - 4;
  return factor * f32(quant);
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
    ggmlType: GgmlType.Q4_K,
    layout: "q4-k-fused-f32-192",
    bytesPerBlock: BROWSER_Q4_K_BLOCK_BYTES,
    valuesPerBlock: K_QUANT_ELEMENTS_PER_BLOCK,
    weightValue: `
fn weight_value(block_word: u32, element: u32) -> f32 {
  let group = element / 32u;
  let lane = element % 32u;
  let factor = bitcast<f32>(packed_weights[block_word + group]);
  let bias = bitcast<f32>(packed_weights[block_word + 8u + group]);
  let lows = packed_byte(block_word, 64u + lane * 4u + group / 2u);
  let quant = (lows >> ((group % 2u) * 4u)) & 15u;
  return factor * f32(quant) - bias;
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
  {
    ggmlType: GgmlType.Q3_K,
    layout: "q3-k-nibble-148",
    bytesPerBlock: BROWSER_Q3_K_BLOCK_BYTES,
    valuesPerBlock: K_QUANT_ELEMENTS_PER_BLOCK,
    weightValue: `
fn weight_value(block_word: u32, element: u32) -> f32 {
  let delta = unpack2x16float(packed_weights[block_word]).x;
  let raw_scale = packed_byte(block_word, 4u + element / 16u);
  let scale = select(i32(raw_scale), i32(raw_scale) - 256, raw_scale >= 128u);
  let packed = packed_byte(block_word, 20u + element / 2u);
  let quant = i32((packed >> ((element % 2u) * 4u)) & 15u) - 4;
  return delta * f32(scale * quant);
}`,
  },
  {
    ggmlType: GgmlType.Q5_K,
    layout: "q5-k-fused-f32-224",
    bytesPerBlock: BROWSER_Q5_K_BLOCK_BYTES,
    valuesPerBlock: K_QUANT_ELEMENTS_PER_BLOCK,
    weightValue: `
fn weight_value(block_word: u32, element: u32) -> f32 {
  let group = element / 32u;
  let lane = element % 32u;
  let factor = bitcast<f32>(packed_weights[block_word + group]);
  let bias = bitcast<f32>(packed_weights[block_word + 8u + group]);
  let lows = packed_byte(block_word, 96u + lane * 4u + group / 2u);
  let low = (lows >> ((group % 2u) * 4u)) & 15u;
  let high = (packed_byte(block_word, 64u + lane) >> group) & 1u;
  return factor * f32(low + high * 16u) - bias;
}`,
  },
  {
    ggmlType: GgmlType.Q6_K,
    layout: "q6-k-fused-f32-256",
    bytesPerBlock: BROWSER_Q6_K_BLOCK_BYTES,
    valuesPerBlock: K_QUANT_ELEMENTS_PER_BLOCK,
    weightValue: `
fn weight_value(block_word: u32, element: u32) -> f32 {
  let group = element / 32u;
  let lane = element % 32u;
  let factor_index = (lane / 16u) * 8u + group;
  let factor = bitcast<f32>(packed_weights[block_word + factor_index]);
  let lows = packed_byte(block_word, 64u + lane * 4u + group / 2u);
  let low = (lows >> ((group % 2u) * 4u)) & 15u;
  let highs = packed_byte(block_word, 192u + lane * 2u + group / 4u);
  let high = (highs >> ((group % 4u) * 2u)) & 3u;
  return factor * f32(i32(low + high * 16u) - 32);
}`,
  },
] as const;

export interface PackedWeightDecoder {
  readonly ggmlType: GgmlType;
  readonly layout: GemvLayout;
  readonly bytesPerBlock: number;
  readonly valuesPerBlock: number;
  /** Requires `packed_weights` and `packed_byte` in the containing shader. */
  readonly source: string;
}

/**
 * Shares the exact register decoder between GEMV and other direct packed-row
 * kernels so quantization field order cannot drift across shader families.
 */
export function packedWeightDecoder(layout: GemvLayout): PackedWeightDecoder {
  const shape = WGSL_SHAPES.find((candidate) => candidate.layout === layout);
  if (shape === undefined) {
    throw new Error("Unsupported packed weight decoder layout");
  }
  return Object.freeze({
    ggmlType: shape.ggmlType,
    layout: shape.layout,
    bytesPerBlock: shape.bytesPerBlock,
    valuesPerBlock: shape.valuesPerBlock,
    source: shape.weightValue,
  });
}

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

var<workgroup> partials: array<f32, 64>;

@compute @workgroup_size(64)
fn packed_gemv(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(num_workgroups) grid: vec3<u32>,
) {
  // Reject surplus 2D invocations before flattening can wrap u32.
  if (group.y > (0xffffffffu - group.x) / grid.x) {
    return;
  }
  let row = group.y * grid.x + group.x;
  if (row >= params.local_rows) {
    return;
  }
  var sum = 0.0f;
  for (var column = local.x; column < params.columns; column += 64u) {
    let block = column / VALUES_PER_BLOCK;
    let element = column % VALUES_PER_BLOCK;
    let block_word = params.weight_word_offset +
      (row * params.blocks_per_row + block) * WORDS_PER_BLOCK;
    sum += weight_value(block_word, element) * activation[column];
  }
  partials[local.x] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) {
      partials[local.x] += partials[local.x + stride];
    }
    workgroupBarrier();
  }
  if (local.x == 0u) {
    output[params.output_row_offset + row] = partials[0];
  }
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
    workgroupSize: 64,
    rowsPerWorkgroup: 1,
    uniformWords: 5,
    bindings: Object.freeze({
      packedWeights: 0,
      activation: 1,
      output: 2,
      uniforms: 3,
    }),
  });
}

const FAST_GEMV_PREAMBLE = /* wgsl */ `
enable f16;
enable subgroups;
struct GemvUniforms {
  local_rows: u32,
  columns: u32,
  blocks_per_row: u32,
  weight_word_offset: u32,
  output_row_offset: u32,
}
@group(0) @binding(0) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: GemvUniforms;
fn packed_byte(base: u32, offset: u32) -> u32 {
  let word = packed_weights[base + offset / 4u];
  return (word >> ((offset % 4u) * 8u)) & 0xffu;
}
fn signed_byte(raw: u32) -> i32 {
  return select(i32(raw), i32(raw) - 256, raw >= 128u);
}
fn nibbles4(word: u32, shift: u32) -> vec4<f16> {
  return vec4<f16>(
    f16((word >> shift) & 15u),
    f16((word >> (shift + 4u)) & 15u),
    f16((word >> (shift + 8u)) & 15u),
    f16((word >> (shift + 12u)) & 15u),
  );
}
fn half4(first: u32, second: u32) -> vec4<f16> {
  let lower = unpack2x16float(first);
  let upper = unpack2x16float(second);
  return vec4<f16>(f16(lower.x), f16(lower.y), f16(upper.x), f16(upper.y));
}
`;

const FAST_ROW_INDEX = /* wgsl */ `
  if (group.y > (0xffffffffu - group.x) / grid.x) { return; }
  let row = group.y * grid.x + group.x;
  if (row >= params.local_rows) { return; }
`;

const Q3_FUSED_GEMV = /* wgsl */ `${FAST_GEMV_PREAMBLE}
@group(0) @binding(1) var<storage, read> activation: array<f32>;
@compute @workgroup_size(64)
fn packed_gemv(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(num_workgroups) grid: vec3<u32>,
) {
  if (group.y > (0xffffffffu - group.x) / grid.x) { return; }
  let workgroup_row = group.y * grid.x + group.x;
  if (workgroup_row > 0xffffffffu / 6u) { return; }
  let subgroup = local_id.x / 32u;
  let local = vec3<u32>(local_id.x % 32u, 0u, 0u);
  var row_sums: array<f32, 3>;
  for (var block = 0u; block < params.blocks_per_row; block += 1u) {
    let input_base = block * 256u + local.x * 8u;
    let first = vec4<f16>(
      f16(activation[input_base]), f16(activation[input_base + 1u]),
      f16(activation[input_base + 2u]), f16(activation[input_base + 3u])
    );
    let second = vec4<f16>(
      f16(activation[input_base + 4u]), f16(activation[input_base + 5u]),
      f16(activation[input_base + 6u]), f16(activation[input_base + 7u])
    );
    for (var slot = 0u; slot < 3u; slot += 1u) {
      let logical_row = workgroup_row * 6u + subgroup + slot * 2u;
      let row = min(logical_row, params.local_rows - 1u);
      let base = params.weight_word_offset +
        (row * params.blocks_per_row + block) * 48u;
      let quant_word = packed_weights[base + 16u + local.x];
      let factor = f16(bitcast<f32>(packed_weights[base + local.x / 2u]));
      row_sums[slot] += f32(factor * (
        dot(nibbles4(quant_word, 0u) - vec4<f16>(4.0h), first) +
        dot(nibbles4(quant_word, 16u) - vec4<f16>(4.0h), second)
      ));
    }
  }
  for (var slot = 0u; slot < 3u; slot += 1u) {
    let logical_row = workgroup_row * 6u + subgroup + slot * 2u;
    let reduced = subgroupAdd(row_sums[slot]);
    if (local.x == 0u && logical_row < params.local_rows) {
      output[params.output_row_offset + logical_row] = reduced;
    }
  }
}`;

const FAST_F32_ACTIVATION = /* wgsl */ `
@group(0) @binding(1) var<storage, read> activation: array<f32>;
fn activation4(block: u32, lane: u32, first_group: u32) -> vec4<f16> {
  let base = block * 256u + first_group * 32u + lane;
  return vec4<f16>(
    f16(activation[base]),
    f16(activation[base + 32u]),
    f16(activation[base + 64u]),
    f16(activation[base + 96u]),
  );
}
`;

const FAST_PACKED_LANE_F16_ACTIVATION = /* wgsl */ `
@group(0) @binding(1) var<storage, read> activation: array<u32>;
fn activation4(block: u32, lane: u32, first_group: u32) -> vec4<f16> {
  let base = block * 128u + (first_group / 2u) * 32u + lane;
  let first = unpack2x16float(activation[base]);
  let second = unpack2x16float(activation[base + 32u]);
  return vec4<f16>(f16(first.x), f16(first.y), f16(second.x), f16(second.y));
}
`;

const Q8_FUSED_GEMV = /* wgsl */ `${FAST_GEMV_PREAMBLE}
@group(0) @binding(1) var<storage, read> activation: array<f32>;
@compute @workgroup_size(128)
fn packed_gemv(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(num_workgroups) grid: vec3<u32>,
) {
  if (group.y > (0xffffffffu - group.x) / grid.x) { return; }
  let workgroup_row = group.y * grid.x + group.x;
  if (workgroup_row > 0xffffffffu / 16u) { return; }
  let subgroup = local_id.x / 32u;
  let lane = local_id.x % 32u;
  var row_sums: array<f32, 4>;
  for (var block_group = 0u; block_group < params.blocks_per_row; block_group += 4u) {
      let block = block_group + lane / 8u;
      if (block >= params.blocks_per_row) { continue; }
      let word_lane = lane % 8u;
      let input_base = block * 32u + word_lane * 4u;
      let input = vec4<f16>(
        f16(activation[input_base]), f16(activation[input_base + 1u]),
        f16(activation[input_base + 2u]), f16(activation[input_base + 3u])
      );
      for (var slot = 0u; slot < 4u; slot += 1u) {
        let logical_row = workgroup_row * 16u + subgroup + slot * 4u;
        let row = min(logical_row, params.local_rows - 1u);
        let base = params.weight_word_offset +
          (row * params.blocks_per_row + block) * 9u;
        let quant_word = packed_weights[base + 1u + word_lane];
        let quant = vec4<f16>(
          f16(signed_byte(quant_word & 255u)),
          f16(signed_byte((quant_word >> 8u) & 255u)),
          f16(signed_byte((quant_word >> 16u) & 255u)),
          f16(signed_byte((quant_word >> 24u) & 255u)),
        );
        let delta = f16(unpack2x16float(packed_weights[base]).x);
        row_sums[slot] += f32(delta * dot(quant, input));
      }
    }
  for (var slot = 0u; slot < 4u; slot += 1u) {
    let logical_row = workgroup_row * 16u + subgroup + slot * 4u;
    let reduced = subgroupAdd(row_sums[slot]);
    if (lane == 0u && logical_row < params.local_rows) {
      output[params.output_row_offset + logical_row] = reduced;
    }
  }
}`;

const Q4_FUSED_GEMV = /* wgsl */ `${FAST_GEMV_PREAMBLE}${FAST_F32_ACTIVATION}
@compute @workgroup_size(64)
fn packed_gemv(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(num_workgroups) grid: vec3<u32>,
) {
  if (group.y > (0xffffffffu - group.x) / grid.x) { return; }
  let workgroup_row = group.y * grid.x + group.x;
  if (workgroup_row > 0xffffffffu / 2u) { return; }
  let logical_row = workgroup_row * 2u + local_id.x / 32u;
  let row = min(logical_row, params.local_rows - 1u);
  let local = vec3<u32>(local_id.x % 32u, 0u, 0u);
  let row_base = params.weight_word_offset + row * params.blocks_per_row * 48u;
  var sum = 0.0f;
  for (var block = 0u; block < params.blocks_per_row; block += 1u) {
    let base = row_base + block * 48u;
    let lows = packed_weights[base + 16u + local.x];
    let weights0 = vec4<f16>(
      f16(bitcast<f32>(packed_weights[base])),
      f16(bitcast<f32>(packed_weights[base + 1u])),
      f16(bitcast<f32>(packed_weights[base + 2u])),
      f16(bitcast<f32>(packed_weights[base + 3u])),
    ) * nibbles4(lows, 0u) - vec4<f16>(
      f16(bitcast<f32>(packed_weights[base + 8u])),
      f16(bitcast<f32>(packed_weights[base + 9u])),
      f16(bitcast<f32>(packed_weights[base + 10u])),
      f16(bitcast<f32>(packed_weights[base + 11u])),
    );
    let weights1 = vec4<f16>(
      f16(bitcast<f32>(packed_weights[base + 4u])),
      f16(bitcast<f32>(packed_weights[base + 5u])),
      f16(bitcast<f32>(packed_weights[base + 6u])),
      f16(bitcast<f32>(packed_weights[base + 7u])),
    ) * nibbles4(lows, 16u) - vec4<f16>(
      f16(bitcast<f32>(packed_weights[base + 12u])),
      f16(bitcast<f32>(packed_weights[base + 13u])),
      f16(bitcast<f32>(packed_weights[base + 14u])),
      f16(bitcast<f32>(packed_weights[base + 15u])),
    );
    sum += f32(
      dot(weights0, activation4(block, local.x, 0u)) +
      dot(weights1, activation4(block, local.x, 4u))
    );
  }
  let reduced = subgroupAdd(sum);
  if (local.x == 0u && logical_row < params.local_rows) {
    output[params.output_row_offset + row] = reduced;
  }
}`;

const Q5_FUSED_GEMV = /* wgsl */ `${FAST_GEMV_PREAMBLE}${FAST_PACKED_LANE_F16_ACTIVATION}
fn high4(high: u32, first: u32) -> vec4<f16> {
  return vec4<f16>(
    f16((high >> first) & 1u),
    f16((high >> (first + 1u)) & 1u),
    f16((high >> (first + 2u)) & 1u),
    f16((high >> (first + 3u)) & 1u),
  ) * vec4<f16>(16.0h);
}
@compute @workgroup_size(128)
fn packed_gemv(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(num_workgroups) grid: vec3<u32>,
) {
  if (group.y > (0xffffffffu - group.x) / grid.x) { return; }
  let workgroup_row = group.y * grid.x + group.x;
  if (workgroup_row > 0xffffffffu / 16u) { return; }
  let subgroup = local_id.x / 32u;
  let lane = local_id.x % 32u;
  var row_sums: array<f32, 1>;
  for (var block = 0u; block < params.blocks_per_row; block += 1u) {
    let input0 = activation4(block, lane, 0u);
    let input1 = activation4(block, lane, 4u);
    for (var slot = 0u; slot < 1u; slot += 1u) {
      let logical_row = workgroup_row * 4u + subgroup + slot * 4u;
      let row = min(logical_row, params.local_rows - 1u);
      let row_base = params.weight_word_offset + row * params.blocks_per_row * 56u;
      let base = row_base + block * 56u;
      let high = packed_byte(base, 64u + lane);
      let lows = packed_weights[base + 24u + lane];
      let q0 = nibbles4(lows, 0u) + high4(high, 0u);
      let q1 = nibbles4(lows, 16u) + high4(high, 4u);
      let weights0 = vec4<f16>(
        f16(bitcast<f32>(packed_weights[base])), f16(bitcast<f32>(packed_weights[base + 1u])),
        f16(bitcast<f32>(packed_weights[base + 2u])), f16(bitcast<f32>(packed_weights[base + 3u]))
      ) * q0 - vec4<f16>(
        f16(bitcast<f32>(packed_weights[base + 8u])), f16(bitcast<f32>(packed_weights[base + 9u])),
        f16(bitcast<f32>(packed_weights[base + 10u])), f16(bitcast<f32>(packed_weights[base + 11u]))
      );
      let weights1 = vec4<f16>(
        f16(bitcast<f32>(packed_weights[base + 4u])), f16(bitcast<f32>(packed_weights[base + 5u])),
        f16(bitcast<f32>(packed_weights[base + 6u])), f16(bitcast<f32>(packed_weights[base + 7u]))
      ) * q1 - vec4<f16>(
        f16(bitcast<f32>(packed_weights[base + 12u])), f16(bitcast<f32>(packed_weights[base + 13u])),
        f16(bitcast<f32>(packed_weights[base + 14u])), f16(bitcast<f32>(packed_weights[base + 15u]))
      );
      row_sums[slot] += f32(dot(weights0, input0) + dot(weights1, input1));
    }
  }
  for (var slot = 0u; slot < 1u; slot += 1u) {
    let logical_row = workgroup_row * 4u + subgroup + slot * 4u;
    let reduced = subgroupAdd(row_sums[slot]);
    if (lane == 0u && logical_row < params.local_rows) {
      output[params.output_row_offset + logical_row] = reduced;
    }
  }
}`;

const Q6_FUSED_GEMV = /* wgsl */ `${FAST_GEMV_PREAMBLE}${FAST_F32_ACTIVATION}
fn high4(high: u32, first: u32) -> vec4<f16> {
  return vec4<f16>(
    f16((high >> (first * 2u)) & 3u),
    f16((high >> ((first + 1u) * 2u)) & 3u),
    f16((high >> ((first + 2u) * 2u)) & 3u),
    f16((high >> ((first + 3u) * 2u)) & 3u),
  );
}
@compute @workgroup_size(128)
fn packed_gemv(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(num_workgroups) grid: vec3<u32>,
) {
  if (group.y > (0xffffffffu - group.x) / grid.x) { return; }
  let workgroup_row = group.y * grid.x + group.x;
  if (workgroup_row > 0xffffffffu / 8u) { return; }
  let subgroup = local_id.x / 32u;
  let lane = local_id.x % 32u;
  var row_sums: array<f32, 2>;
  for (var block = 0u; block < params.blocks_per_row; block += 1u) {
    let input0 = activation4(block, lane, 0u);
    let input1 = activation4(block, lane, 4u);
    for (var slot = 0u; slot < 2u; slot += 1u) {
      let logical_row = workgroup_row * 8u + subgroup + slot * 4u;
      let row = min(logical_row, params.local_rows - 1u);
      let base = params.weight_word_offset +
        (row * params.blocks_per_row + block) * 64u;
      let lows = packed_weights[base + 16u + lane];
      let high = packed_byte(base, 192u + lane * 2u) |
        (packed_byte(base, 193u + lane * 2u) << 8u);
      let q0 = nibbles4(lows, 0u) + high4(high, 0u) * vec4<f16>(16.0h) - vec4<f16>(32.0h);
      let q1 = nibbles4(lows, 16u) + high4(high, 4u) * vec4<f16>(16.0h) - vec4<f16>(32.0h);
      let scale_base = select(0u, 8u, lane >= 16u);
      let weights0 = vec4<f16>(
        f16(bitcast<f32>(packed_weights[base + scale_base])),
        f16(bitcast<f32>(packed_weights[base + scale_base + 1u])),
        f16(bitcast<f32>(packed_weights[base + scale_base + 2u])),
        f16(bitcast<f32>(packed_weights[base + scale_base + 3u]))
      ) * q0;
      let weights1 = vec4<f16>(
        f16(bitcast<f32>(packed_weights[base + scale_base + 4u])),
        f16(bitcast<f32>(packed_weights[base + scale_base + 5u])),
        f16(bitcast<f32>(packed_weights[base + scale_base + 6u])),
        f16(bitcast<f32>(packed_weights[base + scale_base + 7u]))
      ) * q1;
      row_sums[slot] += f32(dot(weights0, input0) + dot(weights1, input1));
    }
  }
  for (var slot = 0u; slot < 2u; slot += 1u) {
    let logical_row = workgroup_row * 8u + subgroup + slot * 4u;
    let reduced = subgroupAdd(row_sums[slot]);
    if (lane == 0u && logical_row < params.local_rows) {
      output[params.output_row_offset + logical_row] = reduced;
    }
  }
}`;

/**
 * Builds the subgroup path directly over the compact GGML-compatible blocks.
 * One subgroup owns four rows and reuses each packed activation block across
 * those rows. This preserves the compact package instead of expanding factors
 * solely to make decode fast.
 */
function compactSubgroupShaderSource(shape: KernelShape): string {
  return /* wgsl */ `
enable f16;
enable subgroups;
const WORDS_PER_BLOCK: u32 = ${shape.bytesPerBlock / 4}u;
struct GemvUniforms {
  local_rows: u32,
  columns: u32,
  blocks_per_row: u32,
  weight_word_offset: u32,
  output_row_offset: u32,
}
@group(0) @binding(0) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(1) var<storage, read> activation: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: GemvUniforms;
fn packed_byte(block_word: u32, byte_offset: u32) -> u32 {
  let word = packed_weights[block_word + byte_offset / 4u];
  return (word >> ((byte_offset % 4u) * 8u)) & 0xffu;
}
${shape.weightValue}
@compute @workgroup_size(128)
fn packed_gemv(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(num_workgroups) grid: vec3<u32>,
) {
  if (group.y > (0xffffffffu - group.x) / grid.x) { return; }
  let workgroup_row = group.y * grid.x + group.x;
  if (workgroup_row > 0xffffffffu / 16u) { return; }
  let subgroup = local_id.x / 32u;
  let lane = local_id.x % 32u;
  var row_sums: array<f32, 4>;
  for (var block = 0u; block < params.blocks_per_row; block += 1u) {
    let input_base = block * 128u + lane * 4u;
    let a01 = unpack2x16float(activation[input_base]);
    let a23 = unpack2x16float(activation[input_base + 1u]);
    let a45 = unpack2x16float(activation[input_base + 2u]);
    let a67 = unpack2x16float(activation[input_base + 3u]);
    let input0 = vec4<f16>(f16(a01.x), f16(a01.y), f16(a23.x), f16(a23.y));
    let input1 = vec4<f16>(f16(a45.x), f16(a45.y), f16(a67.x), f16(a67.y));
    for (var slot = 0u; slot < 4u; slot += 1u) {
      let logical_row = workgroup_row * 16u + subgroup + slot * 4u;
      let row = min(logical_row, params.local_rows - 1u);
      let base = params.weight_word_offset +
        (row * params.blocks_per_row + block) * WORDS_PER_BLOCK;
      let element = lane * 8u;
      var partial = 0.0f;
      for (var index = 0u; index < 4u; index += 1u) {
        partial += f32(f16(weight_value(base, element + index)) * input0[index]);
        partial += f32(f16(weight_value(base, element + 4u + index)) * input1[index]);
      }
      row_sums[slot] += partial;
    }
  }
  for (var slot = 0u; slot < 4u; slot += 1u) {
    let logical_row = workgroup_row * 16u + subgroup + slot * 4u;
    let reduced = subgroupAdd(row_sums[slot]);
    if (lane == 0u && logical_row < params.local_rows) {
      output[params.output_row_offset + logical_row] = reduced;
    }
  }
}`;
}

function fastKernel(input: {
  readonly ggmlType: GgmlType;
  readonly layout: GemvLayout;
  readonly bytesPerBlock: number;
  readonly valuesPerBlock?: 32 | 256;
  readonly source: string;
  readonly workgroupSize?: 32 | 64 | 128;
  readonly rowsPerWorkgroup?: 1 | 2 | 4 | 6 | 8 | 16;
}): LanguageGemvKernel {
  return Object.freeze({
    id: `${input.layout}-gemv-shared-mobile-f16-subgroup`,
    operation: "gemv" as const,
    ggmlType: input.ggmlType,
    layout: input.layout,
    phase: "shared" as const,
    profile: "mobile-f16-subgroup" as const,
    abi: Object.freeze({
      layout: input.layout,
      phase: "shared" as const,
      profile: "mobile-f16-subgroup" as const,
      wordsPerBlock: input.bytesPerBlock / 4,
      bytesPerBlock: input.bytesPerBlock,
      valuesPerBlock: input.valuesPerBlock ?? 256,
      workgroupSize: input.workgroupSize ?? 32,
      rowsPerWorkgroup: input.rowsPerWorkgroup ?? 1,
      uniformWords: 5 as const,
      bindings: Object.freeze({ packedWeights: 0 as const, activation: 1 as const, output: 2 as const, uniforms: 3 as const }),
    }),
    source: input.source,
  });
}

const FAST_LANGUAGE_GEMV_KERNELS: readonly LanguageGemvKernel[] = Object.freeze([
  fastKernel({ ggmlType: GgmlType.Q8_0, layout: "q8-0-36", bytesPerBlock: WEBGPU_Q8_0_BLOCK_BYTES, valuesPerBlock: 32, source: Q8_FUSED_GEMV, workgroupSize: 128, rowsPerWorkgroup: 16 }),
  ...WGSL_SHAPES.filter((shape) => [
    "q3-k-112", "q3-k-nibble-148", "q4-k-144", "q5-k-176", "q6-k-212",
  ].includes(shape.layout)).map((shape) => fastKernel({
    ggmlType: shape.ggmlType,
    layout: shape.layout,
    bytesPerBlock: shape.bytesPerBlock,
    source: compactSubgroupShaderSource(shape),
    workgroupSize: 128,
    rowsPerWorkgroup: 16,
  })),
  fastKernel({ ggmlType: GgmlType.Q3_K, layout: "q3-k-fused-f32-192", bytesPerBlock: BROWSER_Q3_FUSED_K_BLOCK_BYTES, source: Q3_FUSED_GEMV, workgroupSize: 64, rowsPerWorkgroup: 6 }),
  fastKernel({ ggmlType: GgmlType.Q4_K, layout: "q4-k-fused-f32-192", bytesPerBlock: BROWSER_Q4_K_BLOCK_BYTES, source: Q4_FUSED_GEMV, workgroupSize: 64, rowsPerWorkgroup: 2 }),
  fastKernel({ ggmlType: GgmlType.Q5_K, layout: "q5-k-fused-f32-224", bytesPerBlock: BROWSER_Q5_K_BLOCK_BYTES, source: Q5_FUSED_GEMV, workgroupSize: 128, rowsPerWorkgroup: 4 }),
  fastKernel({ ggmlType: GgmlType.Q6_K, layout: "q6-k-fused-f32-256", bytesPerBlock: BROWSER_Q6_K_BLOCK_BYTES, source: Q6_FUSED_GEMV, workgroupSize: 128, rowsPerWorkgroup: 8 }),
]);

export const LANGUAGE_GEMV_KERNELS: readonly LanguageGemvKernel[] =
  Object.freeze(
    [...WGSL_SHAPES.map((shape) =>
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
    ), ...FAST_LANGUAGE_GEMV_KERNELS],
  );

export interface LanguageGemvRegistryDefinition extends KernelDefinition {
  readonly ggmlType: GgmlType;
  readonly abi: Omit<GemvKernelAbi, "phase" | "profile"> & {
    readonly phase: "prefill" | "decode";
    readonly profile: GemvKernelProfile;
  };
}

/**
 * The same row kernel is valid in both phases, but registry keys keep the
 * caller's phase explicit so a future specialized kernel cannot be selected
 * by accident.
 */
export function languageGemvRegistryDefinitions(input: {
  readonly phase: "prefill" | "decode";
  readonly profile: GemvKernelProfile;
}): readonly LanguageGemvRegistryDefinition[] {
  if (input.phase !== "prefill" && input.phase !== "decode") {
    throw new Error("Unsupported GEMV kernel phase");
  }
  if (input.profile !== "portable-f32" && input.profile !== "mobile-f16-subgroup") {
    throw new Error("Unsupported GEMV kernel profile");
  }
  return Object.freeze(
    LANGUAGE_GEMV_KERNELS.filter((kernel) => kernel.profile === input.profile).map((kernel) =>
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

function kernelFor(
  layout: GemvLayout,
  profile: GemvKernelProfile = "portable-f32",
): LanguageGemvKernel {
  const kernel = LANGUAGE_GEMV_KERNELS.find(
    (candidate) => candidate.layout === layout && candidate.profile === profile,
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
  /** Must match the kernel that will execute this dispatch. */
  readonly profile?: GemvKernelProfile;
  readonly ggmlType?: GgmlType;
  readonly localRows: number;
  readonly columns: number;
  readonly packedByteOffset?: number;
  readonly outputRowOffset?: number;
  readonly maxWorkgroupsPerDimension?: number;
}): GemvDispatchPlan {
  const kernel = kernelFor(input.layout, input.profile);
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
  const dispatchRows = Math.ceil(input.localRows / kernel.abi.rowsPerWorkgroup);
  const x = Math.min(dispatchRows, maxWorkgroups);
  const y = Math.ceil(dispatchRows / x);
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
  readonly profile?: GemvKernelProfile;
  readonly rows: number;
  readonly columns: number;
  readonly shards: readonly LogicalMatrixShard[];
  readonly maxWorkgroupsPerDimension?: number;
}): readonly (GemvDispatchPlan & { readonly shardIndex: number })[] {
  const kernel = kernelFor(input.layout, input.profile);
  requireU32(input.rows, "GEMV matrix rows");
  const probe = planGemvDispatch({
    layout: input.layout,
    ...(input.profile === undefined ? {} : { profile: input.profile }),
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
      ...(input.profile === undefined ? {} : { profile: input.profile }),
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
