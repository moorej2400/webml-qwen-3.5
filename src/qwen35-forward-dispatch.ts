import { diagnosticError } from "./diagnostics.js";
import { GgmlType } from "./gguf.js";
import {
  LANGUAGE_GEMV_KERNELS,
  planGemvDispatch,
  type GemvLayout,
  type LanguageGemvKernel,
} from "./mixed-gemv.js";
import {
  PACKED_EMBEDDING_KERNELS,
  planPackedEmbeddingRow,
  type PackedEmbeddingKernel,
} from "./qwen-embedding.js";
import type {
  Qwen35PhysicalRowView,
  Qwen35TensorWeightView,
  Qwen35WeightDirectoryView,
} from "./qwen35-weight-directory.js";
import type { Qwen35StagedPackedRows } from "./qwen35-disk-backed-tied-embedding.js";
import type {
  Qwen35BufferBinding,
  Qwen35DispatchRequest,
  Qwen35KernelSource,
} from "./qwen35-webgpu-executor.js";

export type Qwen35ForwardDeviceLimits = Readonly<{
  minStorageBufferOffsetAlignment: number;
  minUniformBufferOffsetAlignment: number;
  maxStorageBufferBindingSize: number;
  maxUniformBufferBindingSize: number;
  maxComputeWorkgroupsPerDimension: number;
  /** False selects WGSL that does not require the optional WebGPU feature. */
  supportsSubgroups?: boolean;
}>;

export interface Qwen35ForwardBufferSlice {
  readonly buffer: object;
  readonly offset: number;
  /** Available bytes from offset; the planner binds only the required prefix. */
  readonly byteLength: number;
}

export interface Qwen35ForwardDispatchPlan extends Qwen35DispatchRequest {
  /** Caller writes these u32 values into this request's uniform binding. */
  readonly uniformWords: readonly number[];
}

export interface Qwen35TiedLogitsDispatchPlan
  extends Qwen35ForwardDispatchPlan {
  readonly tileIndex: number;
  /** Logical vocabulary row represented by tile element zero. */
  readonly vocabularyStart: number;
  readonly tileRows: number;
  readonly pieceOutputOffset: number;
  readonly pieceRows: number;
  /** Top-k runs once after the piece with this marker completes. */
  readonly completesTile: boolean;
}

export interface Qwen35FusedTiedLogitsDispatchPlan
  extends Qwen35ForwardDispatchPlan {
  readonly candidateStart: number;
  readonly candidateCount: number;
}

export interface Qwen35TiedLogitsGeometry {
  readonly modelRows: 248_320;
  readonly decodableRows: 248_070;
  readonly logicalTileRows: 1_024;
  readonly mathematicalTileCount: 243;
  readonly finalTileRows: 262;
  readonly physicalPieceCount: number;
  readonly reductionDispatchCount: 244;
  readonly uniformCount: number;
}

const MAX_LOGITS_TILE_ROWS = 1_024;
// Resident logits use smaller row ranges than the staged path. This exposes
// enough independent workgroups to saturate a mobile GPU without expanding
// the packed vocabulary matrix.
const FUSED_LOGITS_ROWS_PER_WORKGROUP = 128;
const QWEN35_HIDDEN_SIZE = 2_560;
const QWEN35_VOCABULARY_SIZE = 248_320;
const QWEN35_DECODABLE_LOGIT_ROWS = 248_070;
const QWEN35_MATHEMATICAL_LOGITS_TILES = 243;
const QWEN35_FINAL_LOGITS_TILE_ROWS = 262;
const QWEN35_LOGITS_REDUCTION_DISPATCHES = 244;
const QWEN35_TOP_K_CANDIDATE_CAPACITY = 2_048;

/**
 * Resident Q6 logits must not materialize and reduce 243 separate 1K tiles.
 * One workgroup owns one 128-row range, eight 32-lane subgroups score rows in
 * parallel, and each physical weight buffer emits only its local winners.
 */
const FUSED_Q6_TIED_LOGITS_KERNEL: Qwen35KernelSource = Object.freeze({
  id: "q6-k-fused-f32-256-tied-top1-mobile-f16-subgroup",
  entryPoint: "main",
  source: /* wgsl */ `
enable f16;
enable subgroups;
struct Params {
  local_rows: u32,
  columns: u32,
  blocks_per_row: u32,
  weight_word_offset: u32,
  first_vocabulary_row: u32,
  candidate_start: u32,
}
@group(0) @binding(0) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(1) var<storage, read> activation: array<u32>;
@group(0) @binding(2) var<storage, read_write> candidate_scores: array<f32>;
@group(0) @binding(3) var<storage, read_write> candidate_token_ids: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> group_scores: array<f32, 8>;
var<workgroup> group_tokens: array<u32, 8>;
var<workgroup> group_found: array<u32, 8>;

fn packed_byte(word_base: u32, byte_offset: u32) -> u32 {
  let word = packed_weights[word_base + byte_offset / 4u];
  return (word >> ((byte_offset % 4u) * 8u)) & 255u;
}
fn nibbles4(word: u32, shift: u32) -> vec4<f16> {
  return vec4<f16>(
    f16((word >> shift) & 15u),
    f16((word >> (shift + 4u)) & 15u),
    f16((word >> (shift + 8u)) & 15u),
    f16((word >> (shift + 12u)) & 15u),
  );
}
fn high4(high: u32, first: u32) -> vec4<f16> {
  return vec4<f16>(
    f16((high >> (first * 2u)) & 3u),
    f16((high >> ((first + 1u) * 2u)) & 3u),
    f16((high >> ((first + 2u) * 2u)) & 3u),
    f16((high >> ((first + 3u) * 2u)) & 3u),
  );
}
fn half4(first: u32, second: u32) -> vec4<f16> {
  let a = unpack2x16float(first);
  let b = unpack2x16float(second);
  return vec4<f16>(f16(a.x), f16(a.y), f16(b.x), f16(b.y));
}
fn activation4(block: u32, lane: u32, first_group: u32) -> vec4<f16> {
  let base = block * 128u + (first_group / 2u) * 32u + lane;
  let first = unpack2x16float(activation[base]);
  let second = unpack2x16float(activation[base + 32u]);
  return vec4<f16>(f16(first.x), f16(first.y), f16(second.x), f16(second.y));
}
@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) lane: u32,
  @builtin(subgroup_id) subgroup: u32,
) {
  let range_start = group.x * 128u;
  let range_end = min(range_start + 128u, params.local_rows);
  var found = false;
  var best_score = 0.0f;
  var best_token = 0xffffffffu;
  var row_sums: array<f32, 16>;
  for (var slot = 0u; slot < 16u; slot += 1u) {
    row_sums[slot] = 0.0f;
  }
  // All eight rows owned by a subgroup use the same activation. Keeping blocks
  // outside rows avoids decoding and loading that activation eight times.
  for (var block = 0u; block < params.blocks_per_row; block += 1u) {
    let activation0 = activation4(block, lane, 0u);
    let activation1 = activation4(block, lane, 4u);
    for (var slot = 0u; slot < 16u; slot += 1u) {
      let row = range_start + subgroup + slot * 8u;
      if (row >= range_end) { continue; }
      let row_base = params.weight_word_offset + row * params.blocks_per_row * 64u;
      let base = row_base + block * 64u;
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
      row_sums[slot] += f32(
        dot(weights0, activation0) +
        dot(weights1, activation1)
      );
    }
  }
  for (var slot = 0u; slot < 16u; slot += 1u) {
    let row = range_start + subgroup + slot * 8u;
    if (row >= range_end) { continue; }
    let score = subgroupAdd(row_sums[slot]);
    if (lane == 0u) {
      let token = params.first_vocabulary_row + row;
      let exponent = bitcast<u32>(score) & 0x7f800000u;
      if (token < 248070u && exponent != 0x7f800000u &&
          (!found || score > best_score ||
           (score == best_score && token < best_token))) {
        found = true;
        best_score = score;
        best_token = token;
      }
    }
  }
  if (lane == 0u) {
    group_scores[subgroup] = best_score;
    group_tokens[subgroup] = best_token;
    group_found[subgroup] = select(0u, 1u, found);
  }
  workgroupBarrier();
  for (var stride = 4u; stride > 0u; stride /= 2u) {
    if (local.x < stride && group_found[local.x + stride] != 0u) {
      let other_score = group_scores[local.x + stride];
      let other_token = group_tokens[local.x + stride];
      if (group_found[local.x] == 0u || other_score > group_scores[local.x] ||
          (other_score == group_scores[local.x] && other_token < group_tokens[local.x])) {
        group_scores[local.x] = other_score;
        group_tokens[local.x] = other_token;
        group_found[local.x] = 1u;
      }
    }
    workgroupBarrier();
  }
  if (local.x == 0u) {
    let slot = params.candidate_start + group.x;
    candidate_scores[slot] = group_scores[0];
    candidate_token_ids[slot] = group_tokens[0];
  }
}`,
});

const VISUAL_EMBEDDING_KERNEL: Qwen35KernelSource = Object.freeze({
  id: "qwen35-visual-embedding-f32",
  entryPoint: "main",
  source: /* wgsl */ `
struct Params { output_elements: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.output_elements) { output[id.x] = source[id.x]; }
}`,
});

const PACK_F16_ACTIVATION_KERNEL: Qwen35KernelSource = Object.freeze({
  id: "qwen35-pack-f16-activation",
  entryPoint: "main",
  source: /* wgsl */ `
enable f16;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let element = id.x * 2u;
  packed[id.x] = pack2x16float(vec2<f32>(source[element], source[element + 1u]));
}`,
});

/** Packs F32 activations into the lane-major FP16 order used by Q5 GEMV. */
const PACK_LANE_F16_ACTIVATION_KERNEL: Qwen35KernelSource = Object.freeze({
  id: "qwen35-pack-lane-f16-activation",
  entryPoint: "main",
  source: /* wgsl */ `
enable f16;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let block = id.x / 128u;
  let within = id.x % 128u;
  let pair = within / 32u;
  let lane = within % 32u;
  let first = block * 256u + pair * 64u + lane;
  packed[id.x] = pack2x16float(vec2<f32>(source[first], source[first + 32u]));
}`,
});

const PACK_TIED_LOGITS_ACTIVATION_KERNEL: Qwen35KernelSource = Object.freeze({
  id: "qwen35-pack-tied-logits-activation-f16",
  entryPoint: "main",
  source: /* wgsl */ `
enable f16;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= 1280u) { return; }
  let block = id.x / 128u;
  let within = id.x % 128u;
  let pair = within / 32u;
  let lane = within % 32u;
  let first = block * 256u + pair * 64u + lane;
  packed[id.x] = pack2x16float(vec2<f32>(source[first], source[first + 32u]));
}`,
});

/** Alpha and beta are small F32 matrices with the same normalized input. */
const TWIN_F32_GEMV_KERNEL: Qwen35KernelSource = Object.freeze({
  id: "f32-twin-gemv-portable",
  entryPoint: "main",
  source: /* wgsl */ `
struct Params { rows: u32, columns: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> first_weights: array<f32>;
@group(0) @binding(1) var<storage, read> second_weights: array<f32>;
@group(0) @binding(2) var<storage, read> activation: array<f32>;
@group(0) @binding(3) var<storage, read_write> first_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> second_output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
// Safari exposes shader-f16 without WebGPU subgroups. Keep this small shared
// projection portable so the compact model ABI does not compile a v2 feature.
var<workgroup> first_partials: array<f32, 32>;
var<workgroup> second_partials: array<f32, 32>;
@compute @workgroup_size(32)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let row = group.x;
  if (row >= params.rows) { return; }
  let row_base = row * params.columns;
  var first_sum = 0.0f;
  var second_sum = 0.0f;
  for (var column = local.x; column < params.columns; column += 32u) {
    let value = activation[column];
    first_sum += first_weights[row_base + column] * value;
    second_sum += second_weights[row_base + column] * value;
  }
  first_partials[local.x] = first_sum;
  second_partials[local.x] = second_sum;
  workgroupBarrier();
  for (var stride = 16u; stride > 0u; stride /= 2u) {
    if (local.x < stride) {
      first_partials[local.x] += first_partials[local.x + stride];
      second_partials[local.x] += second_partials[local.x + stride];
    }
    workgroupBarrier();
  }
  if (local.x == 0u) {
    first_output[row] = first_partials[0];
    second_output[row] = second_partials[0];
  }
}`,
});

/**
 * Gate and up consume the same normalized vector and have equal Q3 geometry.
 * One combined grid gives the GPU enough rows to reach bandwidth while one
 * packed activation replaces the two conversions used by separate GEMVs.
 */
const TWIN_Q3_GEMV_KERNEL: Qwen35KernelSource = Object.freeze({
  id: "q3-k-fused-f32-192-swiglu-gemv-mobile-f16-subgroup",
  entryPoint: "main",
  source: /* wgsl */ `
enable f16;
enable subgroups;
struct Params {
  rows: u32,
  columns: u32,
  blocks_per_row: u32,
  pad0: u32,
  pad1: u32,
}
@group(0) @binding(0) var<storage, read> first_weights: array<u32>;
@group(0) @binding(1) var<storage, read> second_weights: array<u32>;
@group(0) @binding(2) var<storage, read> activation: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
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
@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(num_workgroups) grid: vec3<u32>,
) {
  if (group.y > (0xffffffffu - group.x) / grid.x) { return; }
  let workgroup_row = group.y * grid.x + group.x;
  if (workgroup_row > 0xffffffffu / 4u) { return; }
  let subgroup = local_id.x / 32u;
  let lane = local_id.x % 32u;
  var row_sums: array<vec2<f32>, 2>;
  row_sums[0] = vec2<f32>(0.0f);
  row_sums[1] = vec2<f32>(0.0f);
  for (var block = 0u; block < params.blocks_per_row; block += 1u) {
    let input_base = block * 256u + lane * 8u;
    let first = vec4<f16>(
      f16(activation[input_base]), f16(activation[input_base + 1u]),
      f16(activation[input_base + 2u]), f16(activation[input_base + 3u])
    );
    let last = vec4<f16>(
      f16(activation[input_base + 4u]), f16(activation[input_base + 5u]),
      f16(activation[input_base + 6u]), f16(activation[input_base + 7u])
    );
    for (var slot = 0u; slot < 2u; slot += 1u) {
      let logical_row = workgroup_row * 4u + subgroup + slot * 2u;
      let row = min(logical_row, params.rows - 1u);
      let base = row * params.blocks_per_row * 48u + block * 48u;
      let first_quant = first_weights[base + 16u + lane];
      let second_quant = second_weights[base + 16u + lane];
      let first_scale = f16(bitcast<f32>(first_weights[base + lane / 2u]));
      let second_scale = f16(bitcast<f32>(second_weights[base + lane / 2u]));
      row_sums[slot] += vec2<f32>(
        f32(first_scale * (
          dot(nibbles4(first_quant, 0u) - vec4<f16>(4.0h), first) +
          dot(nibbles4(first_quant, 16u) - vec4<f16>(4.0h), last)
        )),
        f32(second_scale * (
          dot(nibbles4(second_quant, 0u) - vec4<f16>(4.0h), first) +
          dot(nibbles4(second_quant, 16u) - vec4<f16>(4.0h), last)
        )),
      );
    }
  }
  for (var slot = 0u; slot < 2u; slot += 1u) {
    let logical_row = workgroup_row * 4u + subgroup + slot * 2u;
    let gate = subgroupAdd(row_sums[slot].x);
    let up = subgroupAdd(row_sums[slot].y);
    if (lane == 0u && logical_row < params.rows) {
      output[logical_row] = (gate / (1.0f + exp(-gate))) * up;
    }
  }
}`,
});

/** Value-equivalent fused Q3 SwiGLU path for browsers without subgroups. */
const TWIN_Q3_PORTABLE_GEMV_KERNEL: Qwen35KernelSource = Object.freeze({
  id: "q3-k-fused-f32-192-swiglu-gemv-portable-f32",
  entryPoint: "main",
  source: /* wgsl */ `
struct Params {
  rows: u32,
  columns: u32,
  blocks_per_row: u32,
  pad0: u32,
  pad1: u32,
}
@group(0) @binding(0) var<storage, read> first_weights: array<u32>;
@group(0) @binding(1) var<storage, read> second_weights: array<u32>;
@group(0) @binding(2) var<storage, read> activation: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> first_partials: array<f32, 64>;
var<workgroup> second_partials: array<f32, 64>;
fn packed_byte(values: ptr<storage, array<u32>, read>, base: u32, offset: u32) -> u32 {
  return ((*values)[base + offset / 4u] >> ((offset % 4u) * 8u)) & 255u;
}
fn weight_value(
  values: ptr<storage, array<u32>, read>,
  block_word: u32,
  element: u32,
) -> f32 {
  let scale_index = element / 16u;
  let factor = bitcast<f32>((*values)[block_word + scale_index]);
  let packed = packed_byte(values, block_word, 64u + element / 2u);
  let quant = i32((packed >> ((element & 1u) * 4u)) & 15u) - 4;
  return factor * f32(quant);
}
@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(num_workgroups) grid: vec3<u32>,
) {
  if (group.y > (0xffffffffu - group.x) / grid.x) { return; }
  let row = group.y * grid.x + group.x;
  if (row >= params.rows) { return; }
  var first_sum = 0.0f;
  var second_sum = 0.0f;
  for (var column = local.x; column < params.columns; column += 64u) {
    let block = column / 256u;
    let element = column % 256u;
    let block_word = (row * params.blocks_per_row + block) * 48u;
    let value = activation[column];
    first_sum += weight_value(&first_weights, block_word, element) * value;
    second_sum += weight_value(&second_weights, block_word, element) * value;
  }
  first_partials[local.x] = first_sum;
  second_partials[local.x] = second_sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) {
      first_partials[local.x] += first_partials[local.x + stride];
      second_partials[local.x] += second_partials[local.x + stride];
    }
    workgroupBarrier();
  }
  if (local.x == 0u) {
    let gate = first_partials[0];
    output[row] = (gate / (1.0f + exp(-gate))) * second_partials[0];
  }
}`,
});

/** DeltaNet gate and QKV share one input but use different packed formats. */
const EMBEDDING_KERNEL_SOURCES = new Map<GemvLayout, Qwen35KernelSource>(
  PACKED_EMBEDDING_KERNELS.map((kernel) => [
    kernel.storageType,
    Object.freeze({
      id: kernel.id,
      source: kernel.source,
      entryPoint: "main",
    }),
  ]),
);

function layoutOf(tensor: Qwen35TensorWeightView): GemvLayout {
  const embedding = PACKED_EMBEDDING_KERNELS.find(
    (kernel) => kernel.storageType === tensor.storageType,
  );
  if (embedding === undefined || embedding.ggmlType !== tensor.ggmlType) {
    throw diagnosticError(
      "forward-weight-layout-invalid",
      "Qwen3.5 forward weight type and packed layout do not match",
    );
  }
  return embedding.storageType;
}

function requireTensor(
  weights: Qwen35WeightDirectoryView,
  name: string,
): Qwen35TensorWeightView {
  const tensor = weights.get(name);
  if (tensor === undefined) {
    throw diagnosticError(
      "forward-weight-missing",
      "A required Qwen3.5 forward weight is unavailable",
    );
  }
  return tensor;
}

function matrixShape(tensor: Qwen35TensorWeightView): {
  readonly columns: number;
  readonly rows: number;
} {
  if (
    tensor.shape.length !== 2 ||
    !Number.isSafeInteger(tensor.shape[0]) ||
    tensor.shape[0]! <= 0 ||
    !Number.isSafeInteger(tensor.shape[1]) ||
    tensor.shape[1]! <= 0 ||
    tensor.rowCount !== tensor.shape[1]
  ) {
    throw diagnosticError(
      "forward-weight-shape-invalid",
      "Qwen3.5 forward weight must be a positive packed matrix",
    );
  }
  return { columns: tensor.shape[0]!, rows: tensor.shape[1]! };
}

function requireTiedEmbeddingShape(shape: {
  readonly columns: number;
  readonly rows: number;
}): void {
  if (
    shape.columns !== QWEN35_HIDDEN_SIZE ||
    shape.rows !== QWEN35_VOCABULARY_SIZE
  ) {
    throw diagnosticError(
      "forward-embedding-shape-invalid",
      "The tied Qwen3.5 embedding matrix has an unexpected model shape",
    );
  }
}

function expectedRowBytes(
  tensor: Qwen35TensorWeightView,
  columns: number,
): number {
  const layout = layoutOf(tensor);
  const kernel = LANGUAGE_GEMV_KERNELS.find(
    (candidate) => candidate.layout === layout,
  );
  if (
    kernel === undefined ||
    columns % kernel.abi.valuesPerBlock !== 0
  ) {
    throw diagnosticError(
      "forward-weight-shape-invalid",
      "Qwen3.5 forward matrix row splits a packed block",
    );
  }
  return (columns / kernel.abi.valuesPerBlock) * kernel.abi.bytesPerBlock;
}

function physicalRows(
  tensor: Qwen35TensorWeightView,
  rows: number,
  rowBytes: number,
): readonly Qwen35PhysicalRowView[] {
  if (
    tensor.rowBytes !== rowBytes ||
    tensor.logicalBytes !== BigInt(rows) * BigInt(rowBytes) ||
    tensor.physicalRows.length === 0
  ) {
    throw diagnosticError(
      "forward-weight-views-invalid",
      "Qwen3.5 physical weight rows do not match the packed matrix",
    );
  }
  let nextRow = 0;
  let nextByte = 0;
  const rangesByBuffer = new Map<object, { start: number; end: number }[]>();
  for (const view of tensor.physicalRows) {
    if (
      !Number.isSafeInteger(view.firstRow) ||
      view.firstRow !== nextRow ||
      !Number.isSafeInteger(view.rowCount) ||
      view.rowCount <= 0 ||
      !Number.isSafeInteger(view.tensorByteOffset) ||
      view.tensorByteOffset !== nextByte ||
      !Number.isSafeInteger(view.bufferByteOffset) ||
      view.bufferByteOffset < 0 ||
      !Number.isSafeInteger(view.byteLength) ||
      view.byteLength !== view.rowCount * rowBytes
    ) {
      throw diagnosticError(
        "forward-weight-views-invalid",
        "Qwen3.5 physical weight rows are not contiguous complete rows",
      );
    }
    const range = {
      start: view.bufferByteOffset,
      end: view.bufferByteOffset + view.byteLength,
    };
    const priorRanges = rangesByBuffer.get(view.buffer) ?? [];
    if (
      !Number.isSafeInteger(range.end) ||
      priorRanges.some(
        (prior) => range.start < prior.end && prior.start < range.end,
      )
    ) {
      throw diagnosticError(
        "forward-weight-views-invalid",
        "Qwen3.5 physical weight ranges overlap in one GPU buffer",
      );
    }
    priorRanges.push(range);
    rangesByBuffer.set(view.buffer, priorRanges);
    nextRow += view.rowCount;
    nextByte += view.byteLength;
  }
  if (nextRow !== rows || nextByte !== Number(tensor.logicalBytes)) {
    throw diagnosticError(
      "forward-weight-views-invalid",
      "Qwen3.5 physical weight rows do not cover the packed matrix",
    );
  }
  return tensor.physicalRows;
}

function positiveLimit(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function binding(
  bindingIndex: number,
  kind: "storage" | "uniform",
  slice: Qwen35ForwardBufferSlice,
  requiredBytes: number,
  limits: Qwen35ForwardDeviceLimits,
): Qwen35BufferBinding {
  const alignment = kind === "storage"
    ? limits.minStorageBufferOffsetAlignment
    : limits.minUniformBufferOffsetAlignment;
  const maximum = kind === "storage"
    ? limits.maxStorageBufferBindingSize
    : limits.maxUniformBufferBindingSize;
  if (
    !Number.isSafeInteger(slice.offset) ||
    slice.offset < 0 ||
    !Number.isSafeInteger(slice.byteLength) ||
    slice.byteLength < requiredBytes ||
    !positiveLimit(alignment) ||
    slice.offset % alignment !== 0 ||
    !positiveLimit(maximum) ||
    !Number.isSafeInteger(requiredBytes) ||
    requiredBytes <= 0 ||
    requiredBytes % 4 !== 0 ||
    requiredBytes > maximum ||
    !Number.isSafeInteger(slice.offset + requiredBytes)
  ) {
    throw diagnosticError(
      "forward-binding-invalid",
      "A Qwen3.5 forward buffer binding is invalid for this device",
    );
  }
  return Object.freeze({
    binding: bindingIndex,
    kind,
    buffer: slice.buffer,
    offset: slice.offset,
    size: requiredBytes,
  });
}

function physicalBinding(
  bindingIndex: number,
  view: Qwen35PhysicalRowView,
  limits: Qwen35ForwardDeviceLimits,
): Qwen35BufferBinding {
  return binding(
    bindingIndex,
    "storage",
    {
      buffer: view.buffer,
      offset: view.bufferByteOffset,
      byteLength: view.byteLength,
    },
    view.byteLength,
    limits,
  );
}

function physicalPackedRangeBinding(
  bindingIndex: number,
  view: Qwen35PhysicalRowView,
  relativeByteOffset: number,
  payloadBytes: number,
  limits: Qwen35ForwardDeviceLimits,
): { readonly binding: Qwen35BufferBinding; readonly wordOffset: number } {
  const alignment = limits.minStorageBufferOffsetAlignment;
  if (
    !positiveLimit(alignment) ||
    view.bufferByteOffset % alignment !== 0 ||
    !Number.isSafeInteger(relativeByteOffset) ||
    relativeByteOffset < 0 ||
    !Number.isSafeInteger(payloadBytes) ||
    payloadBytes <= 0 ||
    relativeByteOffset + payloadBytes > view.byteLength
  ) {
    throw diagnosticError(
      "forward-binding-invalid",
      "A packed Qwen3.5 weight range cannot be bound on this device",
    );
  }
  const alignedRelativeOffset =
    Math.floor(relativeByteOffset / alignment) * alignment;
  const prefixBytes = relativeByteOffset - alignedRelativeOffset;
  const requiredBytes = prefixBytes + payloadBytes;
  if (prefixBytes % 4 !== 0 || !Number.isSafeInteger(requiredBytes)) {
    throw diagnosticError(
      "forward-binding-invalid",
      "A packed Qwen3.5 weight range cannot use a word offset",
    );
  }
  return Object.freeze({
    binding: binding(
      bindingIndex,
      "storage",
      {
        buffer: view.buffer,
        offset: view.bufferByteOffset + alignedRelativeOffset,
        byteLength: view.byteLength - alignedRelativeOffset,
      },
      requiredBytes,
      limits,
    ),
    wordOffset: prefixBytes / 4,
  });
}

function bindingsOverlap(
  left: Qwen35BufferBinding,
  right: Qwen35BufferBinding,
): boolean {
  if (left.buffer !== right.buffer) {
    return false;
  }
  return (
    left.offset < right.offset + right.size &&
    right.offset < left.offset + left.size
  );
}

function requireWritableOutputDisjoint(
  output: Qwen35BufferBinding,
  reads: readonly Qwen35BufferBinding[],
): void {
  if (reads.some((read) => bindingsOverlap(output, read))) {
    throw diagnosticError(
      "forward-buffer-alias-invalid",
      "A Qwen3.5 writable output overlaps a read binding",
    );
  }
}

function requireDistinctUniformRanges(
  uniforms: readonly Qwen35BufferBinding[],
): void {
  for (let index = 0; index < uniforms.length; index += 1) {
    for (let prior = 0; prior < index; prior += 1) {
      if (bindingsOverlap(uniforms[index]!, uniforms[prior]!)) {
        throw diagnosticError(
          "forward-uniform-alias-invalid",
          "Batched Qwen3.5 dispatches require distinct uniform ranges",
        );
      }
    }
  }
}

function dispatchPlan(input: {
  readonly kernel: Qwen35KernelSource;
  readonly bindings: readonly Qwen35BufferBinding[];
  readonly uniformWords: readonly number[];
  readonly workgroups: { readonly x: number; readonly y: number; readonly z: number };
}): Qwen35ForwardDispatchPlan {
  return Object.freeze({
    kernel: input.kernel,
    bindings: Object.freeze([...input.bindings]),
    uniformWords: Object.freeze([...input.uniformWords]),
    workgroups: Object.freeze({ ...input.workgroups }),
  });
}

function logitsPieces(
  views: readonly Qwen35PhysicalRowView[],
  vocabularyRows: number,
  rowBytes: number,
  limits: Qwen35ForwardDeviceLimits,
): readonly {
  readonly view: Qwen35PhysicalRowView;
  readonly firstLocalRow: number;
  readonly rowCount: number;
  readonly tileIndex: number;
  readonly vocabularyStart: number;
  readonly tileRows: number;
  readonly pieceOutputOffset: number;
  readonly completesTile: boolean;
}[] {
  const alignment = limits.minStorageBufferOffsetAlignment;
  const maximum = limits.maxStorageBufferBindingSize;
  if (!positiveLimit(alignment) || !positiveLimit(maximum)) {
    throw diagnosticError(
      "forward-binding-invalid",
      "A Qwen3.5 forward buffer binding is invalid for this device",
    );
  }
  const pieces: {
    readonly view: Qwen35PhysicalRowView;
    readonly firstLocalRow: number;
    readonly rowCount: number;
    readonly tileIndex: number;
    readonly vocabularyStart: number;
    readonly tileRows: number;
    readonly pieceOutputOffset: number;
    readonly completesTile: boolean;
  }[] = [];
  let viewIndex = 0;
  for (
    let vocabularyStart = 0, tileIndex = 0;
    vocabularyStart < vocabularyRows;
    vocabularyStart += MAX_LOGITS_TILE_ROWS, tileIndex += 1
  ) {
    const tileRows = Math.min(
      MAX_LOGITS_TILE_ROWS,
      vocabularyRows - vocabularyStart,
    );
    const tileEnd = vocabularyStart + tileRows;
    let pieceStart = vocabularyStart;
    while (pieceStart < tileEnd) {
      while (
        viewIndex < views.length &&
        pieceStart >= views[viewIndex]!.firstRow + views[viewIndex]!.rowCount
      ) {
        viewIndex += 1;
      }
      const view = views[viewIndex];
      if (
        view === undefined ||
        pieceStart < view.firstRow ||
        view.bufferByteOffset % alignment !== 0
      ) {
        throw diagnosticError(
          "forward-weight-views-invalid",
          "A Qwen3.5 logits tile has no valid physical weight range",
        );
      }
      const firstLocalRow = pieceStart - view.firstRow;
      const relativeByteOffset = firstLocalRow * rowBytes;
      const alignedRelativeOffset =
        Math.floor(relativeByteOffset / alignment) * alignment;
      const prefixBytes = relativeByteOffset - alignedRelativeOffset;
      const maximumPayloadBytes = maximum - prefixBytes;
      const maximumPieceRows = Math.floor(maximumPayloadBytes / rowBytes);
      const physicalEnd = Math.min(
        tileEnd,
        view.firstRow + view.rowCount,
      );
      const rowCount = Math.min(
        physicalEnd - pieceStart,
        maximumPieceRows,
      );
      if (rowCount < 1) {
        throw diagnosticError(
          "forward-binding-invalid",
          "A Qwen3.5 logits row exceeds the device storage binding limit",
        );
      }
      const nextPieceStart = pieceStart + rowCount;
      pieces.push(Object.freeze({
        view,
        firstLocalRow,
        rowCount,
        tileIndex,
        vocabularyStart,
        tileRows,
        pieceOutputOffset: pieceStart - vocabularyStart,
        completesTile: nextPieceStart === tileEnd,
      }));
      pieceStart = nextPieceStart;
    }
  }
  return Object.freeze(pieces);
}

function embeddingKernel(
  tensor: Qwen35TensorWeightView,
): { readonly layout: GemvLayout; readonly kernel: PackedEmbeddingKernel; readonly source: Qwen35KernelSource } {
  const layout = layoutOf(tensor);
  const kernel = PACKED_EMBEDDING_KERNELS.find(
    (candidate) => candidate.storageType === layout,
  );
  const source = EMBEDDING_KERNEL_SOURCES.get(layout);
  if (kernel === undefined || source === undefined) {
    throw diagnosticError(
      "forward-weight-layout-invalid",
      "Qwen3.5 packed embedding kernel is unavailable",
    );
  }
  return { layout, kernel, source };
}

/** Plans one direct packed token-row decode from the tied embedding table. */
export function planQwen35PackedEmbeddingDispatch(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly tokenId: number;
  readonly output: Qwen35ForwardBufferSlice;
  readonly uniform: Qwen35ForwardBufferSlice;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35ForwardDispatchPlan {
  const tensor = requireTensor(input.weights, "token_embd.weight");
  const shape = matrixShape(tensor);
  requireTiedEmbeddingShape(shape);
  const selected = embeddingKernel(tensor);
  const rowBytes = expectedRowBytes(tensor, shape.columns);
  const views = physicalRows(tensor, shape.rows, rowBytes);
  let logical: ReturnType<typeof planPackedEmbeddingRow>;
  try {
    logical = planPackedEmbeddingRow({
      ggmlType: selected.kernel.ggmlType,
      storageType: selected.layout,
      tokenId: input.tokenId,
      vocabSize: shape.rows,
      embeddingLength: shape.columns,
      maxComputeWorkgroupsPerDimension:
        input.limits.maxComputeWorkgroupsPerDimension,
    });
  } catch {
    throw diagnosticError(
      "forward-embedding-invalid",
      "Qwen3.5 packed embedding dispatch is invalid",
    );
  }
  const view = views.find(
    (candidate) =>
      input.tokenId >= candidate.firstRow &&
      input.tokenId < candidate.firstRow + candidate.rowCount,
  );
  if (view === undefined) {
    throw diagnosticError(
      "forward-weight-views-invalid",
      "Qwen3.5 token row has no physical weight buffer",
    );
  }
  const relativeByteOffset = logical.packedByteOffset - view.tensorByteOffset;
  if (
    relativeByteOffset < 0 ||
    relativeByteOffset + logical.packedByteLength > view.byteLength ||
    relativeByteOffset % 4 !== 0
  ) {
    throw diagnosticError(
      "forward-weight-views-invalid",
      "Qwen3.5 token row crosses a physical weight buffer",
    );
  }
  const packedRange = physicalPackedRangeBinding(
    selected.kernel.abi.bindings.packedTable,
    view,
    relativeByteOffset,
    logical.packedByteLength,
    input.limits,
  );
  const outputBinding = binding(
    selected.kernel.abi.bindings.output,
    "storage",
    input.output,
    shape.columns * 4,
    input.limits,
  );
  const uniformBinding = binding(
    selected.kernel.abi.bindings.uniforms,
    "uniform",
    input.uniform,
    selected.kernel.abi.uniformWords * 4,
    input.limits,
  );
  requireWritableOutputDisjoint(outputBinding, [
    packedRange.binding,
    uniformBinding,
  ]);
  return dispatchPlan({
    kernel: selected.source,
    bindings: [
      packedRange.binding,
      outputBinding,
      uniformBinding,
    ],
    uniformWords: [
      packedRange.wordOffset,
      logical.outputElements,
      shape.columns / selected.kernel.abi.valuesPerBlock,
      0,
    ],
    workgroups: logical.workgroups,
  });
}

/** Plans one exact Q6_K row already staged in the bounded tied-input cache. */
export function planQwen35StagedPackedEmbeddingDispatch(input: {
  readonly rows: Qwen35StagedPackedRows;
  readonly output: Qwen35ForwardBufferSlice;
  readonly uniform: Qwen35ForwardBufferSlice;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35ForwardDispatchPlan {
  const selected = PACKED_EMBEDDING_KERNELS.find(
    (candidate) => candidate.storageType === "q6-k-fused-f32-256",
  );
  const source = EMBEDDING_KERNEL_SOURCES.get("q6-k-fused-f32-256");
  if (
    selected === undefined ||
    source === undefined ||
    input.rows.tensorName !== "token_embd.weight" ||
    input.rows.storageType !== "q6-k-fused-f32-256" ||
    input.rows.rowCount !== 1 ||
    input.rows.rowBytes !== 2_560 ||
    input.rows.byteLength !== 2_560 ||
    !Number.isSafeInteger(input.rows.firstRow) ||
    input.rows.firstRow < 0 ||
    input.rows.firstRow >= QWEN35_VOCABULARY_SIZE ||
    !Number.isSafeInteger(input.rows.bufferOffset) ||
    input.rows.bufferOffset < 0 ||
    input.rows.bufferOffset % 4 !== 0 ||
    !Number.isSafeInteger(input.rows.bufferOffset + input.rows.byteLength)
  ) {
    throw diagnosticError(
      "forward-staged-embedding-invalid",
      "The staged Qwen3.5 embedding row is invalid",
    );
  }
  let logical: ReturnType<typeof planPackedEmbeddingRow>;
  try {
    logical = planPackedEmbeddingRow({
      ggmlType: selected.ggmlType,
      storageType: selected.storageType,
      tokenId: 0,
      vocabSize: 1,
      embeddingLength: QWEN35_HIDDEN_SIZE,
      maxComputeWorkgroupsPerDimension:
        input.limits.maxComputeWorkgroupsPerDimension,
    });
  } catch {
    throw diagnosticError(
      "forward-staged-embedding-invalid",
      "The staged Qwen3.5 embedding row is invalid",
    );
  }
  const alignment = input.limits.minStorageBufferOffsetAlignment;
  if (!positiveLimit(alignment)) {
    throw diagnosticError(
      "forward-binding-invalid",
      "A packed Qwen3.5 weight range cannot be bound on this device",
    );
  }
  const alignedOffset = Math.floor(input.rows.bufferOffset / alignment) * alignment;
  const prefixBytes = input.rows.bufferOffset - alignedOffset;
  // Cache slots are row-aligned, not WebGPU-binding-aligned. Bind the safe
  // prefix and pass its u32 distance to the existing Q6_K kernel ABI.
  const stagedView: Qwen35PhysicalRowView = Object.freeze({
    buffer: input.rows.buffer,
    firstRow: 0,
    rowCount: 1,
    tensorByteOffset: 0,
    bufferByteOffset: alignedOffset,
    byteLength: prefixBytes + input.rows.rowBytes,
  });
  const packedRange = physicalPackedRangeBinding(
    selected.abi.bindings.packedTable,
    stagedView,
    prefixBytes,
    input.rows.rowBytes,
    input.limits,
  );
  const outputBinding = binding(
    selected.abi.bindings.output,
    "storage",
    input.output,
    QWEN35_HIDDEN_SIZE * 4,
    input.limits,
  );
  const uniformBinding = binding(
    selected.abi.bindings.uniforms,
    "uniform",
    input.uniform,
    selected.abi.uniformWords * 4,
    input.limits,
  );
  requireWritableOutputDisjoint(outputBinding, [
    packedRange.binding,
    uniformBinding,
  ]);
  return dispatchPlan({
    kernel: source,
    bindings: [packedRange.binding, outputBinding, uniformBinding],
    uniformWords: [
      packedRange.wordOffset,
      logical.outputElements,
      QWEN35_HIDDEN_SIZE / selected.abi.valuesPerBlock,
      0,
    ],
    workgroups: logical.workgroups,
  });
}

function gemvKernelForLayout(
  layout: GemvLayout,
  limits: Qwen35ForwardDeviceLimits,
): { readonly layout: GemvLayout; readonly kernel: LanguageGemvKernel; readonly source: Qwen35KernelSource } {
  const preferredProfile = limits.supportsSubgroups === true
    ? "mobile-f16-subgroup"
    : "portable-f32";
  const kernel = LANGUAGE_GEMV_KERNELS.find(
    (candidate) => candidate.layout === layout && candidate.profile === preferredProfile,
  ) ?? LANGUAGE_GEMV_KERNELS.find(
    (candidate) => candidate.layout === layout && candidate.profile === "portable-f32",
  );
  if (kernel === undefined) {
    throw diagnosticError(
      "forward-weight-layout-invalid",
      "Qwen3.5 packed GEMV kernel is unavailable",
    );
  }
  const source = Object.freeze({
    id: kernel.id,
    source: kernel.source,
    entryPoint: "packed_gemv",
  });
  return { layout, kernel, source };
}

function gemvKernel(
  tensor: Qwen35TensorWeightView,
  limits: Qwen35ForwardDeviceLimits,
): ReturnType<typeof gemvKernelForLayout> {
  return gemvKernelForLayout(layoutOf(tensor), limits);
}

/** Copies one projected visual token into the language hidden workspace. */
export function planQwen35VisualEmbeddingDispatch(input: {
  readonly source: Qwen35ForwardBufferSlice;
  readonly output: Qwen35ForwardBufferSlice;
  readonly uniform: Qwen35ForwardBufferSlice;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35ForwardDispatchPlan {
  const elements = QWEN35_HIDDEN_SIZE;
  const sourceBinding = binding(0, "storage", input.source, elements * 4, input.limits);
  const outputBinding = binding(1, "storage", input.output, elements * 4, input.limits);
  const uniformBinding = binding(2, "uniform", input.uniform, 16, input.limits);
  requireWritableOutputDisjoint(outputBinding, [sourceBinding, uniformBinding]);
  return dispatchPlan({
    kernel: VISUAL_EMBEDDING_KERNEL,
    bindings: [sourceBinding, outputBinding, uniformBinding],
    uniformWords: [elements, 0, 0, 0],
    workgroups: { x: Math.ceil(elements / 256), y: 1, z: 1 },
  });
}

function tiedLogitsGeometryData(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly limits: Qwen35ForwardDeviceLimits;
}): {
  readonly shape: { readonly columns: number; readonly rows: number };
  readonly selected: ReturnType<typeof gemvKernel>;
  readonly rowBytes: number;
  readonly pieces: ReturnType<typeof logitsPieces>;
} {
  const tensor = requireTensor(input.weights, "token_embd.weight");
  const shape = matrixShape(tensor);
  requireTiedEmbeddingShape(shape);
  const selected = gemvKernel(tensor, input.limits);
  const rowBytes = expectedRowBytes(tensor, shape.columns);
  const views = physicalRows(tensor, shape.rows, rowBytes);
  const pieces = logitsPieces(
    views,
    QWEN35_DECODABLE_LOGIT_ROWS,
    rowBytes,
    input.limits,
  );
  return { shape, selected, rowBytes, pieces };
}

/** Derives uniform capacity from physical rows before uniform allocation. */
export function planQwen35TiedLogitsGeometry(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35TiedLogitsGeometry {
  const data = tiedLogitsGeometryData(input);
  return Object.freeze({
    modelRows: QWEN35_VOCABULARY_SIZE,
    decodableRows: QWEN35_DECODABLE_LOGIT_ROWS,
    logicalTileRows: MAX_LOGITS_TILE_ROWS,
    mathematicalTileCount: QWEN35_MATHEMATICAL_LOGITS_TILES,
    finalTileRows: QWEN35_FINAL_LOGITS_TILE_ROWS,
    physicalPieceCount: data.pieces.length,
    reductionDispatchCount: QWEN35_LOGITS_REDUCTION_DISPATCHES,
    uniformCount: data.pieces.length + QWEN35_LOGITS_REDUCTION_DISPATCHES,
  });
}

function fusedQ6TiedLogitsViews(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly limits: Qwen35ForwardDeviceLimits;
}): {
  readonly tensor: Qwen35TensorWeightView;
  readonly views: readonly { readonly view: Qwen35PhysicalRowView; readonly rowCount: number }[];
  readonly candidateCount: number;
} | null {
  const tensor = requireTensor(input.weights, "token_embd.weight");
  const shape = matrixShape(tensor);
  requireTiedEmbeddingShape(shape);
  // The specialized top-1 shader uses subgroups. Portable devices retain the
  // exact tiled logits path over the same packed Q6 values.
  if (
    input.limits.supportsSubgroups !== true ||
    layoutOf(tensor) !== "q6-k-fused-f32-256"
  ) return null;
  const rowBytes = expectedRowBytes(tensor, shape.columns);
  const views = physicalRows(tensor, shape.rows, rowBytes)
    .map((view) => Object.freeze({
      view,
      rowCount: Math.max(
        0,
        Math.min(view.rowCount, QWEN35_DECODABLE_LOGIT_ROWS - view.firstRow),
      ),
    }))
    .filter(({ rowCount }) => rowCount > 0);
  const candidateCount = views.reduce(
    (sum, { rowCount }) => sum + Math.ceil(rowCount / FUSED_LOGITS_ROWS_PER_WORKGROUP),
    0,
  );
  if (
    views.length === 0 ||
    candidateCount < QWEN35_MATHEMATICAL_LOGITS_TILES ||
    candidateCount > QWEN35_TOP_K_CANDIDATE_CAPACITY
  ) {
    throw diagnosticError(
      "forward-tied-logits-geometry-invalid",
      "The resident Qwen3.5 tied-logits geometry is invalid",
    );
  }
  return Object.freeze({ tensor, views: Object.freeze(views), candidateCount });
}

export function planQwen35FusedTiedLogitsGeometry(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Readonly<{
  readonly physicalPieceCount: number;
  readonly candidateCount: number;
  readonly uniformCount: number;
}> | null {
  const data = fusedQ6TiedLogitsViews(input);
  if (data === null) return null;
  return Object.freeze({
    physicalPieceCount: data.views.length,
    candidateCount: data.candidateCount,
    uniformCount: data.views.length + 1,
  });
}

/** Packs the final normalized hidden vector into the Q6 lane-major read order. */
export function planQwen35PackTiedLogitsActivationDispatch(input: {
  readonly activation: Qwen35ForwardBufferSlice;
  readonly packedActivation: Qwen35ForwardBufferSlice;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35ForwardDispatchPlan {
  const source = binding(
    0, "storage", input.activation, QWEN35_HIDDEN_SIZE * 4, input.limits,
  );
  const packed = binding(
    1, "storage", input.packedActivation, QWEN35_HIDDEN_SIZE * 2, input.limits,
  );
  requireWritableOutputDisjoint(packed, [source]);
  return dispatchPlan({
    kernel: PACK_TIED_LOGITS_ACTIVATION_KERNEL,
    bindings: [source, packed],
    uniformWords: [],
    workgroups: { x: QWEN35_HIDDEN_SIZE / 512, y: 1, z: 1 },
  });
}

/** Plans one fused top-1 dispatch per resident Q6 physical buffer. */
export function planQwen35FusedTiedLogitsDispatches(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly activation: Qwen35ForwardBufferSlice;
  readonly candidateScores: Qwen35ForwardBufferSlice;
  readonly candidateTokenIds: Qwen35ForwardBufferSlice;
  readonly uniforms: readonly Qwen35ForwardBufferSlice[];
  readonly limits: Qwen35ForwardDeviceLimits;
}): readonly Qwen35FusedTiedLogitsDispatchPlan[] | null {
  const data = fusedQ6TiedLogitsViews(input);
  if (data === null) return null;
  if (input.uniforms.length !== data.views.length) {
    throw diagnosticError(
      "forward-uniform-count-invalid",
      "Qwen3.5 fused tied logits require one uniform slot per physical buffer",
    );
  }
  const activation = binding(1, "storage", input.activation, QWEN35_HIDDEN_SIZE * 2, input.limits);
  const scores = binding(
    2,
    "storage",
    input.candidateScores,
    QWEN35_TOP_K_CANDIDATE_CAPACITY * 4,
    input.limits,
  );
  const tokenIds = binding(
    3,
    "storage",
    input.candidateTokenIds,
    QWEN35_TOP_K_CANDIDATE_CAPACITY * 4,
    input.limits,
  );
  const uniforms = input.uniforms.map((uniform) =>
    binding(4, "uniform", uniform, 24, input.limits)
  );
  requireDistinctUniformRanges(uniforms);
  requireWritableOutputDisjoint(scores, [activation, tokenIds, ...uniforms]);
  requireWritableOutputDisjoint(tokenIds, [activation, scores, ...uniforms]);

  let candidateStart = 0;
  const plans = data.views.map(({ view, rowCount }, index) => {
    const packedRange = physicalPackedRangeBinding(
      0,
      view,
      0,
      rowCount * data.tensor.rowBytes,
      input.limits,
    );
    const candidateCount = Math.ceil(rowCount / FUSED_LOGITS_ROWS_PER_WORKGROUP);
    const start = candidateStart;
    candidateStart += candidateCount;
    requireWritableOutputDisjoint(scores, [packedRange.binding]);
    requireWritableOutputDisjoint(tokenIds, [packedRange.binding]);
    return Object.freeze({
      ...dispatchPlan({
        kernel: FUSED_Q6_TIED_LOGITS_KERNEL,
        bindings: [packedRange.binding, activation, scores, tokenIds, uniforms[index]!],
        uniformWords: [
          rowCount,
          QWEN35_HIDDEN_SIZE,
          QWEN35_HIDDEN_SIZE / 256,
          packedRange.wordOffset,
          view.firstRow,
          start,
        ],
        workgroups: { x: candidateCount, y: 1, z: 1 },
      }),
      candidateStart: start,
      candidateCount,
    });
  });
  if (candidateStart !== data.candidateCount) {
    throw diagnosticError(
      "forward-tied-logits-geometry-invalid",
      "The resident Qwen3.5 tied-logits candidate coverage is invalid",
    );
  }
  return Object.freeze(plans);
}

/** Fuses the two F32 DeltaNet parameter projections over one activation read. */
export function planQwen35TwinF32GemvDispatch(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly firstTensorName: string;
  readonly secondTensorName: string;
  readonly activation: Qwen35ForwardBufferSlice;
  readonly firstOutput: Qwen35ForwardBufferSlice;
  readonly secondOutput: Qwen35ForwardBufferSlice;
  readonly uniform: Qwen35ForwardBufferSlice;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35ForwardDispatchPlan | null {
  const firstTensor = requireTensor(input.weights, input.firstTensorName);
  const secondTensor = requireTensor(input.weights, input.secondTensorName);
  const firstShape = matrixShape(firstTensor);
  const secondShape = matrixShape(secondTensor);
  const firstViews = physicalRows(
    firstTensor,
    firstShape.rows,
    expectedRowBytes(firstTensor, firstShape.columns),
  );
  const secondViews = physicalRows(
    secondTensor,
    secondShape.rows,
    expectedRowBytes(secondTensor, secondShape.columns),
  );
  if (
    firstTensor.ggmlType !== GgmlType.F32 ||
    secondTensor.ggmlType !== GgmlType.F32 ||
    firstTensor.storageType !== "f32" ||
    secondTensor.storageType !== "f32" ||
    firstShape.columns !== secondShape.columns ||
    firstShape.rows !== secondShape.rows ||
    firstShape.columns % 32 !== 0 ||
    firstShape.rows > input.limits.maxComputeWorkgroupsPerDimension ||
    firstViews.length !== 1 ||
    secondViews.length !== 1
  ) {
    return null;
  }
  const firstWeight = physicalBinding(0, firstViews[0]!, input.limits);
  const secondWeight = physicalBinding(1, secondViews[0]!, input.limits);
  const activation = binding(
    2, "storage", input.activation, firstShape.columns * 4, input.limits,
  );
  const firstOutput = binding(
    3, "storage", input.firstOutput, firstShape.rows * 4, input.limits,
  );
  const secondOutput = binding(
    4, "storage", input.secondOutput, secondShape.rows * 4, input.limits,
  );
  const uniform = binding(5, "uniform", input.uniform, 16, input.limits);
  requireWritableOutputDisjoint(firstOutput, [
    firstWeight, secondWeight, activation, secondOutput, uniform,
  ]);
  requireWritableOutputDisjoint(secondOutput, [
    firstWeight, secondWeight, activation, firstOutput, uniform,
  ]);
  return dispatchPlan({
    kernel: TWIN_F32_GEMV_KERNEL,
    bindings: [firstWeight, secondWeight, activation, firstOutput, secondOutput, uniform],
    uniformWords: [firstShape.rows, firstShape.columns, 0, 0],
    workgroups: { x: firstShape.rows, y: 1, z: 1 },
  });
}

/** Plans one packed GEMV request for each row-sharded physical weight buffer. */
export function planQwen35TwinQ3GemvDispatches(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly firstTensorName: string;
  readonly secondTensorName: string;
  readonly activation: Qwen35ForwardBufferSlice;
  readonly packedActivation: Qwen35ForwardBufferSlice;
  readonly output: Qwen35ForwardBufferSlice;
  readonly uniform: Qwen35ForwardBufferSlice;
  readonly limits: Qwen35ForwardDeviceLimits;
}): readonly Qwen35ForwardDispatchPlan[] | null {
  const firstTensor = requireTensor(input.weights, input.firstTensorName);
  const secondTensor = requireTensor(input.weights, input.secondTensorName);
  const firstShape = matrixShape(firstTensor);
  const secondShape = matrixShape(secondTensor);
  const firstRowBytes = expectedRowBytes(firstTensor, firstShape.columns);
  const secondRowBytes = expectedRowBytes(secondTensor, secondShape.columns);
  const firstViews = physicalRows(firstTensor, firstShape.rows, firstRowBytes);
  const secondViews = physicalRows(secondTensor, secondShape.rows, secondRowBytes);
  if (
    firstTensor.storageType !== "q3-k-fused-f32-192" ||
    secondTensor.storageType !== "q3-k-fused-f32-192" ||
    firstTensor.ggmlType !== secondTensor.ggmlType ||
    firstShape.columns !== secondShape.columns ||
    firstShape.rows !== secondShape.rows ||
    firstRowBytes !== secondRowBytes ||
    firstShape.columns % 512 !== 0 ||
    firstViews.length !== 1 ||
    secondViews.length !== 1
  ) {
    return null;
  }
  const maximum = input.limits.maxComputeWorkgroupsPerDimension;
  if (!Number.isSafeInteger(firstShape.rows) || !positiveLimit(maximum)) {
    return null;
  }
  const portable = input.limits.supportsSubgroups !== true;
  const workgroups = Math.ceil(firstShape.rows / (portable ? 1 : 4));
  const x = Math.min(workgroups, maximum);
  const y = Math.ceil(workgroups / x);
  if (y > maximum) return null;
  const sourceActivation = binding(
    0, "storage", input.activation, firstShape.columns * 4, input.limits,
  );
  const packedActivation = binding(
    2,
    "storage",
    input.activation,
    firstShape.columns * 4,
    input.limits,
  );
  const firstWeight = physicalBinding(0, firstViews[0]!, input.limits);
  const secondWeight = physicalBinding(1, secondViews[0]!, input.limits);
  const output = binding(
    3, "storage", input.output, firstShape.rows * 4, input.limits,
  );
  const uniform = binding(4, "uniform", input.uniform, 20, input.limits);
  requireWritableOutputDisjoint(output, [
    firstWeight, secondWeight, packedActivation, uniform,
  ]);
  const fused = dispatchPlan({
    kernel: portable ? TWIN_Q3_PORTABLE_GEMV_KERNEL : TWIN_Q3_GEMV_KERNEL,
    bindings: [firstWeight, secondWeight, packedActivation, output, uniform],
    uniformWords: [
      firstShape.rows,
      firstShape.columns,
      firstShape.columns / 256,
      0,
      0,
    ],
    workgroups: { x, y, z: 1 },
  });
  return Object.freeze([fused]);
}

export function planQwen35PackedGemvDispatches(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly tensorName: string;
  readonly activation: Qwen35ForwardBufferSlice;
  readonly packedActivation?: Qwen35ForwardBufferSlice;
  readonly output: Qwen35ForwardBufferSlice;
  readonly uniforms: readonly Qwen35ForwardBufferSlice[];
  readonly limits: Qwen35ForwardDeviceLimits;
}): readonly Qwen35ForwardDispatchPlan[] {
  if (input.tensorName === "token_embd.weight") {
    throw diagnosticError(
      "forward-tied-logits-required",
      "The tied embedding projection requires bounded logits tiles",
    );
  }
  const tensor = requireTensor(input.weights, input.tensorName);
  const shape = matrixShape(tensor);
  const selected = gemvKernel(tensor, input.limits);
  const rowBytes = expectedRowBytes(tensor, shape.columns);
  const views = physicalRows(tensor, shape.rows, rowBytes);
  if (input.uniforms.length !== views.length) {
    throw diagnosticError(
      "forward-uniform-count-invalid",
      "Qwen3.5 GEMV requires one explicit uniform slot per physical row range",
    );
  }
  const uniformBindings = input.uniforms.map((uniform) => binding(
    selected.kernel.abi.bindings.uniforms,
    "uniform",
    uniform,
    selected.kernel.abi.uniformWords * 4,
    input.limits,
  ));
  requireDistinctUniformRanges(uniformBindings);
  const usesPackedActivation = selected.kernel.profile === "mobile-f16-subgroup" && (
    selected.layout === "q3-k-112" ||
    selected.layout === "q3-k-nibble-148" ||
    selected.layout === "q4-k-144" ||
    selected.layout === "q5-k-176" ||
    selected.layout === "q6-k-212" ||
    selected.layout === "q5-k-fused-f32-224"
  );
  if (usesPackedActivation && shape.columns % 512 !== 0) {
    throw diagnosticError(
      "forward-gemv-packed-activation-invalid",
      "Qwen3.5 packed activation width must be a complete conversion workgroup",
    );
  }
  const sourceActivationBinding = binding(
    0,
    "storage",
    input.activation,
    shape.columns * 4,
    input.limits,
  );
  const activationBinding = binding(
    selected.kernel.abi.bindings.activation,
    "storage",
    usesPackedActivation
      ? input.packedActivation ?? (() => {
          throw diagnosticError(
            "forward-gemv-packed-activation-missing",
            "Qwen3.5 browser Q3 GEMV requires its bounded FP16 activation scratch",
          );
        })()
      : input.activation,
    shape.columns * (usesPackedActivation ? 2 : 4),
    input.limits,
  );
  const outputBinding = binding(
    selected.kernel.abi.bindings.output,
    "storage",
    input.output,
    shape.rows * 4,
    input.limits,
  );
  const plans = views.map((view, index) => {
    let logical: ReturnType<typeof planGemvDispatch>;
    try {
      logical = planGemvDispatch({
        layout: selected.layout,
        profile: selected.kernel.profile,
        ggmlType: selected.kernel.ggmlType,
        localRows: view.rowCount,
        columns: shape.columns,
        outputRowOffset: view.firstRow,
        maxWorkgroupsPerDimension:
          input.limits.maxComputeWorkgroupsPerDimension,
      });
    } catch {
      throw diagnosticError(
        "forward-gemv-invalid",
        "Qwen3.5 packed GEMV dispatch is invalid",
      );
    }
    const weightBinding = physicalBinding(
      selected.kernel.abi.bindings.packedWeights,
      view,
      input.limits,
    );
    const uniformBinding = uniformBindings[index]!;
    requireWritableOutputDisjoint(outputBinding, [
      weightBinding,
      activationBinding,
      uniformBinding,
    ]);
    return dispatchPlan({
      kernel: selected.source,
      bindings: [
        weightBinding,
        activationBinding,
        outputBinding,
        uniformBinding,
      ],
      uniformWords: [
        logical.uniforms.localRows,
        logical.uniforms.columns,
        logical.uniforms.blocksPerRow,
        0,
        logical.uniforms.outputRowOffset,
      ],
      workgroups: logical.workgroups,
    });
  });
  if (!usesPackedActivation) return Object.freeze(plans);
  const packedOutputBinding = binding(
    1,
    "storage",
    input.packedActivation!,
    shape.columns * 2,
    input.limits,
  );
  requireWritableOutputDisjoint(packedOutputBinding, [sourceActivationBinding]);
  const conversion = dispatchPlan({
    kernel: selected.layout === "q5-k-fused-f32-224"
      ? PACK_LANE_F16_ACTIVATION_KERNEL
      : PACK_F16_ACTIVATION_KERNEL,
    bindings: [sourceActivationBinding, packedOutputBinding],
    uniformWords: [],
    workgroups: { x: shape.columns / 512, y: 1, z: 1 },
  });
  return Object.freeze([conversion, ...plans]);
}

/** Logits reuse token_embd.weight; the package has no second output matrix. */
export function planQwen35TiedLogitsDispatches(input: Omit<
  Parameters<typeof planQwen35PackedGemvDispatches>[0],
  "tensorName"
>): readonly Qwen35TiedLogitsDispatchPlan[] {
  const { shape, selected, rowBytes, pieces } = tiedLogitsGeometryData(input);
  if (input.uniforms.length !== pieces.length) {
    throw diagnosticError(
      "forward-uniform-count-invalid",
      "Qwen3.5 tied logits require one explicit uniform slot per piece",
    );
  }
  const uniformBindings = input.uniforms.map((uniform) => binding(
    selected.kernel.abi.bindings.uniforms,
    "uniform",
    uniform,
    selected.kernel.abi.uniformWords * 4,
    input.limits,
  ));
  requireDistinctUniformRanges(uniformBindings);
  const activationBinding = binding(
    selected.kernel.abi.bindings.activation,
    "storage",
    input.activation,
    shape.columns * 4,
    input.limits,
  );
  const plans = pieces.map((piece, index) => {
    let logical: ReturnType<typeof planGemvDispatch>;
    try {
      logical = planGemvDispatch({
        layout: selected.layout,
        profile: selected.kernel.profile,
        ggmlType: selected.kernel.ggmlType,
        localRows: piece.rowCount,
        columns: shape.columns,
        outputRowOffset: piece.pieceOutputOffset,
        maxWorkgroupsPerDimension:
          input.limits.maxComputeWorkgroupsPerDimension,
      });
    } catch {
      throw diagnosticError(
        "forward-gemv-invalid",
        "Qwen3.5 tied-logits GEMV dispatch is invalid",
      );
    }
    const outputBinding = binding(
      selected.kernel.abi.bindings.output,
      "storage",
      input.output,
      piece.tileRows * 4,
      input.limits,
    );
    const relativeByteOffset = piece.firstLocalRow * rowBytes;
    const packedRange = physicalPackedRangeBinding(
      selected.kernel.abi.bindings.packedWeights,
      piece.view,
      relativeByteOffset,
      piece.rowCount * rowBytes,
      input.limits,
    );
    const uniformBinding = uniformBindings[index]!;
    requireWritableOutputDisjoint(outputBinding, [
      packedRange.binding,
      activationBinding,
      uniformBinding,
    ]);
    const base = dispatchPlan({
      kernel: selected.source,
      bindings: [
        packedRange.binding,
        activationBinding,
        outputBinding,
        uniformBinding,
      ],
      uniformWords: [
        logical.uniforms.localRows,
        logical.uniforms.columns,
        logical.uniforms.blocksPerRow,
        packedRange.wordOffset,
        logical.uniforms.outputRowOffset,
      ],
      workgroups: logical.workgroups,
    });
    return Object.freeze({
      ...base,
      tileIndex: piece.tileIndex,
      vocabularyStart: piece.vocabularyStart,
      tileRows: piece.tileRows,
      pieceOutputOffset: piece.pieceOutputOffset,
      pieceRows: piece.rowCount,
      completesTile: piece.completesTile,
    });
  });
  return Object.freeze(plans);
}

/** Plans one complete logical logits tile from the bounded Q6_K output cache. */
export function planQwen35StagedTiedLogitsDispatch(input: {
  readonly tile: Qwen35StagedPackedRows;
  readonly activation: Qwen35ForwardBufferSlice;
  readonly output: Qwen35ForwardBufferSlice;
  readonly uniform: Qwen35ForwardBufferSlice;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35TiedLogitsDispatchPlan {
  const selection = gemvKernelForLayout("q6-k-fused-f32-256", input.limits);
  const selected = selection.kernel;
  const source = selection.source;
  const tileIndex = input.tile.firstRow / MAX_LOGITS_TILE_ROWS;
  const expectedRows = Math.min(
    MAX_LOGITS_TILE_ROWS,
    QWEN35_DECODABLE_LOGIT_ROWS - input.tile.firstRow,
  );
  if (
    input.tile.tensorName !== "token_embd.weight" ||
    input.tile.storageType !== "q6-k-fused-f32-256" ||
    input.tile.rowBytes !== 2_560 ||
    !Number.isSafeInteger(input.tile.firstRow) ||
    input.tile.firstRow < 0 ||
    input.tile.firstRow >= QWEN35_DECODABLE_LOGIT_ROWS ||
    input.tile.firstRow % MAX_LOGITS_TILE_ROWS !== 0 ||
    !Number.isSafeInteger(tileIndex) ||
    input.tile.rowCount !== expectedRows ||
    input.tile.byteLength !== input.tile.rowCount * input.tile.rowBytes ||
    !Number.isSafeInteger(input.tile.bufferOffset) ||
    input.tile.bufferOffset < 0 ||
    input.tile.bufferOffset % 4 !== 0 ||
    !Number.isSafeInteger(input.tile.bufferOffset + input.tile.byteLength)
  ) {
    throw diagnosticError(
      "forward-staged-logits-invalid",
      "The staged Qwen3.5 logits tile is invalid",
    );
  }
  let logical: ReturnType<typeof planGemvDispatch>;
  try {
    logical = planGemvDispatch({
      layout: selected.layout,
      profile: selected.profile,
      ggmlType: selected.ggmlType,
      localRows: input.tile.rowCount,
      columns: QWEN35_HIDDEN_SIZE,
      outputRowOffset: 0,
      maxWorkgroupsPerDimension:
        input.limits.maxComputeWorkgroupsPerDimension,
    });
  } catch {
    throw diagnosticError(
      "forward-staged-logits-invalid",
      "The staged Qwen3.5 logits tile is invalid",
    );
  }
  const alignment = input.limits.minStorageBufferOffsetAlignment;
  if (!positiveLimit(alignment)) {
    throw diagnosticError(
      "forward-binding-invalid",
      "A packed Qwen3.5 weight range cannot be bound on this device",
    );
  }
  const alignedOffset = Math.floor(input.tile.bufferOffset / alignment) * alignment;
  const prefixBytes = input.tile.bufferOffset - alignedOffset;
  const stagedView: Qwen35PhysicalRowView = Object.freeze({
    buffer: input.tile.buffer,
    firstRow: 0,
    rowCount: input.tile.rowCount,
    tensorByteOffset: 0,
    bufferByteOffset: alignedOffset,
    byteLength: prefixBytes + input.tile.byteLength,
  });
  const packedRange = physicalPackedRangeBinding(
    selected.abi.bindings.packedWeights,
    stagedView,
    prefixBytes,
    input.tile.byteLength,
    input.limits,
  );
  const activationBinding = binding(
    selected.abi.bindings.activation,
    "storage",
    input.activation,
    QWEN35_HIDDEN_SIZE * 4,
    input.limits,
  );
  const outputBinding = binding(
    selected.abi.bindings.output,
    "storage",
    input.output,
    input.tile.rowCount * 4,
    input.limits,
  );
  const uniformBinding = binding(
    selected.abi.bindings.uniforms,
    "uniform",
    input.uniform,
    selected.abi.uniformWords * 4,
    input.limits,
  );
  requireWritableOutputDisjoint(outputBinding, [
    packedRange.binding,
    activationBinding,
    uniformBinding,
  ]);
  const base = dispatchPlan({
    kernel: source,
    bindings: [
      packedRange.binding,
      activationBinding,
      outputBinding,
      uniformBinding,
    ],
    uniformWords: [
      logical.uniforms.localRows,
      logical.uniforms.columns,
      logical.uniforms.blocksPerRow,
      packedRange.wordOffset,
      0,
    ],
    workgroups: logical.workgroups,
  });
  return Object.freeze({
    ...base,
    tileIndex,
    vocabularyStart: input.tile.firstRow,
    tileRows: input.tile.rowCount,
    pieceOutputOffset: 0,
    pieceRows: input.tile.rowCount,
    completesTile: true,
  });
}
