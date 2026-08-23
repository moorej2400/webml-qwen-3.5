import type { KernelDefinition } from "./kernel-registry.js";

export type QwenPrimitiveOperation =
  | "rms-norm"
  | "residual-add"
  | "residual-rms-norm"
  | "silu"
  | "swiglu"
  | "attention-output-gate"
  | "qk-rms-norm"
  | "partial-mrope"
  | "top-k-merge";

export interface QwenPrimitiveAbi {
  readonly operation: QwenPrimitiveOperation;
  readonly layout: "f32";
  readonly phase: "shared";
  readonly profile: "portable-f32";
  readonly workgroupSize: 1 | 256;
  readonly bindings: Readonly<Record<string, number>>;
  readonly uniformWords: number;
  readonly outputCoverage?: "all-elements";
  readonly coordinateCount?: 3;
}

export interface QwenPrimitiveKernel {
  readonly id: string;
  readonly operation: QwenPrimitiveOperation;
  readonly layout: "f32";
  readonly phase: "shared";
  readonly profile: "portable-f32";
  readonly abi: QwenPrimitiveAbi;
  readonly source: string;
}

function sameLength(
  left: ArrayLike<number>,
  right: ArrayLike<number>,
  label: string,
): void {
  if (left.length !== right.length) {
    throw new Error(`${label} shape mismatch`);
  }
}

function requireEpsilon(epsilon: number): void {
  if (!Number.isFinite(epsilon) || epsilon < 0) {
    throw new Error("RMSNorm epsilon must be finite and non-negative");
  }
}

function inverseRms(
  input: Float32Array,
  start: number,
  length: number,
  epsilon: number,
): number {
  // fround on every term/add models shader f32 accumulation instead of hiding
  // cancellation or overflow behind JavaScript's f64 arithmetic.
  let sum = Math.fround(0);
  for (let index = 0; index < length; index += 1) {
    const value = input[start + index]!;
    sum = Math.fround(sum + Math.fround(value * value));
  }
  return Math.fround(1 / Math.sqrt(Math.fround(sum / length + epsilon)));
}

export function rmsNormCpu(
  input: Float32Array,
  weight: Float32Array,
  epsilon: number,
): Float32Array {
  sameLength(input, weight, "RMSNorm input and weight");
  requireEpsilon(epsilon);
  if (input.length === 0) {
    throw new Error("RMSNorm input must not be empty");
  }
  const scale = inverseRms(input, 0, input.length, epsilon);
  return Float32Array.from(
    input,
    (value, index) => Math.fround(Math.fround(value * scale) * weight[index]!),
  );
}

export function residualAddCpu(
  input: Float32Array,
  residual: Float32Array,
): Float32Array {
  sameLength(input, residual, "Residual add");
  return Float32Array.from(input, (value, index) =>
    Math.fround(value + residual[index]!),
  );
}

function sigmoid(value: number): number {
  if (value >= 0) {
    return 1 / (1 + Math.exp(-value));
  }
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

export function siluCpu(value: number): number {
  return Math.fround(value * sigmoid(value));
}

export function fusedSwiGluCpu(
  gate: Float32Array,
  up: Float32Array,
): Float32Array {
  sameLength(gate, up, "SwiGLU gate and up");
  return Float32Array.from(gate, (value, index) =>
    Math.fround(siluCpu(value) * up[index]!),
  );
}

export function attentionOutputGateCpu(
  attention: Float32Array,
  gate: Float32Array,
): Float32Array {
  sameLength(attention, gate, "Attention output gate");
  return Float32Array.from(attention, (value, index) =>
    Math.fround(value * sigmoid(gate[index]!)),
  );
}

export function qkRmsNormPerHeadCpu(
  input: Float32Array,
  weight: Float32Array,
  shape: {
    readonly headCount: number;
    readonly headDimension: number;
    readonly epsilon: number;
  },
): Float32Array {
  if (
    !Number.isSafeInteger(shape.headCount) ||
    shape.headCount < 1 ||
    !Number.isSafeInteger(shape.headDimension) ||
    shape.headDimension < 1 ||
    input.length !== shape.headCount * shape.headDimension ||
    weight.length !== shape.headDimension
  ) {
    throw new Error("Q/K RMSNorm per-head shape mismatch");
  }
  requireEpsilon(shape.epsilon);
  const output = new Float32Array(input.length);
  for (let head = 0; head < shape.headCount; head += 1) {
    const start = head * shape.headDimension;
    const scale = inverseRms(input, start, shape.headDimension, shape.epsilon);
    for (let lane = 0; lane < shape.headDimension; lane += 1) {
      output[start + lane] = Math.fround(
        Math.fround(input[start + lane]! * scale) * weight[lane]!,
      );
    }
  }
  return output;
}

export interface PartialMropeOptions {
  readonly headCount: number;
  readonly headDimension: number;
  readonly rotaryDimension: number;
  /** Frequency counts assigned to the temporal, height, and width positions. */
  readonly sections: readonly [number, number, number];
  readonly positions: readonly [number, number, number];
  readonly theta: number;
}

function requireMropeShape(
  input: Float32Array,
  options: PartialMropeOptions,
): void {
  if (
    !Number.isSafeInteger(options.headCount) ||
    options.headCount < 1 ||
    !Number.isSafeInteger(options.headDimension) ||
    options.headDimension < 1 ||
    input.length !== options.headCount * options.headDimension ||
    !Number.isSafeInteger(options.rotaryDimension) ||
    options.rotaryDimension < 2 ||
    options.rotaryDimension > options.headDimension ||
    options.rotaryDimension % 2 !== 0
  ) {
    throw new Error("M-RoPE input or rotary shape is invalid");
  }
  if (
    options.sections.some((section) => !Number.isSafeInteger(section) || section < 0) ||
    options.sections.reduce((sum, section) => sum + section, 0) * 2 !==
      options.rotaryDimension
  ) {
    throw new Error("M-RoPE sections must cover the rotary dimension");
  }
  if (
    options.positions.some((position) => !Number.isSafeInteger(position) || position < 0) ||
    !Number.isFinite(options.theta) ||
    options.theta <= 0
  ) {
    throw new Error("M-RoPE positions and theta must be non-negative and finite");
  }
}

export function mropeFrequencyOwners(
  sections: readonly [number, number, number],
): readonly number[] {
  if (
    sections.some(
      (section) => !Number.isSafeInteger(section) || section < 0,
    )
  ) {
    throw new Error("M-RoPE sections must be non-negative safe integers");
  }
  const frequencyCount = sections.reduce((sum, section) => sum + section, 0);
  const owners = new Array<number>(frequencyCount).fill(0);
  for (let coordinate = 1; coordinate < 3; coordinate += 1) {
    for (
      let frequency = coordinate;
      frequency < sections[coordinate]! * 3 && frequency < frequencyCount;
      frequency += 3
    ) {
      owners[frequency] = coordinate;
    }
  }
  return Object.freeze(owners);
}

/**
 * Applies Qwen3.5's interleaved temporal-height-width frequency schedule and
 * split-half rotate_half layout. Lanes outside rotaryDimension are copied.
 */
export function partialMropeCpu(
  input: Float32Array,
  options: PartialMropeOptions,
): Float32Array {
  requireMropeShape(input, options);
  const output = new Float32Array(input);
  const frequencyCount = options.rotaryDimension / 2;
  const owners = mropeFrequencyOwners(options.sections);
  for (let head = 0; head < options.headCount; head += 1) {
    const base = head * options.headDimension;
    for (let frequency = 0; frequency < frequencyCount; frequency += 1) {
      const position = options.positions[owners[frequency]!]!;
      const exponent = Math.fround(
        Math.fround(2 * frequency) /
          Math.fround(options.rotaryDimension),
      );
      const divisor = Math.fround(
        Math.pow(Math.fround(options.theta), exponent),
      );
      const angle = Math.fround(Math.fround(position) / divisor);
      const tau = Math.fround(2 * Math.PI);
      // Explicit f32 range reduction avoids device-specific large-angle trig
      // reduction from breaking CPU/WebGPU parity at the context boundary.
      const turns = Math.floor(Math.fround(angle / tau));
      const reducedAngle = Math.fround(
        angle - Math.fround(Math.fround(turns) * tau),
      );
      const cosine = Math.fround(Math.cos(reducedAngle));
      const sine = Math.fround(Math.sin(reducedAngle));
      const firstIndex = base + frequency;
      const secondIndex = firstIndex + frequencyCount;
      const first = input[firstIndex]!;
      const second = input[secondIndex]!;
      output[firstIndex] = Math.fround(
        Math.fround(first * cosine) - Math.fround(second * sine),
      );
      output[secondIndex] = Math.fround(
        Math.fround(first * sine) + Math.fround(second * cosine),
      );
    }
  }
  return output;
}

export interface TopKTile {
  readonly startIndex: number;
  readonly scores: Float32Array;
}

export interface TopKEntry {
  readonly index: number;
  readonly score: number;
}

/**
 * Merges finite tile candidates by score and then vocabulary index. Rejecting
 * non-finite scores prevents NaN comparison order from changing the result.
 */
export function stableTiledTopK(
  tiles: readonly TopKTile[],
  k: number,
): readonly TopKEntry[] {
  if (!Number.isSafeInteger(k) || k < 1) {
    throw new Error("Top-k count must be a positive safe integer");
  }
  const candidates: TopKEntry[] = [];
  for (const tile of tiles) {
    if (!Number.isSafeInteger(tile.startIndex) || tile.startIndex < 0) {
      throw new Error("Top-k tile start index must be non-negative");
    }
    for (let offset = 0; offset < tile.scores.length; offset += 1) {
      const score = tile.scores[offset]!;
      if (Number.isFinite(score)) {
        candidates.push(
          Object.freeze({ index: tile.startIndex + offset, score }),
        );
      }
    }
  }
  candidates.sort(
    (left, right) => right.score - left.score || left.index - right.index,
  );
  return Object.freeze(candidates.slice(0, k));
}

const RMS_NORM_WGSL = /* wgsl */ `
struct Params { element_count: u32, width: u32, epsilon: f32, pad: u32 }
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> partials: array<f32, 256>;
@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let base = group.x * params.width;
  var sum: f32 = 0.0f;
  for (var lane = local.x; lane < params.width; lane += 256u) {
    let value = input_values[base + lane];
    sum += value * value;
  }
  partials[local.x] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partials[local.x] += partials[local.x + stride]; }
    workgroupBarrier();
  }
  let inverse_rms = inverseSqrt(partials[0] / f32(params.width) + params.epsilon);
  for (var lane = local.x; lane < params.width; lane += 256u) {
    let index = base + lane;
    // GGUF conversion makes normal Qwen3.5 norm weights multiplicative.
    output_values[index] = input_values[index] * inverse_rms * weights[lane];
  }
}`;

const RESIDUAL_RMS_NORM_WGSL = /* wgsl */ `
struct Params { element_count: u32, width: u32, epsilon: f32, pad: u32 }
@group(0) @binding(0) var<storage, read_write> normalized_values: array<f32>;
@group(0) @binding(1) var<storage, read> residual_values: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> residual_output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> partials: array<f32, 256>;
@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let base = group.x * params.width;
  var sum = 0.0f;
  for (var lane = local.x; lane < params.width; lane += 256u) {
    let index = base + lane;
    let combined = normalized_values[index] + residual_values[index];
    residual_output[index] = combined;
    sum += combined * combined;
  }
  partials[local.x] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partials[local.x] += partials[local.x + stride]; }
    workgroupBarrier();
  }
  let inverse_rms = inverseSqrt(partials[0] / f32(params.width) + params.epsilon);
  for (var lane = local.x; lane < params.width; lane += 256u) {
    let index = base + lane;
    normalized_values[index] = residual_output[index] * inverse_rms * weights[lane];
  }
}`;

const RESIDUAL_WGSL = /* wgsl */ `
struct Params { element_count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> residual: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  if (invocation.x < params.element_count) {
    output_values[invocation.x] = input_values[invocation.x] + residual[invocation.x];
  }
}`;

const SILU_WGSL = /* wgsl */ `
struct Params { element_count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
fn sigmoid(value: f32) -> f32 { return 1.0f / (1.0f + exp(-value)); }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  if (invocation.x < params.element_count) {
    let value = input_values[invocation.x];
    output_values[invocation.x] = value * sigmoid(value);
  }
}`;

const SWIGLU_WGSL = /* wgsl */ `
struct Params { element_count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  if (invocation.x < params.element_count) {
    let value = gate[invocation.x];
    output_values[invocation.x] = value / (1.0f + exp(-value)) * up[invocation.x];
  }
}`;

const ATTENTION_GATE_WGSL = /* wgsl */ `
struct Params { element_count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> attention: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  if (invocation.x < params.element_count) {
    output_values[invocation.x] =
      attention[invocation.x] / (1.0f + exp(-gate[invocation.x]));
  }
}`;

const QK_NORM_WGSL = /* wgsl */ `
struct Params { element_count: u32, head_dimension: u32, epsilon: f32, pad: u32 }
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= params.element_count) { return; }
  let base = (index / params.head_dimension) * params.head_dimension;
  var sum: f32 = 0.0f;
  for (var lane = 0u; lane < params.head_dimension; lane += 1u) {
    let value = input_values[base + lane];
    sum += value * value;
  }
  let inverse_rms =
    inverseSqrt(sum / f32(params.head_dimension) + params.epsilon);
  output_values[index] =
    input_values[index] * inverse_rms * weights[index % params.head_dimension];
}`;

const MROPE_WGSL = /* wgsl */ `
struct Params {
  element_count: u32,
  head_dimension: u32,
  rotary_dimension: u32,
  frequency_count: u32,
  theta: f32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(2) var<storage, read> sections: array<u32>;
@group(0) @binding(3) var<storage, read> positions: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
fn frequency_owner(frequency: u32) -> u32 {
  let coordinate = frequency % 3u;
  if (coordinate > 0u && frequency < sections[coordinate] * 3u) {
    return coordinate;
  }
  return 0u;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= params.element_count) { return; }
  let lane = index % params.head_dimension;
  if (lane >= params.rotary_dimension) {
    output_values[index] = input_values[index];
    return;
  }
  let half = params.frequency_count;
  let frequency = lane % half;
  let head_base = index - lane;
  let partner_index = select(
    head_base + lane - half,
    head_base + lane + half,
    lane < half,
  );
  let angle = f32(positions[frequency_owner(frequency)]) /
    pow(params.theta, f32(frequency * 2u) / f32(params.rotary_dimension));
  let tau = 6.28318548f;
  let reduced_angle = angle - floor(angle / tau) * tau;
  let cosine = cos(reduced_angle);
  let sine = sin(reduced_angle);
  let value = input_values[index];
  let partner = input_values[partner_index];
  output_values[index] = select(
    value * cosine + partner * sine,
    value * cosine - partner * sine,
    lane < half,
  );
}`;

const TOP_K_WGSL = /* wgsl */ `
struct Params { candidate_count: u32, k: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_valid_count: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  if (invocation.x != 0u) { return; }
  var valid_count = 0u;
  for (var slot = 0u; slot < params.k; slot += 1u) {
    var found = false;
    var best_score = 0.0f;
    var best_index = 0u;
    for (var index = 0u; index < params.candidate_count; index += 1u) {
      let score = scores[index];
      let exponent = bitcast<u32>(score) & 0x7f800000u;
      if (exponent == 0x7f800000u) { continue; }
      var used = false;
      for (var prior = 0u; prior < slot; prior += 1u) {
        used = used || output_indices[prior] == index;
      }
      if (!used && (!found || score > best_score ||
          (score == best_score && index < best_index))) {
        found = true;
        best_score = score;
        best_index = index;
      }
    }
    if (!found) { break; }
    output_scores[slot] = best_score;
    output_indices[slot] = best_index;
    valid_count += 1u;
  }
  output_valid_count[0] = valid_count;
}`;

function primitive(
  operation: QwenPrimitiveOperation,
  source: string,
  workgroupSize: 1 | 256,
  bindings: Readonly<Record<string, number>>,
  uniformWords: number,
  extras: Pick<QwenPrimitiveAbi, "outputCoverage" | "coordinateCount"> = {},
): QwenPrimitiveKernel {
  return Object.freeze({
    id: `${operation}-shared-portable-f32`,
    operation,
    layout: "f32",
    phase: "shared",
    profile: "portable-f32",
    source,
    abi: Object.freeze({
      operation,
      layout: "f32",
      phase: "shared",
      profile: "portable-f32",
      workgroupSize,
      bindings: Object.freeze({ ...bindings }),
      uniformWords,
      ...extras,
    }),
  });
}

export const QWEN_PRIMITIVE_KERNELS: readonly QwenPrimitiveKernel[] =
  Object.freeze([
    primitive("rms-norm", RMS_NORM_WGSL, 256, { input: 0, weight: 1, output: 2, uniforms: 3 }, 4),
    primitive("residual-add", RESIDUAL_WGSL, 256, { input: 0, residual: 1, output: 2, uniforms: 3 }, 4),
    primitive(
      "residual-rms-norm",
      RESIDUAL_RMS_NORM_WGSL,
      256,
      {
        inputOutput: 0,
        residual: 1,
        weight: 2,
        residualOutput: 3,
        uniforms: 4,
      },
      4,
    ),
    primitive("silu", SILU_WGSL, 256, { input: 0, output: 1, uniforms: 2 }, 4),
    primitive("swiglu", SWIGLU_WGSL, 256, { gate: 0, up: 1, output: 2, uniforms: 3 }, 4),
    primitive("attention-output-gate", ATTENTION_GATE_WGSL, 256, { attention: 0, gate: 1, output: 2, uniforms: 3 }, 4),
    primitive("qk-rms-norm", QK_NORM_WGSL, 256, { input: 0, weight: 1, output: 2, uniforms: 3 }, 4),
    primitive(
      "partial-mrope",
      MROPE_WGSL,
      256,
      { input: 0, output: 1, sections: 2, positions: 3, uniforms: 4 },
      8,
      { outputCoverage: "all-elements", coordinateCount: 3 },
    ),
    primitive(
      "top-k-merge",
      TOP_K_WGSL,
      1,
      {
        scores: 0,
        outputScores: 1,
        outputIndices: 2,
        outputValidCount: 3,
        uniforms: 4,
      },
      4,
    ),
  ]);

export interface QwenPrimitiveRegistryDefinition extends KernelDefinition {
  readonly abi: Omit<QwenPrimitiveAbi, "phase"> & {
    readonly phase: "prefill" | "decode";
  };
}

export function qwenPrimitiveRegistryDefinitions(input: {
  readonly phase: "prefill" | "decode";
  readonly profile: "portable-f32";
}): readonly QwenPrimitiveRegistryDefinition[] {
  if (input.phase !== "prefill" && input.phase !== "decode") {
    throw new Error("Unsupported Qwen primitive phase");
  }
  if (input.profile !== "portable-f32") {
    throw new Error("Unsupported Qwen primitive profile");
  }
  return Object.freeze(
    QWEN_PRIMITIVE_KERNELS.map((kernel) =>
      Object.freeze({
        id: `${kernel.operation}-${input.phase}-${input.profile}`,
        key: Object.freeze({
          operation: kernel.operation,
          layout: kernel.layout,
          phase: input.phase,
          profile: input.profile,
        }),
        source: kernel.source,
        abi: Object.freeze({ ...kernel.abi, phase: input.phase }),
      }),
    ),
  );
}

export interface PrimitiveDispatchPlan {
  readonly operation: QwenPrimitiveOperation;
  readonly workgroups: { readonly x: number; readonly y: 1; readonly z: 1 };
  readonly elementCount: number;
  readonly headDimension?: number;
  readonly rotaryDimension?: number;
}

export function planPrimitiveDispatch(input: {
  readonly operation: QwenPrimitiveOperation;
  readonly elementCount: number;
  readonly width?: number;
  readonly headDimension?: number;
  readonly rotaryDimension?: number;
}): PrimitiveDispatchPlan {
  if (
    !Number.isSafeInteger(input.elementCount) ||
    input.elementCount < 1 ||
    input.elementCount > 0xffff_ffff
  ) {
    throw new Error("Primitive element count must fit a positive u32");
  }
  const kernel = QWEN_PRIMITIVE_KERNELS.find(
    (candidate) => candidate.operation === input.operation,
  );
  if (kernel === undefined) {
    throw new Error("Unsupported Qwen primitive operation");
  }
  if (input.operation === "qk-rms-norm") {
    if (
      input.headDimension === undefined ||
      !Number.isSafeInteger(input.headDimension) ||
      input.headDimension < 1 ||
      input.elementCount % input.headDimension !== 0
    ) {
      throw new Error("Q/K RMSNorm element count must contain complete heads");
    }
  }
  if (
    (input.operation === "rms-norm" ||
      input.operation === "residual-rms-norm") &&
    (input.width === undefined ||
      !Number.isSafeInteger(input.width) ||
      input.width < 1 ||
      input.elementCount % input.width !== 0)
  ) {
    throw new Error("RMSNorm element count must contain complete width rows");
  }
  if (input.operation === "partial-mrope") {
    if (
      input.headDimension === undefined ||
      input.rotaryDimension === undefined ||
      !Number.isSafeInteger(input.headDimension) ||
      input.headDimension < 1 ||
      input.elementCount % input.headDimension !== 0 ||
      !Number.isSafeInteger(input.rotaryDimension) ||
      input.rotaryDimension < 2 ||
      input.rotaryDimension > input.headDimension ||
      input.rotaryDimension % 2 !== 0
    ) {
      throw new Error(
        "M-RoPE dispatch must contain complete heads and an even rotary dimension",
      );
    }
  }
  const x =
    input.operation === "top-k-merge"
      ? 1
      : input.operation === "rms-norm" ||
          input.operation === "residual-rms-norm"
        ? input.elementCount / input.width!
      : Math.ceil(input.elementCount / kernel.abi.workgroupSize);
  return Object.freeze({
    operation: input.operation,
    workgroups: Object.freeze({ x, y: 1, z: 1 }),
    elementCount: input.elementCount,
    ...(input.headDimension === undefined
      ? {}
      : { headDimension: input.headDimension }),
    ...(input.rotaryDimension === undefined
      ? {}
      : { rotaryDimension: input.rotaryDimension }),
  });
}
