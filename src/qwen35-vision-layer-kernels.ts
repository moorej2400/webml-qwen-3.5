import { diagnosticError } from "./diagnostics.js";
import type { KernelDefinition, KernelRegistry } from "./kernel-registry.js";
import type { Qwen35ForwardDeviceLimits } from "./qwen35-forward-dispatch.js";
import {
  assertAuthenticatedQwen35VisionGpuStagedGroup,
  type Qwen35VisionGpuStagedGroup,
  type Qwen35VisionGpuTensorView,
} from "./qwen35-vision-gpu-staging.js";
import { QWEN35_VISION_FOUNDATION_KERNELS } from "./qwen35-vision-foundation-kernels.js";
import type { Qwen35BufferBinding, Qwen35DispatchRequest, Qwen35WebGpuBuffer } from "./qwen35-webgpu-executor.js";

const HIDDEN_SIZE = 1_024;
const HEAD_COUNT = 16;
const HEAD_DIMENSION = 64;
const FEED_FORWARD_SIZE = 4_096;
const LAYER_NORM_EPSILON = 0.000001;
const MAX_PATCH_COUNT = 16_384;

function fail(code: string, message: string): never {
  throw diagnosticError(code, message);
}

function f32(value: number): number {
  return Math.fround(value);
}

function integer(value: number, minimum: number, maximum: number, code: string): void {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    fail(code, "Vision transformer layer input is invalid");
  }
}

function finite(values: ArrayLike<number>, code: string): void {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index]!)) fail(code, "Vision transformer layer input is invalid");
  }
}

const BF16_SCRATCH = new DataView(new ArrayBuffer(4));

/** Decodes a GGUF BF16 word directly to F32 without an FP16 intermediate. */
export function visionDecodeBf16(word: number): number {
  if (!Number.isSafeInteger(word) || word < 0 || word > 0xffff) {
    fail("vision-layer-bf16-invalid", "Vision transformer BF16 data is invalid");
  }
  BF16_SCRATCH.setUint32(0, word << 16, true);
  return BF16_SCRATCH.getFloat32(0, true);
}

/** Reduced CPU reference for Qwen's origin-shifted FP32 LayerNorm. */
export function visionLayerNormCpu(input: {
  readonly input: Float32Array;
  readonly tokenCount: number;
  readonly hiddenSize: number;
  readonly weight: Float32Array;
  readonly bias: Float32Array;
  readonly epsilon?: number;
}): Float32Array {
  integer(input.tokenCount, 1, MAX_PATCH_COUNT, "vision-layernorm-invalid");
  integer(input.hiddenSize, 1, HIDDEN_SIZE, "vision-layernorm-invalid");
  const epsilon = input.epsilon ?? LAYER_NORM_EPSILON;
  if (!Number.isFinite(epsilon) || epsilon <= 0 || input.input.length !== input.tokenCount * input.hiddenSize || input.weight.length !== input.hiddenSize || input.bias.length !== input.hiddenSize) {
    fail("vision-layernorm-invalid", "Vision transformer LayerNorm input is invalid");
  }
  finite(input.input, "vision-layernorm-invalid"); finite(input.weight, "vision-layernorm-invalid"); finite(input.bias, "vision-layernorm-invalid");
  const output = new Float32Array(input.input.length);
  for (let token = 0; token < input.tokenCount; token += 1) {
    const base = token * input.hiddenSize;
    const origin = input.input[base]!;
    let centeredSum = 0;
    for (let lane = 0; lane < input.hiddenSize; lane += 1) centeredSum = f32(centeredSum + f32(input.input[base + lane]! - origin));
    const mean = f32(origin + f32(centeredSum / input.hiddenSize));
    let variance = 0;
    for (let lane = 0; lane < input.hiddenSize; lane += 1) {
      const delta = f32(input.input[base + lane]! - mean);
      variance = f32(variance + f32(delta * delta));
    }
    const inverseDeviation = f32(1 / Math.sqrt(f32(variance / input.hiddenSize) + epsilon));
    for (let lane = 0; lane < input.hiddenSize; lane += 1) {
      output[base + lane] = f32(f32(f32(input.input[base + lane]! - mean) * inverseDeviation) * input.weight[lane]! + input.bias[lane]!);
    }
  }
  return output;
}

/** Reduced CPU reference for the GGUF input-contiguous BF16 matrix ABI. */
export function visionLinearBf16Cpu(input: {
  readonly input: Float32Array;
  readonly tokenCount: number;
  readonly inputWidth: number;
  readonly outputWidth: number;
  readonly weight: Uint16Array;
  readonly bias: Float32Array;
}): Float32Array {
  integer(input.tokenCount, 1, MAX_PATCH_COUNT, "vision-layer-linear-invalid");
  integer(input.inputWidth, 1, FEED_FORWARD_SIZE, "vision-layer-linear-invalid");
  integer(input.outputWidth, 1, FEED_FORWARD_SIZE * 3, "vision-layer-linear-invalid");
  if (input.input.length !== input.tokenCount * input.inputWidth || input.weight.length !== input.inputWidth * input.outputWidth || input.bias.length !== input.outputWidth) {
    fail("vision-layer-linear-invalid", "Vision transformer BF16 linear input is invalid");
  }
  finite(input.input, "vision-layer-linear-invalid"); finite(input.bias, "vision-layer-linear-invalid");
  const output = new Float32Array(input.tokenCount * input.outputWidth);
  for (let token = 0; token < input.tokenCount; token += 1) {
    for (let row = 0; row < input.outputWidth; row += 1) {
      let sum = input.bias[row]!;
      for (let column = 0; column < input.inputWidth; column += 1) {
        sum = f32(sum + f32(input.input[token * input.inputWidth + column]! * visionDecodeBf16(input.weight[row * input.inputWidth + column]!)));
      }
      output[token * input.outputWidth + row] = f32(sum);
    }
  }
  return output;
}

/** Applies the tanh approximation required by each Qwen vision transformer MLP. */
export function visionTanhGeluCpu(values: Float32Array): Float32Array {
  finite(values, "vision-layer-gelu-invalid");
  const output = new Float32Array(values.length);
  const coefficient = Math.sqrt(2 / Math.PI);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    output[index] = f32(0.5 * value * (1 + Math.tanh(coefficient * (value + 0.044715 * value * value * value))));
  }
  return output;
}

function validateSegments(tokenCount: number, offsets: Uint32Array): void {
  if (offsets.length < 2 || offsets[0] !== 0 || offsets[offsets.length - 1] !== tokenCount) {
    fail("vision-layer-segments-invalid", "Vision attention segments are invalid");
  }
  for (let index = 1; index < offsets.length; index += 1) {
    if (offsets[index]! <= offsets[index - 1]!) fail("vision-layer-segments-invalid", "Vision attention segments are invalid");
  }
}

/**
 * Computes segment-local non-causal attention with online max/sum state.
 * It intentionally stores no score matrix, matching the production shader.
 */
export function visionOnlineAttentionCpu(input: {
  readonly query: Float32Array;
  readonly key: Float32Array;
  readonly value: Float32Array;
  readonly tokenCount: number;
  readonly headCount: number;
  readonly headDimension: number;
  /** Exclusive token boundaries; each image/frame is one independent segment. */
  readonly segmentOffsets: Uint32Array;
}): Float32Array {
  integer(input.tokenCount, 1, MAX_PATCH_COUNT, "vision-layer-attention-invalid");
  integer(input.headCount, 1, HEAD_COUNT, "vision-layer-attention-invalid");
  integer(input.headDimension, 2, HEAD_DIMENSION, "vision-layer-attention-invalid");
  const vectorLength = input.tokenCount * input.headCount * input.headDimension;
  if (input.query.length !== vectorLength || input.key.length !== vectorLength || input.value.length !== vectorLength) {
    fail("vision-layer-attention-invalid", "Vision attention vectors are invalid");
  }
  validateSegments(input.tokenCount, input.segmentOffsets);
  finite(input.query, "vision-layer-attention-invalid"); finite(input.key, "vision-layer-attention-invalid"); finite(input.value, "vision-layer-attention-invalid");
  const output = new Float32Array(vectorLength);
  const scale = f32(1 / Math.sqrt(input.headDimension));
  for (let segment = 0; segment < input.segmentOffsets.length - 1; segment += 1) {
    const start = input.segmentOffsets[segment]!; const end = input.segmentOffsets[segment + 1]!;
    for (let token = start; token < end; token += 1) for (let head = 0; head < input.headCount; head += 1) {
      const base = (token * input.headCount + head) * input.headDimension;
      const accumulator = new Float32Array(input.headDimension);
      let maximum = -Infinity; let denominator = 0;
      for (let keyToken = start; keyToken < end; keyToken += 1) {
        const keyBase = (keyToken * input.headCount + head) * input.headDimension;
        let dot = 0;
        for (let lane = 0; lane < input.headDimension; lane += 1) dot = f32(dot + f32(input.query[base + lane]! * input.key[keyBase + lane]!));
        const score = f32(dot * scale); const nextMaximum = Math.max(maximum, score);
        const priorScale = maximum === -Infinity ? 0 : f32(Math.exp(maximum - nextMaximum));
        const scoreScale = f32(Math.exp(score - nextMaximum));
        denominator = f32(f32(denominator * priorScale) + scoreScale);
        for (let lane = 0; lane < input.headDimension; lane += 1) accumulator[lane] = f32(f32(accumulator[lane]! * priorScale) + f32(scoreScale * input.value[keyBase + lane]!));
        maximum = nextMaximum;
      }
      for (let lane = 0; lane < input.headDimension; lane += 1) output[base + lane] = f32(accumulator[lane]! / denominator);
    }
  }
  return output;
}

function applyRope(query: Float32Array, key: Float32Array, rope: Float32Array, tokenCount: number, headCount: number, headDimension: number): void {
  const frequencies = headDimension / 2;
  if (rope.length !== tokenCount * frequencies * 2) fail("vision-layer-rope-invalid", "Vision attention rotary data is invalid");
  finite(rope, "vision-layer-rope-invalid");
  for (let token = 0; token < tokenCount; token += 1) for (let head = 0; head < headCount; head += 1) for (let lane = 0; lane < frequencies; lane += 1) {
    const base = (token * headCount + head) * headDimension;
    const ropeBase = (token * frequencies + lane) * 2;
    const cosine = rope[ropeBase]!; const sine = rope[ropeBase + 1]!; const partner = lane + frequencies;
    const qLeft = query[base + lane]!; const qRight = query[base + partner]!; const kLeft = key[base + lane]!; const kRight = key[base + partner]!;
    query[base + lane] = f32(qLeft * cosine - qRight * sine); query[base + partner] = f32(qRight * cosine + qLeft * sine);
    key[base + lane] = f32(kLeft * cosine - kRight * sine); key[base + partner] = f32(kRight * cosine + kLeft * sine);
  }
}

function residual(left: Float32Array, right: Float32Array): Float32Array {
  if (left.length !== right.length) fail("vision-layer-residual-invalid", "Vision residual inputs are invalid");
  const output = new Float32Array(left.length);
  for (let index = 0; index < output.length; index += 1) output[index] = f32(left[index]! + right[index]!);
  return output;
}

/** Reduced, fixed-order CPU oracle for one Qwen vision transformer layer. */
export function visionTransformerLayerCpu(input: {
  readonly input: Float32Array; readonly tokenCount: number; readonly hiddenSize: number; readonly headCount: number;
  readonly qkvWeight: Uint16Array; readonly qkvBias: Float32Array;
  readonly attentionOutputWeight: Uint16Array; readonly attentionOutputBias: Float32Array;
  readonly preAttentionWeight: Float32Array; readonly preAttentionBias: Float32Array;
  readonly preMlpWeight: Float32Array; readonly preMlpBias: Float32Array;
  readonly mlpUpWeight: Uint16Array; readonly mlpUpBias: Float32Array;
  readonly mlpDownWeight: Uint16Array; readonly mlpDownBias: Float32Array;
  readonly feedForwardSize: number; readonly rope: Float32Array; readonly segmentOffsets: Uint32Array;
}): Float32Array {
  integer(input.hiddenSize, 2, HIDDEN_SIZE, "vision-layer-invalid"); integer(input.headCount, 1, HEAD_COUNT, "vision-layer-invalid");
  if (input.hiddenSize % input.headCount !== 0 || (input.hiddenSize / input.headCount) % 4 !== 0) {
    fail("vision-layer-invalid", "Vision transformer head shape is invalid");
  }
  const normalizedAttention = visionLayerNormCpu({ input: input.input, tokenCount: input.tokenCount, hiddenSize: input.hiddenSize, weight: input.preAttentionWeight, bias: input.preAttentionBias });
  const qkv = visionLinearBf16Cpu({ input: normalizedAttention, tokenCount: input.tokenCount, inputWidth: input.hiddenSize, outputWidth: input.hiddenSize * 3, weight: input.qkvWeight, bias: input.qkvBias });
  const vectorLength = input.tokenCount * input.hiddenSize; const query = new Float32Array(vectorLength); const key = new Float32Array(vectorLength); const value = new Float32Array(vectorLength);
  for (let token = 0; token < input.tokenCount; token += 1) { const source = token * input.hiddenSize * 3; const destination = token * input.hiddenSize; query.set(qkv.subarray(source, source + input.hiddenSize), destination); key.set(qkv.subarray(source + input.hiddenSize, source + input.hiddenSize * 2), destination); value.set(qkv.subarray(source + input.hiddenSize * 2, source + input.hiddenSize * 3), destination); }
  applyRope(query, key, input.rope, input.tokenCount, input.headCount, input.hiddenSize / input.headCount);
  const attention = visionOnlineAttentionCpu({ query, key, value, tokenCount: input.tokenCount, headCount: input.headCount, headDimension: input.hiddenSize / input.headCount, segmentOffsets: input.segmentOffsets });
  const attentionOutput = visionLinearBf16Cpu({ input: attention, tokenCount: input.tokenCount, inputWidth: input.hiddenSize, outputWidth: input.hiddenSize, weight: input.attentionOutputWeight, bias: input.attentionOutputBias });
  const afterAttention = residual(input.input, attentionOutput);
  const normalizedMlp = visionLayerNormCpu({ input: afterAttention, tokenCount: input.tokenCount, hiddenSize: input.hiddenSize, weight: input.preMlpWeight, bias: input.preMlpBias });
  const mlpUp = visionLinearBf16Cpu({ input: normalizedMlp, tokenCount: input.tokenCount, inputWidth: input.hiddenSize, outputWidth: input.feedForwardSize, weight: input.mlpUpWeight, bias: input.mlpUpBias });
  const mlpDown = visionLinearBf16Cpu({ input: visionTanhGeluCpu(mlpUp), tokenCount: input.tokenCount, inputWidth: input.feedForwardSize, outputWidth: input.hiddenSize, weight: input.mlpDownWeight, bias: input.mlpDownBias });
  return residual(afterAttention, mlpDown);
}

const LAYERNORM_WGSL = /* wgsl */ `
struct Params { token_count: u32, width: u32, epsilon_bits: u32, pad0: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let token = id.y; if (token >= params.token_count || params.width != 1024u) { return; }
  let base = token * 1024u; let origin = input[base]; var centered = 0.0f;
  for (var lane = 0u; lane < 1024u; lane += 1u) { centered = centered + (input[base + lane] - origin); }
  let mean = origin + centered / 1024.0f; var variance = 0.0f;
  for (var lane = 0u; lane < 1024u; lane += 1u) { let delta = input[base + lane] - mean; variance = variance + delta * delta; }
  let inverse = inverseSqrt(variance / 1024.0f + bitcast<f32>(params.epsilon_bits));
  for (var lane = 0u; lane < 1024u; lane += 1u) { output[base + lane] = (input[base + lane] - mean) * inverse * weight[lane] + bias[lane]; }
}`;

const BF16_LINEAR_WGSL = /* wgsl */ `
struct Params { token_count: u32, input_width: u32, output_width: u32, pad0: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> packed_weight: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
fn bf16(word: u32, lane: u32) -> f32 { let bits = select(word >> 16u, word & 0xffffu, (lane & 1u) == 0u); return bitcast<f32>(bits << 16u); }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x; let token = id.y; if (token >= params.token_count || row >= params.output_width || (params.input_width & 1u) != 0u) { return; }
  var sum = bias[row]; let row_base = row * params.input_width;
  for (var column = 0u; column < params.input_width; column += 1u) { let packed = packed_weight[(row_base + column) / 2u]; sum = sum + input[token * params.input_width + column] * bf16(packed, column); }
  output[token * params.output_width + row] = sum;
}`;

// QKV must be planar here. The existing RoPE ABI takes independent contiguous
// Q and K ranges, while a token-major [Q,K,V] record would require a gather.
const QKV_BF16_LINEAR_WGSL = /* wgsl */ `
struct Params { token_count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> packed_weight: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
fn bf16(word: u32, lane: u32) -> f32 { let bits = select(word >> 16u, word & 0xffffu, (lane & 1u) == 0u); return bitcast<f32>(bits << 16u); }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x; let token = id.y; if (token >= params.token_count || row >= 3072u) { return; }
  var sum = bias[row]; let row_base = row * 1024u;
  for (var column = 0u; column < 1024u; column += 1u) { let packed = packed_weight[(row_base + column) / 2u]; sum = sum + input[token * 1024u + column] * bf16(packed, column); }
  let branch = row / 1024u; let lane = row % 1024u; output[(branch * params.token_count + token) * 1024u + lane] = sum;
}`;

const ONLINE_ATTENTION_WGSL = /* wgsl */ `
struct Params { token_count: u32, segment_count: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> segment_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
var<workgroup> partial: array<f32, 64>; var<workgroup> maximum: f32; var<workgroup> denominator: f32; var<workgroup> prior: f32; var<workgroup> current: f32; var<workgroup> start: u32; var<workgroup> end: u32; var<workgroup> has_key: u32;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let lane = local.x; let token = group.y; let head = group.z; if (token >= params.token_count || head >= 16u) { return; }
  if (lane == 0u) { var found = false; for (var segment = 0u; segment < params.segment_count; segment += 1u) { if (token >= segment_offsets[segment] && token < segment_offsets[segment + 1u]) { start = segment_offsets[segment]; end = segment_offsets[segment + 1u]; found = true; } } if (!found) { start = 0u; end = 0u; } maximum = -3.402823466e+38f; denominator = 0.0; has_key = 0u; }
  workgroupBarrier(); if (start >= end) { return; } let base = (token * 16u + head) * 64u; var accumulator = 0.0f;
  for (var key_token = start; key_token < end; key_token += 1u) { let key_base = (key_token * 16u + head) * 64u; partial[lane] = query[base + lane] * key[key_base + lane]; workgroupBarrier();
    var stride = 32u; loop { if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; } workgroupBarrier(); if (stride == 1u) { break; } stride = stride / 2u; }
    if (lane == 0u) { let score = partial[0] * 0.125f; let next_maximum = max(maximum, score); prior = select(exp(maximum - next_maximum), 0.0f, has_key == 0u); current = exp(score - next_maximum); denominator = denominator * prior + current; maximum = next_maximum; has_key = 1u; }
    workgroupBarrier(); accumulator = accumulator * prior + current * value[key_base + lane]; workgroupBarrier();
  }
  output[base + lane] = accumulator / denominator;
}`;

const TANH_GELU_WGSL = /* wgsl */ `
struct Params { element_count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { let index = id.y * 4096u + id.x; if (index >= params.element_count) { return; } let value = values[index]; values[index] = 0.5f * value * (1.0f + tanh(0.7978845608f * (value + 0.044715f * value * value * value))); }`;

const RESIDUAL_WGSL = /* wgsl */ `
struct Params { element_count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read> update: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { let index = id.y * 1024u + id.x; if (index >= params.element_count) { return; } input[index] = input[index] + update[index]; }`;

function definition(id: string, operation: string, source: string): KernelDefinition {
  return Object.freeze({ id, key: Object.freeze({ operation, layout: "f32", phase: "vision", profile: "portable-f32" }), source });
}

/** Fixed kernels for the 24 identical Qwen3.5 vision transformer layers. */
export const QWEN35_VISION_LAYER_KERNELS: readonly KernelDefinition[] = Object.freeze([
  definition("qwen35-vision-layernorm-f32", "vision-layernorm", LAYERNORM_WGSL),
  definition("qwen35-vision-bf16-linear-f32", "vision-bf16-linear", BF16_LINEAR_WGSL),
  definition("qwen35-vision-qkv-bf16-linear-f32", "vision-qkv-bf16-linear", QKV_BF16_LINEAR_WGSL),
  definition("qwen35-vision-online-attention-f32", "vision-online-attention", ONLINE_ATTENTION_WGSL),
  definition("qwen35-vision-tanh-gelu-f32", "vision-tanh-gelu", TANH_GELU_WGSL),
  definition("qwen35-vision-residual-add-f32", "vision-residual-add", RESIDUAL_WGSL),
]);

export function registerQwen35VisionLayerKernels(registry: Pick<KernelRegistry, "register">): void {
  for (const kernel of QWEN35_VISION_LAYER_KERNELS) registry.register(kernel);
}

export const QWEN35_VISION_LAYER_LIMITS = Object.freeze({ HIDDEN_SIZE, HEAD_COUNT, HEAD_DIMENSION, FEED_FORWARD_SIZE, LAYER_NORM_EPSILON, MAX_PATCH_COUNT });

export interface Qwen35VisionLayerStorage {
  readonly buffer: Qwen35WebGpuBuffer;
  /** The writable range beginning at byteOffset, not the whole physical arena. */
  readonly byteLength: number;
  readonly byteOffset?: number;
}

/**
 * Reuses only short-lived layer activations. Weight ownership remains in the
 * authenticated streamed group and is released after the layer retires.
 */
export interface Qwen35VisionLayerWorkspace {
  readonly hidden: Qwen35VisionLayerStorage;
  readonly normalized: Qwen35VisionLayerStorage;
  /** Three planar [Q tokens][K tokens][V tokens] regions. */
  readonly qkv: Qwen35VisionLayerStorage;
  readonly attention: Qwen35VisionLayerStorage;
  readonly mlp: Qwen35VisionLayerStorage;
  readonly rope: Qwen35VisionLayerStorage;
  /** u32 exclusive boundaries for isolated image/frame attention segments. */
  readonly segmentOffsets: Qwen35VisionLayerStorage;
  /** Ten 16-byte slots; the two identical residual dispatches share slot five. */
  readonly uniforms: readonly Qwen35VisionLayerStorage[];
}

export interface Qwen35VisionLayerDispatchPlan extends Qwen35DispatchRequest {
  readonly uniformWords: readonly [number, number, number, number];
}

function f32Bits(value: number): number {
  BF16_SCRATCH.setFloat32(0, value, true);
  return BF16_SCRATCH.getUint32(0, true);
}

function validateLimits(limits: Qwen35ForwardDeviceLimits): void {
  for (const value of [limits.minStorageBufferOffsetAlignment, limits.minUniformBufferOffsetAlignment, limits.maxStorageBufferBindingSize, limits.maxUniformBufferBindingSize, limits.maxComputeWorkgroupsPerDimension]) {
    integer(value, 1, Number.MAX_SAFE_INTEGER, "vision-layer-limits-invalid");
  }
}

function storageBinding(binding: number, storage: Qwen35VisionLayerStorage, requiredBytes: number, limits: Qwen35ForwardDeviceLimits): Qwen35BufferBinding {
  const offset = storage.byteOffset ?? 0;
  if (
    typeof storage.buffer !== "object" || storage.buffer === null || !Number.isSafeInteger(storage.byteLength) || storage.byteLength < requiredBytes ||
    !Number.isSafeInteger(offset) || offset < 0 || offset % limits.minStorageBufferOffsetAlignment !== 0 || requiredBytes < 4 || requiredBytes % 4 !== 0 || requiredBytes > limits.maxStorageBufferBindingSize
  ) fail("vision-layer-binding-invalid", "Vision transformer layer GPU binding is invalid");
  return Object.freeze({ binding, kind: "storage", buffer: storage.buffer, offset, size: requiredBytes });
}

function uniformBinding(binding: number, storage: Qwen35VisionLayerStorage, limits: Qwen35ForwardDeviceLimits): Qwen35BufferBinding {
  const offset = storage.byteOffset ?? 0;
  if (
    typeof storage.buffer !== "object" || storage.buffer === null || storage.byteLength !== 16 || !Number.isSafeInteger(offset) || offset < 0 ||
    offset % limits.minUniformBufferOffsetAlignment !== 0 || 16 > limits.maxUniformBufferBindingSize
  ) fail("vision-layer-uniform-invalid", "Vision transformer layer uniform binding is invalid");
  return Object.freeze({ binding, kind: "uniform", buffer: storage.buffer, offset, size: 16 });
}

function validateUniformSlots(slots: readonly Qwen35VisionLayerStorage[], limits: Qwen35ForwardDeviceLimits): void {
  const seen = new Map<object, Set<number>>();
  for (const slot of slots) {
    const binding = uniformBinding(0, slot, limits);
    const buffer = binding.buffer as object;
    const offsets = seen.get(buffer);
    if (offsets?.has(binding.offset)) {
      fail("vision-layer-uniform-alias-invalid", "Vision transformer layer uniform slots must be distinct");
    }
    if (offsets === undefined) seen.set(buffer, new Set([binding.offset]));
    else offsets.add(binding.offset);
  }
}

function storageSlice(storage: Qwen35VisionLayerStorage, byteOffset: number): Qwen35VisionLayerStorage {
  if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset > storage.byteLength) {
    fail("vision-layer-workspace-invalid", "Vision transformer layer workspace is invalid");
  }
  return Object.freeze({ buffer: storage.buffer, byteOffset: (storage.byteOffset ?? 0) + byteOffset, byteLength: storage.byteLength - byteOffset });
}

function tensor(group: Qwen35VisionGpuStagedGroup, name: string, shape: readonly number[], precision: "f32" | "bf16", orientation: "element-contiguous" | "input-width-contiguous"): Qwen35VisionGpuTensorView {
  const value = group.tensors.find((candidate) => candidate.name === name);
  if (
    value === undefined || value.shape.join(",") !== shape.join(",") || value.precision !== precision ||
    value.storageType !== (precision === "f32" ? "f32" : "raw") || value.orientation.kind !== orientation || value.segments.length !== 1
  ) fail("vision-layer-tensor-invalid", "Vision transformer layer tensor ABI is invalid");
  if (
    orientation === "element-contiguous" && value.orientation.contiguousDimension !== "element" ||
    orientation === "input-width-contiguous" && (
      value.orientation.contiguousDimension !== "input-width" || value.orientation.manifestShape.join(",") !== "input-width,output-rows"
    )
  ) fail("vision-layer-tensor-invalid", "Vision transformer layer tensor ABI is invalid");
  const expectedBytes = shape.reduce((product, dimension) => product * dimension, 1) * (precision === "f32" ? 4 : 2);
  const segment = value.segments[0]!;
  if (segment.tensorOffset !== 0 || segment.byteLength !== expectedBytes || segment.bufferOffset < 0 || segment.bufferOffset % 4 !== 0) {
    fail("vision-layer-tensor-invalid", "Vision transformer layer tensor ABI is invalid");
  }
  return value;
}

function tensorBinding(binding: number, value: Qwen35VisionGpuTensorView, limits: Qwen35ForwardDeviceLimits): Qwen35BufferBinding {
  const segment = value.segments[0]!;
  if (segment.bufferOffset % limits.minStorageBufferOffsetAlignment !== 0 || segment.byteLength > limits.maxStorageBufferBindingSize) {
    fail("vision-layer-binding-invalid", "Vision transformer layer GPU binding is invalid");
  }
  return Object.freeze({ binding, kind: "storage", buffer: segment.buffer, offset: segment.bufferOffset, size: segment.byteLength });
}

function kernel(operation: string): { readonly id: string; readonly source: string; readonly entryPoint: string } {
  const definition = [...QWEN35_VISION_LAYER_KERNELS, ...QWEN35_VISION_FOUNDATION_KERNELS].find((candidate) => candidate.key.operation === operation);
  if (definition === undefined) fail("vision-layer-kernel-missing", "Vision transformer layer kernel is unavailable");
  return Object.freeze({ id: definition.id, source: definition.source, entryPoint: "main" });
}

function validateWorkgroups(workgroups: { readonly x: number; readonly y: number; readonly z: number }, limits: Qwen35ForwardDeviceLimits): void {
  if (workgroups.x > limits.maxComputeWorkgroupsPerDimension || workgroups.y > limits.maxComputeWorkgroupsPerDimension || workgroups.z > limits.maxComputeWorkgroupsPerDimension) {
    fail("vision-layer-dispatch-invalid", "Vision transformer layer dispatch exceeds device limits");
  }
}

/**
 * Assembles one exact transformer-layer sequence. The QKV projection writes
 * planar Q/K/V regions so the existing RoPE kernel can bind Q and K directly.
 */
export function planQwen35VisionLayerDispatches(input: {
  readonly layer: number;
  readonly staged: Qwen35VisionGpuStagedGroup;
  readonly workspace: Qwen35VisionLayerWorkspace;
  readonly tokenCount: number;
  readonly segmentCount: number;
  readonly limits: Qwen35ForwardDeviceLimits;
}): readonly Qwen35VisionLayerDispatchPlan[] {
  integer(input.layer, 0, 23, "vision-layer-invalid"); integer(input.tokenCount, 1, MAX_PATCH_COUNT, "vision-layer-invalid"); integer(input.segmentCount, 1, input.tokenCount, "vision-layer-invalid"); validateLimits(input.limits);
  const staged = assertAuthenticatedQwen35VisionGpuStagedGroup(input.staged);
  if (staged.layer !== input.layer || input.workspace.uniforms.length !== 10) fail("vision-layer-invalid", "Vision transformer layer input is invalid");
  // Slot five is reused by both equal residual payloads. The ten logical slots
  // themselves stay distinct so callers cannot overwrite another stage's data.
  validateUniformSlots(input.workspace.uniforms, input.limits);
  const prefix = `v.blk.${input.layer}`; const f32 = Float32Array.BYTES_PER_ELEMENT;
  const ln1Weight = tensor(staged, `${prefix}.ln1.weight`, [HIDDEN_SIZE], "f32", "element-contiguous"); const ln1Bias = tensor(staged, `${prefix}.ln1.bias`, [HIDDEN_SIZE], "f32", "element-contiguous");
  const qkvWeight = tensor(staged, `${prefix}.attn_qkv.weight`, [HIDDEN_SIZE, HIDDEN_SIZE * 3], "bf16", "input-width-contiguous"); const qkvBias = tensor(staged, `${prefix}.attn_qkv.bias`, [HIDDEN_SIZE * 3], "f32", "element-contiguous");
  const outWeight = tensor(staged, `${prefix}.attn_out.weight`, [HIDDEN_SIZE, HIDDEN_SIZE], "bf16", "input-width-contiguous"); const outBias = tensor(staged, `${prefix}.attn_out.bias`, [HIDDEN_SIZE], "f32", "element-contiguous");
  const ln2Weight = tensor(staged, `${prefix}.ln2.weight`, [HIDDEN_SIZE], "f32", "element-contiguous"); const ln2Bias = tensor(staged, `${prefix}.ln2.bias`, [HIDDEN_SIZE], "f32", "element-contiguous");
  const upWeight = tensor(staged, `${prefix}.ffn_up.weight`, [HIDDEN_SIZE, FEED_FORWARD_SIZE], "bf16", "input-width-contiguous"); const upBias = tensor(staged, `${prefix}.ffn_up.bias`, [FEED_FORWARD_SIZE], "f32", "element-contiguous");
  const downWeight = tensor(staged, `${prefix}.ffn_down.weight`, [FEED_FORWARD_SIZE, HIDDEN_SIZE], "bf16", "input-width-contiguous"); const downBias = tensor(staged, `${prefix}.ffn_down.bias`, [HIDDEN_SIZE], "f32", "element-contiguous");
  const hiddenBytes = input.tokenCount * HIDDEN_SIZE * f32; const qkvBytes = hiddenBytes * 3; const mlpBytes = input.tokenCount * FEED_FORWARD_SIZE * f32; const ropeBytes = input.tokenCount * (HEAD_DIMENSION / 2) * 2 * f32;
  const uniform = (index: number) => uniformBinding(4, input.workspace.uniforms[index]!, input.limits);
  const plan = (operation: string, bindings: readonly Qwen35BufferBinding[], workgroups: { readonly x: number; readonly y: number; readonly z: number }, uniformWords: readonly [number, number, number, number]): Qwen35VisionLayerDispatchPlan => {
    validateWorkgroups(workgroups, input.limits); return Object.freeze({ kernel: kernel(operation), bindings: Object.freeze(bindings), workgroups: Object.freeze(workgroups), uniformWords: Object.freeze(uniformWords) as readonly [number, number, number, number] });
  };
  const normWords = Object.freeze([input.tokenCount, HIDDEN_SIZE, f32Bits(LAYER_NORM_EPSILON), 0]) as readonly [number, number, number, number];
  const residualWords = Object.freeze([input.tokenCount * HIDDEN_SIZE, 0, 0, 0]) as readonly [number, number, number, number];
  return Object.freeze([
    plan("vision-layernorm", [storageBinding(0, input.workspace.hidden, hiddenBytes, input.limits), tensorBinding(1, ln1Weight, input.limits), tensorBinding(2, ln1Bias, input.limits), storageBinding(3, input.workspace.normalized, hiddenBytes, input.limits), uniform(0)], { x: 1, y: input.tokenCount, z: 1 }, normWords),
    plan("vision-qkv-bf16-linear", [storageBinding(0, input.workspace.normalized, hiddenBytes, input.limits), tensorBinding(1, qkvWeight, input.limits), tensorBinding(2, qkvBias, input.limits), storageBinding(3, input.workspace.qkv, qkvBytes, input.limits), uniform(1)], { x: 48, y: input.tokenCount, z: 1 }, Object.freeze([input.tokenCount, 0, 0, 0]) as readonly [number, number, number, number]),
    plan("vision-apply-2d-rope", [storageBinding(0, storageSlice(input.workspace.qkv, 0), hiddenBytes, input.limits), storageBinding(1, storageSlice(input.workspace.qkv, hiddenBytes), hiddenBytes, input.limits), storageBinding(2, input.workspace.rope, ropeBytes, input.limits), uniformBinding(3, input.workspace.uniforms[2]!, input.limits)], { x: 1, y: input.tokenCount, z: HEAD_COUNT }, Object.freeze([input.tokenCount, 0, 0, 0]) as readonly [number, number, number, number]),
    plan("vision-online-attention", [storageBinding(0, storageSlice(input.workspace.qkv, 0), hiddenBytes, input.limits), storageBinding(1, storageSlice(input.workspace.qkv, hiddenBytes), hiddenBytes, input.limits), storageBinding(2, storageSlice(input.workspace.qkv, hiddenBytes * 2), hiddenBytes, input.limits), storageBinding(3, input.workspace.segmentOffsets, (input.segmentCount + 1) * 4, input.limits), storageBinding(4, input.workspace.attention, hiddenBytes, input.limits), uniformBinding(5, input.workspace.uniforms[3]!, input.limits)], { x: 1, y: input.tokenCount, z: HEAD_COUNT }, Object.freeze([input.tokenCount, input.segmentCount, 0, 0]) as readonly [number, number, number, number]),
    plan("vision-bf16-linear", [storageBinding(0, input.workspace.attention, hiddenBytes, input.limits), tensorBinding(1, outWeight, input.limits), tensorBinding(2, outBias, input.limits), storageBinding(3, input.workspace.normalized, hiddenBytes, input.limits), uniform(4)], { x: 16, y: input.tokenCount, z: 1 }, Object.freeze([input.tokenCount, HIDDEN_SIZE, HIDDEN_SIZE, 0]) as readonly [number, number, number, number]),
    plan("vision-residual-add", [storageBinding(0, input.workspace.hidden, hiddenBytes, input.limits), storageBinding(1, input.workspace.normalized, hiddenBytes, input.limits), uniformBinding(2, input.workspace.uniforms[5]!, input.limits)], { x: 16, y: input.tokenCount, z: 1 }, residualWords),
    plan("vision-layernorm", [storageBinding(0, input.workspace.hidden, hiddenBytes, input.limits), tensorBinding(1, ln2Weight, input.limits), tensorBinding(2, ln2Bias, input.limits), storageBinding(3, input.workspace.normalized, hiddenBytes, input.limits), uniform(6)], { x: 1, y: input.tokenCount, z: 1 }, normWords),
    plan("vision-bf16-linear", [storageBinding(0, input.workspace.normalized, hiddenBytes, input.limits), tensorBinding(1, upWeight, input.limits), tensorBinding(2, upBias, input.limits), storageBinding(3, input.workspace.mlp, mlpBytes, input.limits), uniform(7)], { x: 64, y: input.tokenCount, z: 1 }, Object.freeze([input.tokenCount, HIDDEN_SIZE, FEED_FORWARD_SIZE, 0]) as readonly [number, number, number, number]),
    plan("vision-tanh-gelu", [storageBinding(0, input.workspace.mlp, mlpBytes, input.limits), uniformBinding(1, input.workspace.uniforms[8]!, input.limits)], { x: 64, y: input.tokenCount, z: 1 }, Object.freeze([input.tokenCount * FEED_FORWARD_SIZE, 0, 0, 0]) as readonly [number, number, number, number]),
    plan("vision-bf16-linear", [storageBinding(0, input.workspace.mlp, mlpBytes, input.limits), tensorBinding(1, downWeight, input.limits), tensorBinding(2, downBias, input.limits), storageBinding(3, input.workspace.normalized, hiddenBytes, input.limits), uniform(9)], { x: 16, y: input.tokenCount, z: 1 }, Object.freeze([input.tokenCount, FEED_FORWARD_SIZE, HIDDEN_SIZE, 0]) as readonly [number, number, number, number]),
    plan("vision-residual-add", [storageBinding(0, input.workspace.hidden, hiddenBytes, input.limits), storageBinding(1, input.workspace.normalized, hiddenBytes, input.limits), uniformBinding(2, input.workspace.uniforms[5]!, input.limits)], { x: 16, y: input.tokenCount, z: 1 }, residualWords),
  ]);
}
