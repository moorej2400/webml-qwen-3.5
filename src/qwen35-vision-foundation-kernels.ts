import { diagnosticError } from "./diagnostics.js";
import type { KernelDefinition, KernelRegistry } from "./kernel-registry.js";
import type {
  Qwen35VisionGpuStagedGroup,
  Qwen35VisionGpuTensorView,
} from "./qwen35-vision-gpu-staging.js";
import { assertAuthenticatedQwen35VisionGpuStagedGroup } from "./qwen35-vision-gpu-staging.js";
import type {
  Qwen35BufferBinding,
  Qwen35DispatchRequest,
  Qwen35WebGpuBuffer,
} from "./qwen35-webgpu-executor.js";
import type { Qwen35ForwardDeviceLimits } from "./qwen35-forward-dispatch.js";

const PATCH_SIZE = 16;
const TEMPORAL_PATCH_SIZE = 2;
const CHANNEL_COUNT = 3;
const HIDDEN_SIZE = 1_024;
const HEAD_COUNT = 16;
const HEAD_DIMENSION = 64;
const ROPE_FREQUENCIES = HEAD_DIMENSION / 2;
const POSITION_TABLE_EDGE = 48;
const POSITION_TABLE_ROWS = POSITION_TABLE_EDGE * POSITION_TABLE_EDGE;
// The selected 4,096 post-merge visual-token contract has four pre-merge
// patches per token. This is a product bound, not a WebGPU feasibility claim.
const MAX_PATCH_COUNT = 16_384;

function fail(code: string, message: string): never {
  throw diagnosticError(code, message);
}

function requireInteger(value: number, minimum: number, maximum: number, code: string): void {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    fail(code, "Vision foundation kernel plan is invalid");
  }
}

function f32(value: number): number {
  return Math.fround(value);
}

function mergeBlockCoordinate(patch: number, gridHeight: number, gridWidth: number): readonly [number, number] {
  const mergedWidth = gridWidth / 2;
  const group = Math.floor(patch / 4);
  const withinGroup = patch % 4;
  return Object.freeze([
    Math.floor(group / mergedWidth) * 2 + Math.floor(withinGroup / 2),
    (group % mergedWidth) * 2 + (withinGroup % 2),
  ]);
}

function requireGrid(gridHeight: number, gridWidth: number): number {
  requireInteger(gridHeight, 2, MAX_PATCH_COUNT, "vision-foundation-grid-invalid");
  requireInteger(gridWidth, 2, MAX_PATCH_COUNT, "vision-foundation-grid-invalid");
  if (gridHeight % 2 !== 0 || gridWidth % 2 !== 0) {
    fail("vision-foundation-grid-invalid", "Vision foundation kernel plan is invalid");
  }
  const patchCount = gridHeight * gridWidth;
  requireInteger(patchCount, 1, MAX_PATCH_COUNT, "vision-foundation-grid-invalid");
  return patchCount;
}

/** Reduced reference used only by deterministic fixtures for packed-layout checks. */
export function visionPatchConv3dReferenceCpu(input: {
  readonly patches: Float32Array;
  readonly patchCount: number;
  readonly hiddenSize: number;
  readonly patchSize: number;
  readonly temporalPatchSize: number;
  readonly weightsTemporalZero: Float32Array;
  readonly weightsTemporalOne: Float32Array;
  readonly bias: Float32Array;
}): Float32Array {
  const channels = input.weightsTemporalZero.length /
    (input.hiddenSize * input.patchSize * input.patchSize);
  if (
    !Number.isInteger(channels) || channels < 1 ||
    input.temporalPatchSize !== 2 ||
    input.patchCount < 1 || input.hiddenSize < 1 || input.patchSize < 1 ||
    input.weightsTemporalOne.length !== input.weightsTemporalZero.length ||
    input.bias.length !== input.hiddenSize ||
    input.patches.length !== input.patchCount * channels * input.temporalPatchSize * input.patchSize * input.patchSize
  ) {
    fail("vision-foundation-patch-invalid", "Vision patch Conv3D inputs are invalid");
  }
  const output = new Float32Array(input.patchCount * input.hiddenSize);
  const patchArea = input.patchSize * input.patchSize;
  for (let patch = 0; patch < input.patchCount; patch += 1) {
    for (let outputChannel = 0; outputChannel < input.hiddenSize; outputChannel += 1) {
      let sum = f32(input.bias[outputChannel]!);
      for (let channel = 0; channel < channels; channel += 1) {
        for (let row = 0; row < input.patchSize; row += 1) {
          for (let column = 0; column < input.patchSize; column += 1) {
            const withinPatch = row * input.patchSize + column;
            const packedBase = (((channel * 2) * patchArea) + withinPatch);
            const patchBase = patch * channels * 2 * patchArea;
            // GGUF dimension zero is contiguous: width, height, channel, output.
            const weight = column + input.patchSize * (row + input.patchSize * (channel + channels * outputChannel));
            sum = f32(sum + f32(input.patches[patchBase + packedBase]! * input.weightsTemporalZero[weight]!));
            sum = f32(sum + f32(input.patches[patchBase + packedBase + patchArea]! * input.weightsTemporalOne[weight]!));
          }
        }
      }
      output[patch * input.hiddenSize + outputChannel] = sum;
    }
  }
  return output;
}

/** CPU oracle for the production 3x2x16x16-to-1024 Conv3D kernel. */
export function visionPatchConv3dCpu(input: {
  readonly patches: Float32Array;
  readonly patchCount: number;
  readonly hiddenSize: number;
  readonly patchSize: number;
  readonly temporalPatchSize: number;
  readonly weightsTemporalZero: Float32Array;
  readonly weightsTemporalOne: Float32Array;
  readonly bias: Float32Array;
}): Float32Array {
  if (
    input.hiddenSize !== HIDDEN_SIZE || input.patchSize !== PATCH_SIZE ||
    input.temporalPatchSize !== TEMPORAL_PATCH_SIZE ||
    input.weightsTemporalZero.length !== PATCH_SIZE * PATCH_SIZE * CHANNEL_COUNT * HIDDEN_SIZE ||
    input.weightsTemporalOne.length !== input.weightsTemporalZero.length || input.bias.length !== HIDDEN_SIZE
  ) fail("vision-foundation-patch-invalid", "Vision patch Conv3D inputs are invalid");
  return visionPatchConv3dReferenceCpu(input);
}

/** Adds the [row][hidden] learned table with align_corners interpolation. */
export function visionAddLearnedPositionCpu(input: {
  readonly embeddings: Float32Array;
  readonly gridHeight: number;
  readonly gridWidth: number;
  readonly hiddenSize: number;
  readonly tableHeight: number;
  readonly tableWidth: number;
  readonly table: Float32Array;
}): Float32Array {
  const patchCount = requireGrid(input.gridHeight, input.gridWidth);
  if (
    input.hiddenSize < 1 || input.tableHeight < 2 || input.tableWidth < 2 ||
    input.embeddings.length !== patchCount * input.hiddenSize ||
    input.table.length !== input.tableHeight * input.tableWidth * input.hiddenSize
  ) fail("vision-foundation-position-invalid", "Vision learned-position inputs are invalid");
  const output = new Float32Array(input.embeddings);
  for (let patch = 0; patch < patchCount; patch += 1) {
    const [row, column] = mergeBlockCoordinate(patch, input.gridHeight, input.gridWidth);
    const sourceY = f32(row * (input.tableHeight - 1) / (input.gridHeight - 1));
    const sourceX = f32(column * (input.tableWidth - 1) / (input.gridWidth - 1));
    const y0 = Math.floor(sourceY); const x0 = Math.floor(sourceX);
    const y1 = Math.min(y0 + 1, input.tableHeight - 1); const x1 = Math.min(x0 + 1, input.tableWidth - 1);
    const fy = f32(sourceY - y0); const fx = f32(sourceX - x0);
    for (let hidden = 0; hidden < input.hiddenSize; hidden += 1) {
      const at = (y: number, x: number) => input.table[(y * input.tableWidth + x) * input.hiddenSize + hidden]!;
      const top = f32(f32(at(y0, x0) * f32(1 - fx)) + f32(at(y0, x1) * fx));
      const bottom = f32(f32(at(y1, x0) * f32(1 - fx)) + f32(at(y1, x1) * fx));
      const interpolated = f32(f32(top * f32(1 - fy)) + f32(bottom * fy));
      output[patch * input.hiddenSize + hidden] = f32(output[patch * input.hiddenSize + hidden]! + interpolated);
    }
  }
  return output;
}

/** Prepares f32 cos/sin pairs and merge-block-major height/width coordinates. */
export function visionPrepare2dRopeCpu(input: {
  readonly gridHeight: number;
  readonly gridWidth: number;
  readonly headDimension: number;
}): Readonly<{ readonly coordinates: Uint32Array; readonly values: Float32Array }> {
  const patchCount = requireGrid(input.gridHeight, input.gridWidth);
  if (input.headDimension < 4 || input.headDimension % 4 !== 0) {
    fail("vision-foundation-rope-invalid", "Vision 2D RoPE inputs are invalid");
  }
  const frequencies = input.headDimension / 2;
  const coordinates = new Uint32Array(patchCount * 2);
  const values = new Float32Array(patchCount * frequencies * 2);
  for (let patch = 0; patch < patchCount; patch += 1) {
    const [height, width] = mergeBlockCoordinate(patch, input.gridHeight, input.gridWidth);
    coordinates[patch * 2] = height;
    coordinates[patch * 2 + 1] = width;
    for (let frequency = 0; frequency < frequencies; frequency += 1) {
      const position = frequency < frequencies / 2 ? height : width;
      const localFrequency = frequency % (frequencies / 2);
      const exponent = f32(f32(2 * localFrequency) / frequencies);
      const inverseFrequency = f32(1 / f32(Math.pow(10_000, exponent)));
      const angle = f32(f32(position) * inverseFrequency);
      const base = (patch * frequencies + frequency) * 2;
      values[base] = f32(Math.cos(angle));
      values[base + 1] = f32(Math.sin(angle));
    }
  }
  return Object.freeze({ coordinates, values });
}

/** Applies the prepared split-half 2D RoPE pairs to independent Q and K arrays. */
export function visionApply2dRopeCpu(input: {
  readonly query: Float32Array;
  readonly key: Float32Array;
  readonly rope: Float32Array;
  readonly patchCount: number;
  readonly headCount: number;
  readonly headDimension: number;
}): Readonly<{ readonly query: Float32Array; readonly key: Float32Array }> {
  const frequencies = input.headDimension / 2;
  if (
    input.patchCount < 1 || input.headCount < 1 || input.headDimension < 4 || input.headDimension % 2 !== 0 ||
    input.query.length !== input.patchCount * input.headCount * input.headDimension ||
    input.key.length !== input.query.length || input.rope.length !== input.patchCount * frequencies * 2
  ) fail("vision-foundation-rope-invalid", "Vision 2D RoPE inputs are invalid");
  const query = new Float32Array(input.query); const key = new Float32Array(input.key);
  for (let patch = 0; patch < input.patchCount; patch += 1) {
    for (let head = 0; head < input.headCount; head += 1) {
      const base = (patch * input.headCount + head) * input.headDimension;
      for (let frequency = 0; frequency < frequencies; frequency += 1) {
        const ropeBase = (patch * frequencies + frequency) * 2;
        const cosine = input.rope[ropeBase]!; const sine = input.rope[ropeBase + 1]!;
        const partner = frequency + frequencies;
        query[base + frequency] = f32(f32(input.query[base + frequency]! * cosine) - f32(input.query[base + partner]! * sine));
        query[base + partner] = f32(f32(input.query[base + partner]! * cosine) + f32(input.query[base + frequency]! * sine));
        key[base + frequency] = f32(f32(input.key[base + frequency]! * cosine) - f32(input.key[base + partner]! * sine));
        key[base + partner] = f32(f32(input.key[base + partner]! * cosine) + f32(input.key[base + frequency]! * sine));
      }
    }
  }
  return Object.freeze({ query, key });
}

const PATCH_CONV3D_WGSL = /* wgsl */ `
struct Params { patch_count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> patches: array<f32>;
@group(0) @binding(1) var<storage, read> weights_t0: array<f32>;
@group(0) @binding(2) var<storage, read> weights_t1: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> embeddings: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let hidden = id.x; let patch = id.y;
  if (hidden >= 1024u || patch >= params.patch_count || params.patch_count > 16384u) { return; }
  var sum = bias[hidden];
  for (var channel = 0u; channel < 3u; channel += 1u) {
    for (var row = 0u; row < 16u; row += 1u) {
      for (var column = 0u; column < 16u; column += 1u) {
        let pixel = row * 16u + column;
        let patch_base = patch * 1536u + channel * 512u + pixel;
        let weight = column + 16u * (row + 16u * (channel + 3u * hidden));
        sum = sum + patches[patch_base] * weights_t0[weight];
        sum = sum + patches[patch_base + 256u] * weights_t1[weight];
      }
    }
  }
  embeddings[patch * 1024u + hidden] = sum;
}`;

const POSITION_WGSL = /* wgsl */ `
struct Params { patch_count: u32, grid_height: u32, grid_width: u32, first_segment_scalars: u32 }
@group(0) @binding(0) var<storage, read_write> embeddings: array<f32>;
@group(0) @binding(1) var<storage, read> position_a: array<f32>;
@group(0) @binding(2) var<storage, read> position_b: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
fn position(index: u32) -> f32 { if (index < params.first_segment_scalars) { return position_a[index]; } return position_b[index - params.first_segment_scalars]; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let hidden = id.x; let patch = id.y;
  if (hidden >= 1024u || patch >= params.patch_count || params.patch_count > 16384u || params.grid_height < 2u || params.grid_width < 2u || (params.grid_height & 1u) != 0u || (params.grid_width & 1u) != 0u) { return; }
  let merged_width = params.grid_width / 2u; let group = patch / 4u; let within = patch % 4u;
  let row = (group / merged_width) * 2u + within / 2u; let column = (group % merged_width) * 2u + within % 2u;
  let source_y = f32(row) * 47.0f / f32(params.grid_height - 1u); let source_x = f32(column) * 47.0f / f32(params.grid_width - 1u);
  let y0 = u32(floor(source_y)); let x0 = u32(floor(source_x)); let y1 = min(y0 + 1u, 47u); let x1 = min(x0 + 1u, 47u);
  let fy = source_y - f32(y0); let fx = source_x - f32(x0);
  let p00 = position((y0 * 48u + x0) * 1024u + hidden); let p01 = position((y0 * 48u + x1) * 1024u + hidden);
  let p10 = position((y1 * 48u + x0) * 1024u + hidden); let p11 = position((y1 * 48u + x1) * 1024u + hidden);
  let top = p00 * (1.0f - fx) + p01 * fx; let bottom = p10 * (1.0f - fx) + p11 * fx;
  embeddings[patch * 1024u + hidden] = embeddings[patch * 1024u + hidden] + (top * (1.0f - fy) + bottom * fy);
}`;

const ROPE_PREPARE_WGSL = /* wgsl */ `
struct Params { patch_count: u32, grid_height: u32, grid_width: u32, pad0: u32 }
@group(0) @binding(0) var<storage, read_write> rope: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let frequency = id.x; let patch = id.y;
  if (frequency >= 32u || patch >= params.patch_count || params.patch_count > 16384u || params.grid_height < 2u || params.grid_width < 2u || (params.grid_height & 1u) != 0u || (params.grid_width & 1u) != 0u) { return; }
  let merged_width = params.grid_width / 2u; let group = patch / 4u; let within = patch % 4u;
  let height = (group / merged_width) * 2u + within / 2u; let width = (group % merged_width) * 2u + within % 2u;
  let position = select(height, width, frequency >= 16u); let local_frequency = frequency % 16u;
  let inverse_frequency = 1.0f / pow(10000.0f, f32(2u * local_frequency) / 32.0f); let angle = f32(position) * inverse_frequency; let destination = (patch * 32u + frequency) * 2u;
  rope[destination] = cos(angle); rope[destination + 1u] = sin(angle);
}`;

const ROPE_APPLY_WGSL = /* wgsl */ `
struct Params { patch_count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read_write> query: array<f32>;
@group(0) @binding(1) var<storage, read_write> key: array<f32>;
@group(0) @binding(2) var<storage, read> rope: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let lane = id.x; let patch = id.y; let head = id.z;
  if (lane >= 32u || head >= 16u || patch >= params.patch_count || params.patch_count > 16384u) { return; }
  let base = (patch * 16u + head) * 64u; let pair = lane + 32u; let rope_base = (patch * 32u + lane) * 2u; let cosine = rope[rope_base]; let sine = rope[rope_base + 1u];
  let q_left = query[base + lane]; let q_right = query[base + pair]; let k_left = key[base + lane]; let k_right = key[base + pair];
  query[base + lane] = q_left * cosine - q_right * sine; query[base + pair] = q_right * cosine + q_left * sine;
  key[base + lane] = k_left * cosine - k_right * sine; key[base + pair] = k_right * cosine + k_left * sine;
}`;

function kernel(id: string, operation: string, source: string): KernelDefinition {
  return Object.freeze({ id, key: Object.freeze({ operation, layout: "f32", phase: "vision", profile: "portable-f32" }), source });
}

/** Fixed Qwen3.5 vision foundation; later attention/MLP kernels are separate. */
export const QWEN35_VISION_FOUNDATION_KERNELS: readonly KernelDefinition[] = Object.freeze([
  kernel("qwen35-vision-patch-conv3d-f32", "vision-patch-conv3d", PATCH_CONV3D_WGSL),
  kernel("qwen35-vision-add-learned-position-f32", "vision-add-learned-position", POSITION_WGSL),
  kernel("qwen35-vision-prepare-2d-rope-f32", "vision-prepare-2d-rope", ROPE_PREPARE_WGSL),
  kernel("qwen35-vision-apply-2d-rope-f32", "vision-apply-2d-rope", ROPE_APPLY_WGSL),
]);

export function registerQwen35VisionFoundationKernels(registry: Pick<KernelRegistry, "register">): void {
  for (const kernel of QWEN35_VISION_FOUNDATION_KERNELS) registry.register(kernel);
}

export const QWEN35_VISION_FOUNDATION_LIMITS = Object.freeze({ PATCH_SIZE, TEMPORAL_PATCH_SIZE, CHANNEL_COUNT, HIDDEN_SIZE, HEAD_COUNT, HEAD_DIMENSION, ROPE_FREQUENCIES, POSITION_TABLE_ROWS, MAX_PATCH_COUNT });

export interface Qwen35VisionFoundationStorage {
  readonly buffer: Qwen35WebGpuBuffer;
  readonly byteLength: number;
}

export interface Qwen35VisionBootstrapFoundationWorkspace {
  readonly patches: Qwen35VisionFoundationStorage;
  readonly embeddings: Qwen35VisionFoundationStorage;
  readonly rope: Qwen35VisionFoundationStorage;
  /** Each is the executor-owned 16-byte uniform slot for its matching dispatch. */
  readonly patchUniform: Qwen35VisionFoundationStorage;
  readonly positionUniform: Qwen35VisionFoundationStorage;
  readonly ropeUniform: Qwen35VisionFoundationStorage;
}

/** Per-layer Q/K storage; invoke only after that layer's QKV projection. */
export interface Qwen35VisionRopeApplyWorkspace {
  readonly rope: Qwen35VisionFoundationStorage;
  /** Q/K pairs are read before either lane is written, so no duplicate output buffers are needed. */
  readonly query: Qwen35VisionFoundationStorage;
  readonly key: Qwen35VisionFoundationStorage;
  readonly applyUniform: Qwen35VisionFoundationStorage;
}

function storageBinding(
  binding: number,
  value: Qwen35VisionFoundationStorage,
  requiredBytes: number = value.byteLength,
): Qwen35BufferBinding {
  if (
    typeof value.buffer !== "object" || value.buffer === null || value.byteLength < requiredBytes ||
    requiredBytes < 4 || requiredBytes % 4 !== 0
  ) {
    fail("vision-foundation-binding-invalid", "Vision foundation GPU binding is invalid");
  }
  return Object.freeze({ binding, kind: "storage" as const, buffer: value.buffer, offset: 0, size: requiredBytes });
}

function uniformBinding(binding: number, value: Qwen35VisionFoundationStorage): Qwen35BufferBinding {
  if (typeof value.buffer !== "object" || value.buffer === null || value.byteLength !== 16) {
    fail("vision-foundation-binding-invalid", "Vision foundation GPU binding is invalid");
  }
  return Object.freeze({ binding, kind: "uniform" as const, buffer: value.buffer, offset: 0, size: value.byteLength });
}

export interface Qwen35VisionFoundationDispatchPlan extends Qwen35DispatchRequest {
  /** Caller writes these exact u32 values into this dispatch's uniform slot. */
  readonly uniformWords: readonly [number, number, number, number];
}

function validateLimits(limits: Qwen35ForwardDeviceLimits): void {
  for (const value of [
    limits.minStorageBufferOffsetAlignment,
    limits.minUniformBufferOffsetAlignment,
    limits.maxStorageBufferBindingSize,
    limits.maxUniformBufferBindingSize,
    limits.maxComputeWorkgroupsPerDimension,
  ]) requireInteger(value, 1, Number.MAX_SAFE_INTEGER, "vision-foundation-limits-invalid");
}

function validatePlanBindings(
  plans: readonly Qwen35VisionFoundationDispatchPlan[],
  limits: Qwen35ForwardDeviceLimits,
): readonly Qwen35VisionFoundationDispatchPlan[] {
  for (const plan of plans) {
    for (const binding of plan.bindings) {
      const alignment = binding.kind === "storage"
        ? limits.minStorageBufferOffsetAlignment
        : limits.minUniformBufferOffsetAlignment;
      const maximum = binding.kind === "storage"
        ? limits.maxStorageBufferBindingSize
        : limits.maxUniformBufferBindingSize;
      if (binding.offset % alignment !== 0 || binding.size > maximum || binding.size % 4 !== 0) {
        fail("vision-foundation-binding-invalid", "Vision foundation GPU binding is invalid");
      }
    }
  }
  return plans;
}

function tensor(group: Qwen35VisionGpuStagedGroup, name: string): Qwen35VisionGpuTensorView {
  const value = group.tensors.find((candidate) => candidate.name === name);
  if (value === undefined) fail("vision-foundation-tensor-invalid", "Vision foundation tensor view is invalid");
  return value;
}

function tensorBinding(binding: number, value: Qwen35VisionGpuTensorView, expectedBytes: number, segments: number): readonly Qwen35BufferBinding[] {
  if (
    value.segments.length !== segments ||
    value.segments.reduce((sum, segment) => sum + segment.byteLength, 0) !== expectedBytes ||
    value.segments.some((segment, index) =>
      segment.byteLength < 4 || segment.byteLength % 4 !== 0 || segment.bufferOffset % 4 !== 0 ||
      segment.tensorOffset !== value.segments.slice(0, index).reduce((sum, prior) => sum + prior.byteLength, 0),
    )
  ) fail("vision-foundation-tensor-invalid", "Vision foundation tensor view is invalid");
  return Object.freeze(value.segments.map((segment, index) => Object.freeze({
    binding: binding + index,
    kind: "storage" as const,
    buffer: segment.buffer as Qwen35WebGpuBuffer,
    offset: segment.bufferOffset,
    size: segment.byteLength,
  })));
}

function source(operation: string) {
  const definition = QWEN35_VISION_FOUNDATION_KERNELS.find((candidate) => candidate.key.operation === operation);
  if (definition === undefined) fail("vision-foundation-kernel-invalid", "Vision foundation kernel is unavailable");
  return Object.freeze({ id: definition.id, source: definition.source, entryPoint: "main" });
}

/**
 * Assembles fixed bootstrap work. It binds the two physical
 * position-table segments directly, so interpolation never reconstructs a
 * 9 MiB logical table on CPU or in an extra GPU buffer.
 */
export function planQwen35VisionBootstrapFoundationDispatches(input: {
  readonly bootstrap: Qwen35VisionGpuStagedGroup;
  readonly workspace: Qwen35VisionBootstrapFoundationWorkspace;
  readonly gridHeight: number;
  readonly gridWidth: number;
  readonly limits: Qwen35ForwardDeviceLimits;
}): readonly Qwen35VisionFoundationDispatchPlan[] {
  const bootstrap = assertAuthenticatedQwen35VisionGpuStagedGroup(input.bootstrap);
  if (bootstrap.layer !== "bootstrap") {
    fail("vision-foundation-bootstrap-invalid", "Vision foundation requires the bootstrap group");
  }
  const patchCount = requireGrid(input.gridHeight, input.gridWidth);
  validateLimits(input.limits);
  if (
    HIDDEN_SIZE / 64 > input.limits.maxComputeWorkgroupsPerDimension ||
    patchCount > input.limits.maxComputeWorkgroupsPerDimension ||
    HEAD_COUNT > input.limits.maxComputeWorkgroupsPerDimension
  ) {
    fail("vision-foundation-dispatch-invalid", "Vision foundation dispatch exceeds device limits");
  }
  const f32 = Float32Array.BYTES_PER_ELEMENT;
  const patchBytes = patchCount * CHANNEL_COUNT * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE * f32;
  const embeddingBytes = patchCount * HIDDEN_SIZE * f32;
  const ropeBytes = patchCount * ROPE_FREQUENCIES * 2 * f32;
  const requireWorkspace = (storage: Qwen35VisionFoundationStorage, bytes: number): void => {
    if (storage.byteLength < bytes) fail("vision-foundation-workspace-invalid", "Vision foundation workspace is too small");
  };
  requireWorkspace(input.workspace.patches, patchBytes);
  requireWorkspace(input.workspace.embeddings, embeddingBytes);
  requireWorkspace(input.workspace.rope, ropeBytes);
  const patchWeight0 = tensor(bootstrap, "v.patch_embd.weight");
  const patchWeight1 = tensor(bootstrap, "v.patch_embd.weight.1");
  const patchBiasView = tensor(bootstrap, "v.patch_embd.bias");
  const positionView = tensor(bootstrap, "v.position_embd.weight");
  if (
    patchWeight0.precision !== "f32" || patchWeight0.storageType !== "f32" || patchWeight0.orientation.kind !== "patch-conv3d" ||
    patchWeight0.shape.join(",") !== "16,16,3,1024" ||
    patchWeight1.precision !== "f32" || patchWeight1.storageType !== "f32" || patchWeight1.orientation.kind !== "patch-conv3d" ||
    patchWeight1.shape.join(",") !== "16,16,3,1024" ||
    patchBiasView.precision !== "f32" || patchBiasView.storageType !== "f32" || patchBiasView.orientation.kind !== "element-contiguous" ||
    patchBiasView.shape.join(",") !== "1024" ||
    positionView.precision !== "f32" || positionView.storageType !== "f32" || positionView.orientation.kind !== "learned-position-hidden-contiguous" ||
    positionView.shape.join(",") !== "1024,2304"
  ) fail("vision-foundation-tensor-invalid", "Vision foundation tensor view is invalid");
  const patchWeights0 = tensorBinding(1, patchWeight0, PATCH_SIZE * PATCH_SIZE * CHANNEL_COUNT * HIDDEN_SIZE * f32, 1);
  const patchWeights1 = tensorBinding(2, patchWeight1, PATCH_SIZE * PATCH_SIZE * CHANNEL_COUNT * HIDDEN_SIZE * f32, 1);
  const patchBias = tensorBinding(3, patchBiasView, HIDDEN_SIZE * f32, 1);
  const position = tensorBinding(1, positionView, POSITION_TABLE_ROWS * HIDDEN_SIZE * f32, 2);
  const common = Object.freeze({ x: HIDDEN_SIZE / 64, y: patchCount, z: 1 });
  const firstSegmentScalars = positionView.segments[0]!.byteLength / f32;
  if (!Number.isSafeInteger(firstSegmentScalars) || firstSegmentScalars < 1) {
    fail("vision-foundation-tensor-invalid", "Vision foundation tensor view is invalid");
  }
  const plans: readonly Qwen35VisionFoundationDispatchPlan[] = Object.freeze([
    Object.freeze({
      kernel: source("vision-patch-conv3d"), workgroups: common,
      bindings: Object.freeze([
        storageBinding(0, input.workspace.patches, patchBytes), ...patchWeights0, ...patchWeights1, ...patchBias,
        storageBinding(4, input.workspace.embeddings, embeddingBytes), uniformBinding(5, input.workspace.patchUniform),
      ]),
      uniformWords: Object.freeze([patchCount, 0, 0, 0]) as readonly [number, number, number, number],
    }),
    Object.freeze({
      kernel: source("vision-add-learned-position"), workgroups: common,
      bindings: Object.freeze([
        storageBinding(0, input.workspace.embeddings, embeddingBytes), ...position, uniformBinding(3, input.workspace.positionUniform),
      ]),
      uniformWords: Object.freeze([patchCount, input.gridHeight, input.gridWidth, firstSegmentScalars]) as readonly [number, number, number, number],
    }),
    Object.freeze({
      kernel: source("vision-prepare-2d-rope"), workgroups: Object.freeze({ x: 1, y: patchCount, z: 1 }),
      bindings: Object.freeze([storageBinding(0, input.workspace.rope, ropeBytes), uniformBinding(1, input.workspace.ropeUniform)]),
      uniformWords: Object.freeze([patchCount, input.gridHeight, input.gridWidth, 0]) as readonly [number, number, number, number],
    }),
  ]);
  return validatePlanBindings(plans, input.limits);
}

/** Plans Q/K rotation for exactly one transformer layer after its QKV dispatch. */
export function planQwen35VisionLayerRopeApplyDispatch(input: {
  readonly workspace: Qwen35VisionRopeApplyWorkspace;
  readonly patchCount: number;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35VisionFoundationDispatchPlan {
  requireInteger(input.patchCount, 1, MAX_PATCH_COUNT, "vision-foundation-dispatch-invalid");
  validateLimits(input.limits);
  if (input.patchCount > input.limits.maxComputeWorkgroupsPerDimension || HEAD_COUNT > input.limits.maxComputeWorkgroupsPerDimension) {
    fail("vision-foundation-dispatch-invalid", "Vision foundation dispatch exceeds device limits");
  }
  const f32 = Float32Array.BYTES_PER_ELEMENT;
  const vectorBytes = input.patchCount * HEAD_COUNT * HEAD_DIMENSION * f32;
  for (const storage of [input.workspace.query, input.workspace.key]) {
    if (storage.byteLength < vectorBytes) fail("vision-foundation-workspace-invalid", "Vision foundation workspace is too small");
  }
  if (input.workspace.rope.byteLength < input.patchCount * ROPE_FREQUENCIES * 2 * f32) {
    fail("vision-foundation-workspace-invalid", "Vision foundation workspace is too small");
  }
  return validatePlanBindings(Object.freeze([Object.freeze({
    kernel: source("vision-apply-2d-rope"), workgroups: Object.freeze({ x: 1, y: input.patchCount, z: HEAD_COUNT }),
    bindings: Object.freeze([
      storageBinding(0, input.workspace.query, vectorBytes), storageBinding(1, input.workspace.key, vectorBytes), storageBinding(2, input.workspace.rope, input.patchCount * ROPE_FREQUENCIES * 2 * f32),
      uniformBinding(3, input.workspace.applyUniform),
    ]),
    uniformWords: Object.freeze([input.patchCount, 0, 0, 0]) as readonly [number, number, number, number],
  })]), input.limits)[0]!;
}
