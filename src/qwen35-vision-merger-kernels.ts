import { diagnosticError } from "./diagnostics.js";
import type { GpuAllocation, GpuAllocationRequest } from "./gpu-arena.js";
import type { KernelDefinition, KernelRegistry } from "./kernel-registry.js";
import type { Qwen35ForwardDeviceLimits } from "./qwen35-forward-dispatch.js";
import { assertAuthenticatedQwen35VisionGpuStagedGroup, type Qwen35VisionGpuStagedGroup, type Qwen35VisionGpuTensorView } from "./qwen35-vision-gpu-staging.js";
import { QWEN35_VISION_LAYER_KERNELS, visionLayerNormCpu, visionLinearBf16Cpu, type Qwen35VisionLayerStorage } from "./qwen35-vision-layer-kernels.js";
import type { Qwen35BufferBinding, Qwen35DispatchRequest, Qwen35WebGpuBuffer } from "./qwen35-webgpu-executor.js";
import type { Qwen35WebGpuExecutor } from "./qwen35-webgpu-executor.js";

const HIDDEN = 1_024;
const MERGED = 4_096;
const PROJECTED = 2_560;
const MAX_PATCHES = 16_384;
const GPU_BUFFER_USAGE_COPY_DST = 0x0008;
const GPU_BUFFER_USAGE_STORAGE = 0x0080;

function fail(code: string, message: string): never { throw diagnosticError(code, message); }
function f32(value: number): number { return Math.fround(value); }
function validCount(patches: number): number {
  if (!Number.isSafeInteger(patches) || patches < 4 || patches > MAX_PATCHES || patches % 4 !== 0) fail("vision-merger-patches-invalid", "Vision merger patch count is invalid");
  return patches / 4;
}
function erf(value: number): number {
  const sign = value < 0 ? -1 : 1; const x = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * x);
  const polynomial = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return f32(sign * (1 - polynomial * Math.exp(-x * x)));
}

/** Numerically stable CPU reference for the projector's erf GELU operation. */
export function visionExactGeluCpu(values: Float32Array): Float32Array {
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (!Number.isFinite(value)) fail("vision-merger-gelu-invalid", "Vision merger GELU input is invalid");
    output[index] = f32(0.5 * value * (1 + erf(value / Math.SQRT2)));
  }
  return output;
}

/**
 * Fixed CPU merger reference. Rows are already merge-block-major, so four
 * consecutive normalized patch rows are one 4096-wide input row by view only.
 */
export function visionMergerCpu(input: {
  readonly hidden: Float32Array;
  readonly patchCount: number;
  readonly postWeight: Float32Array;
  readonly postBias: Float32Array;
  readonly inputWeight: Uint16Array;
  readonly inputBias: Float32Array;
  readonly outputWeight: Uint16Array;
  readonly outputBias: Float32Array;
}): Float32Array {
  return visionMergerReferenceCpu({ ...input, hiddenSize: HIDDEN, intermediateSize: MERGED, outputSize: PROJECTED });
}

/** Reduced oracle sharing the fixed operation order for pinned PyTorch fixtures. */
export function visionMergerReferenceCpu(input: {
  readonly hidden: Float32Array; readonly patchCount: number; readonly hiddenSize: number; readonly intermediateSize: number; readonly outputSize: number;
  readonly postWeight: Float32Array; readonly postBias: Float32Array; readonly inputWeight: Uint16Array; readonly inputBias: Float32Array; readonly outputWeight: Uint16Array; readonly outputBias: Float32Array;
}): Float32Array {
  const tokens = validCount(input.patchCount);
  const mergedWidth = input.hiddenSize * 4;
  if (input.hidden.length !== input.patchCount * input.hiddenSize || input.postWeight.length !== input.hiddenSize || input.postBias.length !== input.hiddenSize || input.inputWeight.length !== mergedWidth * input.intermediateSize || input.inputBias.length !== input.intermediateSize || input.outputWeight.length !== input.intermediateSize * input.outputSize || input.outputBias.length !== input.outputSize) fail("vision-merger-input-invalid", "Vision merger input is invalid");
  const normalized = visionLayerNormCpu({ input: input.hidden, tokenCount: input.patchCount, hiddenSize: input.hiddenSize, weight: input.postWeight, bias: input.postBias });
  // `normalized` stays contiguous. This is a logical [tokens, 4096] view,
  // not a reshape shader or a second hidden-state allocation.
  const up = visionLinearBf16Cpu({ input: normalized, tokenCount: tokens, inputWidth: mergedWidth, outputWidth: input.intermediateSize, weight: input.inputWeight, bias: input.inputBias });
  return visionLinearBf16Cpu({ input: visionExactGeluCpu(up), tokenCount: tokens, inputWidth: input.intermediateSize, outputWidth: input.outputSize, weight: input.outputWeight, bias: input.outputBias });
}

const EXACT_GELU_WGSL = /* wgsl */ `
struct Params { token_count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
fn erf_approx(x: f32) -> f32 { let sign = select(1.0f, -1.0f, x < 0.0f); let a = abs(x); let t = 1.0f / (1.0f + 0.3275911f * a); let p = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t; return sign * (1.0f - p * exp(-a * a)); }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { let row = id.x; let token = id.y; if (token >= params.token_count || row >= 4096u) { return; } let index = token * 4096u + row; let value = values[index]; values[index] = 0.5f * value * (1.0f + erf_approx(value * 0.7071067811865475f)); }`;

function definition(): KernelDefinition { return Object.freeze({ id: "qwen35-vision-exact-gelu-f32", key: Object.freeze({ operation: "vision-exact-gelu", layout: "f32", phase: "vision", profile: "portable-f32" }), source: EXACT_GELU_WGSL }); }
export const QWEN35_VISION_MERGER_KERNELS: readonly KernelDefinition[] = Object.freeze([definition()]);
export function registerQwen35VisionMergerKernels(registry: Pick<KernelRegistry, "register">): void { for (const kernel of QWEN35_VISION_MERGER_KERNELS) registry.register(kernel); }

export interface Qwen35VisionMergerWorkspace {
  readonly hidden: Qwen35VisionLayerStorage;
  readonly normalized: Qwen35VisionLayerStorage;
  readonly intermediate: Qwen35VisionLayerStorage;
  readonly projected: Qwen35VisionLayerStorage;
  readonly uniforms: readonly Qwen35VisionLayerStorage[];
}
export interface Qwen35VisionMergerDispatchPlan extends Qwen35DispatchRequest { readonly uniformWords: readonly [number, number, number, number]; }

function storage(binding: number, value: Qwen35VisionLayerStorage, required: number, limits: Qwen35ForwardDeviceLimits): Qwen35BufferBinding {
  const offset = value.byteOffset ?? 0;
  if (typeof value.buffer !== "object" || value.buffer === null || !Number.isSafeInteger(value.byteLength) || value.byteLength < required || !Number.isSafeInteger(offset) || offset < 0 || offset % limits.minStorageBufferOffsetAlignment !== 0 || required % 4 !== 0 || required > limits.maxStorageBufferBindingSize) fail("vision-merger-binding-invalid", "Vision merger GPU binding is invalid");
  return Object.freeze({ binding, kind: "storage", buffer: value.buffer, offset, size: required });
}
function uniform(binding: number, value: Qwen35VisionLayerStorage, limits: Qwen35ForwardDeviceLimits): Qwen35BufferBinding {
  const offset = value.byteOffset ?? 0;
  if (typeof value.buffer !== "object" || value.buffer === null || value.byteLength !== 16 || !Number.isSafeInteger(offset) || offset < 0 || offset % limits.minUniformBufferOffsetAlignment !== 0 || 16 > limits.maxUniformBufferBindingSize) fail("vision-merger-uniform-invalid", "Vision merger uniform is invalid");
  return Object.freeze({ binding, kind: "uniform", buffer: value.buffer, offset, size: 16 });
}
function tensor(group: Qwen35VisionGpuStagedGroup, name: string, shape: readonly number[], precision: "f32" | "bf16", orientation: "element-contiguous" | "input-width-contiguous"): Qwen35VisionGpuTensorView {
  const value = group.tensors.find((candidate) => candidate.name === name);
  const expectedBytes = shape.reduce((total, dimension) => total * dimension, 1) * (precision === "f32" ? 4 : 2);
  const segment = value?.segments[0];
  if (value === undefined || value.shape.join(",") !== shape.join(",") || value.precision !== precision || value.storageType !== (precision === "f32" ? "f32" : "raw") || value.orientation.kind !== orientation || value.segments.length !== 1 || (orientation === "element-contiguous" && value.orientation.contiguousDimension !== "element") || (orientation === "input-width-contiguous" && (value.orientation.contiguousDimension !== "input-width" || value.orientation.manifestShape.join(",") !== "input-width,output-rows")) || segment === undefined || segment.tensorOffset !== 0 || segment.byteLength !== expectedBytes || !Number.isSafeInteger(segment.bufferOffset) || segment.bufferOffset < 0 || segment.bufferOffset % 4 !== 0) fail("vision-merger-tensor-invalid", "Vision merger tensor ABI is invalid");
  return value;
}
function weight(binding: number, value: Qwen35VisionGpuTensorView, limits: Qwen35ForwardDeviceLimits): Qwen35BufferBinding {
  const segment = value.segments[0]!;
  if (segment.bufferOffset % limits.minStorageBufferOffsetAlignment !== 0 || segment.byteLength > limits.maxStorageBufferBindingSize || segment.byteLength % 4 !== 0) fail("vision-merger-binding-invalid", "Vision merger tensor binding is invalid");
  return Object.freeze({ binding, kind: "storage", buffer: segment.buffer as Qwen35WebGpuBuffer, offset: segment.bufferOffset, size: segment.byteLength });
}
function kernel(operation: string): { readonly id: string; readonly source: string; readonly entryPoint: string } {
  const found = [...QWEN35_VISION_LAYER_KERNELS, ...QWEN35_VISION_MERGER_KERNELS].find((item) => item.key.operation === operation);
  if (found === undefined) fail("vision-merger-kernel-invalid", "Vision merger kernel is unavailable");
  return Object.freeze({ id: found.id, source: found.source, entryPoint: "main" });
}

/** Plans post-LN, logical contiguous 2x2 view, mm.0, exact GELU, and mm.2. */
export function planQwen35VisionMergerDispatches(input: { readonly bootstrap: Qwen35VisionGpuStagedGroup; readonly workspace: Qwen35VisionMergerWorkspace; readonly patchCount: number; readonly limits: Qwen35ForwardDeviceLimits }): readonly Qwen35VisionMergerDispatchPlan[] {
  const bootstrap = assertAuthenticatedQwen35VisionGpuStagedGroup(input.bootstrap); const tokenCount = validCount(input.patchCount);
  if (bootstrap.layer !== "bootstrap" || input.workspace.uniforms.length !== 4) fail("vision-merger-invalid", "Vision merger input is invalid");
  // Capability flags share the planner profile but are not numeric WebGPU limits.
  for (const limit of Object.values(input.limits)) if (typeof limit === "number" && (!Number.isSafeInteger(limit) || limit < 1)) fail("vision-merger-limits-invalid", "Vision merger limits are invalid");
  const seenUniforms = new Map<object, Set<number>>();
  for (const slot of input.workspace.uniforms) {
    const bound = uniform(0, slot, input.limits); const buffer = bound.buffer as object;
    const offsets = seenUniforms.get(buffer);
    if (offsets?.has(bound.offset)) fail("vision-merger-uniform-invalid", "Vision merger uniforms must be distinct");
    if (offsets === undefined) seenUniforms.set(buffer, new Set([bound.offset])); else offsets.add(bound.offset);
  }
  if (input.patchCount > input.limits.maxComputeWorkgroupsPerDimension || tokenCount > input.limits.maxComputeWorkgroupsPerDimension || MERGED / 64 > input.limits.maxComputeWorkgroupsPerDimension || PROJECTED / 64 > input.limits.maxComputeWorkgroupsPerDimension) fail("vision-merger-dispatch-invalid", "Vision merger dispatch exceeds device limits");
  const postWeight = tensor(bootstrap, "v.post_ln.weight", [HIDDEN], "f32", "element-contiguous"); const postBias = tensor(bootstrap, "v.post_ln.bias", [HIDDEN], "f32", "element-contiguous");
  const inputWeight = tensor(bootstrap, "mm.0.weight", [MERGED, MERGED], "bf16", "input-width-contiguous"); const inputBias = tensor(bootstrap, "mm.0.bias", [MERGED], "f32", "element-contiguous");
  const outputWeight = tensor(bootstrap, "mm.2.weight", [MERGED, PROJECTED], "bf16", "input-width-contiguous"); const outputBias = tensor(bootstrap, "mm.2.bias", [PROJECTED], "f32", "element-contiguous");
  const hiddenBytes = input.patchCount * HIDDEN * 4; const intermediateBytes = tokenCount * MERGED * 4; const projectedBytes = tokenCount * PROJECTED * 4;
  const plan = (operation: string, bindings: readonly Qwen35BufferBinding[], workgroups: { readonly x: number; readonly y: number; readonly z: number }, uniformWords: readonly [number, number, number, number]): Qwen35VisionMergerDispatchPlan => Object.freeze({ kernel: kernel(operation), bindings: Object.freeze(bindings), workgroups: Object.freeze(workgroups), uniformWords: Object.freeze(uniformWords) as readonly [number, number, number, number] });
  const epsilon = new DataView(new ArrayBuffer(4)); epsilon.setFloat32(0, 0.000001, true);
  return Object.freeze([
    plan("vision-layernorm", [storage(0, input.workspace.hidden, hiddenBytes, input.limits), weight(1, postWeight, input.limits), weight(2, postBias, input.limits), storage(3, input.workspace.normalized, hiddenBytes, input.limits), uniform(4, input.workspace.uniforms[0]!, input.limits)], { x: 1, y: input.patchCount, z: 1 }, [input.patchCount, HIDDEN, epsilon.getUint32(0, true), 0]),
    plan("vision-bf16-linear", [storage(0, input.workspace.normalized, hiddenBytes, input.limits), weight(1, inputWeight, input.limits), weight(2, inputBias, input.limits), storage(3, input.workspace.intermediate, intermediateBytes, input.limits), uniform(4, input.workspace.uniforms[1]!, input.limits)], { x: MERGED / 64, y: tokenCount, z: 1 }, [tokenCount, MERGED, MERGED, 0]),
    plan("vision-exact-gelu", [storage(0, input.workspace.intermediate, intermediateBytes, input.limits), uniform(1, input.workspace.uniforms[2]!, input.limits)], { x: MERGED / 64, y: tokenCount, z: 1 }, [tokenCount, 0, 0, 0]),
    plan("vision-bf16-linear", [storage(0, input.workspace.intermediate, intermediateBytes, input.limits), weight(1, outputWeight, input.limits), weight(2, outputBias, input.limits), storage(3, input.workspace.projected, projectedBytes, input.limits), uniform(4, input.workspace.uniforms[3]!, input.limits)], { x: PROJECTED / 64, y: tokenCount, z: 1 }, [tokenCount, MERGED, PROJECTED, 0]),
  ]);
}

export interface Qwen35VisionProjectedOutput { readonly storage: Qwen35VisionLayerStorage; dispose(): Promise<void>; }
export async function createQwen35VisionProjectedOutput(input: { readonly arena: { allocate(request: GpuAllocationRequest): Promise<GpuAllocation> }; readonly queue: { onSubmittedWorkDone(): Promise<void> }; readonly visualTokenCount: number; readonly maxStorageBufferBindingSize: number; readonly allocationId: string }): Promise<Qwen35VisionProjectedOutput> {
  if (!Number.isSafeInteger(input.visualTokenCount) || input.visualTokenCount < 1 || input.visualTokenCount > MAX_PATCHES / 4 || !Number.isSafeInteger(input.maxStorageBufferBindingSize) || input.maxStorageBufferBindingSize < 1 || input.allocationId.trim().length === 0) fail("vision-projected-output-invalid", "Vision projected output options are invalid");
  const bytes = input.visualTokenCount * PROJECTED * 4;
  if (bytes > input.maxStorageBufferBindingSize) fail("vision-projected-output-invalid", "Vision projected output options are invalid");
  const allocation = await input.arena.allocate({ id: input.allocationId, category: "activation", byteLength: BigInt(bytes), usage: GPU_BUFFER_USAGE_COPY_DST | GPU_BUFFER_USAGE_STORAGE, alignment: 4, requiredShardQuantumBytes: BigInt(bytes) });
  if (allocation.shards.length !== 1 || allocation.logicalBytes !== BigInt(bytes) || allocation.shards[0]!.logicalByteOffset !== 0n || allocation.shards[0]!.logicalByteLength !== BigInt(bytes) || allocation.shards[0]!.allocatedByteLength < BigInt(bytes)) { allocation.destroy(); fail("vision-projected-output-invalid", "Vision projected output must be one bindable range"); }
  let disposal: Promise<void> | null = null;
  return Object.freeze({ storage: Object.freeze({ buffer: allocation.shards[0]!.buffer, byteLength: bytes }), dispose: () => {
    if (disposal !== null) return disposal;
    disposal = (async () => { let first: unknown; try { await input.queue.onSubmittedWorkDone(); } catch (error) { first ??= error; } try { allocation.destroy(); } catch (error) { first ??= error; } if (first !== undefined) throw first; })(); return disposal;
  } });
}

export interface Qwen35VisionMergerUniformSlot { update(words: Uint32Array): void; }
function aborted(signal: AbortSignal): Error | null { return signal.aborted ? Object.assign(new Error("Vision merger cancelled"), { name: "AbortError" }) : null; }

/** Executes one already-authenticated merger plan. Bootstrap lifetime stays with its caller. */
export async function executeQwen35VisionMerger(input: {
  readonly plans: readonly Qwen35VisionMergerDispatchPlan[];
  readonly uniforms: readonly Qwen35VisionMergerUniformSlot[];
  readonly gpu: Pick<Qwen35WebGpuExecutor, "dispatchBatch" | "submittedWorkDone" | "dispose">;
  readonly projected: Qwen35VisionProjectedOutput;
  readonly signal?: AbortSignal;
}): Promise<Qwen35VisionProjectedOutput> {
  const signal = input.signal ?? new AbortController().signal;
  let primary: unknown;
  let submissionAttempted = false;
  let retirementSucceeded = false;
  try {
    if (input.plans.length !== 4 || input.uniforms.length !== 4) fail("vision-merger-execution-invalid", "Vision merger execution plan is invalid");
    const before = aborted(signal); if (before !== null) throw before;
    for (let index = 0; index < 4; index += 1) input.uniforms[index]!.update(Uint32Array.from(input.plans[index]!.uniformWords));
    const afterUniforms = aborted(signal); if (afterUniforms !== null) throw afterUniforms;
    submissionAttempted = true;
    await input.gpu.dispatchBatch(input.plans);
    await input.gpu.submittedWorkDone();
    retirementSucceeded = true;
    const afterRetirement = aborted(signal); if (afterRetirement !== null) throw afterRetirement;
    return input.projected;
  } catch (error) {
    primary = error;
    // Queue ownership is uncertain only after a submitted dispatch or failed
    // retirement. Validation, upload, and post-retirement cancellation leave
    // the shared executor reusable.
    if (submissionAttempted && !retirementSucceeded) {
      try { await input.gpu.dispose(); } catch { /* Preserve compute failure. */ }
    }
    try { await input.projected.dispose(); } catch { /* Preserve compute failure. */ }
    throw primary;
  }
}
