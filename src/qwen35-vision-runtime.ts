import { AllocationLedger } from "./allocation-ledger.js";
import { diagnosticError } from "./diagnostics.js";
import { GpuArena, type GpuAllocation } from "./gpu-arena.js";
import type { DeviceProfile } from "./device-profile.js";
import type { Qwen35ForwardBufferSlice, Qwen35ForwardDeviceLimits } from "./qwen35-forward-dispatch.js";
import {
  planQwen35VisionBootstrapFoundationDispatches,
  type Qwen35VisionBootstrapFoundationWorkspace,
  type Qwen35VisionFoundationDispatchPlan,
} from "./qwen35-vision-foundation-kernels.js";
import {
  createQwen35VisionStreamingExecutor,
  createQwen35VisionStreamingOwnedActivationWorkspace,
  createQwen35VisionStreamingOwnedResources,
  type Qwen35VisionActivationQueue,
  type Qwen35VisionStreamingExecutor,
  type Qwen35VisionStreamingOwnedActivationWorkspace,
  type Qwen35VisionStreamingOwnedResources,
} from "./qwen35-vision-streaming-executor.js";
import {
  createQwen35VisionProjectedOutput,
  executeQwen35VisionMerger,
  planQwen35VisionMergerDispatches,
  type Qwen35VisionMergerDispatchPlan,
  type Qwen35VisionProjectedOutput,
} from "./qwen35-vision-merger-kernels.js";
import {
  stageQwen35VisionGpuGroup,
  type Qwen35VisionGpuAllocator,
  type Qwen35VisionGpuQueue,
  type Qwen35VisionGpuStagedGroup,
} from "./qwen35-vision-gpu-staging.js";
import { loadProductionQwen35VisionPackage } from "./qwen35-vision-package-bootstrap.js";
import type { Qwen35IntegrityValidatedVisionPackage, Qwen35VisionPackagePins } from "./qwen35-vision-package-loader.js";
import { createQwen35VisionProgram, type Qwen35VisionProgram } from "./qwen35-vision-program.js";
import {
  Qwen35VisionEncoder,
  type Qwen35VisionEncoderBootstrap,
  type Qwen35VisionProjectedTokens,
} from "./qwen35-vision-encoder.js";
import type { Qwen35WebGpuDevice, Qwen35WebGpuExecutor } from "./qwen35-webgpu-executor.js";
import type { Qwen35PerformanceWriteQueue } from "./qwen35-performance.js";
import { createQwen35UniformArena, type Qwen35UniformArena } from "./qwen35-uniform-arena.js";

const GPU_BUFFER_USAGE_COPY_DST = 0x0008;
const GPU_BUFFER_USAGE_STORAGE = 0x0080;
const VISION_PATCH_SCALARS = 3 * 2 * 16 * 16;
const VISION_UNIFORM_SLOTS = 17;
const VISION_UNIFORM_WORDS = 4;
const VISION_LAYER_UNIFORM_OFFSET = 3;

type SharedGpuExecutor = Pick<
  Qwen35WebGpuExecutor,
  "dispatchBatch" | "submittedWorkDone" | "releaseBindGroups" | "dispose"
>;

export interface Qwen35VisionRuntimeInput {
  readonly device: Qwen35WebGpuDevice;
  readonly profile: Pick<DeviceProfile, "uploadLaneBytes" | "bufferShardCapBytes">;
  readonly arena: GpuArena;
  readonly ledger: AllocationLedger;
  readonly executor: SharedGpuExecutor;
  readonly performanceQueue?: Qwen35PerformanceWriteQueue;
  readonly fetchImplementation?: typeof fetch;
  readonly visionPackagePins?: Qwen35VisionPackagePins;
}

export interface Qwen35VisionRuntime {
  encode(input: {
    readonly patches: Float32Array;
    readonly gridHeight: number;
    readonly gridWidth: number;
    readonly signal: AbortSignal;
  }): Promise<{
    readonly tokenCount: number;
    readonly storage: Qwen35ForwardBufferSlice;
    dispose(): Promise<void>;
  }>;
  dispose(): Promise<void>;
}

interface VisionWorkspace {
  readonly patchAllocation: GpuAllocation;
  readonly patchStorage: { readonly buffer: object; readonly byteLength: number };
  readonly activation: Qwen35VisionStreamingOwnedActivationWorkspace;
  readonly streamingResources: Qwen35VisionStreamingOwnedResources;
  readonly uniformArena: Qwen35UniformArena;
}

interface VisionOperation {
  readonly encoder: Qwen35VisionEncoder<Qwen35VisionFoundationDispatchPlan, Qwen35VisionMergerDispatchPlan>;
  dispose(): Promise<void>;
}

function fail(code: string, message: string): never {
  throw diagnosticError(code, message);
}

function limits(device: Qwen35WebGpuDevice): Qwen35ForwardDeviceLimits {
  return Object.freeze({
    minStorageBufferOffsetAlignment: device.limits.minStorageBufferOffsetAlignment,
    minUniformBufferOffsetAlignment: device.limits.minUniformBufferOffsetAlignment,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxUniformBufferBindingSize: device.limits.maxUniformBufferBindingSize,
    maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
  });
}

function queueForVision(
  device: Qwen35WebGpuDevice,
  performanceQueue?: Qwen35PerformanceWriteQueue,
): Qwen35VisionGpuQueue {
  const queue = performanceQueue ?? device.queue;
  return {
    writeBuffer(buffer, bufferOffset, data, dataOffset = 0, size = data.byteLength - dataOffset) {
      queue.writeBuffer(buffer, bufferOffset, data.buffer as ArrayBuffer, data.byteOffset + dataOffset, size);
    },
    onSubmittedWorkDone: () => queue.onSubmittedWorkDone(),
  };
}

function activationQueueForVision(
  device: Qwen35WebGpuDevice,
  performanceQueue?: Qwen35PerformanceWriteQueue,
): Qwen35VisionActivationQueue {
  const queue = performanceQueue ?? device.queue;
  return {
    writeBuffer(buffer, bufferOffset, data, dataOffset, size) {
      queue.writeBuffer(buffer, bufferOffset, data, dataOffset, size);
    },
    onSubmittedWorkDone: () => queue.onSubmittedWorkDone(),
  };
}

function allocatorForVision(device: Qwen35WebGpuDevice): Qwen35VisionGpuAllocator {
  return {
    createBuffer(descriptor) {
      return device.createBuffer(descriptor);
    },
  };
}

function forwardSlice(
  buffer: object,
  offset: number,
  byteLength: number,
): Qwen35ForwardBufferSlice {
  return Object.freeze({ buffer, offset, byteLength });
}

function storageSlice(value: {
  readonly buffer: object;
  readonly byteLength?: number;
  readonly byteOffset?: number;
  readonly size?: number;
  readonly offset?: number;
}): {
  readonly buffer: object;
  readonly byteLength: number;
  readonly byteOffset?: number;
} {
  const byteLength = value.byteLength ?? value.size;
  const byteOffset = value.byteOffset ?? value.offset;
  if (byteLength === undefined) {
    fail("vision-runtime-storage-slice-invalid", "Vision storage slice is missing its byte length");
  }
  return Object.freeze({
    buffer: value.buffer,
    byteLength,
    ...(byteOffset === undefined ? {} : { byteOffset }),
  });
}


function singleRange(allocation: GpuAllocation, expectedBytes: number, code: string): {
  readonly buffer: object;
  readonly byteLength: number;
} {
  const shard = allocation.shards[0];
  if (
    allocation.shards.length !== 1 || shard === undefined ||
    allocation.logicalBytes !== BigInt(expectedBytes) ||
    shard.logicalByteOffset !== 0n || shard.logicalByteLength !== BigInt(expectedBytes) ||
    shard.allocatedByteLength < BigInt(expectedBytes)
  ) {
    try { allocation.destroy(); } catch { /* Preserve the stable ABI error. */ }
    fail(code, "Vision allocation is not one bindable range");
  }
  return Object.freeze({ buffer: shard.buffer, byteLength: expectedBytes });
}

async function allocatePatches(input: {
  readonly arena: GpuArena;
  readonly patchBytes: number;
  readonly allocationId: string;
}): Promise<{ readonly allocation: GpuAllocation; readonly storage: { readonly buffer: object; readonly byteLength: number } }> {
  let allocation: GpuAllocation;
  try {
    allocation = await input.arena.allocate({
      id: input.allocationId,
      category: "activation",
      byteLength: BigInt(input.patchBytes),
      usage: GPU_BUFFER_USAGE_COPY_DST | GPU_BUFFER_USAGE_STORAGE,
      alignment: 4,
      requiredShardQuantumBytes: BigInt(input.patchBytes),
    });
  } catch {
    fail("vision-patches-allocation-failed", "Vision patch allocation failed");
  }
  return { allocation, storage: singleRange(allocation, input.patchBytes, "vision-patches-allocation-invalid") };
}

async function createWorkspace(input: {
  readonly arena: GpuArena;
  readonly queue: Qwen35VisionActivationQueue;
  readonly gridHeight: number;
  readonly gridWidth: number;
  readonly device: Qwen35WebGpuDevice;
  readonly allocationIdPrefix: string;
}): Promise<VisionWorkspace> {
  const patchCount = input.gridHeight * input.gridWidth;
  let activation: Qwen35VisionStreamingOwnedActivationWorkspace | null = null;
  let uniformArena: Qwen35UniformArena | null = null;
  let patches: { readonly allocation: GpuAllocation; readonly storage: { readonly buffer: object; readonly byteLength: number } } | null = null;
  try {
    activation = await createQwen35VisionStreamingOwnedActivationWorkspace({
      arena: input.arena,
      queue: input.queue,
      tokenCount: patchCount,
      segmentOffsets: new Uint32Array([0, patchCount]),
      minStorageBufferOffsetAlignment: input.device.limits.minStorageBufferOffsetAlignment,
      maxStorageBufferBindingSize: input.device.limits.maxStorageBufferBindingSize,
      allocationIdPrefix: `${input.allocationIdPrefix}-activations`,
    });
    uniformArena = await createQwen35UniformArena({
      arena: input.arena,
      queue: input.queue,
      allocationId: "qwen35-vision-uniform-arena",
      slotCount: VISION_UNIFORM_SLOTS,
      slotWordCapacity: VISION_UNIFORM_WORDS,
      minUniformBufferOffsetAlignment: input.device.limits.minUniformBufferOffsetAlignment,
      maxUniformBufferBindingSize: input.device.limits.maxUniformBufferBindingSize,
    });
    const patchBytes = patchCount * VISION_PATCH_SCALARS * Float32Array.BYTES_PER_ELEMENT;
    patches = await allocatePatches({ arena: input.arena, patchBytes, allocationId: `${input.allocationIdPrefix}-patches` });
    const streamingResources = createQwen35VisionStreamingOwnedResources({
      activationWorkspace: activation,
      uniformArena,
      uniformSlotOffset: VISION_LAYER_UNIFORM_OFFSET,
    });
    return Object.freeze({ patchAllocation: patches.allocation, patchStorage: patches.storage, activation, streamingResources, uniformArena });
  } catch (error) {
    try { await uniformArena?.dispose(); } catch { /* Preserve the allocation failure. */ }
    try { await activation?.dispose(); } catch { /* Preserve the allocation failure. */ }
    if (patches === null) {
      // The patch allocator owns rollback on its own rejection.
    } else {
      try { patches.allocation.destroy(); } catch { /* Preserve the allocation failure. */ }
    }
    throw error;
  }
}

function foundationWorkspace(workspace: VisionWorkspace): Qwen35VisionBootstrapFoundationWorkspace {
  const hidden = storageSlice(workspace.activation.hidden);
  const rope = storageSlice(workspace.activation.rope);
  const patches = storageSlice(workspace.patchStorage);
  const slot = (index: number) => storageSlice(workspace.uniformArena.slot(index, VISION_UNIFORM_WORDS).binding);
  return Object.freeze({ patches, embeddings: hidden, rope, patchUniform: slot(0), positionUniform: slot(1), ropeUniform: slot(2) });
}

function activationWorkspace(workspace: VisionWorkspace): {
  readonly hidden: { readonly buffer: object; readonly byteLength: number };
  readonly normalized: { readonly buffer: object; readonly byteLength: number };
  readonly intermediate: { readonly buffer: object; readonly byteLength: number };
  readonly projected: { readonly buffer: object; readonly byteLength: number; readonly byteOffset?: number };
} {
  return Object.freeze({
    hidden: storageSlice(workspace.activation.hidden),
    normalized: storageSlice(workspace.activation.normalized),
    intermediate: storageSlice(workspace.activation.mlp),
    projected: storageSlice(workspace.activation.hidden),
  });
}

function asProjectedTokens(projected: Qwen35VisionProjectedOutput, tokenCount: number): Qwen35VisionProjectedTokens {
  const offset = projected.storage.byteOffset ?? 0;
  return Object.freeze({
    tokenCount,
    storage: forwardSlice(projected.storage.buffer, offset, projected.storage.byteLength),
    dispose: () => projected.dispose(),
  });
}

/**
 * Composes the authenticated foundation, streamed layers, and merger over one
 * shared language executor. It deliberately has no generic tensor scheduler.
 */
export async function createQwen35VisionOperation(input: {
  readonly package: Qwen35IntegrityValidatedVisionPackage;
  readonly program: Qwen35VisionProgram;
  readonly runtime: Qwen35VisionRuntimeInput;
  readonly patches: Float32Array;
  readonly gridHeight: number;
  readonly gridWidth: number;
  readonly signal: AbortSignal;
}): Promise<VisionOperation> {
  const { runtime } = input;
  const patchCount = input.gridHeight * input.gridWidth;
  if (!Number.isSafeInteger(patchCount) || patchCount < 4 || patchCount > 16_384 || patchCount % 4 !== 0 || input.patches.length !== patchCount * VISION_PATCH_SCALARS) {
    fail("vision-runtime-input-invalid", "Vision runtime input is invalid");
  }
  const queue = queueForVision(runtime.device, runtime.performanceQueue);
  const activationQueue = activationQueueForVision(runtime.device, runtime.performanceQueue);
  const workspace = await createWorkspace({
    arena: runtime.arena,
    queue: activationQueue,
    gridHeight: input.gridHeight,
    gridWidth: input.gridWidth,
    device: runtime.device,
    allocationIdPrefix: "qwen35-vision-image",
  });
  const patchStorage = singleRange(workspace.patchAllocation, input.patches.byteLength, "vision-patches-workspace-invalid");
  let bootstrap: Qwen35VisionGpuStagedGroup | null = null;
  let streaming: Qwen35VisionStreamingExecutor | null = null;
  let projected: Qwen35VisionProjectedOutput | null = null;
  let patchReleased = false;
  let bootstrapReleased = false;
  let releasePromise: Promise<void> | null = null;
  const foundationSlots = [0, 1, 2].map((index) => workspace.uniformArena.slot(index, VISION_UNIFORM_WORDS));
  const mergerSlots = [13, 14, 15, 16].map((index) => workspace.uniformArena.slot(index, VISION_UNIFORM_WORDS));
  const limitsView = limits(runtime.device);
  const allocator = allocatorForVision(runtime.device);
  const stageBootstrap = async (signal: AbortSignal): Promise<Qwen35VisionEncoderBootstrap> => {
    bootstrap = await stageQwen35VisionGpuGroup({
      package: input.package,
      program: input.program,
      layer: "bootstrap",
      ledger: runtime.ledger,
      allocator,
      queue,
      allocationId: "qwen35-vision-bootstrap",
      uploadLaneBytes: runtime.profile.uploadLaneBytes,
      bufferCapBytes: runtime.profile.bufferShardCapBytes,
      signal,
    });
    return bootstrap;
  };
  const uploadPatches = async (patches: Float32Array, signal: AbortSignal): Promise<void> => {
    signal.throwIfAborted();
    if (patches.byteLength !== input.patches.byteLength) fail("vision-patches-input-invalid", "Vision patch byte length changed");
    const patchQueue = runtime.performanceQueue ?? runtime.device.queue;
    patchQueue.writeBuffer(patchStorage.buffer, 0, patches.buffer as ArrayBuffer, patches.byteOffset, patches.byteLength);
    await patchQueue.onSubmittedWorkDone();
  };
  const foundation = (staged: Qwen35VisionGpuStagedGroup): readonly Qwen35VisionFoundationDispatchPlan[] => planQwen35VisionBootstrapFoundationDispatches({ bootstrap: staged, workspace: foundationWorkspace(workspace), gridHeight: input.gridHeight, gridWidth: input.gridWidth, limits: limitsView });
  const runFoundation = async (plans: readonly Qwen35VisionFoundationDispatchPlan[], signal: AbortSignal): Promise<void> => {
    signal.throwIfAborted();
    for (const [index, plan] of plans.entries()) foundationSlots[index]!.update(Uint32Array.from(plan.uniformWords));
    await runtime.executor.dispatchBatch(plans);
    await runtime.executor.submittedWorkDone();
    signal.throwIfAborted();
  };
  const runLayers = async (phase: { readonly bootstrap: Qwen35VisionEncoderBootstrap; readonly signal: AbortSignal }): Promise<void> => {
    const staged = phase.bootstrap as Qwen35VisionGpuStagedGroup;
    streaming = createQwen35VisionStreamingExecutor({
      package: input.package,
      program: input.program,
      ledger: runtime.ledger,
      allocator,
      queue,
      allocationIdPrefix: "qwen35-vision",
      limits: limitsView,
      resources: workspace.streamingResources,
      gpu: runtime.executor,
      disposeGpu: false,
    });
    await streaming.run({ tokenCount: patchCount, segmentCount: 1, signal: phase.signal });
    // Keep the streamed activation workspace live until the merger consumes it.
    void staged;
  };
  const createProjected = async (phase: { readonly visualTokenCount: number }): Promise<Qwen35VisionProjectedTokens> => {
    projected = await createQwen35VisionProjectedOutput({
      arena: runtime.arena,
      queue,
      visualTokenCount: phase.visualTokenCount,
      maxStorageBufferBindingSize: runtime.device.limits.maxStorageBufferBindingSize,
      allocationId: "qwen35-vision-projected",
    });
    return asProjectedTokens(projected, phase.visualTokenCount);
  };
  const merger = (staged: Qwen35VisionGpuStagedGroup, output: Qwen35VisionProjectedTokens): readonly Qwen35VisionMergerDispatchPlan[] => {
    if (projected === null) fail("vision-merger-output-invalid", "Vision merger output is unavailable");
    const activation = activationWorkspace(workspace);
    return planQwen35VisionMergerDispatches({
      bootstrap: staged,
      patchCount,
      limits: limitsView,
      workspace: {
        hidden: activation.hidden,
        normalized: activation.normalized,
        intermediate: activation.intermediate,
        projected: storageSlice(projected.storage),
        uniforms: mergerSlots.map((slot) => storageSlice(slot.binding)),
      },
    });
  };
  const runMerger = async (plans: readonly Qwen35VisionMergerDispatchPlan[], output: Qwen35VisionProjectedTokens, signal: AbortSignal): Promise<Qwen35VisionProjectedTokens> => {
    if (projected === null) fail("vision-merger-output-invalid", "Vision merger output is unavailable");
    await executeQwen35VisionMerger({ plans, uniforms: mergerSlots, gpu: runtime.executor, projected, signal });
    return output;
  };
  const releaseTransient = (): Promise<void> => {
    if (releasePromise !== null) return releasePromise;
    const work = (async () => {
      let first: unknown;
      try { await streaming?.dispose(); } catch (error) { first ??= error; }
      try { await queue.onSubmittedWorkDone(); } catch (error) { first ??= error; }
      try { runtime.executor.releaseBindGroups(); } catch (error) { first ??= error; }
      if (!patchReleased) {
        try {
          workspace.patchAllocation.destroy();
          patchReleased = true;
        } catch (error) { first ??= error; }
      }
      if (!bootstrapReleased && bootstrap !== null) {
        try {
          await bootstrap.destroy();
          bootstrapReleased = true;
        } catch (error) { first ??= error; }
      }
      if (first !== undefined) throw first;
    })();
    releasePromise = work;
    void work.catch(() => {
      if (releasePromise === work) releasePromise = null;
    });
    return work;
  };
  const encoder = new Qwen35VisionEncoder<Qwen35VisionFoundationDispatchPlan, Qwen35VisionMergerDispatchPlan>({
    stageBootstrap,
    uploadPatches,
    planFoundation: foundation,
    runFoundation,
    runLayers,
    createProjectedOutput: createProjected,
    planMerger: merger,
    runMerger,
    releaseTransient,
  });
  return Object.freeze({
    encoder,
    dispose: async () => {
      let first: unknown;
      try { await encoder.dispose(); } catch (error) { first ??= error; }
      try { await releaseTransient(); } catch (error) { first ??= error; }
      if (first !== undefined) throw first;
    },
  });
}

/** Lazy metadata bootstrap keeps text-only model loads free of projector fetches. */
export function createQwen35LazyVisionRuntime(input: Qwen35VisionRuntimeInput): Qwen35VisionRuntime {
  let packagePromise: Promise<{ readonly package: Qwen35IntegrityValidatedVisionPackage; readonly program: Qwen35VisionProgram }> | null = null;
  let active: VisionOperation | null = null;
  let busy = false;
  let disposed = false;
  return {
    async encode(request) {
      if (disposed) fail("vision-runtime-disposed", "Vision runtime is disposed");
      if (busy) fail("vision-runtime-busy", "Vision runtime already has an active image");
      busy = true;
      try {
        const packageOptions = input.fetchImplementation === undefined
          ? { signal: request.signal, ...(input.visionPackagePins === undefined ? {} : { pins: input.visionPackagePins }) }
          : { fetchImplementation: input.fetchImplementation, signal: request.signal, ...(input.visionPackagePins === undefined ? {} : { pins: input.visionPackagePins }) };
        packagePromise ??= loadProductionQwen35VisionPackage(packageOptions).then((package_) => Object.freeze({ package: package_, program: createQwen35VisionProgram(package_) }));
        const loaded = await packagePromise;
        request.signal.throwIfAborted();
        const operation = await createQwen35VisionOperation({ ...request, package: loaded.package, program: loaded.program, runtime: input });
        active = operation;
        try {
          const output = await operation.encoder.encode({ patches: request.patches, gridHeight: request.gridHeight, gridWidth: request.gridWidth, signal: request.signal });
          active = null;
          return Object.freeze({
            tokenCount: output.tokenCount,
            storage: forwardSlice(output.storage.buffer, 0, output.storage.byteLength),
            dispose: output.dispose,
          });
        } catch (error) {
          try { await operation.dispose(); } catch { /* Preserve the inference error. */ }
          active = null;
          throw error;
        }
      } finally {
        busy = false;
      }
    },
    async dispose() {
      disposed = true;
      const operation = active;
      active = null;
      await operation?.dispose();
    },
  };
}
