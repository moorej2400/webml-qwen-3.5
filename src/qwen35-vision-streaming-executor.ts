import { diagnosticError } from "./diagnostics.js";
import {
  planQwen35VisionLayerDispatches,
  type Qwen35VisionLayerDispatchPlan,
  type Qwen35VisionLayerWorkspace,
} from "./qwen35-vision-layer-kernels.js";
import {
  stageQwen35VisionGpuGroup,
  type Qwen35VisionGpuAllocator,
  type Qwen35VisionGpuQueue,
  type Qwen35VisionGpuStagedGroup,
} from "./qwen35-vision-gpu-staging.js";
import type { Qwen35IntegrityValidatedVisionPackage } from "./qwen35-vision-package-loader.js";
import type { Qwen35VisionProgram } from "./qwen35-vision-program.js";
import type { Qwen35ForwardDeviceLimits } from "./qwen35-forward-dispatch.js";
import type { Qwen35WebGpuExecutor } from "./qwen35-webgpu-executor.js";
import type { AllocationLedger } from "./allocation-ledger.js";
import type { Qwen35UniformArena } from "./qwen35-uniform-arena.js";
import type { GpuAllocation, GpuAllocationRequest } from "./gpu-arena.js";

const LAYER_COUNT = 24;
const STAGE_TO_UNIFORM_SLOT = Object.freeze([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5]);
const GPU_BUFFER_USAGE_COPY_DST = 0x0008;
const GPU_BUFFER_USAGE_STORAGE = 0x0080;

export interface Qwen35VisionStreamingUniformSlot {
  readonly index: number;
  /** This exact range is also given to the layer planner. */
  readonly binding: Qwen35VisionLayerWorkspace["uniforms"][number];
  update(words: Uint32Array): void;
}

/** All activation buffers are owned separately from a temporarily staged weight group. */
export interface Qwen35VisionStreamingOwnedResources {
  readonly workspace: Omit<Qwen35VisionLayerWorkspace, "uniforms">;
  readonly uniformSlots: readonly Qwen35VisionStreamingUniformSlot[];
  readonly preparedTokenCount: number;
  readonly preparedSegmentCount: number;
  /** Releases activation and uniform allocations, including their ledger handles. */
  dispose(): Promise<void>;
}

/** A model-specific activation allocation with a single, explicit release path. */
export interface Qwen35VisionStreamingOwnedActivationWorkspace extends Omit<Qwen35VisionLayerWorkspace, "uniforms"> {
  readonly tokenCount: number;
  readonly segmentCount: number;
  dispose(): Promise<void>;
}

export interface Qwen35VisionActivationArena {
  allocate(request: GpuAllocationRequest): Promise<GpuAllocation>;
}

export interface Qwen35VisionActivationQueue {
  writeBuffer(buffer: object, bufferOffset: number, data: ArrayBuffer, dataOffset: number, size: number): void;
  onSubmittedWorkDone(): Promise<void>;
}

/**
 * Allocates the fixed first-correctness activation layout. Each allocation is
 * one bindable range; a rejection informs a later tiled experiment, not a
 * device-wide model-size conclusion.
 */
export async function createQwen35VisionStreamingOwnedActivationWorkspace(input: {
  readonly arena: Qwen35VisionActivationArena;
  readonly queue: Qwen35VisionActivationQueue;
  readonly tokenCount: number;
  /** Exact exclusive token boundaries, copied before asynchronous allocation. */
  readonly segmentOffsets: Uint32Array;
  readonly minStorageBufferOffsetAlignment: number;
  readonly maxStorageBufferBindingSize: number;
  readonly allocationIdPrefix: string;
}): Promise<Qwen35VisionStreamingOwnedActivationWorkspace> {
  const segmentOffsetsSnapshot = new Uint32Array(input.segmentOffsets);
  const segmentCount = segmentOffsetsSnapshot.length - 1;
  if (!Number.isSafeInteger(input.tokenCount) || input.tokenCount < 1 || input.tokenCount > 16_384 || segmentCount < 1 || segmentCount > input.tokenCount || segmentOffsetsSnapshot[0] !== 0 || segmentOffsetsSnapshot[segmentOffsetsSnapshot.length - 1] !== input.tokenCount || !Number.isSafeInteger(input.minStorageBufferOffsetAlignment) || input.minStorageBufferOffsetAlignment < 1 || !Number.isSafeInteger(input.maxStorageBufferBindingSize) || input.maxStorageBufferBindingSize < 4 || input.allocationIdPrefix.length === 0) {
    fail("vision-activation-options-invalid", "Vision activation workspace options are invalid");
  }
  for (let index = 1; index < segmentOffsetsSnapshot.length; index += 1) {
    if (segmentOffsetsSnapshot[index]! <= segmentOffsetsSnapshot[index - 1]!) fail("vision-activation-options-invalid", "Vision activation workspace options are invalid");
  }
  const f32 = 4;
  const hiddenBytes = input.tokenCount * 1_024 * f32;
  if (hiddenBytes % input.minStorageBufferOffsetAlignment !== 0) {
    fail("vision-activation-binding-invalid", "Vision activation workspace QKV planes are not bindable");
  }
  const bytes = {
    hidden: hiddenBytes,
    normalized: hiddenBytes,
    qkv: hiddenBytes * 3,
    attention: hiddenBytes,
    mlp: input.tokenCount * 4_096 * f32,
    rope: input.tokenCount * 64 / 2 * 2 * f32,
    segments: segmentOffsetsSnapshot.byteLength,
  } as const;
  const owned: GpuAllocation[] = [];
  const allocate = async (name: keyof typeof bytes): Promise<Qwen35VisionLayerWorkspace["hidden"]> => {
    const byteLength = bytes[name];
    if (!Number.isSafeInteger(byteLength) || byteLength < 4 || byteLength > input.maxStorageBufferBindingSize || byteLength % 4 !== 0) {
      fail("vision-activation-binding-invalid", "Vision activation workspace range is invalid");
    }
    const allocation = await input.arena.allocate({
      id: `${input.allocationIdPrefix}-${name}`,
      category: "activation",
      byteLength: BigInt(byteLength),
      usage: GPU_BUFFER_USAGE_COPY_DST | GPU_BUFFER_USAGE_STORAGE,
      // Independent buffers bind at offset zero. Four-byte allocation alignment
      // keeps the tiny segment table valid while the QKV plane check above
      // protects the only non-zero storage offsets in this first layout.
      alignment: 4,
      requiredShardQuantumBytes: BigInt(byteLength),
    });
    if (allocation.shards.length !== 1 || allocation.logicalBytes !== BigInt(byteLength) || allocation.shards[0]!.logicalByteOffset !== 0n || allocation.shards[0]!.logicalByteLength !== BigInt(byteLength)) {
      allocation.destroy();
      fail("vision-activation-binding-invalid", "Vision activation workspace must be one bindable range");
    }
    owned.push(allocation);
    return Object.freeze({ buffer: allocation.shards[0]!.buffer, byteLength });
  };
  try {
    const hidden = await allocate("hidden");
    const normalized = await allocate("normalized");
    const qkv = await allocate("qkv");
    const attention = await allocate("attention");
    const mlp = await allocate("mlp");
    const rope = await allocate("rope");
    const segmentOffsets = await allocate("segments");
    try {
      input.queue.writeBuffer(
        segmentOffsets.buffer,
        0,
        segmentOffsetsSnapshot.buffer,
        segmentOffsetsSnapshot.byteOffset,
        segmentOffsetsSnapshot.byteLength,
      );
    } catch {
      fail("vision-activation-upload-failed", "Vision activation segment upload failed");
    }
    let disposal: Promise<void> | null = null;
    return Object.freeze({
      hidden, normalized, qkv, attention, mlp, rope, segmentOffsets,
      tokenCount: input.tokenCount,
      segmentCount,
      dispose: () => {
        if (disposal !== null) return disposal;
        disposal = (async () => {
          let firstError: unknown;
          try { await input.queue.onSubmittedWorkDone(); } catch (error) { firstError ??= error; }
          for (const allocation of [...owned].reverse()) {
            try { allocation.destroy(); } catch (error) { firstError ??= error; }
          }
          if (firstError !== undefined) throw firstError;
        })();
        return disposal;
      },
    });
  } catch (error) {
    for (const allocation of [...owned].reverse()) {
      try { allocation.destroy(); } catch { /* Preserve the allocation failure. */ }
    }
    throw error;
  }
}

/**
 * Joins reusable activation ownership to the ten fixed uniform slots. Callers
 * create both backing allocations with their AllocationLedger-aware arenas.
 */
export function createQwen35VisionStreamingOwnedResources(input: {
  readonly activationWorkspace: Qwen35VisionStreamingOwnedActivationWorkspace;
  readonly uniformArena: Qwen35UniformArena;
}): Qwen35VisionStreamingOwnedResources {
  const slots = Object.freeze(Array.from({ length: 10 }, (_, index) => input.uniformArena.slot(index, 4)));
  const uniforms = Object.freeze(slots.map((slot) => Object.freeze({
    buffer: slot.binding.buffer,
    byteOffset: slot.binding.offset,
    byteLength: slot.binding.byteLength,
  })));
  const workspace = Object.freeze({
    hidden: input.activationWorkspace.hidden,
    normalized: input.activationWorkspace.normalized,
    qkv: input.activationWorkspace.qkv,
    attention: input.activationWorkspace.attention,
    mlp: input.activationWorkspace.mlp,
    rope: input.activationWorkspace.rope,
    segmentOffsets: input.activationWorkspace.segmentOffsets,
  });
  let disposal: Promise<void> | null = null;
  return Object.freeze({
    workspace,
    preparedTokenCount: input.activationWorkspace.tokenCount,
    preparedSegmentCount: input.activationWorkspace.segmentCount,
    uniformSlots: Object.freeze(slots.map((slot, index) => Object.freeze({
      index,
      binding: uniforms[index]!,
      update: (words: Uint32Array) => slot.update(words as Uint32Array<ArrayBuffer>),
    }))),
    dispose: () => {
      if (disposal !== null) return disposal;
      disposal = (async () => {
        let firstError: unknown;
        try { await input.uniformArena.dispose(); } catch (error) { firstError ??= error; }
        try { await input.activationWorkspace.dispose(); } catch (error) { firstError ??= error; }
        if (firstError !== undefined) throw firstError;
      })();
      return disposal;
    },
  });
}

export interface Qwen35VisionStreamingMetrics {
  readonly state: "ready" | "running" | "complete" | "cancelled" | "failed" | "disposed";
  readonly currentLayer: number | null;
  readonly completedLayers: number;
  readonly dispatchCount: number;
  readonly maxResidentLayerGroups: number;
  readonly unresolvedLayerGroups: number;
  readonly phase: "idle" | "staging" | "uniform-upload" | "dispatch" | "retirement" | "release" | "complete" | "cancelled" | "failed" | "disposed";
  readonly layers: readonly Qwen35VisionStreamingLayerMetrics[];
}

export interface Qwen35VisionStreamingLayerMetrics {
  readonly layer: number;
  readonly stageMs: number;
  readonly uniformUploadMs: number;
  readonly dispatchMs: number;
  readonly retirementMs: number;
  readonly releaseMs: number;
}

export interface Qwen35VisionStreamingResult {
  readonly layerOrder: readonly number[];
  readonly metrics: Qwen35VisionStreamingMetrics;
}

export interface Qwen35VisionStreamingDependencies {
  readonly limits: Qwen35ForwardDeviceLimits;
  readonly resources: Qwen35VisionStreamingOwnedResources;
  readonly stageLayer: (layer: number, signal: AbortSignal) => Promise<Qwen35VisionGpuStagedGroup>;
  readonly planLayer: (input: {
    readonly layer: number;
    readonly staged: Qwen35VisionGpuStagedGroup;
    readonly workspace: Qwen35VisionLayerWorkspace;
    readonly tokenCount: number;
    readonly segmentCount: number;
    readonly limits: Qwen35ForwardDeviceLimits;
  }) => readonly Qwen35VisionLayerDispatchPlan[];
  readonly gpu: Pick<Qwen35WebGpuExecutor, "dispatchBatch" | "submittedWorkDone" | "dispose">;
  /** Monotonic measurement source; only finite elapsed durations are recorded. */
  readonly now?: () => number;
  /** Test hook only. Production callers do not need a layer-complete callback. */
  readonly onLayerComplete?: (layer: number) => void;
}

export interface CreateQwen35VisionStreamingExecutorOptions {
  readonly package: Qwen35IntegrityValidatedVisionPackage;
  readonly program: Qwen35VisionProgram;
  readonly ledger: AllocationLedger;
  readonly allocator: Qwen35VisionGpuAllocator;
  readonly queue: Qwen35VisionGpuQueue;
  readonly allocationIdPrefix: string;
  readonly limits: Qwen35ForwardDeviceLimits;
  readonly resources: Qwen35VisionStreamingOwnedResources;
  readonly gpu: Pick<Qwen35WebGpuExecutor, "dispatchBatch" | "submittedWorkDone" | "dispose">;
}

function fail(code: string, message: string): never {
  throw diagnosticError(code, message);
}

function abortError(): Error {
  return Object.assign(new Error("Vision execution cancelled"), { name: "AbortError" });
}

function assertNotAborted(signal: AbortSignal): void {
  if (signal.aborted) throw abortError();
}

function isAbort(error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { readonly name?: unknown }).name === "AbortError";
}

function validateResources(resources: Qwen35VisionStreamingOwnedResources): void {
  if (resources.uniformSlots.length !== 10) {
    fail("vision-streaming-uniforms-invalid", "Vision streaming executor uniform slots are invalid");
  }
  const indexes = new Set<number>();
  const slots = new Set<object>();
  const bindings = new Map<object, Set<number>>();
  for (const slot of resources.uniformSlots) {
    const binding = slot.binding;
    const offset = binding.byteOffset ?? 0;
    const offsets = typeof binding.buffer === "object" && binding.buffer !== null
      ? bindings.get(binding.buffer) : undefined;
    if (!Number.isSafeInteger(slot.index) || slot.index < 0 || slot.index >= 10 || indexes.has(slot.index) || slots.has(slot) || binding.byteLength !== 16 || !Number.isSafeInteger(offset) || offset < 0 || offsets?.has(offset)) {
      fail("vision-streaming-uniforms-invalid", "Vision streaming executor uniform slots are invalid");
    }
    if (offsets === undefined) bindings.set(binding.buffer, new Set([offset]));
    else offsets.add(offset);
    indexes.add(slot.index); slots.add(slot);
  }
}

/**
 * Runs the fixed 24 vision layers. It owns reusable activations, but owns only
 * one authenticated layer-weight group at a time and retires it before staging
 * the next layer.
 */
export class Qwen35VisionStreamingExecutor {
  readonly #dependencies: Qwen35VisionStreamingDependencies;
  readonly #workspace: Qwen35VisionLayerWorkspace;
  readonly #now: () => number;
  #state: Qwen35VisionStreamingMetrics["state"] = "ready";
  #phase: Qwen35VisionStreamingMetrics["phase"] = "idle";
  #currentLayer: number | null = null;
  #completedLayers = 0;
  #dispatchCount = 0;
  #residentLayerGroups = 0;
  #maxResidentLayerGroups = 0;
  #unresolvedStaged: Qwen35VisionGpuStagedGroup | null = null;
  #layers: Qwen35VisionStreamingLayerMetrics[] = [];
  #runPromise: Promise<Qwen35VisionStreamingResult> | null = null;
  #disposePromise: Promise<void> | null = null;
  #resourcesReleasePromise: Promise<void> | null = null;
  #gpuDisposePromise: Promise<void> | null = null;
  #disposeRequested = false;

  constructor(dependencies: Qwen35VisionStreamingDependencies) {
    validateResources(dependencies.resources);
    if (!Number.isSafeInteger(dependencies.resources.preparedTokenCount) || !Number.isSafeInteger(dependencies.resources.preparedSegmentCount) || dependencies.resources.preparedTokenCount < 1 || dependencies.resources.preparedSegmentCount < 1 || dependencies.resources.preparedSegmentCount > dependencies.resources.preparedTokenCount) {
      fail("vision-streaming-resources-invalid", "Vision streaming executor resources are invalid");
    }
    this.#dependencies = dependencies;
    this.#workspace = Object.freeze({
      ...dependencies.resources.workspace,
      uniforms: Object.freeze(dependencies.resources.uniformSlots.map((slot) => slot.binding)),
    });
    this.#now = dependencies.now ?? (() => performance.now());
  }

  getMetrics(): Qwen35VisionStreamingMetrics {
    return Object.freeze({ state: this.#state, currentLayer: this.#currentLayer, completedLayers: this.#completedLayers, dispatchCount: this.#dispatchCount, maxResidentLayerGroups: this.#maxResidentLayerGroups, unresolvedLayerGroups: this.#unresolvedStaged === null ? 0 : 1, phase: this.#phase, layers: Object.freeze([...this.#layers]) });
  }

  run(input: { readonly tokenCount: number; readonly segmentCount: number; readonly signal?: AbortSignal }): Promise<Qwen35VisionStreamingResult> {
    if (this.#state !== "ready" || this.#disposeRequested) {
      return Promise.reject(diagnosticError("vision-streaming-unusable", "Vision streaming executor is not ready"));
    }
    if (this.#runPromise !== null) return this.#runPromise;
    if (!Number.isSafeInteger(input.tokenCount) || input.tokenCount < 1 || input.tokenCount > 16_384 || !Number.isSafeInteger(input.segmentCount) || input.segmentCount < 1 || input.segmentCount > input.tokenCount || input.tokenCount !== this.#dependencies.resources.preparedTokenCount || input.segmentCount !== this.#dependencies.resources.preparedSegmentCount) {
      return Promise.reject(diagnosticError("vision-streaming-input-invalid", "Vision streaming executor input is invalid"));
    }
    this.#state = "running";
    this.#runPromise = this.#run(input, input.signal ?? new AbortController().signal);
    return this.#runPromise;
  }

  async #run(input: { readonly tokenCount: number; readonly segmentCount: number }, signal: AbortSignal): Promise<Qwen35VisionStreamingResult> {
    const order: number[] = [];
    let primaryError: unknown;
    try {
      for (let layer = 0; layer < LAYER_COUNT; layer += 1) {
        assertNotAborted(signal);
        this.#currentLayer = layer;
        this.#phase = "staging";
        let staged: Qwen35VisionGpuStagedGroup | undefined;
        let layerError: unknown;
        let submissionAttempted = false;
        const timing = { layer, stageMs: 0, uniformUploadMs: 0, dispatchMs: 0, retirementMs: 0, releaseMs: 0 };
        try {
          const stagingStarted = this.#timestamp();
          staged = await this.#dependencies.stageLayer(layer, signal);
          timing.stageMs = this.#elapsed(stagingStarted);
          this.#residentLayerGroups += 1;
          this.#maxResidentLayerGroups = Math.max(this.#maxResidentLayerGroups, this.#residentLayerGroups);
          if (this.#residentLayerGroups !== 1) fail("vision-streaming-residency-invalid", "Vision streaming executor staged more than one layer group");
          assertNotAborted(signal);
          const plans = this.#dependencies.planLayer({ layer, staged, workspace: this.#workspace, tokenCount: input.tokenCount, segmentCount: input.segmentCount, limits: this.#dependencies.limits });
          if (plans.length !== 11) fail("vision-streaming-plan-invalid", "Vision streaming executor layer plan is invalid");
          this.#phase = "uniform-upload";
          const uploadStarted = this.#timestamp();
          for (let stage = 0; stage < plans.length; stage += 1) {
            const plan = plans[stage]!;
            const slot = this.#dependencies.resources.uniformSlots[STAGE_TO_UNIFORM_SLOT[stage]!]!;
            slot.update(Uint32Array.from(plan.uniformWords));
          }
          timing.uniformUploadMs = this.#elapsed(uploadStarted);
          assertNotAborted(signal);
          this.#phase = "dispatch";
          const dispatchStarted = this.#timestamp();
          submissionAttempted = true;
          await this.#dependencies.gpu.dispatchBatch(plans);
          timing.dispatchMs = this.#elapsed(dispatchStarted);
          this.#dispatchCount += plans.length;
          this.#phase = "retirement";
          const retirementStarted = this.#timestamp();
          await this.#dependencies.gpu.submittedWorkDone();
          timing.retirementMs = this.#elapsed(retirementStarted);
        } catch (error) {
          layerError = error;
        }
        if (layerError !== undefined && submissionAttempted) {
          // A failed submission or retirement leaves queue ownership uncertain.
          // Fence/poison the executor before staged buffers can be released.
          try { await this.#disposeGpu(); } catch { /* Preserve the execution error. */ }
        }
        if (staged !== undefined) {
          this.#phase = "release";
          const releaseStarted = this.#timestamp();
          try {
            await staged.destroy();
          } catch (error) {
            // A dispatch or retirement failure explains the failed computation;
            // a later release failure must not conceal it.
            layerError ??= error;
            this.#unresolvedStaged = staged;
          } finally {
            if (this.#unresolvedStaged !== staged) this.#residentLayerGroups -= 1;
            timing.releaseMs = this.#elapsed(releaseStarted);
          }
        }
        this.#layers.push(Object.freeze(timing));
        if (layerError !== undefined) throw layerError;
        if (this.#residentLayerGroups !== 0) {
          fail("vision-streaming-residency-invalid", "Vision streaming executor did not release a layer group");
        }
        order.push(layer);
        this.#completedLayers += 1;
        this.#dependencies.onLayerComplete?.(layer);
      }
      this.#state = "complete"; this.#phase = "complete"; this.#currentLayer = null;
      return Object.freeze({ layerOrder: Object.freeze(order), metrics: this.getMetrics() });
    } catch (error) {
      primaryError = error;
      this.#state = isAbort(error) ? "cancelled" : "failed";
      this.#phase = this.#state;
      this.#currentLayer = null;
      try { await this.#releaseResources(); } catch { /* Preserve the original execution error. */ }
      throw primaryError;
    }
  }

  dispose(): Promise<void> {
    if (this.#disposePromise !== null) return this.#disposePromise;
    this.#disposeRequested = true;
    this.#disposePromise = this.#disposeAfterRun();
    return this.#disposePromise;
  }

  async #disposeAfterRun(): Promise<void> {
    const activeRun = this.#runPromise;
    if (activeRun !== null) {
      try { await activeRun; } catch { /* The run already retained its primary error. */ }
    }
    await this.#releaseResources();
    this.#state = "disposed"; this.#phase = "disposed";
  }

  #timestamp(): number {
    try {
      const value = this.#now();
      return Number.isFinite(value) ? value : 0;
    } catch {
      return 0;
    }
  }

  #elapsed(started: number): number {
    const elapsed = this.#timestamp() - started;
    return Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : 0;
  }

  #releaseResources(): Promise<void> {
    if (this.#resourcesReleasePromise !== null) return this.#resourcesReleasePromise;
    const releasing = this.#releaseResourcesInner();
    this.#resourcesReleasePromise = releasing;
    void releasing.catch(() => {
      if (this.#resourcesReleasePromise === releasing) this.#resourcesReleasePromise = null;
    });
    return releasing;
  }

  async #releaseResourcesInner(): Promise<void> {
    let firstError: unknown;
    try { await this.#releaseUnresolvedStaged(); } catch (error) { firstError ??= error; }
    try { await this.#dependencies.resources.dispose(); } catch (error) { firstError ??= error; }
    try { await this.#disposeGpu(); } catch (error) { firstError ??= error; }
    if (firstError !== undefined) throw firstError;
  }

  #disposeGpu(): Promise<void> {
    if (this.#gpuDisposePromise !== null) return this.#gpuDisposePromise;
    this.#gpuDisposePromise = this.#dependencies.gpu.dispose();
    return this.#gpuDisposePromise;
  }

  async #releaseUnresolvedStaged(): Promise<void> {
    const staged = this.#unresolvedStaged;
    if (staged === null) return;
    await staged.destroy();
    this.#unresolvedStaged = null;
    this.#residentLayerGroups -= 1;
  }
}

/** Creates the production wiring: integrity-checked package staging plus the fixed layer planner. */
export function createQwen35VisionStreamingExecutor(input: CreateQwen35VisionStreamingExecutorOptions): Qwen35VisionStreamingExecutor {
  if (input.allocationIdPrefix.length === 0) fail("vision-streaming-options-invalid", "Vision streaming executor options are invalid");
  return new Qwen35VisionStreamingExecutor({
    limits: input.limits,
    resources: input.resources,
    gpu: input.gpu,
    planLayer: planQwen35VisionLayerDispatches,
    stageLayer: (layer, signal) => stageQwen35VisionGpuGroup({ package: input.package, program: input.program, layer, ledger: input.ledger, allocator: input.allocator, queue: input.queue, allocationId: `${input.allocationIdPrefix}-layer-${layer}`, signal }),
  });
}
