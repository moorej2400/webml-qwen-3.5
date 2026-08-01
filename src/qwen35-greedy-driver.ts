import { diagnosticError } from "./diagnostics.js";
import type { Qwen35HybridState } from "./hybrid-state.js";
import {
  createQwen35ActivationWorkspace,
  planQwen35ActivationWorkspace,
  type Qwen35ActivationResourceKind,
  type Qwen35ActivationWorkspace,
} from "./qwen35-activation-workspace.js";
import { createQwen35AllocationClearer } from "./qwen35-allocation-clear.js";
import {
  planQwen35DeltaNetLayerDispatch,
  planQwen35DeltaNetLayerGeometry,
} from "./qwen35-deltanet-dispatch.js";
import { planQwen35FinalNormDispatch } from "./qwen35-final-dispatch.js";
import {
  planQwen35VisualEmbeddingDispatch,
  planQwen35PackedEmbeddingDispatch,
  type Qwen35ForwardBufferSlice,
  type Qwen35ForwardDeviceLimits,
} from "./qwen35-forward-dispatch.js";
import {
  planQwen35FullAttentionLayerDispatch,
  planQwen35FullAttentionLayerGeometry,
} from "./qwen35-full-attention-dispatch.js";
import {
  assembleQwen35TiledLogitsCommands,
  planQwen35TiledLogitsUniformCount,
} from "./qwen35-logits-dispatch.js";
import { QWEN35_NO_SELECTED_TOKEN } from "./qwen35-logits-reduction.js";
import type {
  Qwen35StateAllocationClearContext,
  Qwen35ExecutionDriverFactory,
  Qwen35DriverFactoryContext,
} from "./qwen35-model-loader.js";
import type { Qwen35Invocation, Qwen35Program } from "./qwen35-program.js";
import type {
  Qwen35DriverGenerateInput,
  Qwen35DriverPrefillInput,
  Qwen35ExecutionDriver,
} from "./qwen35-session.js";
import {
  createQwen35UniformArena,
  type Qwen35UniformArena,
  type Qwen35UniformSlot,
} from "./qwen35-uniform-arena.js";
import { Qwen35WebGpuExecutor } from "./qwen35-webgpu-executor.js";
import type {
  Qwen35DispatchRequest,
  Qwen35WebGpuDevice,
} from "./qwen35-webgpu-executor.js";
import type { Qwen35WeightDirectoryView } from "./qwen35-weight-directory.js";

const DECODABLE_TOKEN_COUNT = 248_070;
const MASKED_MODEL_ROWS = 250;
const MAX_UNIFORM_WORDS = 8;

type AttentionInvocation = Extract<
  Qwen35Invocation,
  { kind: "gated-deltanet" | "full-attention" }
>;

export interface Qwen35GreedyLayerGeometry {
  readonly layer: number;
  readonly kind: AttentionInvocation["kind"];
  readonly uniformStart: number;
  readonly uniformCount: number;
}

export interface Qwen35GreedyUniformGeometry {
  readonly maxUniformWords: 8;
  readonly embeddingUniform: number;
  readonly layers: readonly Qwen35GreedyLayerGeometry[];
  readonly finalNormUniform: number;
  readonly logitsUniformStart: number;
  readonly logitsUniformCount: number;
  readonly uniformSlotCount: number;
}

function requireRunnableProgram(program: Qwen35Program): void {
  if (
    program.model !== "qwen35-4b" ||
    program.runnable !== true ||
    "blockedBy" in program
  ) {
    throw diagnosticError(
      "greedy-program-invalid",
      "The Qwen3.5 greedy program is not runnable",
    );
  }
}

function attentionInvocations(program: Qwen35Program): readonly AttentionInvocation[] {
  const invocations = program.invocations.filter(
    (invocation): invocation is AttentionInvocation =>
      invocation.kind === "gated-deltanet" ||
      invocation.kind === "full-attention",
  );
  if (
    invocations.length !== 32 ||
    invocations.some((invocation, layer) => invocation.layer !== layer)
  ) {
    throw diagnosticError(
      "greedy-program-invalid",
      "The Qwen3.5 greedy layer sequence is invalid",
    );
  }
  return Object.freeze(invocations);
}

function finalInvocation(program: Qwen35Program): Extract<
  Qwen35Invocation,
  { kind: "rms-norm" }
> {
  const invocation = program.invocations.at(-3);
  if (
    invocation?.kind !== "rms-norm" ||
    invocation.layer !== "final" ||
    invocation.site !== "final"
  ) {
    throw diagnosticError(
      "greedy-program-invalid",
      "The Qwen3.5 greedy output sequence is invalid",
    );
  }
  return invocation;
}

/** Derives every stable uniform slot before any driver-owned GPU allocation. */
export function planQwen35GreedyUniformGeometry(input: {
  readonly program: Qwen35Program;
  readonly weights: Qwen35WeightDirectoryView;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35GreedyUniformGeometry {
  requireRunnableProgram(input.program);
  finalInvocation(input.program);
  let cursor = 1;
  const layers = attentionInvocations(input.program).map((invocation) => {
    const geometry = invocation.kind === "gated-deltanet"
      ? planQwen35DeltaNetLayerGeometry({
          program: input.program,
          invocation,
          weights: input.weights,
        })
      : planQwen35FullAttentionLayerGeometry({
          program: input.program,
          invocation,
          weights: input.weights,
        });
    const item = Object.freeze({
      layer: invocation.layer,
      kind: invocation.kind,
      uniformStart: cursor,
      uniformCount: geometry.uniformCount,
    });
    cursor += geometry.uniformCount;
    return item;
  });
  const finalNormUniform = cursor;
  cursor += 1;
  const logitsUniformStart = cursor;
  const logitsUniformCount = planQwen35TiledLogitsUniformCount({
    weights: input.weights,
    limits: input.limits,
  });
  cursor += logitsUniformCount;
  if (!Number.isSafeInteger(cursor) || cursor < 1) {
    throw diagnosticError(
      "greedy-uniform-geometry-invalid",
      "The Qwen3.5 greedy uniform geometry is invalid",
    );
  }
  return Object.freeze({
    maxUniformWords: MAX_UNIFORM_WORDS,
    embeddingUniform: 0,
    layers: Object.freeze(layers),
    finalNormUniform,
    logitsUniformStart,
    logitsUniformCount,
    uniformSlotCount: cursor,
  });
}

export interface Qwen35GreedyTokenStep {
  readonly tokenId: number;
  readonly embeddingOverride?: Qwen35ForwardBufferSlice;
  readonly predict: boolean;
  readonly phase: "prefill" | "generation";
  readonly signal: AbortSignal;
}

/** Narrow boundary used by the lifecycle driver and its deterministic tests. */
export interface Qwen35GreedyTokenEngine {
  readonly capacity: number;
  readonly position: number;
  readonly poisoned: boolean;
  execute(step: Qwen35GreedyTokenStep): Promise<number | null>;
  reset(): Promise<void>;
  dispose(): Promise<void>;
}

function abortError(): DOMException {
  return new DOMException("Operation cancelled", "AbortError");
}

function requireTokenId(tokenId: number, code: string): void {
  if (
    !Number.isSafeInteger(tokenId) ||
    tokenId < 0 ||
    tokenId >= DECODABLE_TOKEN_COUNT
  ) {
    throw diagnosticError(code, "A Qwen3.5 token id is invalid");
  }
}

function linkedSignal(
  external: AbortSignal,
  internal: AbortSignal,
): { readonly signal: AbortSignal; release(): void } {
  const controller = new AbortController();
  const abort = (): void => controller.abort(abortError());
  if (external.aborted || internal.aborted) abort();
  else {
    external.addEventListener("abort", abort, { once: true });
    internal.addEventListener("abort", abort, { once: true });
  }
  return Object.freeze({
    signal: controller.signal,
    release() {
      external.removeEventListener("abort", abort);
      internal.removeEventListener("abort", abort);
    },
  });
}

class Qwen35GreedyTextDriver implements Qwen35ExecutionDriver {
  readonly #engine: Qwen35GreedyTokenEngine;
  readonly sharedGpuExecutor?: Pick<
    Qwen35WebGpuExecutor,
    "dispatchBatch" | "submittedWorkDone" | "dispose"
  >;
  #operation: "prefill" | "generation" | null = null;
  #controller: AbortController | null = null;
  #gpuWork: Promise<unknown> | null = null;
  #cachedNextToken: number | null = null;
  #pendingToken: number | null = null;
  #resetRequired = false;
  #disposed = false;
  #disposePromise: Promise<void> | null = null;

  constructor(
    engine: Qwen35GreedyTokenEngine,
    sharedGpuExecutor?: Pick<
      Qwen35WebGpuExecutor,
      "dispatchBatch" | "submittedWorkDone" | "dispose"
    >,
  ) {
    this.#engine = engine;
    if (sharedGpuExecutor !== undefined) {
      this.sharedGpuExecutor = sharedGpuExecutor;
    }
  }

  async prefill(input: Qwen35DriverPrefillInput): Promise<void> {
    this.#assertReady();
    const visualByToken = new Map<number, { readonly tokenCount: number; readonly source: Qwen35ForwardBufferSlice }>();
    for (const visual of input.visualEmbeddings ?? []) {
      if (
        !Number.isSafeInteger(visual.tokenId) ||
        !Number.isSafeInteger(visual.tokenCount) ||
        visual.tokenCount < 1 ||
        visual.tokenCount > 16_384 ||
        visualByToken.has(visual.tokenId) ||
        visual.source.byteLength < visual.tokenCount * 2_560 * 4
      ) {
        throw diagnosticError("greedy-visual-embedding-invalid", "The Qwen3.5 visual embedding range is invalid");
      }
      visualByToken.set(visual.tokenId, visual);
    }
    const expandedTokenCount = input.tokenIds.reduce(
      (total, tokenId) => total + (visualByToken.get(tokenId)?.tokenCount ?? 1),
      0,
    );
    if (
      !Array.isArray(input.tokenIds) ||
      input.tokenIds.length < 1 ||
      !Number.isSafeInteger(expandedTokenCount) ||
      this.#engine.position + expandedTokenCount > this.#engine.capacity
    ) {
      throw diagnosticError(
        "greedy-prefill-range-invalid",
        "The Qwen3.5 prefill range is invalid",
      );
    }
    for (const tokenId of input.tokenIds) {
      const visual = visualByToken.get(tokenId);
      if (visual !== undefined) continue;
      requireTokenId(tokenId, "greedy-prefill-token-invalid");
    }
    const startPosition = this.#engine.position;
    const controller = this.#begin("prefill");
    const linked = linkedSignal(input.signal, controller.signal);
    try {
      for (let index = 0; index < input.tokenIds.length; index += 1) {
        const tokenId = input.tokenIds[index]!;
        const visual = visualByToken.get(tokenId);
        const visualCount = visual?.tokenCount ?? 1;
        for (let visualIndex = 0; visualIndex < visualCount; visualIndex += 1) {
          linked.signal.throwIfAborted();
          const isLast = index === input.tokenIds.length - 1 && visualIndex === visualCount - 1;
          const step: Qwen35GreedyTokenStep = {
            tokenId: visual === undefined ? tokenId : 0,
            predict: isLast,
            phase: "prefill",
            signal: linked.signal,
            ...(visual === undefined ? {} : {
              embeddingOverride: {
                buffer: visual.source.buffer,
                offset: visual.source.offset + visualIndex * 2_560 * 4,
                byteLength: 2_560 * 4,
              },
            }),
          };
          const selected = await this.#execute(step);
          if (isLast) {
            this.#cachedNextToken = this.#selectedToken(selected);
            this.#pendingToken = null;
          }
        }
      }
    } catch (error) {
      this.#cachedNextToken = null;
      this.#pendingToken = null;
      if (this.#engine.position !== startPosition || !this.#engine.poisoned) {
        this.#resetRequired = true;
      }
      throw error;
    } finally {
      linked.release();
      this.#end(controller);
    }
  }

  generate(input: Qwen35DriverGenerateInput): AsyncIterable<number> {
    return this.#generate(input);
  }

  async *#generate(input: Qwen35DriverGenerateInput): AsyncGenerator<number> {
    this.#assertReady();
    if (
      !Number.isSafeInteger(input.maxNewTokens) ||
      input.maxNewTokens < 1 ||
      input.logitMask.start !== DECODABLE_TOKEN_COUNT ||
      input.logitMask.count !== MASKED_MODEL_ROWS
    ) {
      throw diagnosticError(
        "greedy-logit-mask-invalid",
        "The Qwen3.5 greedy generation contract is invalid",
      );
    }
    if (
      this.#cachedNextToken === null &&
      this.#pendingToken === null
    ) {
      throw diagnosticError(
        "greedy-token-cache-missing",
        "Qwen3.5 generation requires a prefetched token",
      );
    }
    const controller = this.#begin("generation");
    const linked = linkedSignal(input.signal, controller.signal);
    try {
      const requiredStateSteps = input.maxNewTokens -
        (this.#pendingToken === null ? 1 : 0);
      const logicalRemaining = this.#engine.capacity - this.#engine.position -
        (this.#pendingToken === null ? 0 : 1);
      if (
        requiredStateSteps < 0 ||
        requiredStateSteps > this.#engine.capacity - this.#engine.position ||
        input.maxNewTokens > logicalRemaining
      ) {
        throw diagnosticError(
          "greedy-context-full",
          "The Qwen3.5 context capacity is exhausted",
        );
      }
      if (this.#cachedNextToken === null) {
        const pending = this.#pendingToken!;
        const selected = await this.#execute({
          tokenId: pending,
          predict: true,
          phase: "generation",
          signal: linked.signal,
        });
        this.#cachedNextToken = this.#selectedToken(selected);
        this.#pendingToken = null;
      }
      for (let index = 0; index < input.maxNewTokens; index += 1) {
        linked.signal.throwIfAborted();
        const emitted = this.#cachedNextToken;
        if (emitted === null) {
          throw diagnosticError(
            "greedy-token-cache-missing",
            "Qwen3.5 generation requires a prefetched token",
          );
        }
        this.#cachedNextToken = null;
        this.#pendingToken = emitted;
        yield emitted;
        if (index + 1 === input.maxNewTokens) continue;
        linked.signal.throwIfAborted();
        const pending = this.#pendingToken;
        if (pending === null) {
          throw diagnosticError(
            "greedy-token-cache-missing",
            "Qwen3.5 generation lost its pending token",
          );
        }
        const selected = await this.#execute({
          tokenId: pending,
          predict: true,
          phase: "generation",
          signal: linked.signal,
        });
        this.#cachedNextToken = this.#selectedToken(selected);
        this.#pendingToken = null;
      }
    } finally {
      linked.release();
      this.#end(controller);
    }
  }

  async cancel(): Promise<void> {
    this.#controller?.abort(abortError());
    const work = this.#gpuWork;
    if (work !== null) {
      try {
        await work;
      } catch {
        // The operation owner reports its stable failure to the session.
      }
    }
    if (this.#engine.poisoned) {
      throw diagnosticError(
        "greedy-driver-poisoned",
        "Qwen3.5 cancellation crossed committed GPU state",
      );
    }
  }

  async reset(): Promise<void> {
    this.#assertLive();
    if (this.#operation !== null || this.#gpuWork !== null) {
      throw diagnosticError(
        "greedy-driver-not-quiescent",
        "Qwen3.5 reset requires a quiescent driver",
      );
    }
    if (this.#engine.poisoned) {
      throw diagnosticError(
        "greedy-driver-poisoned",
        "Qwen3.5 execution state must be disposed",
      );
    }
    await this.#engine.reset();
    this.#cachedNextToken = null;
    this.#pendingToken = null;
    this.#resetRequired = false;
  }

  dispose(): Promise<void> {
    this.#disposePromise ??= this.#dispose();
    return this.#disposePromise;
  }

  async #dispose(): Promise<void> {
    this.#disposed = true;
    this.#controller?.abort(abortError());
    const work = this.#gpuWork;
    if (work !== null) {
      try {
        await work;
      } catch {
        // Disposal still releases every driver-owned GPU resource.
      }
    }
    this.#cachedNextToken = null;
    this.#pendingToken = null;
    await this.#engine.dispose();
  }

  #begin(operation: "prefill" | "generation"): AbortController {
    if (this.#operation !== null) {
      throw diagnosticError(
        "greedy-driver-busy",
        "Qwen3.5 execution driver already has an active operation",
      );
    }
    const controller = new AbortController();
    this.#operation = operation;
    this.#controller = controller;
    return controller;
  }

  #end(controller: AbortController): void {
    if (this.#controller !== controller) return;
    this.#controller = null;
    this.#operation = null;
  }

  async #execute(step: Qwen35GreedyTokenStep): Promise<number | null> {
    const operation = this.#engine.execute(step);
    this.#gpuWork = operation;
    try {
      return await operation;
    } finally {
      if (this.#gpuWork === operation) this.#gpuWork = null;
    }
  }

  #selectedToken(value: number | null): number {
    if (value === null) {
      throw diagnosticError(
        "greedy-token-selection-invalid",
        "Qwen3.5 greedy selection did not return a token",
      );
    }
    requireTokenId(value, "greedy-token-selection-invalid");
    return value;
  }

  #assertLive(): void {
    if (this.#disposed) {
      throw diagnosticError(
        "greedy-driver-disposed",
        "Qwen3.5 execution driver is disposed",
      );
    }
  }

  #assertReady(): void {
    this.#assertLive();
    if (this.#engine.poisoned) {
      throw diagnosticError(
        "greedy-driver-poisoned",
        "Qwen3.5 execution state must be disposed",
      );
    }
    if (this.#resetRequired) {
      throw diagnosticError(
        "greedy-driver-reset-required",
        "Qwen3.5 execution state must be reset",
      );
    }
    if (this.#operation !== null) {
      throw diagnosticError(
        "greedy-driver-busy",
        "Qwen3.5 execution driver already has an active operation",
      );
    }
  }
}

export function createQwen35GreedyTextDriver(
  engine: Qwen35GreedyTokenEngine,
  sharedGpuExecutor?: Pick<
    Qwen35WebGpuExecutor,
    "dispatchBatch" | "submittedWorkDone" | "dispose"
  >,
): Qwen35ExecutionDriver {
  return new Qwen35GreedyTextDriver(engine, sharedGpuExecutor);
}

interface UniformCommand extends Qwen35DispatchRequest {
  readonly uniformWords: readonly number[];
}

export interface Qwen35GreedyUniformUploadSlot {
  readonly binding: Qwen35ForwardBufferSlice;
  update(values: Uint32Array<ArrayBuffer>): void;
}

/** Uploads each command payload to the exact uniform binding assigned by its planner. */
export function uploadQwen35GreedyUniformCommands(input: {
  readonly commands: readonly UniformCommand[];
  readonly slots: readonly Qwen35GreedyUniformUploadSlot[];
  readonly expectedSlotCount: number;
}): void {
  if (
    !Number.isSafeInteger(input.expectedSlotCount) ||
    input.expectedSlotCount < 1 ||
    input.expectedSlotCount > input.slots.length
  ) {
    throw diagnosticError(
      "greedy-uniform-schedule-invalid",
      "The Qwen3.5 uniform schedule is invalid",
    );
  }
  const slotsByBuffer = new Map<object, Map<number, number>>();
  for (const [index, slot] of input.slots.entries()) {
    const byOffset = slotsByBuffer.get(slot.binding.buffer) ?? new Map();
    if (byOffset.has(slot.binding.offset)) {
      throw diagnosticError(
        "greedy-uniform-schedule-invalid",
        "The Qwen3.5 uniform slots overlap",
      );
    }
    byOffset.set(slot.binding.offset, index);
    slotsByBuffer.set(slot.binding.buffer, byOffset);
  }
  const used = new Set<number>();
  for (const command of input.commands) {
    if (command.uniformWords.length === 0) continue;
    const bindings = command.bindings.filter(
      (binding) => binding.kind === "uniform",
    );
    if (
      command.uniformWords.length > MAX_UNIFORM_WORDS ||
      bindings.length !== 1
    ) {
      throw diagnosticError(
        "greedy-uniform-schedule-invalid",
        "A Qwen3.5 command has an invalid uniform payload",
      );
    }
    const binding = bindings[0]!;
    const slotIndex = slotsByBuffer.get(binding.buffer)?.get(binding.offset) ?? -1;
    const slot = input.slots[slotIndex];
    if (
      slotIndex < 0 ||
      slotIndex >= input.expectedSlotCount ||
      slot === undefined ||
      slot.binding.byteLength < binding.size ||
      used.has(slotIndex)
    ) {
      throw diagnosticError(
        "greedy-uniform-schedule-invalid",
        "A Qwen3.5 command references an invalid uniform slot",
      );
    }
    const words = new Uint32Array(MAX_UNIFORM_WORDS);
    words.set(command.uniformWords);
    slot.update(words);
    used.add(slotIndex);
  }
  if (
    used.size !== input.expectedSlotCount ||
    Array.from(
      { length: input.expectedSlotCount },
      (_, index) => index,
    ).some((index) => !used.has(index))
  ) {
    throw diagnosticError(
      "greedy-uniform-schedule-invalid",
      "The Qwen3.5 uniform schedule does not cover every required slot",
    );
  }
}

export interface Qwen35GreedyGpuBatchExecutor {
  dispatchBatch(requests: readonly Qwen35DispatchRequest[]): Promise<void>;
  submittedWorkDone(): Promise<void>;
  readU32(source: object, byteOffset: number): Promise<number>;
}

export interface Qwen35GreedyGpuBatchState {
  advance(tokens: number): unknown;
}

export interface Qwen35GreedySelectedTokenView {
  readonly buffer: object;
  readonly offset: number;
}

/**
 * Executes the fixed post-planning GPU schedule. This boundary keeps queue
 * retirement, state advancement, and scalar readback in one testable order.
 */
export async function executeQwen35GreedyGpuBatch(input: {
  readonly commands: readonly Qwen35DispatchRequest[];
  readonly executor: Qwen35GreedyGpuBatchExecutor;
  readonly state: Qwen35GreedyGpuBatchState;
  readonly selected: Qwen35GreedySelectedTokenView | null;
  readonly step: Qwen35GreedyTokenStep;
  readonly poison: () => void;
}): Promise<number | null> {
  if (input.step.signal.aborted) throw abortError();
  let submissionAttempted = false;
  let retired = false;
  let advanced = false;
  let readbackStarted = false;
  let readbackCompleted = false;
  try {
    submissionAttempted = true;
    await input.executor.dispatchBatch(input.commands);
    await input.executor.submittedWorkDone();
    retired = true;
    input.state.advance(1);
    advanced = true;
    if (input.step.signal.aborted) {
      if (input.step.phase === "generation") input.poison();
      throw abortError();
    }
    if (input.selected === null) return null;
    readbackStarted = true;
    const token = await input.executor.readU32(
      input.selected.buffer,
      input.selected.offset,
    );
    readbackCompleted = true;
    if (input.step.signal.aborted) {
      if (input.step.phase === "generation") input.poison();
      throw abortError();
    }
    if (
      token === QWEN35_NO_SELECTED_TOKEN ||
      token < 0 ||
      token >= DECODABLE_TOKEN_COUNT
    ) {
      input.poison();
      throw diagnosticError(
        "greedy-token-selection-invalid",
        token === QWEN35_NO_SELECTED_TOKEN
          ? "Qwen3.5 greedy selection found no finite logits"
          : `Qwen3.5 greedy selection returned token ${token} outside the decodable range`,
      );
    }
    return token;
  } catch (error) {
    if (
      submissionAttempted &&
      (!retired || !advanced || (readbackStarted && !readbackCompleted))
    ) {
      input.poison();
    }
    if (
      error instanceof DOMException &&
      error.name === "AbortError"
    ) {
      throw abortError();
    }
    if (
      typeof error === "object" &&
      error !== null &&
      (error as { readonly code?: unknown }).code ===
        "greedy-token-selection-invalid"
    ) {
      throw error;
    }
    throw diagnosticError(
      "greedy-gpu-batch-failed",
      "Qwen3.5 GPU token execution failed",
    );
  }
}

export async function disposeQwen35GreedyOwnedResources(input: {
  readonly executor: Pick<Qwen35WebGpuExecutor, "dispose">;
  readonly uniformArena: Pick<Qwen35UniformArena, "dispose"> | null;
  readonly workspace: Pick<Qwen35ActivationWorkspace, "dispose"> | null;
}): Promise<void> {
  let failed = false;
  try {
    // The executor fence must settle before any bound arena buffer is released.
    await input.executor.dispose();
  } catch {
    failed = true;
  }
  const released = await Promise.allSettled([
    input.uniformArena?.dispose() ?? Promise.resolve(),
    input.workspace?.dispose() ?? Promise.resolve(),
  ]);
  if (failed || released.some((result) => result.status === "rejected")) {
    throw diagnosticError(
      "greedy-engine-cleanup-failed",
      "Qwen3.5 GPU execution cleanup did not complete",
    );
  }
}

function workspaceSlice(
  workspace: Qwen35ActivationWorkspace,
  kind: Qwen35ActivationResourceKind,
): Qwen35ForwardBufferSlice {
  const view = workspace.get(kind);
  return Object.freeze({
    buffer: view.binding.buffer,
    offset: view.binding.offset,
    byteLength: view.binding.size,
  });
}

class Qwen35GpuTokenEngine implements Qwen35GreedyTokenEngine {
  readonly #program: Qwen35Program;
  readonly #weights: Qwen35WeightDirectoryView;
  readonly #state: Qwen35HybridState;
  readonly #workspace: Qwen35ActivationWorkspace;
  readonly #uniformArena: Qwen35UniformArena;
  readonly #uniformSlots: readonly Qwen35UniformSlot[];
  readonly #geometry: Qwen35GreedyUniformGeometry;
  readonly #executor: Qwen35WebGpuExecutor;
  readonly #limits: Qwen35ForwardDeviceLimits;
  readonly #layers: readonly AttentionInvocation[];
  readonly #final: ReturnType<typeof finalInvocation>;
  #poisoned = false;
  #disposePromise: Promise<void> | null = null;

  constructor(input: {
    readonly device: Qwen35WebGpuDevice;
    readonly program: Qwen35Program;
    readonly weights: Qwen35WeightDirectoryView;
    readonly state: Qwen35HybridState;
    readonly workspace: Qwen35ActivationWorkspace;
    readonly uniformArena: Qwen35UniformArena;
    readonly geometry: Qwen35GreedyUniformGeometry;
    readonly executor: Qwen35WebGpuExecutor;
  }) {
    this.#program = input.program;
    this.#weights = input.weights;
    this.#state = input.state;
    this.#workspace = input.workspace;
    this.#uniformArena = input.uniformArena;
    this.#geometry = input.geometry;
    this.#executor = input.executor;
    this.#limits = input.device.limits;
    this.#layers = attentionInvocations(input.program);
    this.#final = finalInvocation(input.program);
    this.#uniformSlots = Object.freeze(Array.from(
      { length: input.geometry.uniformSlotCount },
      (_, index) => input.uniformArena.slot(index, MAX_UNIFORM_WORDS),
    ));
  }

  get capacity(): number {
    return this.#state.capacity;
  }

  get position(): number {
    return this.#state.position;
  }

  get poisoned(): boolean {
    return this.#poisoned;
  }

  async execute(step: Qwen35GreedyTokenStep): Promise<number | null> {
    if (this.#poisoned) {
      throw diagnosticError(
        "greedy-engine-poisoned",
        "Qwen3.5 GPU execution state must be disposed",
      );
    }
    step.signal.throwIfAborted();
    if (this.position >= this.capacity) {
      throw diagnosticError(
        "greedy-context-full",
        "The Qwen3.5 context capacity is exhausted",
      );
    }

    const commands: UniformCommand[] = [];
    const embedding = step.embeddingOverride === undefined
      ? planQwen35PackedEmbeddingDispatch({
          weights: this.#weights,
          tokenId: step.tokenId,
          output: workspaceSlice(this.#workspace, "packed-embedding-output"),
          uniform: this.#uniformSlots[this.#geometry.embeddingUniform]!.binding,
          limits: this.#limits,
        })
      : planQwen35VisualEmbeddingDispatch({
          source: step.embeddingOverride,
          output: workspaceSlice(this.#workspace, "packed-embedding-output"),
          uniform: this.#uniformSlots[this.#geometry.embeddingUniform]!.binding,
          limits: this.#limits,
        });
    commands.push(Object.freeze({ ...embedding, uniformWords: embedding.uniformWords }));

    for (const [index, invocation] of this.#layers.entries()) {
      const geometry = this.#geometry.layers[index]!;
      const uniforms = this.#uniformSlots
        .slice(geometry.uniformStart, geometry.uniformStart + geometry.uniformCount)
        .map((slot) => slot.binding);
      const state = this.#state.getLayerResources(invocation.layer);
      const plan = invocation.kind === "gated-deltanet"
        ? planQwen35DeltaNetLayerDispatch({
            program: this.#program,
            invocation,
            weights: this.#weights,
            workspace: this.#workspace,
            deltanetParameterLiveness: WORKSPACE_LIVENESS,
            state,
            limits: this.#limits,
            uniforms,
          })
        : planQwen35FullAttentionLayerDispatch({
            program: this.#program,
            invocation,
            weights: this.#weights,
            workspace: this.#workspace,
            state,
            position: this.position,
            capacity: this.capacity,
            mropePositions: [this.position, this.position, this.position],
            limits: this.#limits,
            uniforms,
          });
      commands.push(...plan.commands);
    }

    let selected: ReturnType<typeof assembleQwen35TiledLogitsCommands>["selectedTokenReadback"] | null = null;
    if (step.predict) {
      const final = planQwen35FinalNormDispatch({
        program: this.#program,
        invocation: this.#final,
        weights: this.#weights,
        workspace: this.#workspace,
        limits: this.#limits,
        uniform: this.#uniformSlots[this.#geometry.finalNormUniform]!.binding,
      });
      commands.push(final.command);
      const logits = assembleQwen35TiledLogitsCommands({
        weights: this.#weights,
        normalizedHidden: workspaceSlice(this.#workspace, "normalized-hidden"),
        workspace: this.#workspace,
        limits: this.#limits,
        uniforms: this.#uniformSlots
          .slice(
            this.#geometry.logitsUniformStart,
            this.#geometry.logitsUniformStart + this.#geometry.logitsUniformCount,
          )
          .map((slot) => slot.binding),
      });
      commands.push(...logits.commands);
      selected = logits.selectedTokenReadback;
    }

    const expectedUniforms = step.predict
      ? this.#geometry.uniformSlotCount
      : this.#geometry.finalNormUniform;
    uploadQwen35GreedyUniformCommands({
      commands,
      slots: this.#uniformSlots,
      expectedSlotCount: expectedUniforms,
    });
    step.signal.throwIfAborted();

    return executeQwen35GreedyGpuBatch({
      commands,
      executor: this.#executor,
      state: this.#state,
      selected,
      step,
      poison: () => { this.#poisoned = true; },
    });
  }

  async reset(): Promise<void> {
    if (this.#poisoned) {
      throw diagnosticError(
        "greedy-engine-poisoned",
        "Qwen3.5 GPU execution state must be disposed",
      );
    }
    try {
      await this.#state.reset();
      await this.#workspace.reset();
    } catch {
      this.#poisoned = true;
      throw diagnosticError(
        "greedy-engine-reset-failed",
        "Qwen3.5 GPU execution state reset failed",
      );
    }
  }

  dispose(): Promise<void> {
    this.#disposePromise ??= this.#dispose();
    return this.#disposePromise;
  }

  async #dispose(): Promise<void> {
    await disposeQwen35GreedyOwnedResources({
      executor: this.#executor,
      uniformArena: this.#uniformArena,
      workspace: this.#workspace,
    });
  }
}

// Kept as one immutable shared object so every layer validates the same ranges.
const WORKSPACE_LIVENESS = planQwen35ActivationWorkspace().deltanetParameterLiveness;

async function createGpuDriver(
  context: Qwen35DriverFactoryContext,
): Promise<Qwen35ExecutionDriver> {
  requireRunnableProgram(context.program);
  const limits = context.device.limits;
  const geometry = planQwen35GreedyUniformGeometry({
    program: context.program,
    weights: context.weightDirectory,
    limits,
  });
  const clearer = createQwen35AllocationClearer(context.device);
  const executor = new Qwen35WebGpuExecutor(context.device);
  let workspace: Qwen35ActivationWorkspace | null = null;
  let uniformArena: Qwen35UniformArena | null = null;
  try {
    workspace = await createQwen35ActivationWorkspace({
      arena: context.arena,
      clearAllocation: (allocation) => clearer.clearAllocation(allocation),
    });
    uniformArena = await createQwen35UniformArena({
      arena: context.arena,
      queue: context.device.queue,
      slotCount: geometry.uniformSlotCount,
      slotWordCapacity: geometry.maxUniformWords,
      minUniformBufferOffsetAlignment: limits.minUniformBufferOffsetAlignment,
      maxUniformBufferBindingSize: limits.maxUniformBufferBindingSize,
    });
    return createQwen35GreedyTextDriver(new Qwen35GpuTokenEngine({
      device: context.device,
      program: context.program,
      weights: context.weightDirectory,
      state: context.hybridState,
      workspace,
      uniformArena,
      geometry,
      executor,
    }), executor);
  } catch (error) {
    try {
      await disposeQwen35GreedyOwnedResources({
        executor,
        uniformArena,
        workspace,
      });
    } catch {
      throw diagnosticError(
        "greedy-driver-rollback-failed",
        "Qwen3.5 driver creation rollback did not complete",
      );
    }
    throw error;
  }
}

/** Public production factory; callers may still inject a test factory. */
export function createQwen35GreedyExecutionDriverFactory(): Qwen35ExecutionDriverFactory {
  const clearers = new WeakMap<object, ReturnType<typeof createQwen35AllocationClearer>>();
  const clearer = (device: Qwen35WebGpuDevice) => {
    const key = device as object;
    let current = clearers.get(key);
    if (current === undefined) {
      current = createQwen35AllocationClearer(device);
      clearers.set(key, current);
    }
    return current;
  };
  return Object.freeze({
    clearStateAllocation(context: Qwen35StateAllocationClearContext) {
      return clearer(context.device).clearAllocation(context.allocation);
    },
    create(context: Qwen35DriverFactoryContext) {
      return createGpuDriver(context);
    },
  });
}
