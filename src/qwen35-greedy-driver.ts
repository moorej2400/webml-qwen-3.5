import { diagnosticError } from "./diagnostics.js";
import type { GpuAllocation } from "./gpu-arena.js";
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
  planQwen35StagedPackedEmbeddingDispatch,
  type Qwen35ForwardBufferSlice,
  type Qwen35ForwardDeviceLimits,
} from "./qwen35-forward-dispatch.js";
import {
  QWEN35_STAGED_LOGITS_TILE_BUFFER_COUNT,
  type Qwen35DiskBackedTiedEmbeddingStore,
  type Qwen35LogitCandidate,
  type Qwen35StagedPackedRows,
} from "./qwen35-disk-backed-tied-embedding.js";
import {
  planQwen35FullAttentionLayerDispatch,
  planQwen35FullAttentionLayerGeometry,
} from "./qwen35-full-attention-dispatch.js";
import {
  assembleQwen35StagedFinalTokenCommand,
  assembleQwen35StagedLogitsTileGpuCommands,
  assembleQwen35StagedLogitsTileCommands,
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
import { createQwen35PerformanceWriteQueue } from "./qwen35-performance.js";
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
import {
  executeQwen35RollingLayerSequence,
  type Qwen35RollingLayerMutation,
  type Qwen35RollingLayerStore,
} from "./qwen35-rolling-layer-weights.js";

const DECODABLE_TOKEN_COUNT = 248_070;
const MASKED_MODEL_ROWS = 250;
const MAX_UNIFORM_WORDS = 8;
const STAGED_LOGITS_UNIFORM_COUNT = 2;
// The shared tied-store capacity makes every batch own a distinct staging
// buffer. The per-tile Q6_K kernel and candidate record ABI stay unchanged.
const STAGED_LOGITS_BATCH_SIZE = QWEN35_STAGED_LOGITS_TILE_BUFFER_COUNT;
const STAGED_LOGITS_TILE_UNIFORM_COUNT =
  STAGED_LOGITS_UNIFORM_COUNT * STAGED_LOGITS_BATCH_SIZE;
const STAGED_LOGITS_TOTAL_UNIFORM_COUNT =
  STAGED_LOGITS_TILE_UNIFORM_COUNT + STAGED_LOGITS_UNIFORM_COUNT;
const GPU_STORAGE_AND_COPY_SRC = 0x0080 | 0x0004;

export function rollingFullAttentionUniformCount(
  program: Qwen35Program,
): 75 | 77 {
  const tiedLayout = program.tensorBindings.get("token_embd.weight")?.tensor.storageType;
  // ABI v2 fuses the streamed Q3 gate/up pair into one command. Each command
  // owns one uniform slot, so reserving the ABI v1 count makes the exact v2
  // full-attention planner reject the layer before dispatch.
  if (tiedLayout === "q6-k-fused-f32-256") return 75;
  if (tiedLayout === "q6-k-212") return 77;
  throw diagnosticError(
    "greedy-program-invalid",
    "Qwen3.5 rolling uniform geometry requires a supported package ABI",
  );
}

function rollingDeltaNetUniformCount(program: Qwen35Program): 10 | 12 {
  const tiedLayout = program.tensorBindings.get("token_embd.weight")?.tensor.storageType;
  // The package ABI determines the fused projection schedule for every streamed
  // layer, even when the tied tensor itself is owned outside the rolling store.
  if (tiedLayout === "q6-k-fused-f32-256") return 10;
  if (tiedLayout === "q6-k-212") return 12;
  throw diagnosticError(
    "greedy-program-invalid",
    "Qwen3.5 rolling uniform geometry requires a supported package ABI",
  );
}
const PREFILL_CHUNK_SIZE = 4;

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
  readonly diskBackedTiedEmbedding?: boolean;
  readonly rollingLayers?: Pick<Qwen35RollingLayerStore, "streamedLayers">;
}): Qwen35GreedyUniformGeometry {
  requireRunnableProgram(input.program);
  finalInvocation(input.program);
  let cursor = 1;
  const rollingLayers = new Set(input.rollingLayers?.streamedLayers ?? []);
  const layers = attentionInvocations(input.program).map((invocation) => {
    // Every streamed tensor is smaller than the 64 MiB minimum product shard
    // policy, so its per-tensor allocation contributes one physical GEMV piece.
    const uniformCount = rollingLayers.has(invocation.layer)
      ? invocation.kind === "gated-deltanet"
        ? rollingDeltaNetUniformCount(input.program)
        : rollingFullAttentionUniformCount(input.program)
      : invocation.kind === "gated-deltanet"
        ? planQwen35DeltaNetLayerGeometry({
            program: input.program,
            invocation,
            weights: input.weights,
          }).uniformCount
        : planQwen35FullAttentionLayerGeometry({
            program: input.program,
            invocation,
            weights: input.weights,
          }).uniformCount;
    const item = Object.freeze({
      layer: invocation.layer,
      kind: invocation.kind,
      uniformStart: cursor,
      uniformCount,
    });
    cursor += uniformCount;
    return item;
  });
  const finalNormUniform = cursor;
  cursor += 1;
  const logitsUniformStart = cursor;
  const logitsUniformCount = input.diskBackedTiedEmbedding === true
    ? STAGED_LOGITS_TOTAL_UNIFORM_COUNT
    : planQwen35TiledLogitsUniformCount({
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

/** Resolves one text or projected-visual input without consulting resident weights. */
export async function stageQwen35GreedyInputEmbedding(input: {
  readonly tiedEmbedding: Pick<
    Qwen35DiskBackedTiedEmbeddingStore,
    "stageInputRow"
  >;
  readonly step: Qwen35GreedyTokenStep;
}): Promise<
  | { readonly kind: "visual"; readonly source: Qwen35ForwardBufferSlice }
  | { readonly kind: "packed"; readonly rows: Qwen35StagedPackedRows }
> {
  if (input.step.embeddingOverride !== undefined) {
    return Object.freeze({
      kind: "visual" as const,
      source: input.step.embeddingOverride,
    });
  }
  const rows = await input.tiedEmbedding.stageInputRow({
    tokenId: input.step.tokenId,
    phase: input.step.phase === "generation" ? "decode" : "prefill",
    signal: input.step.signal,
  });
  return Object.freeze({ kind: "packed" as const, rows });
}

/** Reduces the streamed tied table to one valid decodable token. */
export async function selectQwen35GreedyTiedToken(input: {
  readonly tiedEmbedding: Pick<
    Qwen35DiskBackedTiedEmbeddingStore,
    "selectTopK"
  >;
  readonly phase: Qwen35GreedyTokenStep["phase"];
  readonly signal: AbortSignal;
  readonly scoreTile: (
    tile: Qwen35StagedPackedRows,
  ) => Promise<readonly Qwen35LogitCandidate[]> | readonly Qwen35LogitCandidate[];
}): Promise<number> {
  const candidates = await input.tiedEmbedding.selectTopK({
    phase: input.phase === "generation" ? "decode" : "prefill",
    topK: 1,
    signal: input.signal,
    scoreTile: input.scoreTile,
  });
  const tokenId = candidates[0]?.tokenId;
  if (
    !Number.isSafeInteger(tokenId) ||
    tokenId === undefined ||
    tokenId < 0 ||
    tokenId >= DECODABLE_TOKEN_COUNT
  ) {
    throw diagnosticError(
      "greedy-token-selection-invalid",
      "Qwen3.5 greedy selection did not return a decodable token",
    );
  }
  return tokenId;
}

/** Reduces streamed GPU candidates without copying intermediate winners to JS. */
export async function selectQwen35GreedyTiedTokenGpu(input: {
  readonly tiedEmbedding: Pick<
    Qwen35DiskBackedTiedEmbeddingStore,
    "selectTopKGpu"
  >;
  readonly phase: Qwen35GreedyTokenStep["phase"];
  readonly signal: AbortSignal;
  readonly scoreTile: (
    tile: Qwen35StagedPackedRows,
    candidateSlot: number,
  ) => Promise<void> | void;
  readonly flush: () => Promise<void> | void;
  readonly finalize: () => Promise<number> | number;
}): Promise<number> {
  const selectTopKGpu = input.tiedEmbedding.selectTopKGpu;
  if (typeof selectTopKGpu !== "function") {
    throw diagnosticError(
      "greedy-token-selection-invalid",
      "Qwen3.5 GPU token selection is unavailable",
    );
  }
  const tokenId = await selectTopKGpu.call(input.tiedEmbedding, {
    phase: input.phase === "generation" ? "decode" : "prefill",
    signal: input.signal,
    scoreTile: input.scoreTile,
    flush: input.flush,
    finalize: input.finalize,
  });
  if (
    !Number.isSafeInteger(tokenId) ||
    tokenId < 0 ||
    tokenId >= DECODABLE_TOKEN_COUNT
  ) {
    throw diagnosticError(
      "greedy-token-selection-invalid",
      "Qwen3.5 GPU token selection did not return a decodable token",
    );
  }
  return tokenId;
}

interface Qwen35StagedLogitsBatcher {
  enqueue(commands: readonly Qwen35DispatchRequest[]): void;
  flush(): Promise<void>;
}

export interface Qwen35ResidentLayerBatcher {
  enqueue(commands: readonly Qwen35DispatchRequest[]): void;
  flush(): Promise<void>;
}

/** Keeps adjacent resident layers in one command encoder and queue submission. */
export function createQwen35ResidentLayerBatcher(
  executor: Pick<Qwen35WebGpuExecutor, "dispatchBatch">,
): Qwen35ResidentLayerBatcher {
  const pending: Qwen35DispatchRequest[] = [];
  return Object.freeze({
    enqueue(commands: readonly Qwen35DispatchRequest[]): void {
      pending.push(...commands);
    },
    async flush(): Promise<void> {
      if (pending.length === 0) return;
      await executor.dispatchBatch(Object.freeze(pending.splice(0)));
    },
  });
}

/** Batches tile kernels until the tied store must recycle an output buffer. */
function createQwen35StagedLogitsBatcher(
  executor: Pick<
    Qwen35WebGpuExecutor,
    "dispatchBatch"
  >,
): Qwen35StagedLogitsBatcher {
  const pending: Qwen35DispatchRequest[] = [];
  return Object.freeze({
    enqueue(commands: readonly Qwen35DispatchRequest[]): void {
      pending.push(...commands);
    },
    async flush(): Promise<void> {
      if (pending.length === 0) return;
      const commands = Object.freeze(pending.splice(0));
      await executor.dispatchBatch(commands);
    },
  });
}

function stagedLogitsTileUniformSlots(
  slots: readonly Qwen35UniformSlot[],
  candidateSlot: number,
): readonly Qwen35UniformSlot[] {
  const start = (candidateSlot % STAGED_LOGITS_BATCH_SIZE) *
    STAGED_LOGITS_UNIFORM_COUNT;
  return slots.slice(start, start + STAGED_LOGITS_UNIFORM_COUNT);
}

function stagedLogitsFinalUniformSlots(
  slots: readonly Qwen35UniformSlot[],
): readonly Qwen35UniformSlot[] {
  return slots.slice(STAGED_LOGITS_TILE_UNIFORM_COUNT);
}

/** Narrow boundary used by the lifecycle driver and its deterministic tests. */
export interface Qwen35GreedyTokenEngine {
  readonly capacity: number;
  readonly position: number;
  readonly poisoned: boolean;
  execute(step: Qwen35GreedyTokenStep): Promise<number | null>;
  /** Executes a bounded resident prefill chunk in layer-major order when supported. */
  prefillChunk?(input: {
    readonly steps: readonly Qwen35GreedyTokenStep[];
    readonly signal: AbortSignal;
  }): Promise<number | null>;
  reset(): Promise<void>;
  dispose(): Promise<void>;
}

export interface Qwen35GreedyResidentState {
  readonly position: number;
  ensureCapacity(requiredEnd: number, signal?: AbortSignal): Promise<void>;
}

/** Ensures planners can never observe a logical token without physical state. */
export async function ensureQwen35GreedyTokenState(
  state: Qwen35GreedyResidentState,
  signal: AbortSignal,
): Promise<void> {
  signal.throwIfAborted();
  if (!Number.isSafeInteger(state.position) || state.position < 0) {
    throw diagnosticError(
      "greedy-state-position-invalid",
      "Qwen3.5 GPU state position is invalid",
    );
  }
  await state.ensureCapacity(state.position + 1, signal);
  signal.throwIfAborted();
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
    "dispatchBatch" | "submittedWorkDone" | "releaseBindGroups" | "dispose"
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
      "dispatchBatch" | "submittedWorkDone" | "releaseBindGroups" | "dispose"
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
      const steps: Qwen35GreedyTokenStep[] = [];
      for (let index = 0; index < input.tokenIds.length; index += 1) {
        const tokenId = input.tokenIds[index]!;
        const visual = visualByToken.get(tokenId);
        const visualCount = visual?.tokenCount ?? 1;
        for (let visualIndex = 0; visualIndex < visualCount; visualIndex += 1) {
          const isLast = index === input.tokenIds.length - 1 && visualIndex === visualCount - 1;
          steps.push({
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
          });
        }
      }
      for (let start = 0; start < steps.length; start += PREFILL_CHUNK_SIZE) {
        const chunk = steps.slice(start, start + PREFILL_CHUNK_SIZE);
        linked.signal.throwIfAborted();
        const selected = await this.#prefillChunk(chunk, linked.signal);
        if (chunk.at(-1)?.predict === true) {
          this.#cachedNextToken = this.#selectedToken(selected);
          this.#pendingToken = null;
        } else if (selected !== null) {
          throw diagnosticError(
            "greedy-token-selection-invalid",
            "Qwen3.5 prefill returned a token before the final prompt step",
          );
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

  async #prefillChunk(
    steps: readonly Qwen35GreedyTokenStep[],
    signal: AbortSignal,
  ): Promise<number | null> {
    if (this.#engine.prefillChunk === undefined) {
      let selected: number | null = null;
      for (const step of steps) selected = await this.#execute(step);
      return selected;
    }
    const operation = this.#engine.prefillChunk({ steps, signal });
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
    "dispatchBatch" | "submittedWorkDone" | "releaseBindGroups" | "dispose"
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
  // The slot copies a successful upload when it needs persistence. Reuse one
  // staging vector here instead of allocating one short typed array per GPU
  // command on every generated token.
  const words = new Uint32Array(MAX_UNIFORM_WORDS);
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
    words.fill(0);
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

/** Uploads the used prefix of a rolling layer's worst-case uniform capacity. */
export function uploadQwen35GreedyRollingPlanUniforms(input: {
  readonly planUniformCount: number;
  readonly reservedUniformCount: number;
  readonly commands: readonly UniformCommand[];
  readonly slots: readonly Qwen35GreedyUniformUploadSlot[];
}): void {
  if (
    !Number.isSafeInteger(input.planUniformCount) ||
    input.planUniformCount < 1 ||
    !Number.isSafeInteger(input.reservedUniformCount) ||
    input.reservedUniformCount < 1 ||
    input.planUniformCount > input.reservedUniformCount ||
    input.slots.length !== input.reservedUniformCount
  ) {
    throw diagnosticError(
      "greedy-uniform-schedule-invalid",
      "The Qwen3.5 rolling uniform schedule exceeds its reserved capacity",
    );
  }
  uploadQwen35GreedyUniformCommands({
    commands: input.commands,
    slots: input.slots,
    expectedSlotCount: input.planUniformCount,
  });
}

export interface Qwen35GreedyGpuBatchExecutor {
  dispatchBatch(requests: readonly Qwen35DispatchRequest[]): Promise<void>;
  dispatchBatchAndReadU32?(
    requests: readonly Qwen35DispatchRequest[],
    source: object,
    byteOffset: number,
  ): Promise<number>;
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

/** Marks possible persistent mutation before one layer submission can reach the queue. */
export async function executeQwen35GreedyRollingLayerDispatch(input: {
  readonly commands: readonly Qwen35DispatchRequest[];
  readonly executor: Pick<
    Qwen35GreedyGpuBatchExecutor,
    "dispatchBatch" | "submittedWorkDone"
  >;
  readonly mutation: Qwen35RollingLayerMutation;
  readonly signal: AbortSignal;
  /** Rolling-layer owners fence before destruction; permanent layers do not. */
  readonly waitForRetirement?: boolean;
}): Promise<void> {
  input.signal.throwIfAborted();
  try {
    input.mutation.markStateMutation();
    await input.executor.dispatchBatch(input.commands);
    if (input.waitForRetirement !== false) {
      await input.executor.submittedWorkDone();
    }
    input.signal.throwIfAborted();
  } catch (error) {
    if (
      input.signal.aborted ||
      (error instanceof DOMException && error.name === "AbortError")
    ) {
      throw abortError();
    }
    throw diagnosticError(
      "greedy-rolling-layer-dispatch-failed",
      "Qwen3.5 rolling layer dispatch failed",
    );
  }
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
    let token: number | null = null;
    if (
      input.selected !== null &&
      input.executor.dispatchBatchAndReadU32 !== undefined
    ) {
      readbackStarted = true;
      token = await input.executor.dispatchBatchAndReadU32(
        input.commands,
        input.selected.buffer,
        input.selected.offset,
      );
      readbackCompleted = true;
      retired = true;
    } else {
      await input.executor.dispatchBatch(input.commands);
    }
    if (input.selected === null) {
      await input.executor.submittedWorkDone();
      retired = true;
    } else if (!retired) {
      // Mapping the queued scalar copy resolves only after this dispatch and
      // every earlier queue operation complete. It is therefore the same
      // ownership proof as a separate submittedWorkDone fence, without the
      // extra wait-before-copy bubble.
      readbackStarted = true;
      token = await input.executor.readU32(
        input.selected.buffer,
        input.selected.offset,
      );
      readbackCompleted = true;
      retired = true;
    }
    input.state.advance(1);
    advanced = true;
    if (input.step.signal.aborted) {
      if (input.step.phase === "generation") input.poison();
      throw abortError();
    }
    if (token === null) return null;
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

/** Executes one streamed logits tile without advancing recurrent token state. */
export async function executeQwen35GreedyTiedTileScore(input: {
  readonly commands: readonly Qwen35DispatchRequest[];
  readonly executor: {
    dispatchBatch(commands: readonly Qwen35DispatchRequest[]): Promise<void>;
    submittedWorkDone(): Promise<void>;
    readU32(buffer: object, byteOffset: number): Promise<number>;
  };
  readonly tile: Qwen35StagedPackedRows;
  readonly candidateScoreReadback: { readonly buffer: object; readonly offset: number };
  readonly candidateTokenReadback: { readonly buffer: object; readonly offset: number };
  readonly signal: AbortSignal;
}): Promise<readonly Qwen35LogitCandidate[]> {
  input.signal.throwIfAborted();
  try {
    await input.executor.dispatchBatch(input.commands);
    await input.executor.submittedWorkDone();
    input.signal.throwIfAborted();
    // Executor error scopes are stack-owned, so the two tiny readbacks remain
    // serial even though their source bytes share one retired scratch buffer.
    const scoreBits = await input.executor.readU32(
      input.candidateScoreReadback.buffer,
      input.candidateScoreReadback.offset,
    );
    const tokenId = await input.executor.readU32(
      input.candidateTokenReadback.buffer,
      input.candidateTokenReadback.offset,
    );
    input.signal.throwIfAborted();
    if (tokenId === QWEN35_NO_SELECTED_TOKEN) return Object.freeze([]);
    const score = new Float32Array(Uint32Array.of(scoreBits).buffer)[0]!;
    if (
      !Number.isFinite(score) ||
      !Number.isSafeInteger(tokenId) ||
      tokenId < input.tile.firstRow ||
      tokenId >= input.tile.firstRow + input.tile.rowCount ||
      tokenId >= DECODABLE_TOKEN_COUNT
    ) {
      throw diagnosticError(
        "greedy-token-selection-invalid",
        "Qwen3.5 staged logits returned an invalid candidate",
      );
    }
    return Object.freeze([Object.freeze({ tokenId, score })]);
  } catch (error) {
    if (
      input.signal.aborted ||
      (error instanceof DOMException && error.name === "AbortError")
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
      "greedy-tied-tile-score-failed",
      "Qwen3.5 staged logits tile scoring failed",
    );
  }
}

/** Executes the final streamed candidate reduction and reads one token id. */
export async function executeQwen35GreedyStagedFinalToken(input: {
  readonly commands: readonly Qwen35DispatchRequest[];
  readonly executor: {
    dispatchBatch(commands: readonly Qwen35DispatchRequest[]): Promise<void>;
    submittedWorkDone(): Promise<void>;
    readU32(buffer: object, byteOffset: number): Promise<number>;
    dispatchBatchAndReadU32?(
      commands: readonly Qwen35DispatchRequest[],
      buffer: object,
      byteOffset: number,
    ): Promise<number>;
  };
  readonly selectedTokenReadback: { readonly buffer: object; readonly offset: number };
  readonly signal: AbortSignal;
}): Promise<number> {
  input.signal.throwIfAborted();
  try {
    const tokenId = input.executor.dispatchBatchAndReadU32 === undefined
      ? await (async () => {
          await input.executor.dispatchBatch(input.commands);
          await input.executor.submittedWorkDone();
          input.signal.throwIfAborted();
          return input.executor.readU32(
            input.selectedTokenReadback.buffer,
            input.selectedTokenReadback.offset,
          );
        })()
      : await input.executor.dispatchBatchAndReadU32(
          input.commands,
          input.selectedTokenReadback.buffer,
          input.selectedTokenReadback.offset,
        );
    input.signal.throwIfAborted();
    if (
      tokenId === QWEN35_NO_SELECTED_TOKEN ||
      !Number.isSafeInteger(tokenId) ||
      tokenId < 0 ||
      tokenId >= DECODABLE_TOKEN_COUNT
    ) {
      throw diagnosticError(
        "greedy-token-selection-invalid",
        "Qwen3.5 staged logits returned an invalid token",
      );
    }
    return tokenId;
  } catch (error) {
    if (
      input.signal.aborted ||
      (error instanceof DOMException && error.name === "AbortError")
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
      "greedy-tied-final-selection-failed",
      "Qwen3.5 staged logits final selection failed",
    );
  }
}

export async function disposeQwen35GreedyOwnedResources(input: {
  readonly executor: Pick<Qwen35WebGpuExecutor, "dispose">;
  readonly uniformArena: Pick<Qwen35UniformArena, "dispose"> | null;
  readonly workspace: Pick<Qwen35ActivationWorkspace, "dispose"> | null;
  readonly prefillUniformArena?: Pick<Qwen35UniformArena, "dispose"> | null;
  readonly prefillWorkspaces?: readonly Pick<Qwen35ActivationWorkspace, "dispose">[];
  readonly candidateScratch?: Pick<GpuAllocation, "destroy">;
}): Promise<void> {
  let failed = false;
  try {
    // The executor fence must settle before any bound arena buffer is released.
    await input.executor.dispose();
  } catch {
    failed = true;
  }
  const released = await Promise.allSettled([
    Promise.resolve().then(() => input.candidateScratch?.destroy()),
    input.uniformArena?.dispose() ?? Promise.resolve(),
    input.workspace?.dispose() ?? Promise.resolve(),
    input.prefillUniformArena?.dispose() ?? Promise.resolve(),
    ...(input.prefillWorkspaces ?? []).map((workspace) => workspace.dispose()),
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

function weightBuffers(weights: Qwen35WeightDirectoryView): readonly object[] {
  const buffers = new Set<object>();
  for (const [, tensor] of weights) {
    for (const row of tensor.physicalRows) buffers.add(row.buffer as object);
  }
  return Object.freeze([...buffers]);
}

function candidateScratchSlice(
  allocation: GpuAllocation,
): Qwen35ForwardBufferSlice {
  const shard = allocation.shards[0];
  if (
    allocation.logicalBytes !== 2_048n ||
    allocation.shards.length !== 1 ||
    shard === undefined ||
    shard.logicalByteOffset !== 0n ||
    shard.logicalByteLength !== 2_048n ||
    shard.allocatedByteLength < 2_048n
  ) {
    throw diagnosticError(
      "greedy-candidate-scratch-invalid",
      "Qwen3.5 staged candidate scratch is invalid",
    );
  }
  return Object.freeze({
    buffer: shard.buffer as object,
    offset: 0,
    byteLength: 2_048,
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
  readonly #tiedEmbedding: Qwen35DiskBackedTiedEmbeddingStore | null;
  readonly #rollingLayers: Qwen35RollingLayerStore | null;
  readonly #candidateScratch: GpuAllocation | null;
  readonly #candidateOutput: Qwen35ForwardBufferSlice | null;
  readonly #prefillWorkspaces: readonly Qwen35ActivationWorkspace[];
  readonly #prefillUniformArena: Qwen35UniformArena | null;
  readonly #prefillUniformSlots: readonly Qwen35UniformSlot[];
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
    readonly tiedEmbedding?: Qwen35DiskBackedTiedEmbeddingStore;
    readonly rollingLayers?: Qwen35RollingLayerStore;
    readonly candidateScratch?: GpuAllocation;
    readonly prefillWorkspaces?: readonly Qwen35ActivationWorkspace[];
    readonly prefillUniformArena?: Qwen35UniformArena;
  }) {
    this.#program = input.program;
    this.#weights = input.weights;
    this.#state = input.state;
    this.#workspace = input.workspace;
    this.#uniformArena = input.uniformArena;
    this.#geometry = input.geometry;
    this.#executor = input.executor;
    this.#limits = snapshotQwen35ForwardDeviceLimits(input.device);
    this.#layers = attentionInvocations(input.program);
    this.#final = finalInvocation(input.program);
    this.#tiedEmbedding = input.tiedEmbedding ?? null;
    this.#rollingLayers = input.rollingLayers ?? null;
    this.#candidateScratch = input.candidateScratch ?? null;
    this.#prefillWorkspaces = Object.freeze([...(input.prefillWorkspaces ?? [])]);
    this.#prefillUniformArena = input.prefillUniformArena ?? null;
    this.#prefillUniformSlots = this.#prefillUniformArena === null
      ? Object.freeze([])
      : Object.freeze(Array.from(
          { length: input.geometry.uniformSlotCount * PREFILL_CHUNK_SIZE },
          (_, index) => this.#prefillUniformArena!.slot(index, MAX_UNIFORM_WORDS),
        ));
    if ((this.#tiedEmbedding === null) !== (this.#candidateScratch === null)) {
      throw diagnosticError(
        "greedy-candidate-scratch-invalid",
        "Qwen3.5 staged candidate ownership is incomplete",
      );
    }
    this.#candidateOutput = this.#candidateScratch === null
      ? null
      : candidateScratchSlice(this.#candidateScratch);
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

  /** Runs a bounded prompt chunk or the serial fallback when pool allocation is unavailable. */
  async prefillChunk(input: {
    readonly steps: readonly Qwen35GreedyTokenStep[];
    readonly signal: AbortSignal;
  }): Promise<number | null> {
    if (
      this.#prefillUniformArena !== null &&
      this.#prefillWorkspaces.length > 0
    ) {
      return this.#prefillChunkLayerMajor(input);
    }
    let selected: number | null = null;
    for (const step of input.steps) selected = await this.execute(step);
    return selected;
  }
  /**
   * Executes a bounded prompt chunk layer-major. This keeps rolling weights
   * resident for one layer while all prompt rows use that layer, then releases
   * the layer before moving on. The same schedule also batches resident layers.
   */
  async #prefillChunkLayerMajor(input: {
    readonly steps: readonly Qwen35GreedyTokenStep[];
    readonly signal: AbortSignal;
  }): Promise<number | null> {
    if (
      input.steps.length < 1 ||
      input.steps.length > PREFILL_CHUNK_SIZE ||
      input.steps.some((step, index) =>
        step.phase !== "prefill" ||
        (index < input.steps.length - 1 && step.predict)
      ) ||
      this.position + input.steps.length > this.capacity
    ) {
      throw diagnosticError(
        "greedy-prefill-chunk-invalid",
        "Qwen3.5 prefill chunk is invalid",
      );
    }
    input.signal.throwIfAborted();
    const startPosition = this.position;
    await this.#state.ensureCapacity(startPosition + input.steps.length, input.signal);
    let persistentSubmission = false;
    const tokenSlots = (index: number): readonly Qwen35UniformSlot[] => {
      const start = index * this.#geometry.uniformSlotCount;
      return this.#prefillUniformSlots.slice(
        start,
        start + this.#geometry.uniformSlotCount,
      );
    };

    try {
      const residentBatcher = createQwen35ResidentLayerBatcher({
        dispatchBatch: (commands) => {
          persistentSubmission = true;
          return this.#executor.dispatchBatch(commands);
        },
      });
      const embeddingCommands: UniformCommand[] = [];
      for (const [index, step] of input.steps.entries()) {
        const workspace = this.#prefillWorkspaces[index];
        if (workspace === undefined) {
          throw diagnosticError(
            "greedy-prefill-chunk-invalid",
            "Qwen3.5 prefill workspace is incomplete",
          );
        }
        const slots = tokenSlots(index);
        const stagedInput = this.#tiedEmbedding === null
          ? null
          : await stageQwen35GreedyInputEmbedding({
              tiedEmbedding: this.#tiedEmbedding,
              step,
            });
        const embedding = stagedInput === null
          ? step.embeddingOverride === undefined
            ? planQwen35PackedEmbeddingDispatch({
                weights: this.#weights,
                tokenId: step.tokenId,
                output: workspaceSlice(workspace, "packed-embedding-output"),
                uniform: slots[this.#geometry.embeddingUniform]!.binding,
                limits: this.#limits,
              })
            : planQwen35VisualEmbeddingDispatch({
                source: step.embeddingOverride,
                output: workspaceSlice(workspace, "packed-embedding-output"),
                uniform: slots[this.#geometry.embeddingUniform]!.binding,
                limits: this.#limits,
              })
          : stagedInput.kind === "visual"
            ? planQwen35VisualEmbeddingDispatch({
                source: stagedInput.source,
                output: workspaceSlice(workspace, "packed-embedding-output"),
                uniform: slots[this.#geometry.embeddingUniform]!.binding,
                limits: this.#limits,
              })
            : planQwen35StagedPackedEmbeddingDispatch({
                rows: stagedInput.rows,
                output: workspaceSlice(workspace, "packed-embedding-output"),
                uniform: slots[this.#geometry.embeddingUniform]!.binding,
                limits: this.#limits,
              });
        uploadQwen35GreedyUniformCommands({
          commands: [embedding],
          slots: [slots[this.#geometry.embeddingUniform]!],
          expectedSlotCount: 1,
        });
        embeddingCommands.push(Object.freeze({
          ...embedding,
          uniformWords: embedding.uniformWords,
        }));
      }

      residentBatcher.enqueue(embeddingCommands);

      const executeLayer = async (layerInput: {
        readonly invocation: AttentionInvocation;
        readonly weights: Qwen35WeightDirectoryView;
        readonly mutation: Qwen35RollingLayerMutation;
        readonly transientWeights?: boolean;
      }): Promise<void> => {
        const { invocation, weights, mutation } = layerInput;
        const transientWeights = layerInput.transientWeights ?? false;
        const geometry = this.#geometry.layers[invocation.layer];
        if (
          geometry === undefined ||
          geometry.layer !== invocation.layer ||
          geometry.kind !== invocation.kind
        ) {
          throw diagnosticError(
            "greedy-uniform-schedule-invalid",
            "Qwen3.5 prefill layer geometry is incomplete",
          );
        }
        const layerCommands: UniformCommand[] = [];
        const state = this.#state.getLayerResources(invocation.layer);
        for (const [tokenIndex] of input.steps.entries()) {
          const workspace = this.#prefillWorkspaces[tokenIndex];
          if (workspace === undefined) {
            throw diagnosticError(
              "greedy-prefill-chunk-invalid",
              "Qwen3.5 prefill workspace is incomplete",
            );
          }
          const slots = tokenSlots(tokenIndex).slice(
            geometry.uniformStart,
            geometry.uniformStart + geometry.uniformCount,
          );
          const plan = invocation.kind === "gated-deltanet"
            ? planQwen35DeltaNetLayerDispatch({
                program: this.#program,
                invocation,
                weights,
                workspace,
                deltanetParameterLiveness: WORKSPACE_LIVENESS,
                state,
                limits: this.#limits,
                uniforms: slots.map((slot) => slot.binding),
              })
            : planQwen35FullAttentionLayerDispatch({
                program: this.#program,
                invocation,
                weights,
                workspace,
                state,
                position: startPosition + tokenIndex,
                capacity: this.capacity,
                mropePositions: [
                  startPosition + tokenIndex,
                  startPosition + tokenIndex,
                  startPosition + tokenIndex,
                ],
                limits: this.#limits,
                uniforms: slots.map((slot) => slot.binding),
              });
          uploadQwen35GreedyRollingPlanUniforms({
            planUniformCount: plan.uniformCount,
            reservedUniformCount: geometry.uniformCount,
            commands: plan.commands,
            slots,
          });
          layerCommands.push(...plan.commands);
        }
        if (!transientWeights) {
          mutation.markStateMutation();
          residentBatcher.enqueue(layerCommands);
          return;
        }
        await residentBatcher.flush();
        persistentSubmission = true;
        await executeQwen35GreedyRollingLayerDispatch({
          commands: layerCommands,
          executor: this.#executor,
          mutation,
          signal: input.signal,
          // The rolling owner supplies the only per-layer fence. Permanent
          // layers remain ordered on the queue and retire with the final batch.
          waitForRetirement: false,
        });
        if (transientWeights) {
          this.#executor.releaseBindGroupsForBuffers(weightBuffers(weights));
        }
      };

      if (this.#rollingLayers === null) {
        for (const invocation of this.#layers) {
          await executeLayer({
            invocation,
            weights: this.#weights,
            mutation: { markStateMutation() {} },
          });
        }
      } else {
        await executeQwen35RollingLayerSequence({
          invocations: this.#layers,
          permanentWeights: this.#weights,
          rollingStore: this.#rollingLayers,
          phase: "prefill",
          signal: input.signal,
          poison: () => { this.#poisoned = true; },
          execute: executeLayer,
        });
      }
      await residentBatcher.flush();

      const lastStep = input.steps.at(-1)!;
      input.signal.throwIfAborted();
      this.#state.advance(input.steps.length);
      if (!lastStep.predict) return null;
      const lastIndex = input.steps.length - 1;
      const lastWorkspace = this.#prefillWorkspaces[lastIndex];
      if (lastWorkspace === undefined) {
        throw diagnosticError(
          "greedy-prefill-chunk-invalid",
          "Qwen3.5 final prefill workspace is incomplete",
        );
      }
      const lastSlots = tokenSlots(lastIndex);
      const finalUniform = lastSlots[this.#geometry.finalNormUniform];
      if (finalUniform === undefined) {
        throw diagnosticError(
          "greedy-uniform-schedule-invalid",
          "Qwen3.5 prefill final uniform slot is incomplete",
        );
      }
      const final = planQwen35FinalNormDispatch({
        program: this.#program,
        invocation: this.#final,
        weights: this.#weights,
        workspace: lastWorkspace,
        limits: this.#limits,
        uniform: finalUniform.binding,
      });

      if (this.#tiedEmbedding === null) {
        const logitsSlots = lastSlots.slice(
          this.#geometry.logitsUniformStart,
          this.#geometry.logitsUniformStart + this.#geometry.logitsUniformCount,
        );
        const logits = assembleQwen35TiledLogitsCommands({
          weights: this.#weights,
          normalizedHidden: workspaceSlice(lastWorkspace, "normalized-hidden"),
          workspace: lastWorkspace,
          limits: this.#limits,
          uniforms: logitsSlots.map((slot) => slot.binding),
        });
        const tailCommands: UniformCommand[] = [final.command, ...logits.commands];
        const tailSlots = [finalUniform, ...logitsSlots];
        uploadQwen35GreedyUniformCommands({
          commands: tailCommands,
          slots: tailSlots,
          expectedSlotCount: tailSlots.length,
        });
        await this.#executor.dispatchBatch(tailCommands);
        await this.#executor.submittedWorkDone();
        this.#executor.releaseBindGroups();
        const token = await this.#executor.readU32(
          logits.selectedTokenReadback.buffer,
          logits.selectedTokenReadback.offset,
        );
        input.signal.throwIfAborted();
        if (
          token === QWEN35_NO_SELECTED_TOKEN ||
          token < 0 ||
          token >= DECODABLE_TOKEN_COUNT
        ) {
          throw diagnosticError(
            "greedy-token-selection-invalid",
            "Qwen3.5 prefill returned an invalid token",
          );
        }
        return token;
      }

      const stagedLogitsSlots = lastSlots.slice(
        this.#geometry.logitsUniformStart,
        this.#geometry.logitsUniformStart + STAGED_LOGITS_TOTAL_UNIFORM_COUNT,
      );
      if (
        stagedLogitsSlots.length !== STAGED_LOGITS_TOTAL_UNIFORM_COUNT ||
        this.#candidateOutput === null
      ) {
        throw diagnosticError(
          "greedy-uniform-schedule-invalid",
          "Qwen3.5 staged logits uniform capacity is incomplete",
        );
      }
      uploadQwen35GreedyUniformCommands({
        commands: [final.command],
        slots: [finalUniform],
        expectedSlotCount: 1,
      });
      await this.#executor.dispatchBatch([final.command]);
      await this.#executor.submittedWorkDone();
      this.#executor.releaseBindGroups();
      const normalizedHidden = workspaceSlice(lastWorkspace, "normalized-hidden");
      const stagedLogitsTileSlots = stagedLogitsTileUniformSlots(
        stagedLogitsSlots,
        0,
      );
      const stagedLogitsFinalSlots = stagedLogitsFinalUniformSlots(stagedLogitsSlots);
      if (
        stagedLogitsTileSlots.length !== STAGED_LOGITS_UNIFORM_COUNT ||
        stagedLogitsFinalSlots.length !== STAGED_LOGITS_UNIFORM_COUNT
      ) {
        throw diagnosticError(
          "greedy-uniform-schedule-invalid",
          "Qwen3.5 staged logits uniform capacity is incomplete",
        );
      }
      if (typeof this.#tiedEmbedding.selectTopKGpu === "function") {
        const batcher = createQwen35StagedLogitsBatcher(this.#executor);
        const token = await selectQwen35GreedyTiedTokenGpu({
          tiedEmbedding: this.#tiedEmbedding,
          phase: lastStep.phase,
          signal: input.signal,
          scoreTile: async (tile, candidateSlot) => {
            const tileUniformSlots = stagedLogitsTileUniformSlots(
              stagedLogitsSlots,
              candidateSlot,
            );
            const planned = assembleQwen35StagedLogitsTileGpuCommands({
              tile,
              normalizedHidden,
              workspace: lastWorkspace,
              candidateOutput: this.#candidateOutput!,
              candidateSlot,
              limits: this.#limits,
              uniforms: tileUniformSlots.map((slot) => slot.binding),
            });
            uploadQwen35GreedyUniformCommands({
              commands: planned.commands,
              slots: tileUniformSlots,
              expectedSlotCount: STAGED_LOGITS_UNIFORM_COUNT,
            });
            batcher.enqueue(planned.commands);
          },
          flush: () => batcher.flush(),
          finalize: async () => {
            const finalSlot = stagedLogitsFinalUniformSlots(stagedLogitsSlots)[1];
            if (finalSlot === undefined) {
              throw diagnosticError(
                "greedy-uniform-schedule-invalid",
                "Qwen3.5 staged logits final uniform slot is incomplete",
              );
            }
            const selectedToken = workspaceSlice(lastWorkspace, "selected-token");
            const finalSelection = assembleQwen35StagedFinalTokenCommand({
              candidateOutput: this.#candidateOutput!,
              selectedToken,
              limits: this.#limits,
              uniform: finalSlot.binding,
            });
            uploadQwen35GreedyUniformCommands({
              commands: [finalSelection],
              slots: [finalSlot],
              expectedSlotCount: 1,
            });
            const token = await executeQwen35GreedyStagedFinalToken({
              commands: [finalSelection],
              executor: this.#executor,
              selectedTokenReadback: {
                buffer: selectedToken.buffer,
                offset: selectedToken.offset,
              },
              signal: input.signal,
            });
            this.#executor.releaseBindGroups();
            return token;
          },
        });
        return token;
      }
      return await selectQwen35GreedyTiedToken({
        tiedEmbedding: this.#tiedEmbedding,
        phase: lastStep.phase,
        signal: input.signal,
        scoreTile: async (tile) => {
          const planned = assembleQwen35StagedLogitsTileCommands({
            tile,
            normalizedHidden,
            workspace: lastWorkspace,
            candidateOutput: this.#candidateOutput!,
            limits: this.#limits,
            uniforms: stagedLogitsTileSlots.map((slot) => slot.binding),
          });
          uploadQwen35GreedyUniformCommands({
            commands: planned.commands,
            slots: stagedLogitsTileSlots,
            expectedSlotCount: STAGED_LOGITS_UNIFORM_COUNT,
          });
          const candidates = await executeQwen35GreedyTiedTileScore({
            commands: planned.commands,
            executor: this.#executor,
            tile,
            candidateScoreReadback: planned.candidateScoreReadback,
            candidateTokenReadback: planned.candidateTokenReadback,
            signal: input.signal,
          });
          this.#executor.releaseBindGroups();
          return candidates;
        },
      });
    } catch (error) {
      if (persistentSubmission) this.#poisoned = true;
      throw error;
    }
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

    // State residency follows the used context. This is inside the tracked GPU
    // operation so cancellation and disposal wait for transactional page growth.
    await ensureQwen35GreedyTokenState(this.#state, step.signal);

    if (this.#rollingLayers !== null) {
      return this.#executeRolling(step);
    }

    const commands: UniformCommand[] = [];
    let uniformCursor = 1;
    const stagedInput = this.#tiedEmbedding === null
      ? null
      : await stageQwen35GreedyInputEmbedding({
          tiedEmbedding: this.#tiedEmbedding,
          step,
        });
    const embedding = stagedInput === null
      ? step.embeddingOverride === undefined
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
          })
      : stagedInput.kind === "visual"
        ? planQwen35VisualEmbeddingDispatch({
            source: stagedInput.source,
            output: workspaceSlice(this.#workspace, "packed-embedding-output"),
            uniform: this.#uniformSlots[this.#geometry.embeddingUniform]!.binding,
            limits: this.#limits,
          })
        : planQwen35StagedPackedEmbeddingDispatch({
            rows: stagedInput.rows,
            output: workspaceSlice(this.#workspace, "packed-embedding-output"),
            uniform: this.#uniformSlots[this.#geometry.embeddingUniform]!.binding,
            limits: this.#limits,
          });
    commands.push(Object.freeze({ ...embedding, uniformWords: embedding.uniformWords }));

    for (const [index, invocation] of this.#layers.entries()) {
      const geometry = this.#geometry.layers[index]!;
      const uniforms = this.#uniformSlots
        .slice(uniformCursor, uniformCursor + geometry.uniformCount)
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
      uniformCursor += plan.uniformCount;
    }

    let selected: ReturnType<typeof assembleQwen35TiledLogitsCommands>["selectedTokenReadback"] | null = null;
    let stagedLogitsSlots: readonly Qwen35UniformSlot[] | null = null;
    if (step.predict) {
      const finalUniform = this.#uniformSlots[uniformCursor];
      if (finalUniform === undefined) {
        throw diagnosticError(
          "greedy-uniform-schedule-invalid",
          "Qwen3.5 final uniform capacity is incomplete",
        );
      }
      const final = planQwen35FinalNormDispatch({
        program: this.#program,
        invocation: this.#final,
        weights: this.#weights,
        workspace: this.#workspace,
        limits: this.#limits,
        uniform: finalUniform.binding,
      });
      commands.push(final.command);
      uniformCursor += 1;
      if (this.#tiedEmbedding !== null) {
        stagedLogitsSlots = this.#uniformSlots.slice(
          uniformCursor,
          uniformCursor + STAGED_LOGITS_TOTAL_UNIFORM_COUNT,
        );
        if (
          this.#geometry.logitsUniformCount !== STAGED_LOGITS_TOTAL_UNIFORM_COUNT ||
          stagedLogitsSlots.length !== STAGED_LOGITS_TOTAL_UNIFORM_COUNT ||
          this.#candidateOutput === null
        ) {
          throw diagnosticError(
            "greedy-uniform-schedule-invalid",
            "Qwen3.5 staged logits uniform capacity is incomplete",
          );
        }
      } else {
        const logits = assembleQwen35TiledLogitsCommands({
          weights: this.#weights,
          normalizedHidden: workspaceSlice(this.#workspace, "normalized-hidden"),
          workspace: this.#workspace,
          limits: this.#limits,
          uniforms: this.#uniformSlots
            .slice(
              uniformCursor,
              uniformCursor + this.#geometry.logitsUniformCount,
            )
            .map((slot) => slot.binding),
        });
        commands.push(...logits.commands);
        uniformCursor += this.#geometry.logitsUniformCount;
        selected = logits.selectedTokenReadback;
      }
    }

    if (uniformCursor > this.#geometry.uniformSlotCount) {
      throw diagnosticError(
        "greedy-uniform-schedule-invalid",
        "Qwen3.5 dynamic uniform schedule exceeds its fixed capacity",
      );
    }
    uploadQwen35GreedyUniformCommands({
      commands,
      slots: this.#uniformSlots,
      expectedSlotCount: uniformCursor,
    });
    step.signal.throwIfAborted();

    const committed = await executeQwen35GreedyGpuBatch({
      commands,
      executor: this.#executor,
      state: this.#state,
      selected,
      step,
      poison: () => { this.#poisoned = true; },
    });
    if (
      !step.predict ||
      this.#tiedEmbedding === null ||
      stagedLogitsSlots === null ||
      this.#candidateOutput === null
    ) {
      return committed;
    }
    try {
      const normalizedHidden = workspaceSlice(this.#workspace, "normalized-hidden");
      const stagedLogitsTileSlots = stagedLogitsSlots.slice(
        0,
        STAGED_LOGITS_UNIFORM_COUNT,
      );
      const stagedLogitsFinalSlots = stagedLogitsSlots.slice(
        STAGED_LOGITS_TILE_UNIFORM_COUNT,
      );
      if (
        stagedLogitsTileSlots.length !== STAGED_LOGITS_UNIFORM_COUNT ||
        stagedLogitsFinalSlots.length !== STAGED_LOGITS_UNIFORM_COUNT
      ) {
        throw diagnosticError(
          "greedy-uniform-schedule-invalid",
          "Qwen3.5 staged logits uniform capacity is incomplete",
        );
      }
      if (typeof this.#tiedEmbedding.selectTopKGpu === "function") {
        const batcher = createQwen35StagedLogitsBatcher(this.#executor);
        return await selectQwen35GreedyTiedTokenGpu({
          tiedEmbedding: this.#tiedEmbedding,
          phase: step.phase,
          signal: step.signal,
          scoreTile: async (tile, candidateSlot) => {
            const tileUniformSlots = stagedLogitsTileUniformSlots(
              stagedLogitsSlots,
              candidateSlot,
            );
            const planned = assembleQwen35StagedLogitsTileGpuCommands({
              tile,
              normalizedHidden,
              workspace: this.#workspace,
              candidateOutput: this.#candidateOutput!,
              candidateSlot,
              limits: this.#limits,
              uniforms: tileUniformSlots.map((slot) => slot.binding),
            });
            uploadQwen35GreedyUniformCommands({
              commands: planned.commands,
              slots: tileUniformSlots,
              expectedSlotCount: STAGED_LOGITS_UNIFORM_COUNT,
            });
            batcher.enqueue(planned.commands);
          },
          flush: () => batcher.flush(),
          finalize: async () => {
            const finalSlot = stagedLogitsFinalSlots[1];
            if (finalSlot === undefined) {
              throw diagnosticError(
                "greedy-uniform-schedule-invalid",
                "Qwen3.5 staged logits final uniform slot is incomplete",
              );
            }
            const final = assembleQwen35StagedFinalTokenCommand({
              candidateOutput: this.#candidateOutput!,
              selectedToken: workspaceSlice(this.#workspace, "selected-token"),
              limits: this.#limits,
              uniform: finalSlot.binding,
            });
            uploadQwen35GreedyUniformCommands({
              commands: [final],
              slots: [finalSlot],
              expectedSlotCount: 1,
            });
            const selectedToken = workspaceSlice(this.#workspace, "selected-token");
            return executeQwen35GreedyStagedFinalToken({
              commands: [final],
              executor: this.#executor,
              selectedTokenReadback: {
                buffer: selectedToken.buffer,
                offset: selectedToken.offset,
              },
              signal: step.signal,
            });
          },
        });
      }
      return await selectQwen35GreedyTiedToken({
        tiedEmbedding: this.#tiedEmbedding,
        phase: step.phase,
        signal: step.signal,
        scoreTile: async (tile) => {
          const planned = assembleQwen35StagedLogitsTileCommands({
            tile,
            normalizedHidden,
            workspace: this.#workspace,
            candidateOutput: this.#candidateOutput!,
            limits: this.#limits,
            uniforms: stagedLogitsTileSlots.map((slot) => slot.binding),
          });
          uploadQwen35GreedyUniformCommands({
            commands: planned.commands,
            slots: stagedLogitsTileSlots,
            expectedSlotCount: STAGED_LOGITS_UNIFORM_COUNT,
          });
          return executeQwen35GreedyTiedTileScore({
            commands: planned.commands,
            executor: this.#executor,
            tile,
            candidateScoreReadback: planned.candidateScoreReadback,
            candidateTokenReadback: planned.candidateTokenReadback,
            signal: step.signal,
          });
        },
      });
    } catch (error) {
      // The base batch has already advanced recurrent and KV state. Any later
      // retry of this token would double-apply it, so this engine cannot recover.
      this.#poisoned = true;
      throw error;
    }
  }

  async #executeRolling(step: Qwen35GreedyTokenStep): Promise<number | null> {
    const stagedInput = this.#tiedEmbedding === null
      ? null
      : await stageQwen35GreedyInputEmbedding({
          tiedEmbedding: this.#tiedEmbedding,
          step,
        });
    const embedding = stagedInput === null
      ? step.embeddingOverride === undefined
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
          })
      : stagedInput.kind === "visual"
        ? planQwen35VisualEmbeddingDispatch({
            source: stagedInput.source,
            output: workspaceSlice(this.#workspace, "packed-embedding-output"),
            uniform: this.#uniformSlots[this.#geometry.embeddingUniform]!.binding,
            limits: this.#limits,
          })
        : planQwen35StagedPackedEmbeddingDispatch({
            rows: stagedInput.rows,
            output: workspaceSlice(this.#workspace, "packed-embedding-output"),
            uniform: this.#uniformSlots[this.#geometry.embeddingUniform]!.binding,
            limits: this.#limits,
          });
    uploadQwen35GreedyUniformCommands({
      commands: [embedding],
      slots: [this.#uniformSlots[this.#geometry.embeddingUniform]!],
      expectedSlotCount: 1,
    });
    const residentBatcher = createQwen35ResidentLayerBatcher(this.#executor);
    residentBatcher.enqueue([embedding]);

    await executeQwen35RollingLayerSequence({
      invocations: this.#layers,
      permanentWeights: this.#weights,
      rollingStore: this.#rollingLayers!,
      phase: step.phase === "generation" ? "decode" : "prefill",
      signal: step.signal,
      poison: () => { this.#poisoned = true; },
      execute: async ({ invocation, weights, mutation, transientWeights }) => {
        const geometry = this.#geometry.layers[invocation.layer];
        if (
          geometry === undefined ||
          geometry.layer !== invocation.layer ||
          geometry.kind !== invocation.kind
        ) {
          throw diagnosticError(
            "greedy-uniform-schedule-invalid",
            "Qwen3.5 rolling layer uniform geometry is invalid",
          );
        }
        const slots = this.#uniformSlots.slice(
          geometry.uniformStart,
          geometry.uniformStart + geometry.uniformCount,
        );
        const state = this.#state.getLayerResources(invocation.layer);
        const plan = invocation.kind === "gated-deltanet"
          ? planQwen35DeltaNetLayerDispatch({
              program: this.#program,
              invocation,
              weights,
              workspace: this.#workspace,
              deltanetParameterLiveness: WORKSPACE_LIVENESS,
              state,
              limits: this.#limits,
              uniforms: slots.map((slot) => slot.binding),
            })
          : planQwen35FullAttentionLayerDispatch({
              program: this.#program,
              invocation,
              weights,
              workspace: this.#workspace,
              state,
              position: this.position,
              capacity: this.capacity,
              mropePositions: [this.position, this.position, this.position],
              limits: this.#limits,
              uniforms: slots.map((slot) => slot.binding),
            });
        uploadQwen35GreedyRollingPlanUniforms({
          planUniformCount: plan.uniformCount,
          reservedUniformCount: geometry.uniformCount,
          commands: plan.commands,
          slots,
        });
        if (!transientWeights) {
          mutation.markStateMutation();
          residentBatcher.enqueue(plan.commands);
          return;
        }
        await residentBatcher.flush();
        await executeQwen35GreedyRollingLayerDispatch({
          commands: plan.commands,
          executor: this.#executor,
          mutation,
          signal: step.signal,
          // The rolling store fences transient buffers before destruction. A
          // later token fence retires permanent-layer work in queue order.
          waitForRetirement: false,
        });
        if (transientWeights) {
          // Temporary layer buffers must not remain strongly held after their
          // owner destroys them; resident-layer groups remain reusable.
          this.#executor.releaseBindGroupsForBuffers(weightBuffers(weights));
        }
      },
    });
    await residentBatcher.flush();

    if (!step.predict) {
      try {
        step.signal.throwIfAborted();
        this.#state.advance(1);
        return null;
      } catch (error) {
        this.#poisoned = true;
        if (
          step.signal.aborted ||
          (error instanceof DOMException && error.name === "AbortError")
        ) {
          throw abortError();
        }
        throw diagnosticError(
          "greedy-gpu-batch-failed",
          "Qwen3.5 rolling token state advancement failed",
        );
      }
    }

    try {
      const finalSlot = this.#uniformSlots[this.#geometry.finalNormUniform];
      if (finalSlot === undefined) {
        throw diagnosticError(
          "greedy-uniform-schedule-invalid",
          "Qwen3.5 final uniform capacity is incomplete",
        );
      }
      const final = planQwen35FinalNormDispatch({
        program: this.#program,
        invocation: this.#final,
        weights: this.#weights,
        workspace: this.#workspace,
        limits: this.#limits,
        uniform: finalSlot.binding,
      });
      let tailCommands: readonly UniformCommand[];
      let tailSlots: readonly Qwen35UniformSlot[];
      let selected: ReturnType<typeof assembleQwen35TiledLogitsCommands>["selectedTokenReadback"] | null = null;
      let stagedLogitsSlots: readonly Qwen35UniformSlot[] | null = null;
      if (this.#tiedEmbedding === null) {
        const logitsSlots = this.#uniformSlots.slice(
          this.#geometry.logitsUniformStart,
          this.#geometry.logitsUniformStart + this.#geometry.logitsUniformCount,
        );
        const logits = assembleQwen35TiledLogitsCommands({
          weights: this.#weights,
          normalizedHidden: workspaceSlice(this.#workspace, "normalized-hidden"),
          workspace: this.#workspace,
          limits: this.#limits,
          uniforms: logitsSlots.map((slot) => slot.binding),
        });
        tailCommands = Object.freeze([final.command, ...logits.commands]);
        tailSlots = Object.freeze([finalSlot, ...logitsSlots]);
        selected = logits.selectedTokenReadback;
      } else {
        stagedLogitsSlots = this.#uniformSlots.slice(
          this.#geometry.logitsUniformStart,
          this.#geometry.logitsUniformStart + STAGED_LOGITS_TOTAL_UNIFORM_COUNT,
        );
        if (
          stagedLogitsSlots.length !== STAGED_LOGITS_TOTAL_UNIFORM_COUNT ||
          this.#candidateOutput === null
        ) {
          throw diagnosticError(
            "greedy-uniform-schedule-invalid",
            "Qwen3.5 staged logits uniform capacity is incomplete",
          );
        }
        tailCommands = Object.freeze([final.command]);
        tailSlots = Object.freeze([finalSlot]);
      }
      uploadQwen35GreedyUniformCommands({
        commands: tailCommands,
        slots: tailSlots,
        expectedSlotCount: tailSlots.length,
      });
      const committed = await executeQwen35GreedyGpuBatch({
        commands: tailCommands,
        executor: this.#executor,
        state: this.#state,
        selected,
        step,
        poison: () => { this.#poisoned = true; },
      });
      if (
        this.#tiedEmbedding === null ||
        stagedLogitsSlots === null ||
        this.#candidateOutput === null
      ) {
        return committed;
      }
      const normalizedHidden = workspaceSlice(this.#workspace, "normalized-hidden");
      const stagedLogitsTileSlots = stagedLogitsTileUniformSlots(
        stagedLogitsSlots,
        0,
      );
      const stagedLogitsFinalSlots = stagedLogitsFinalUniformSlots(stagedLogitsSlots);
      if (
        stagedLogitsTileSlots.length !== STAGED_LOGITS_UNIFORM_COUNT ||
        stagedLogitsFinalSlots.length !== STAGED_LOGITS_UNIFORM_COUNT
      ) {
        throw diagnosticError(
          "greedy-uniform-schedule-invalid",
          "Qwen3.5 staged logits uniform capacity is incomplete",
        );
      }
      if (typeof this.#tiedEmbedding.selectTopKGpu === "function") {
        const batcher = createQwen35StagedLogitsBatcher(this.#executor);
        return await selectQwen35GreedyTiedTokenGpu({
          tiedEmbedding: this.#tiedEmbedding,
          phase: step.phase,
          signal: step.signal,
          scoreTile: async (tile, candidateSlot) => {
            const tileUniformSlots = stagedLogitsTileUniformSlots(
              stagedLogitsSlots,
              candidateSlot,
            );
            const planned = assembleQwen35StagedLogitsTileGpuCommands({
              tile,
              normalizedHidden,
              workspace: this.#workspace,
              candidateOutput: this.#candidateOutput!,
              candidateSlot,
              limits: this.#limits,
              uniforms: tileUniformSlots.map((slot) => slot.binding),
            });
            uploadQwen35GreedyUniformCommands({
              commands: planned.commands,
              slots: tileUniformSlots,
              expectedSlotCount: STAGED_LOGITS_UNIFORM_COUNT,
            });
            batcher.enqueue(planned.commands);
          },
          flush: () => batcher.flush(),
          finalize: async () => {
            const finalSlot = stagedLogitsFinalSlots[1];
            if (finalSlot === undefined) {
              throw diagnosticError(
                "greedy-uniform-schedule-invalid",
                "Qwen3.5 staged logits final uniform slot is incomplete",
              );
            }
            const final = assembleQwen35StagedFinalTokenCommand({
              candidateOutput: this.#candidateOutput!,
              selectedToken: workspaceSlice(this.#workspace, "selected-token"),
              limits: this.#limits,
              uniform: finalSlot.binding,
            });
            uploadQwen35GreedyUniformCommands({
              commands: [final],
              slots: [finalSlot],
              expectedSlotCount: 1,
            });
            const selectedToken = workspaceSlice(this.#workspace, "selected-token");
            return executeQwen35GreedyStagedFinalToken({
              commands: [final],
              executor: this.#executor,
              selectedTokenReadback: {
                buffer: selectedToken.buffer,
                offset: selectedToken.offset,
              },
              signal: step.signal,
            });
          },
        });
      }
      return await selectQwen35GreedyTiedToken({
        tiedEmbedding: this.#tiedEmbedding,
        phase: step.phase,
        signal: step.signal,
        scoreTile: async (tile) => {
          const planned = assembleQwen35StagedLogitsTileCommands({
            tile,
            normalizedHidden,
            workspace: this.#workspace,
            candidateOutput: this.#candidateOutput!,
            limits: this.#limits,
            uniforms: stagedLogitsTileSlots.map((slot) => slot.binding),
          });
          uploadQwen35GreedyUniformCommands({
            commands: planned.commands,
            slots: stagedLogitsTileSlots,
            expectedSlotCount: STAGED_LOGITS_UNIFORM_COUNT,
          });
          return executeQwen35GreedyTiedTileScore({
            commands: planned.commands,
            executor: this.#executor,
            tile,
            candidateScoreReadback: planned.candidateScoreReadback,
            candidateTokenReadback: planned.candidateTokenReadback,
            signal: step.signal,
          });
        },
      });
    } catch (error) {
      // All 32 layers have already mutated persistent token state at this point.
      this.#poisoned = true;
      throw error;
    }
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
      await Promise.all([
        this.#workspace.reset(),
        ...this.#prefillWorkspaces.map((workspace) => workspace.reset()),
      ]);
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
      prefillUniformArena: this.#prefillUniformArena,
      prefillWorkspaces: this.#prefillWorkspaces,
      ...(this.#candidateScratch === null
        ? {}
        : { candidateScratch: this.#candidateScratch }),
    });
  }
}

/**
 * Copies the limit fields by name because WebIDL objects may expose them as
 * prototype getters. Object spread drops those fields in Safari and leaves the
 * execution planner without its required device limits.
 */
export function snapshotQwen35ForwardDeviceLimits(
  device: Pick<Qwen35WebGpuDevice, "limits" | "features">,
): Qwen35ForwardDeviceLimits {
  const subgroupFeature = device.features !== undefined &&
    Array.from(device.features).includes("subgroups");
  const subgroupLimitsAbsent = device.limits.minSubgroupSize === undefined &&
    device.limits.maxSubgroupSize === undefined;
  const subgroupLimitsAre32 = device.limits.minSubgroupSize === 32 &&
    device.limits.maxSubgroupSize === 32;
  return Object.freeze({
    minStorageBufferOffsetAlignment:
      device.limits.minStorageBufferOffsetAlignment,
    minUniformBufferOffsetAlignment:
      device.limits.minUniformBufferOffsetAlignment,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxUniformBufferBindingSize: device.limits.maxUniformBufferBindingSize,
    maxComputeWorkgroupsPerDimension:
      device.limits.maxComputeWorkgroupsPerDimension,
    // Current Chrome exposes the subgroup feature but omits the provisional
    // size-limit fields. Its optimized kernels are covered by the physical GPU
    // parity harness. If a browser does expose size limits, require SIMD32.
    supportsSubgroups: subgroupFeature &&
      (subgroupLimitsAbsent || subgroupLimitsAre32),
  });
}

// Kept as one immutable shared object so every layer validates the same ranges.
const WORKSPACE_LIVENESS = planQwen35ActivationWorkspace().deltanetParameterLiveness;

async function createGpuDriver(
  context: Qwen35DriverFactoryContext,
): Promise<Qwen35ExecutionDriver> {
  requireRunnableProgram(context.program);
  // Geometry and execution must use the same capability snapshot. Passing the
  // raw WebIDL limits here made Chrome reserve portable logits uniforms while
  // the engine selected subgroup kernels from device feature evidence.
  const limits = snapshotQwen35ForwardDeviceLimits(context.device);
  const geometry = planQwen35GreedyUniformGeometry({
    program: context.program,
    weights: context.weightDirectory,
    limits,
    diskBackedTiedEmbedding: context.tiedEmbedding !== undefined,
    ...(context.rollingLayers === undefined
      ? {}
      : { rollingLayers: context.rollingLayers }),
  });
  const clearer = createQwen35AllocationClearer(
    context.device,
    context.performanceCounters,
  );
  const executionQueue = context.performanceCounters === undefined
    ? context.device.queue
    : createQwen35PerformanceWriteQueue({
        queue: context.device.queue,
        counters: context.performanceCounters,
      });
  const executor = new Qwen35WebGpuExecutor(
    context.device,
    context.performanceCounters,
  );
  let workspace: Qwen35ActivationWorkspace | null = null;
  let uniformArena: Qwen35UniformArena | null = null;
  let prefillUniformArena: Qwen35UniformArena | null = null;
  const prefillWorkspaces: Qwen35ActivationWorkspace[] = [];
  let candidateScratch: GpuAllocation | null = null;
  try {
    if (context.tiedEmbedding !== undefined) {
      candidateScratch = await context.arena.allocate({
        id: "tied-logits-candidate-scratch",
        category: "scratch",
        byteLength: 2_048n,
        usage: GPU_STORAGE_AND_COPY_SRC,
        alignment: 4,
        requiredShardQuantumBytes: 2_048n,
      });
      candidateScratchSlice(candidateScratch);
    }
    workspace = await createQwen35ActivationWorkspace({
      arena: context.arena,
      clearAllocation: (allocation) => clearer.clearAllocation(allocation),
    });
    uniformArena = await createQwen35UniformArena({
      arena: context.arena,
      queue: executionQueue,
      slotCount: geometry.uniformSlotCount,
      slotWordCapacity: geometry.maxUniformWords,
      minUniformBufferOffsetAlignment: limits.minUniformBufferOffsetAlignment,
      maxUniformBufferBindingSize: limits.maxUniformBufferBindingSize,
    });
    // The pool is useful for both resident and rolling paths. On a constrained
    // device it is optional: keep the proven serial executor if the extra
    // transient allocations do not fit the live ledger.
    try {
      for (let index = 0; index < PREFILL_CHUNK_SIZE; index += 1) {
        prefillWorkspaces.push(await createQwen35ActivationWorkspace({
          arena: context.arena,
          clearAllocation: (allocation) => clearer.clearAllocation(allocation),
          allocationIdPrefix: `qwen35-prefill-${index}`,
        }));
      }
      prefillUniformArena = await createQwen35UniformArena({
        arena: context.arena,
        queue: executionQueue,
        allocationId: "qwen35-prefill-uniforms",
        slotCount: geometry.uniformSlotCount * PREFILL_CHUNK_SIZE,
        slotWordCapacity: geometry.maxUniformWords,
        minUniformBufferOffsetAlignment: limits.minUniformBufferOffsetAlignment,
        maxUniformBufferBindingSize: limits.maxUniformBufferBindingSize,
      });
    } catch {
      await Promise.all(prefillWorkspaces.splice(0).map((item) => item.dispose()));
      prefillUniformArena = null;
    }
    return createQwen35GreedyTextDriver(new Qwen35GpuTokenEngine({
      device: context.device,
      program: context.program,
      weights: context.weightDirectory,
      state: context.hybridState,
      workspace,
      uniformArena,
      geometry,
      executor,
      ...(context.tiedEmbedding === undefined
        ? {}
        : { tiedEmbedding: context.tiedEmbedding }),
      ...(context.rollingLayers === undefined
        ? {}
        : { rollingLayers: context.rollingLayers }),
      ...(candidateScratch === null ? {} : { candidateScratch }),
      prefillWorkspaces,
      ...(prefillUniformArena === null ? {} : { prefillUniformArena }),
    }), executor);
  } catch (error) {
    try {
      await disposeQwen35GreedyOwnedResources({
        executor,
        uniformArena,
        workspace,
        prefillUniformArena,
        prefillWorkspaces,
        ...(candidateScratch === null ? {} : { candidateScratch }),
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
  const clearers = new WeakMap<object, {
    readonly performanceCounters: Qwen35DriverFactoryContext["performanceCounters"];
    readonly clearer: ReturnType<typeof createQwen35AllocationClearer>;
  }>();
  const clearer = (
    device: Qwen35WebGpuDevice,
    performanceCounters: Qwen35DriverFactoryContext["performanceCounters"],
  ) => {
    const key = device as object;
    let current = clearers.get(key);
    if (
      current === undefined ||
      current.performanceCounters !== performanceCounters
    ) {
      current = {
        performanceCounters,
        clearer: createQwen35AllocationClearer(device, performanceCounters),
      };
      clearers.set(key, current);
    }
    return current.clearer;
  };
  return Object.freeze({
    clearStateAllocation(context: Qwen35StateAllocationClearContext) {
      return clearer(context.device, context.performanceCounters)
        .clearAllocation(context.allocation);
    },
    create(context: Qwen35DriverFactoryContext) {
      return createGpuDriver(context);
    },
  });
}
