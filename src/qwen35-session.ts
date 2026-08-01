import { diagnosticError } from "./diagnostics.js";
import {
  QWEN35_PRODUCT_CONTEXT_TOKENS,
  assembleQwen35Conversation,
  type Qwen35ChatMessage,
} from "./qwen-chat-template.js";
import {
  OriginModelLock,
  OriginModelLockCleanupError,
  browserExclusiveLockManager,
  type ExclusiveLockManager,
} from "./origin-model-lock.js";
import type { Qwen35Tokenizer } from "./qwen-tokenizer.js";
import type { Qwen35ForwardBufferSlice } from "./qwen35-forward-dispatch.js";
import type { Qwen35VisionPatchBatch } from "./qwen35-vision-preprocess.js";
import type { Qwen35WebGpuExecutor } from "./qwen35-webgpu-executor.js";

export type Qwen35SessionState =
  | "idle"
  | "loading"
  | "ready"
  | "prefilling"
  | "generating"
  | "cancelling"
  | "resetting"
  | "disposing"
  | "disposed"
  | "failed";

export type TextOrImageConversation = readonly Qwen35ChatMessage[];

export interface SequenceState {
  readonly rendered: string;
  readonly contextTokens: number;
  readonly remainingContextTokens: number;
}

export interface GenerateOptions {
  readonly maxNewTokens: number;
  readonly temperature?: 0;
  readonly topK?: 1;
}

export interface GeneratedToken {
  readonly id: number;
  readonly text: string;
  readonly index: number;
}

export interface Qwen35DriverPrefillInput {
  readonly tokenIds: readonly number[];
  /** Replaces one image-pad token with projected rows during prefill. */
  readonly visualEmbeddings?: readonly {
    readonly tokenId: number;
    readonly tokenCount: number;
    readonly source: Qwen35ForwardBufferSlice;
  }[];
  readonly signal: AbortSignal;
}

export interface Qwen35DriverGenerateInput {
  readonly maxNewTokens: number;
  readonly signal: AbortSignal;
  readonly logitMask: {
    readonly start: number;
    readonly count: number;
  };
}

/** Model-specific scheduler boundary; it is not a reusable tensor API. */
export interface Qwen35ExecutionDriver {
  /** Borrowed by the lazy vision runtime; the driver remains its owner. */
  readonly sharedGpuExecutor?: Pick<
    Qwen35WebGpuExecutor,
    "dispatchBatch" | "submittedWorkDone" | "dispose"
  >;
  /** A rejected or cancelled prefill may be partial; reset must clear it fully. */
  prefill(input: Qwen35DriverPrefillInput): Promise<void>;
  generate(input: Qwen35DriverGenerateInput): AsyncIterable<number>;
  /** Resolves only after active driver work is quiescent. */
  cancel(): Promise<void>;
  reset(): Promise<void>;
  dispose(): Promise<void>;
}

export interface Qwen35LoadedResources {
  readonly tokenizer: Qwen35Tokenizer;
  readonly driver: Qwen35ExecutionDriver;
  readonly cacheHit: boolean;
  readonly trackedCpuBytes: number;
  readonly trackedGpuBytes: number;
  readonly gpuByteMetrics?: () => {
    readonly currentBytes: number;
    readonly peakBytes: number;
  };
  readonly deviceLost?: Promise<unknown>;
  readonly vision?: {
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
  };
  dispose(): Promise<void>;
}

export interface LoadOptions {
  readonly signal?: AbortSignal;
  readonly [key: string]: unknown;
}

export interface Qwen35SessionRuntime {
  readonly lockManager: ExclusiveLockManager;
  readonly now?: () => number;
  load(
    signal: AbortSignal,
    options: LoadOptions,
  ): Promise<Qwen35LoadedResources>;
}

export interface RuntimePhaseMetrics {
  readonly count: number;
  readonly totalMilliseconds: number;
  readonly lastMilliseconds: number;
}

export interface RuntimeMetrics {
  readonly state: Qwen35SessionState;
  readonly phases: Readonly<Record<string, RuntimePhaseMetrics>>;
  readonly cacheHit: boolean | null;
  readonly trackedCpuBytes: number;
  readonly trackedGpuBytes: number;
  readonly peakTrackedGpuBytes: number;
  readonly contextTokens: number;
  readonly timeToFirstTokenMilliseconds: number | null;
  readonly prefillTokens: number;
  readonly prefillTokensPerSecond: number | null;
  readonly generatedTokens: number;
  readonly generatedTokensPerSecond: number | null;
  readonly cancellationCount: number;
  readonly deviceLostCount: number;
}

interface MutablePhaseMetrics {
  count: number;
  totalMilliseconds: number;
  lastMilliseconds: number;
}

interface Deferred<T> {
  readonly promise: Promise<T>;
  resolve(value: T): void;
  reject(error: unknown): void;
}

interface GenerationOperation {
  readonly settled: Deferred<void>;
  driverIterator: AsyncIterator<number> | null;
  cancellationPromise: Promise<void> | null;
  retirementPromise: Promise<void> | null;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((accept, decline) => {
    resolve = accept;
    reject = decline;
  });
  return { promise, resolve, reject };
}

function requireBoundedInteger(value: number, label: string): number {
  if (
    !Number.isSafeInteger(value) ||
    value < 1 ||
    value > QWEN35_PRODUCT_CONTEXT_TOKENS
  ) {
    throw diagnosticError(
      "generate-option-invalid",
      `${label} is outside the product context bound`,
    );
  }
  return value;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === "AbortError"
    : error instanceof Error && error.name === "AbortError";
}

function productionRuntime(): Qwen35SessionRuntime {
  const locks = globalThis.navigator?.locks;
  return {
    lockManager:
      locks === undefined
        ? {
            async request() {
              throw diagnosticError(
                "web-locks-unavailable",
                "The browser Web Locks API is unavailable",
              );
            },
          }
        : browserExclusiveLockManager(locks),
    async load(signal, options) {
      const { loadQwen35BrowserResources } = await import(
        "./qwen35-model-loader.js"
      );
      return loadQwen35BrowserResources(signal, options);
    },
  };
}

/** Owns one loaded Qwen3.5 model from origin-lock acquisition through cleanup. */
export class Qwen35Session {
  readonly #runtime: Qwen35SessionRuntime;
  readonly #now: () => number;
  readonly #phases = new Map<string, MutablePhaseMetrics>();
  #state: Qwen35SessionState = "idle";
  #resources: Qwen35LoadedResources | null = null;
  #lock: OriginModelLock | null = null;
  #lockPromise: Promise<void> | null = null;
  #lifetime: Deferred<void> | null = null;
  #disposePromise: Promise<void> | null = null;
  #disposeRequested = false;
  #loadCancellationRequested = false;
  #operationController: AbortController | null = null;
  #operationSettled: Deferred<void> | null = null;
  #generationOperation: GenerationOperation | null = null;
  #activeGenerate = false;
  #operationCancellationPromise: Promise<void> | null = null;
  #sequenceTokenIds: readonly number[] | null = null;
  #contextTokens = 0;
  #cacheHit: boolean | null = null;
  #trackedCpuBytes = 0;
  #trackedGpuBytes = 0;
  #peakTrackedGpuBytes = 0;
  #ttft: number | null = null;
  #prefillTokens = 0;
  #prefillMilliseconds = 0;
  #generatedTokens = 0;
  #generationMilliseconds = 0;
  #cancellationCount = 0;
  #deviceLostCount = 0;

  constructor(runtime?: Qwen35SessionRuntime) {
    this.#runtime = runtime ?? productionRuntime();
    this.#now =
      runtime?.now ??
      (() => globalThis.performance?.now() ?? Date.now());
  }

  get state(): Qwen35SessionState {
    return this.#state;
  }

  async load(options: LoadOptions): Promise<void> {
    if (this.#state !== "idle") {
      throw diagnosticError(
        "session-load-illegal",
        "Session load is legal only from idle",
      );
    }
    this.#state = "loading";
    const ready = deferred<void>();
    const lifetime = deferred<void>();
    this.#lifetime = lifetime;
    const lock = new OriginModelLock(this.#runtime.lockManager);
    this.#lock = lock;
    const started = this.#now();

    // The lock callback stays pending for the complete loaded lifetime. Returning
    // from load only resolves `ready`; only dispose resolves `lifetime`.
    const lockPromise = lock.runExclusive({
      run: async (signal) => {
        try {
          options.signal?.throwIfAborted();
          const resources = await this.#runtime.load(signal, options);
          // Ownership transfers before either cancellation check so lock-held
          // cleanup also covers a runtime that finishes concurrently with abort.
          this.#resources = resources;
          signal.throwIfAborted();
          if (options.signal?.aborted) {
            throw options.signal.reason;
          }
          this.#cacheHit = resources.cacheHit;
          this.#trackedCpuBytes = resources.trackedCpuBytes;
          this.#trackedGpuBytes = resources.trackedGpuBytes;
          this.#peakTrackedGpuBytes = resources.trackedGpuBytes;
          if (resources.deviceLost !== undefined) {
            void resources.deviceLost.then(
              () => this.#onDeviceLost(),
              () => this.#onDeviceLost(),
            );
          }
          this.#state = "ready";
          this.#recordPhase("load", started);
          ready.resolve(undefined);
          await lifetime.promise;
        } catch (error) {
          ready.reject(error);
          throw error;
        }
      },
      cleanup: async () => {
        const resources = this.#resources;
        await resources?.dispose();
        const gpuMetrics = resources?.gpuByteMetrics?.();
        if (gpuMetrics !== undefined) {
          this.#peakTrackedGpuBytes = Math.max(
            this.#peakTrackedGpuBytes,
            gpuMetrics.peakBytes,
          );
        }
        // Successful cleanup releases every GPU allocation. Preserve only the
        // high-water mark after resource ownership is gone.
        this.#trackedGpuBytes = 0;
        this.#resources = null;
      },
    });
    this.#lockPromise = lockPromise;
    // Observe the lifetime promise immediately. Load and dispose await the same
    // promise for their phase-specific error handling.
    void lockPromise.catch(() => undefined);

    const abort = (): void => lock.cancel();
    options.signal?.addEventListener("abort", abort, { once: true });
    try {
      // A lock request can be cancelled before its callback starts, so the
      // callback-owned `ready` promise cannot be the only load completion path.
      await Promise.race([ready.promise, lockPromise]);
    } catch (error) {
      const lockError = await lockPromise.then(
        () => null,
        (failure: unknown) => failure,
      );
      const stateAfterLock = this.#state as Qwen35SessionState;
      if (stateAfterLock !== "disposing" && stateAfterLock !== "disposed") {
        this.#state = "failed";
      }
      throw lockError ?? error;
    } finally {
      options.signal?.removeEventListener("abort", abort);
    }
  }

  async prefill(input: TextOrImageConversation): Promise<SequenceState> {
    this.#requireReady();
    if (this.#sequenceTokenIds !== null) {
      throw diagnosticError(
        "session-reset-required",
        "Reset the session before replacing the prefetched conversation",
      );
    }
    const resources = this.#resources!;
    const images = input.flatMap((message) =>
      Array.isArray(message.content)
        ? message.content.filter((part): part is Extract<typeof part, { type: "image" }> => part.type === "image")
        : [],
    );
    // The template reserves one image-pad token before preprocessing. Once the
    // browser has selected an actual visual-token budget, use that same count
    // for both context accounting and the driver expansion. Keeping the
    // default here is correct for an unprocessed image, while a smaller local
    // smoke-test budget must not make the driver walk past its projection.
    const visualTokensPerImage = images.length === 1 && images[0]!.patches !== undefined
      ? images[0]!.patches.projectedVisualTokens
      : undefined;
    const assembled = assembleQwen35Conversation(
      resources.tokenizer,
      input,
      visualTokensPerImage === undefined ? {} : { visualTokensPerImage },
    );
    if (!assembled.ok) {
      throw diagnosticError(
        "context-limit-exceeded",
        "Conversation exceeds the 16384-token context limit",
      );
    }
    const controller = new AbortController();
    const operationSettled = deferred<void>();
    this.#operationCancellationPromise = null;
    this.#operationController = controller;
    this.#operationSettled = operationSettled;
    this.#state = "prefilling";
    const started = this.#now();
    let projected: { readonly tokenCount: number; readonly storage: Qwen35ForwardBufferSlice; dispose(): Promise<void> } | null = null;
    try {
      if (images.length > 0) {
        if (resources.vision === undefined || images.some((image) => image.patches === undefined)) {
          throw diagnosticError("vision-not-loaded", "Vision input is not available in the loaded runtime");
        }
        if (images.length !== 1) {
          throw diagnosticError("vision-image-count-invalid", "Only one image is supported per user turn");
        }
        const image = images[0]!;
        const patches = image.patches as Qwen35VisionPatchBatch;
        projected = await resources.vision.encode({
          patches: patches.patches,
          gridHeight: patches.gridTHW[1],
          gridWidth: patches.gridTHW[2],
          signal: controller.signal,
        });
      }
      await resources.driver.prefill({
        tokenIds: assembled.tokenIds,
        ...(projected === null ? {} : {
          visualEmbeddings: [{
            tokenId: resources.tokenizer.addedTokenId("<|image_pad|>")!,
            tokenCount: projected.tokenCount,
            source: projected.storage,
          }],
        }),
        signal: controller.signal,
      });
      await projected?.dispose();
      projected = null;
      controller.signal.throwIfAborted();
      this.#sequenceTokenIds = assembled.tokenIds;
      this.#contextTokens = assembled.promptTokenCount;
      this.#prefillTokens += assembled.promptTokenCount;
      const elapsed = this.#recordPhase("prefill", started);
      this.#prefillMilliseconds += elapsed;
      this.#state = "ready";
      return Object.freeze({
        rendered: assembled.rendered,
        contextTokens: assembled.promptTokenCount,
        remainingContextTokens: assembled.remainingContextTokens,
      });
    } catch (error) {
      try { await projected?.dispose(); } catch { /* Preserve prefill failure. */ }
      projected = null;
      if (controller.signal.aborted && this.#operationCancellationPromise !== null) {
        let cancellationFailure: unknown;
        let rollbackFailure: unknown;
        try {
          // Reset must not overlap driver cancellation. Every observer waits on
          // this same promise before ownership can become ready or failed.
          await this.#operationCancellationPromise;
        } catch (error) {
          cancellationFailure = error;
        }
        if (cancellationFailure !== undefined) {
          if (!this.#disposeRequested) {
            this.#state = "failed";
          }
          throw cancellationFailure;
        }
        try {
          // A successful cancel proves the driver is quiescent. Only then may
          // reset clear partial prefill state for reuse.
          await resources.driver.reset();
          this.#sequenceTokenIds = null;
          this.#contextTokens = 0;
        } catch (error) {
          rollbackFailure = error;
        }
        if (rollbackFailure !== undefined) {
          if (!this.#disposeRequested) {
            this.#state = "failed";
          }
          throw diagnosticError(
            "prefill-rollback-failed",
            "Cancelled prefill rollback did not complete",
          );
        }
        if (!this.#disposeRequested) {
          this.#state = "ready";
        }
      } else {
        if (!this.#disposeRequested) {
          this.#state = "failed";
        }
      }
      throw error;
    } finally {
      this.#operationController = null;
      this.#operationSettled = null;
      operationSettled.resolve(undefined);
    }
  }

  generate(options: GenerateOptions): AsyncIterable<GeneratedToken> {
    if (this.#activeGenerate) {
      throw diagnosticError(
        "session-generate-active",
        "Only one generation iterator may be active",
      );
    }
    this.#requireReady();
    if (this.#sequenceTokenIds === null) {
      throw diagnosticError(
        "session-prefill-required",
        "Generation requires a prefetched conversation",
      );
    }
    if (options.temperature !== undefined && options.temperature !== 0) {
      throw diagnosticError(
        "sampling-not-supported",
        "Only greedy generation is available",
      );
    }
    if (options.topK !== undefined && options.topK !== 1) {
      throw diagnosticError(
        "sampling-not-supported",
        "Only greedy generation is available",
      );
    }
    const maxNewTokens = requireBoundedInteger(
      options.maxNewTokens,
      "Maximum generated token count",
    );
    if (this.#contextTokens + maxNewTokens > QWEN35_PRODUCT_CONTEXT_TOKENS) {
      throw diagnosticError(
        "context-limit-exceeded",
        "Generation would exceed the 16384-token context limit",
      );
    }

    this.#activeGenerate = true;
    this.#operationCancellationPromise = null;
    this.#ttft = null;
    this.#state = "generating";
    const controller = new AbortController();
    const operation: GenerationOperation = {
      settled: deferred<void>(),
      driverIterator: null,
      cancellationPromise: null,
      retirementPromise: null,
    };
    this.#operationSettled = operation.settled;
    this.#generationOperation = operation;
    this.#operationController = controller;
    return this.#generateIterator(
      maxNewTokens,
      controller,
      operation,
    );
  }

  async cancel(): Promise<void> {
    if (this.#state === "loading") {
      if (this.#loadCancellationRequested) {
        return;
      }
      this.#loadCancellationRequested = true;
      this.#cancellationCount += 1;
      this.#lock?.cancel();
      return;
    }
    if (
      this.#disposeRequested &&
      this.#operationCancellationPromise !== null
    ) {
      try {
        await this.#operationCancellationPromise;
      } finally {
        this.#settleCancelledGeneration();
      }
      return;
    }
    if (
      this.#state !== "prefilling" &&
      this.#state !== "generating" &&
      this.#state !== "cancelling"
    ) {
      return;
    }
    try {
      await this.#requestOperationCancellation();
    } finally {
      this.#settleCancelledGeneration();
    }
  }

  async reset(): Promise<void> {
    this.#requireReady();
    const started = this.#now();
    const deviceLostCount = this.#deviceLostCount;
    const operationSettled = deferred<void>();
    this.#operationSettled = operationSettled;
    this.#state = "resetting";
    try {
      await this.#resources!.driver.reset();
      if (this.#state !== "resetting") {
        if (this.#deviceLostCount !== deviceLostCount) {
          throw diagnosticError(
            "device-lost",
            "WebGPU device ownership was lost during reset",
          );
        }
        throw diagnosticError(
          "session-operation-superseded",
          "Session reset was superseded by a terminal lifecycle operation",
        );
      }
      this.#sequenceTokenIds = null;
      this.#contextTokens = 0;
      this.#operationCancellationPromise = null;
      this.#recordPhase("reset", started);
      this.#state = "ready";
    } catch (error) {
      if (this.#state === "resetting") {
        this.#state = "failed";
      }
      throw error;
    } finally {
      if (this.#operationSettled === operationSettled) {
        this.#operationSettled = null;
      }
      operationSettled.resolve(undefined);
    }
  }

  dispose(): Promise<void> {
    this.#disposeRequested = true;
    this.#disposePromise ??= this.#dispose();
    return this.#disposePromise;
  }

  getMetrics(): RuntimeMetrics {
    const gpuMetrics = this.#resources?.gpuByteMetrics?.();
    const phases = Object.fromEntries(
      [...this.#phases]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([name, phase]) => [name, Object.freeze({ ...phase })]),
    );
    return Object.freeze({
      state: this.#state,
      phases: Object.freeze(phases),
      cacheHit: this.#cacheHit,
      trackedCpuBytes: this.#trackedCpuBytes,
      trackedGpuBytes: gpuMetrics?.currentBytes ?? this.#trackedGpuBytes,
      peakTrackedGpuBytes:
        gpuMetrics?.peakBytes ?? this.#peakTrackedGpuBytes,
      contextTokens: this.#contextTokens,
      timeToFirstTokenMilliseconds: this.#ttft,
      prefillTokens: this.#prefillTokens,
      prefillTokensPerSecond:
        this.#prefillMilliseconds > 0
          ? (this.#prefillTokens * 1_000) / this.#prefillMilliseconds
          : null,
      generatedTokens: this.#generatedTokens,
      generatedTokensPerSecond:
        this.#generationMilliseconds > 0
          ? (this.#generatedTokens * 1_000) / this.#generationMilliseconds
          : null,
      cancellationCount: this.#cancellationCount,
      deviceLostCount: this.#deviceLostCount,
    });
  }

  async *#generateIterator(
    maxNewTokens: number,
    controller: AbortController,
    operation: GenerationOperation,
  ): AsyncGenerator<GeneratedToken> {
    if (this.#generationOperation !== operation) {
      await operation.cancellationPromise;
      return;
    }
    const resources = this.#resources!;
    const decoder = resources.tokenizer.createStreamingDecoder({
      skipSpecialTokens: true,
    });
    const started = this.#now();
    let completed = false;
    let generationFailed = false;
    let generated = 0;
    let lastTokenId: number | null = null;
    try {
      const tokens = resources.driver.generate({
        maxNewTokens,
        signal: controller.signal,
        logitMask: Object.freeze({
          start: resources.tokenizer.decodableTokenCount,
          count: resources.tokenizer.undecodableLogitRows,
        }),
      });
      const iterator = tokens[Symbol.asyncIterator]();
      operation.driverIterator = iterator;
      while (true) {
        // A session-side cancellation can detach this consumer while it is
        // paused at yield. Check before asking the driver for more work.
        controller.signal.throwIfAborted();
        const item = await iterator.next();
        controller.signal.throwIfAborted();
        if (item.done) break;
        const id = item.value;
        if (generated >= maxNewTokens) {
          throw diagnosticError(
            "driver-token-count-exceeded",
            "Execution driver exceeded the requested token count",
          );
        }
        if (!resources.tokenizer.isDecodableTokenId(id)) {
          throw diagnosticError(
            "driver-token-id-invalid",
            "Execution driver returned an unmapped token id",
          );
        }
        const text = decoder.push(id);
        lastTokenId = id;
        generated += 1;
        this.#generatedTokens += 1;
        this.#contextTokens += 1;
        if (this.#ttft === null) {
          this.#ttft = Math.max(0, this.#now() - started);
        }
        yield Object.freeze({ id, text, index: generated - 1 });
      }
      const suffix = decoder.finish();
      if (suffix.length > 0 && lastTokenId !== null) {
        yield Object.freeze({
          id: lastTokenId,
          text: suffix,
          index: generated - 1,
        });
      }
      completed = true;
    } catch (error) {
      const explicitlyCancelled =
        this.#operationCancellationPromise !== null && isAbortError(error);
      generationFailed = !explicitlyCancelled;
      throw error;
    } finally {
      let cancellationFailure: unknown;
      const ownedAtEntry = this.#generationOperation === operation;
      if (!completed) {
        try {
          if (ownedAtEntry) {
            await this.#requestOperationCancellation();
          } else {
            await operation.cancellationPromise;
          }
        } catch (error) {
          cancellationFailure = error;
          generationFailed = true;
        }
      }
      const ownsOperation = this.#generationOperation === operation;
      if (ownsOperation) {
        const elapsed = this.#recordPhase("generate", started);
        this.#generationMilliseconds += elapsed;
        this.#activeGenerate = false;
        this.#operationController = null;
        operation.settled.resolve(undefined);
        this.#operationSettled = null;
        this.#generationOperation = null;
        if (this.#disposeRequested) {
          // Disposal owns the terminal state transition.
        } else if (generationFailed) {
          this.#state = "failed";
        } else if (
          this.#state === "generating" ||
          this.#state === "cancelling"
        ) {
          this.#state = "ready";
        }
      }
      if (cancellationFailure !== undefined) {
        throw cancellationFailure;
      }
    }
  }

  #requestOperationCancellation(): Promise<void> {
    if (this.#operationCancellationPromise !== null) {
      return this.#operationCancellationPromise;
    }
    this.#cancellationCount += 1;
    if (!this.#disposeRequested) {
      this.#state = "cancelling";
    }
    this.#operationController?.abort();
    const generationOperation = this.#generationOperation;
    this.#operationCancellationPromise = (async () => {
      try {
        await this.#resources!.driver.cancel();
      } catch {
        const error = diagnosticError(
          "driver-cancel-failed",
          "Execution driver cancellation did not complete",
        );
        if (!this.#disposeRequested) {
          this.#state = "failed";
        }
        throw error;
      }
      if (generationOperation !== null) {
        await this.#retireGenerationIterator(generationOperation);
      }
    })();
    if (generationOperation !== null) {
      generationOperation.cancellationPromise =
        this.#operationCancellationPromise;
    }
    return this.#operationCancellationPromise;
  }

  #retireGenerationIterator(
    operation: GenerationOperation,
  ): Promise<void> {
    if (operation.retirementPromise !== null) {
      return operation.retirementPromise;
    }
    const iterator = operation.driverIterator;
    operation.retirementPromise = (async () => {
      try {
        await iterator?.return?.();
      } catch {
        if (!this.#disposeRequested) {
          this.#state = "failed";
        }
        throw diagnosticError(
          "driver-iterator-retire-failed",
          "Execution driver iterator retirement did not complete",
        );
      }
    })();
    return operation.retirementPromise;
  }

  async #dispose(): Promise<void> {
    if (this.#state === "disposed") {
      return;
    }
    if (this.#state === "idle") {
      this.#state = "disposed";
      return;
    }
    const started = this.#now();
    const wasLoading = this.#state === "loading";
    const wasFailed = this.#state === "failed";
    let cancellationFailure: unknown;
    this.#state = "disposing";
    if (wasLoading) {
      this.#lock?.cancel();
    } else {
      if (this.#operationCancellationPromise !== null) {
        try {
          await this.#operationCancellationPromise;
        } catch (error) {
          cancellationFailure = error;
        }
      }
    }
    if (!wasLoading && this.#operationSettled !== null) {
      const operationSettled = this.#operationSettled;
      if (this.#operationController !== null) {
        try {
          await this.#requestOperationCancellation();
        } catch (error) {
          cancellationFailure ??= error;
        } finally {
          this.#settleCancelledGeneration();
        }
      }
      await operationSettled.promise;
    }
    this.#lifetime?.resolve(undefined);
    try {
      await this.#lockPromise;
    } catch (error) {
      if (wasLoading && isAbortError(error)) {
        this.#recordPhase("dispose", started);
        this.#state = "disposed";
        return;
      }
      if (wasFailed && !(error instanceof OriginModelLockCleanupError)) {
        this.#recordPhase("dispose", started);
        this.#state = "disposed";
        return;
      }
      this.#state = "failed";
      throw error;
    }
    this.#recordPhase("dispose", started);
    if (cancellationFailure !== undefined) {
      this.#state = "failed";
      throw cancellationFailure;
    }
    this.#state = "disposed";
  }

  #requireReady(): void {
    if (this.#state !== "ready") {
      throw diagnosticError(
        "session-not-ready",
        "Session operation requires the ready state",
      );
    }
  }

  #recordPhase(name: string, started: number): number {
    const elapsed = Math.max(0, this.#now() - started);
    const phase = this.#phases.get(name) ?? {
      count: 0,
      totalMilliseconds: 0,
      lastMilliseconds: 0,
    };
    phase.count += 1;
    phase.totalMilliseconds += elapsed;
    phase.lastMilliseconds = elapsed;
    this.#phases.set(name, phase);
    return elapsed;
  }

  #onDeviceLost(): void {
    if (
      this.#disposeRequested ||
      this.#state === "disposed" ||
      this.#state === "disposing"
    ) {
      return;
    }
    this.#deviceLostCount += 1;
    this.#operationController?.abort();
    this.#state = "failed";
  }

  #settleCancelledGeneration(): void {
    if (!this.#activeGenerate) {
      return;
    }
    const operation = this.#generationOperation;
    const operationSettled = operation?.settled ?? this.#operationSettled;
    const cancellation = this.#operationCancellationPromise;
    if (operation !== null && cancellation !== null) {
      operation.cancellationPromise = cancellation;
    }
    this.#activeGenerate = false;
    this.#operationController = null;
    operationSettled?.resolve(undefined);
    this.#operationSettled = null;
    this.#generationOperation = null;
    if (!this.#disposeRequested && this.#state === "cancelling") {
      this.#state = "ready";
    }
  }
}
