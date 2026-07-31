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
  /** A rejected or cancelled prefill may be partial; reset must clear it fully. */
  prefill(input: Qwen35DriverPrefillInput): Promise<void>;
  generate(input: Qwen35DriverGenerateInput): AsyncIterable<number>;
  cancel?(): Promise<void>;
  reset(): Promise<void>;
  dispose(): Promise<void>;
}

export interface Qwen35LoadedResources {
  readonly tokenizer: Qwen35Tokenizer;
  readonly driver: Qwen35ExecutionDriver;
  readonly cacheHit: boolean;
  readonly trackedCpuBytes: number;
  readonly trackedGpuBytes: number;
  readonly deviceLost?: Promise<unknown>;
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
  #loadCancellationRequested = false;
  #operationController: AbortController | null = null;
  #operationSettled: Deferred<void> | null = null;
  #activeGenerate = false;
  #generateStarted = false;
  #operationCancellationRequested = false;
  #sequenceTokenIds: readonly number[] | null = null;
  #contextTokens = 0;
  #cacheHit: boolean | null = null;
  #trackedCpuBytes = 0;
  #trackedGpuBytes = 0;
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
        await this.#resources?.dispose();
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
    if (
      input.some((message) =>
        Array.isArray(message.content) &&
        message.content.some((part) => part.type === "image"),
      )
    ) {
      throw diagnosticError(
        "vision-not-loaded",
        "Vision input is not available in this runtime milestone",
      );
    }
    const resources = this.#resources!;
    const assembled = assembleQwen35Conversation(
      resources.tokenizer,
      input,
    );
    if (!assembled.ok) {
      throw diagnosticError(
        "context-limit-exceeded",
        "Conversation exceeds the 16384-token context limit",
      );
    }
    const controller = new AbortController();
    const operationSettled = deferred<void>();
    this.#operationCancellationRequested = false;
    this.#operationController = controller;
    this.#operationSettled = operationSettled;
    this.#state = "prefilling";
    const started = this.#now();
    try {
      await resources.driver.prefill({
        tokenIds: assembled.tokenIds,
        signal: controller.signal,
      });
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
      if (controller.signal.aborted && this.#operationCancellationRequested) {
        try {
          // Prefill may commit a prefix before it observes abort. Clear that
          // partial state before this session becomes reusable.
          await resources.driver.reset();
          this.#sequenceTokenIds = null;
          this.#contextTokens = 0;
          this.#state = "ready";
        } catch {
          this.#state = "failed";
          throw diagnosticError(
            "prefill-rollback-failed",
            "Cancelled prefill rollback did not complete",
          );
        }
      } else {
        this.#state = "failed";
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
    this.#generateStarted = false;
    this.#operationCancellationRequested = false;
    this.#ttft = null;
    this.#state = "generating";
    const controller = new AbortController();
    this.#operationSettled = deferred<void>();
    this.#operationController = controller;
    return this.#generateIterator(maxNewTokens, controller);
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
      this.#state !== "prefilling" &&
      this.#state !== "generating" &&
      this.#state !== "cancelling"
    ) {
      return;
    }
    await this.#requestOperationCancellation();
    this.#settleUnstartedGeneration();
  }

  async reset(): Promise<void> {
    this.#requireReady();
    const started = this.#now();
    const operationSettled = deferred<void>();
    this.#operationSettled = operationSettled;
    this.#state = "resetting";
    try {
      await this.#resources!.driver.reset();
      this.#sequenceTokenIds = null;
      this.#contextTokens = 0;
      this.#recordPhase("reset", started);
      this.#state = "ready";
    } catch (error) {
      this.#state = "failed";
      throw error;
    } finally {
      this.#operationSettled = null;
      operationSettled.resolve(undefined);
    }
  }

  dispose(): Promise<void> {
    this.#disposePromise ??= this.#dispose();
    return this.#disposePromise;
  }

  getMetrics(): RuntimeMetrics {
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
      trackedGpuBytes: this.#trackedGpuBytes,
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
  ): AsyncGenerator<GeneratedToken> {
    if (this.#operationController !== controller) {
      return;
    }
    this.#generateStarted = true;
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
      for await (const id of tokens) {
        controller.signal.throwIfAborted();
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
        this.#operationCancellationRequested &&
        isAbortError(error);
      generationFailed = !explicitlyCancelled;
      throw error;
    } finally {
      if (!completed) {
        await this.#requestOperationCancellation();
      }
      const elapsed = this.#recordPhase("generate", started);
      this.#generationMilliseconds += elapsed;
      this.#activeGenerate = false;
      this.#generateStarted = false;
      this.#operationController = null;
      this.#operationSettled?.resolve(undefined);
      this.#operationSettled = null;
      if (generationFailed) {
        this.#state = "failed";
      } else if (
        this.#state === "generating" ||
        this.#state === "cancelling"
      ) {
        this.#state = "ready";
      }
    }
  }

  async #requestOperationCancellation(): Promise<void> {
    if (this.#operationCancellationRequested) {
      return;
    }
    this.#operationCancellationRequested = true;
    this.#cancellationCount += 1;
    this.#state = "cancelling";
    this.#operationController?.abort();
    await this.#resources?.driver.cancel?.();
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
    this.#state = "disposing";
    if (wasLoading) {
      this.#lock?.cancel();
    } else if (this.#operationSettled !== null) {
      const operationSettled = this.#operationSettled;
      if (this.#operationController !== null) {
        await this.#requestOperationCancellation();
        this.#settleUnstartedGeneration();
      }
      await operationSettled.promise;
    }
    this.#lifetime?.resolve(undefined);
    try {
      await this.#lockPromise;
      this.#recordPhase("dispose", started);
      this.#state = "disposed";
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
    if (this.#state === "disposed" || this.#state === "disposing") {
      return;
    }
    this.#deviceLostCount += 1;
    this.#operationController?.abort();
    this.#state = "failed";
  }

  #settleUnstartedGeneration(): void {
    if (!this.#activeGenerate || this.#generateStarted) {
      return;
    }
    this.#activeGenerate = false;
    this.#operationController = null;
    this.#operationSettled?.resolve(undefined);
    this.#operationSettled = null;
    if (this.#state === "cancelling") {
      this.#state = "ready";
    }
  }
}
