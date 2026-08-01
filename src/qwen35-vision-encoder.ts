import { diagnosticError } from "./diagnostics.js";

export interface Qwen35VisionEncoderBootstrap {
  destroy(): Promise<void>;
}

export interface Qwen35VisionProjectedTokens {
  readonly storage: {
    readonly buffer: object;
    readonly byteLength: number;
  };
  dispose(): Promise<void>;
}

export interface Qwen35VisionEncoderInput {
  readonly gridHeight: number;
  readonly gridWidth: number;
  readonly signal?: AbortSignal;
}

export interface Qwen35VisionEncoderDependencies<
  FoundationPlan = unknown,
  MergerPlan = unknown,
> {
  stageBootstrap(signal: AbortSignal): Promise<Qwen35VisionEncoderBootstrap>;
  planFoundation(
    bootstrap: Qwen35VisionEncoderBootstrap,
    input: Readonly<{ gridHeight: number; gridWidth: number }>,
  ): readonly FoundationPlan[];
  runFoundation(plans: readonly FoundationPlan[], signal: AbortSignal): Promise<void>;
  runLayers(input: {
    readonly bootstrap: Qwen35VisionEncoderBootstrap;
    readonly gridHeight: number;
    readonly gridWidth: number;
    readonly signal: AbortSignal;
  }): Promise<void>;
  createProjectedOutput(input: {
    readonly visualTokenCount: number;
  }): Promise<Qwen35VisionProjectedTokens>;
  planMerger(
    bootstrap: Qwen35VisionEncoderBootstrap,
    projected: Qwen35VisionProjectedTokens,
  ): readonly MergerPlan[];
  runMerger(
    plans: readonly MergerPlan[],
    projected: Qwen35VisionProjectedTokens,
    signal: AbortSignal,
  ): Promise<Qwen35VisionProjectedTokens>;
}

type EncoderState = "ready" | "running" | "complete" | "failed" | "disposed";

function abortError(): Error {
  return Object.assign(new Error("Vision encoding cancelled"), { name: "AbortError" });
}

function assertGrid(value: number): void {
  if (!Number.isSafeInteger(value) || value < 2 || value > 128 || value % 2 !== 0) {
    throw diagnosticError("vision-encoder-grid-invalid", "Vision encoder grid is invalid");
  }
}

/**
 * Owns the fixed vision sequence from authenticated bootstrap through projected
 * visual tokens. The bootstrap remains live across foundation and merger because
 * both stages bind its weights directly; streamed transformer weights do not.
 */
export class Qwen35VisionEncoder<FoundationPlan = unknown, MergerPlan = unknown> {
  readonly #dependencies: Qwen35VisionEncoderDependencies<FoundationPlan, MergerPlan>;
  #state: EncoderState = "ready";
  #bootstrap: Qwen35VisionEncoderBootstrap | null = null;
  #projected: Qwen35VisionProjectedTokens | null = null;
  #run: Promise<Qwen35VisionProjectedTokens> | null = null;
  #dispose: Promise<void> | null = null;

  constructor(dependencies: Qwen35VisionEncoderDependencies<FoundationPlan, MergerPlan>) {
    this.#dependencies = dependencies;
  }

  get state(): EncoderState { return this.#state; }

  encode(input: Qwen35VisionEncoderInput): Promise<Qwen35VisionProjectedTokens> {
    if (this.#state !== "ready") {
      return Promise.reject(diagnosticError("vision-encoder-unusable", "Vision encoder is not ready"));
    }
    assertGrid(input.gridHeight);
    assertGrid(input.gridWidth);
    if (input.gridHeight * input.gridWidth > 16_384) {
      return Promise.reject(diagnosticError("vision-encoder-grid-invalid", "Vision encoder grid exceeds the product token contract"));
    }
    this.#state = "running";
    this.#run = this.#encode(input, input.signal ?? new AbortController().signal);
    return this.#run;
  }

  async #encode(
    input: Qwen35VisionEncoderInput,
    signal: AbortSignal,
  ): Promise<Qwen35VisionProjectedTokens> {
    try {
      signal.throwIfAborted();
      const bootstrap = await this.#dependencies.stageBootstrap(signal);
      this.#bootstrap = bootstrap;
      signal.throwIfAborted();
      const geometry = Object.freeze({ gridHeight: input.gridHeight, gridWidth: input.gridWidth });
      await this.#dependencies.runFoundation(this.#dependencies.planFoundation(bootstrap, geometry), signal);
      signal.throwIfAborted();
      await this.#dependencies.runLayers({ bootstrap, ...geometry, signal });
      signal.throwIfAborted();
      const projected = await this.#dependencies.createProjectedOutput({
        visualTokenCount: input.gridHeight * input.gridWidth / 4,
      });
      this.#projected = projected;
      signal.throwIfAborted();
      const output = await this.#dependencies.runMerger(
        this.#dependencies.planMerger(bootstrap, projected),
        projected,
        signal,
      );
      if (output !== projected) {
        throw diagnosticError("vision-encoder-output-invalid", "Vision encoder merger returned an unowned output");
      }
      signal.throwIfAborted();
      this.#state = "complete";
      return output;
    } catch (error) {
      this.#state = "failed";
      // A failed release must stay retryable but never conceal the computation
      // error that tells the caller why this encoding attempt failed.
      try { await this.#releaseOwned(); } catch { /* Preserve the primary failure. */ }
      if (signal.aborted) throw abortError();
      throw error;
    }
  }

  dispose(): Promise<void> {
    if (this.#dispose !== null) return this.#dispose;
    this.#dispose = this.#disposeAfterRun();
    return this.#dispose;
  }

  async #disposeAfterRun(): Promise<void> {
    try { await this.#run; } catch { /* The caller retains the run failure. */ }
    await this.#releaseOwned();
    this.#state = "disposed";
  }

  async #releaseOwned(): Promise<void> {
    let first: unknown;
    const bootstrap = this.#bootstrap;
    if (bootstrap !== null) {
      try {
        await bootstrap.destroy();
        this.#bootstrap = null;
      } catch (error) { first ??= error; }
    }
    const projected = this.#projected;
    if (projected !== null) {
      try {
        await projected.dispose();
        this.#projected = null;
      } catch (error) { first ??= error; }
    }
    if (first !== undefined) throw first;
  }
}
