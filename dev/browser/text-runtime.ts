import { QWEN35_PRODUCT_CONTEXT_TOKENS } from "../../src/qwen-chat-template.js";
import type { Qwen35ConversationOptions } from "../../src/qwen-chat-template.js";
import { diagnosticError } from "../../src/diagnostics.js";
import type { Qwen35BrowserLoadOptions } from "../../src/qwen35-model-loader.js";
import type { Qwen35RuntimeCoordinator } from "../../src/chat-app.js";
import type {
  GenerateOptions,
  GeneratedToken,
  Qwen35SessionState,
  RuntimeMetrics,
  SequenceState,
  TextOrImageConversation,
} from "../../src/qwen35-session.js";

const DEFAULT_MAX_NEW_TOKENS = 128;
const MAX_PROMPT_BYTES = 64 * 1024;
const PROMPT_FIELDS = new Set(["prompt", "maxNewTokens"]);
const REFERENCE_PROMPT = "Write one short sentence about WebGPU.";
const REFERENCE_TOKEN_IDS = Object.freeze([
  5_793, 48_213, 369, 264, 6_278, 13_775, 5_165, 5_995,
  310, 3_300, 1_496, 55_549, 11, 3_238, 11_258, 2_528,
]);

export interface TextRuntimeSession {
  readonly state: Qwen35SessionState;
  load(options: Qwen35BrowserLoadOptions): Promise<void>;
  prefill(
    input: TextOrImageConversation,
    options?: Pick<Qwen35ConversationOptions, "enableThinking">,
  ): Promise<SequenceState>;
  generate(options: GenerateOptions): AsyncIterable<GeneratedToken>;
  cancel(): Promise<void>;
  reset(): Promise<void>;
  dispose(): Promise<void>;
  getMetrics(): RuntimeMetrics;
}

export interface TextRuntimeState {
  readonly modelState: string;
  readonly generationState: string;
  readonly cacheState: string;
  readonly deviceState: string;
  readonly loaded: boolean;
  readonly generating: boolean;
  readonly contextTokens: number;
  readonly maxContextTokens: number;
  readonly cpuBytes: number;
  readonly gpuBytes: number;
}

export interface TextRuntimeController {
  load(payload?: unknown): Promise<void>;
  runPrompt(payload?: unknown): Promise<void>;
  cancelPrompt(): Promise<void>;
  dispose(): Promise<void>;
  getState(): TextRuntimeState;
}

/** Per-command counter deltas that identify data movement without user content. */
export interface TextRuntimePerformanceMetrics {
  readonly diskReadBytes: number;
  readonly gpuUploadBytes: number;
  readonly dispatchCount: number;
  readonly queueSubmissionCount: number;
  readonly queueRetirementCount: number;
  readonly gpuReadbackCount: number;
}

/** A single completed command measurement with no user or model content. */
export interface TextRuntimeRunMetrics {
  readonly contextTokens: number;
  readonly trackedCpuBytes: number;
  readonly trackedGpuBytes: number;
  readonly peakTrackedGpuBytes: number;
  readonly prefillTokens: number;
  readonly prefillDurationMilliseconds: number;
  readonly prefillTokensPerSecond: number | null;
  readonly generatedTokens: number;
  /** Fresh controlled prompts emit one cached prefill token before decode starts. */
  readonly targetStepCount: number;
  /** Time spent awaiting only model target steps; excludes the cached first token and observers. */
  readonly targetStepDurationMilliseconds: number;
  readonly targetStepsPerSecond: number | null;
  readonly generationDurationMilliseconds: number;
  /** End-to-end emission rate retained for lifecycle diagnostics, not decode comparison. */
  readonly generatedTokensPerSecond: number | null;
  readonly timeToFirstTokenMilliseconds: number | null;
  /** Count only; the local control path never records decoded model text. */
  readonly decodedTextCodeUnits: number;
  /** Fixed public GGUF oracle size; zero means this command was not the oracle. */
  readonly referenceTokenCount: number;
  /** Zero is a pass only when referenceTokenCount is nonzero. */
  readonly referenceTokenMismatchCount: number;
  /** Fixed-oracle diagnostic only; absent for non-oracle prompts and exact matches. */
  readonly referenceFirstMismatchIndex?: number;
  readonly referenceExpectedTokenId?: number;
  readonly referenceObservedTokenId?: number;
  /** Zero, one, or two snapshots were available at the command boundaries. */
  readonly performanceSnapshotCount: number;
  /** Present only when both command boundaries expose valid monotonic counters. */
  readonly performance?: TextRuntimePerformanceMetrics;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const requireEmptyPayload = (payload: unknown): void => {
  if (payload === undefined) return;
  if (!isRecord(payload) || Object.keys(payload).length !== 0) {
    throw new TypeError("Load payload cannot override local runtime configuration");
  }
};

const parsePromptPayload = (
  payload: unknown,
): { readonly prompt: string; readonly maxNewTokens: number } => {
  if (!isRecord(payload)) throw new TypeError("Prompt payload must be an object");
  for (const field of Object.keys(payload)) {
    if (!PROMPT_FIELDS.has(field)) throw new TypeError("Prompt payload contains an unsupported field");
  }
  if (
    typeof payload.prompt !== "string" ||
    payload.prompt.length === 0 ||
    new TextEncoder().encode(payload.prompt).byteLength > MAX_PROMPT_BYTES
  ) {
    throw new TypeError("Prompt must contain between 1 and 65536 UTF-8 bytes");
  }
  const maxNewTokens = payload.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS;
  if (
    !Number.isSafeInteger(maxNewTokens) ||
    (maxNewTokens as number) < 1 ||
    (maxNewTokens as number) > QWEN35_PRODUCT_CONTEXT_TOKENS
  ) {
    throw new TypeError("Maximum generated token count is invalid");
  }
  return Object.freeze({ prompt: payload.prompt, maxNewTokens: maxNewTokens as number });
};

const isLoadedState = (state: Qwen35SessionState): boolean =>
  state === "ready" || state === "prefilling" || state === "generating" ||
  state === "cancelling" || state === "resetting";

const nonnegativeDifference = (after: number, before: number): number =>
  Math.max(0, after - before);

const PERFORMANCE_COUNTERS = Object.freeze([
  "diskReadBytes",
  "gpuUploadBytes",
  "dispatchCount",
  "queueSubmissionCount",
  "queueRetirementCount",
  "gpuReadbackCount",
] as const);

/**
 * The session owns cumulative counters across its lifetime. Emit a command
 * delta only when both snapshots are monotonic safe integers, so a malformed
 * optional diagnostic can never turn into a misleading performance record.
 */
const performanceDelta = (
  before: RuntimeMetrics["performance"],
  after: RuntimeMetrics["performance"],
): TextRuntimePerformanceMetrics | undefined => {
  if (before === undefined || after === undefined) return undefined;
  const values: Partial<Record<keyof TextRuntimePerformanceMetrics, number>> = {};
  for (const field of PERFORMANCE_COUNTERS) {
    const initial = before[field];
    const final = after[field];
    if (
      !Number.isSafeInteger(initial) ||
      !Number.isSafeInteger(final) ||
      initial < 0 ||
      final < initial
    ) {
      return undefined;
    }
    values[field] = final - initial;
  }
  return Object.freeze(values as TextRuntimePerformanceMetrics);
};

const completedRunMetrics = (
  before: RuntimeMetrics,
  after: RuntimeMetrics,
  decodedTextCodeUnits: number,
  targetStepDurationMilliseconds: number,
  timeToFirstTokenMilliseconds: number | null,
  referenceTokenCount: number,
  referenceTokenMismatchCount: number,
  referenceFirstMismatch?: Readonly<{
    index: number;
    expected: number;
    observed: number;
  }>,
): TextRuntimeRunMetrics => {
  const prefillTokens = nonnegativeDifference(after.prefillTokens, before.prefillTokens);
  const generatedTokens = nonnegativeDifference(after.generatedTokens, before.generatedTokens);
  const prefillDurationMilliseconds = after.phases.prefill?.lastMilliseconds ?? 0;
  const generationDurationMilliseconds = after.phases.generate?.lastMilliseconds ?? 0;
  const performance = performanceDelta(before.performance, after.performance);
  return Object.freeze({
    contextTokens: after.contextTokens,
    trackedCpuBytes: after.trackedCpuBytes,
    trackedGpuBytes: after.trackedGpuBytes,
    peakTrackedGpuBytes: after.peakTrackedGpuBytes,
    prefillTokens,
    prefillDurationMilliseconds,
    prefillTokensPerSecond: prefillDurationMilliseconds > 0
      ? (prefillTokens * 1_000) / prefillDurationMilliseconds
      : null,
    generatedTokens,
    // runPrompt always prefills a fresh sequence. Its first yielded token is
    // therefore cached logits from prefill, not a decode target step.
    targetStepCount: generatedTokens === 0 ? 0 : generatedTokens - 1,
    targetStepDurationMilliseconds,
    targetStepsPerSecond:
      generatedTokens > 1 && targetStepDurationMilliseconds > 0
        ? ((generatedTokens - 1) * 1_000) / targetStepDurationMilliseconds
        : null,
    generationDurationMilliseconds,
    generatedTokensPerSecond: generationDurationMilliseconds > 0
      ? (generatedTokens * 1_000) / generationDurationMilliseconds
      : null,
    timeToFirstTokenMilliseconds,
    decodedTextCodeUnits,
    referenceTokenCount,
    referenceTokenMismatchCount,
    ...(referenceFirstMismatch === undefined
      ? {}
      : {
          referenceFirstMismatchIndex: referenceFirstMismatch.index,
          referenceExpectedTokenId: referenceFirstMismatch.expected,
          referenceObservedTokenId: referenceFirstMismatch.observed,
        }),
    performanceSnapshotCount:
      (before.performance === undefined ? 0 : 1) +
      (after.performance === undefined ? 0 : 1),
    ...(performance === undefined ? {} : { performance }),
  });
};

type TextRuntimeControllerOptions = {
  readonly onPromptStart?: (prompt: string) => void;
  readonly onText?: (text: string) => void;
  readonly now?: () => number;
  /** Observers receive only operation deltas after successful generation. */
  readonly onRunMetrics?: (metrics: TextRuntimeRunMetrics) => void;
} & (
  | {
      readonly coordinator: Qwen35RuntimeCoordinator;
    }
  | {
      readonly session: TextRuntimeSession;
      readonly loadOptions: Qwen35BrowserLoadOptions;
    }
);

export const createTextRuntimeController = (
  options: TextRuntimeControllerOptions,
): TextRuntimeController => {
  let hasSequence = false;
  let activePrompt: Promise<void> | null = null;
  const now = options.now ?? (() => globalThis.performance?.now() ?? Date.now());

  const state = (): Qwen35SessionState =>
    "coordinator" in options
      ? options.coordinator.state
      : options.session.state;

  const metrics = (): RuntimeMetrics =>
    "coordinator" in options
      ? options.coordinator.getMetrics()
      : options.session.getMetrics();

  const load = async (payload?: unknown): Promise<void> => {
    requireEmptyPayload(payload);
    // The polished UI and local control share this loaded session. A repeated
    // load command must not replace it while another owner is using it.
    if (isLoadedState(state())) return;
    if ("coordinator" in options) {
      await options.coordinator.load();
      return;
    }
    await options.session.load(options.loadOptions);
  };

  const runPrompt = async (payload?: unknown): Promise<void> => {
    const parsed = parsePromptPayload(payload);
    if (activePrompt !== null) throw new Error("A prompt is already running");
    const operation = (async () => {
      const commandStarted = now();
      const metricsBeforeRun = metrics();
      let decodedTextCodeUnits = 0;
      let targetStepDurationMilliseconds = 0;
      const compareReference =
        parsed.prompt === REFERENCE_PROMPT &&
        parsed.maxNewTokens === REFERENCE_TOKEN_IDS.length;
      const generatedTokenIds: number[] = [];
      let firstLogicalTokenAt: number | null = null;
      options.onPromptStart?.(parsed.prompt);
      if ("coordinator" in options) {
        await options.coordinator.replaceConversation([
          Object.freeze({ role: "user", content: parsed.prompt }),
        ], compareReference ? { enableThinking: false } : undefined);
      } else {
        if (hasSequence) {
          await options.session.reset();
          hasSequence = false;
        }
        await options.session.prefill([
          Object.freeze({ role: "user", content: parsed.prompt }),
        ], compareReference ? { enableThinking: false } : undefined);
        hasSequence = true;
      }
      const generated = "coordinator" in options
        ? options.coordinator.generate({
            maxNewTokens: parsed.maxNewTokens,
            temperature: 0,
            topK: 1,
          })
        : options.session.generate({
            maxNewTokens: parsed.maxNewTokens,
            temperature: 0,
            topK: 1,
          });
      const iterator = generated[Symbol.asyncIterator]();
      let emittedTokens = 0;
      let lastTokenIndex: number | null = null;
      while (true) {
        const started = now();
        const item = await iterator.next();
        const completedAt = now();
        const elapsed = Math.max(0, completedAt - started);
        if (item.done) break;
        const token = item.value;
        // Incremental UTF-8 flushes may repeat the last token index. They are
        // text fragments, not additional model steps or oracle token IDs.
        const isLogicalToken = token.index !== lastTokenIndex;
        if (!isLogicalToken) {
          decodedTextCodeUnits += token.text.length;
          options.onText?.(token.text);
          continue;
        }
        if (firstLogicalTokenAt === null) firstLogicalTokenAt = completedAt;
        // The first item is cached by prefill. Every later next() performs one
        // target step before it resolves, while observer work happens below.
        if (emittedTokens > 0) targetStepDurationMilliseconds += elapsed;
        emittedTokens += 1;
        lastTokenIndex = token.index;
        if (compareReference) generatedTokenIds.push(token.id);
        decodedTextCodeUnits += token.text.length;
        options.onText?.(token.text);
      }
      // Session metrics are lifetime counters. Emit an operation delta so
      // successive controlled prompts remain independently comparable.
      const referenceTokenMismatchCount = compareReference
        ? Math.abs(generatedTokenIds.length - REFERENCE_TOKEN_IDS.length) +
          REFERENCE_TOKEN_IDS
            .slice(0, Math.min(generatedTokenIds.length, REFERENCE_TOKEN_IDS.length))
            .reduce(
              (count, expected, index) =>
                count + (generatedTokenIds[index] === expected ? 0 : 1),
              0,
            )
        : 0;
      const firstMismatchIndex = compareReference
        ? REFERENCE_TOKEN_IDS.findIndex(
            (expected, index) => generatedTokenIds[index] !== expected,
          )
        : -1;
      const firstMismatch = firstMismatchIndex < 0
        ? undefined
        : Object.freeze({
            index: firstMismatchIndex,
            expected: REFERENCE_TOKEN_IDS[firstMismatchIndex]!,
            // The vocabulary IDs are non-negative. Use the vocabulary size as
            // a bounded sentinel only when generation ended before this slot.
            observed: generatedTokenIds[firstMismatchIndex] ?? 248_320,
          });
      const runMetrics = completedRunMetrics(
        metricsBeforeRun,
        metrics(),
        decodedTextCodeUnits,
        targetStepDurationMilliseconds,
        firstLogicalTokenAt === null ? null : Math.max(0, firstLogicalTokenAt - commandStarted),
        compareReference ? REFERENCE_TOKEN_IDS.length : 0,
        referenceTokenMismatchCount,
        firstMismatch,
      );
      options.onRunMetrics?.(runMetrics);
      if (compareReference && referenceTokenMismatchCount !== 0) {
        throw diagnosticError(
          "reference-token-mismatch",
          "Controlled generation did not match the pinned token oracle",
        );
      }
    })();
    activePrompt = operation;
    try {
      await operation;
    } finally {
      if (activePrompt === operation) activePrompt = null;
    }
  };

  const getState = (): TextRuntimeState => {
    const snapshot = metrics();
    const loaded = isLoadedState(snapshot.state);
    const generating = snapshot.state === "generating" || snapshot.state === "cancelling";
    return Object.freeze({
      modelState: snapshot.state,
      generationState: generating ? "active" : "idle",
      cacheState: snapshot.cacheHit === null ? "unknown" : snapshot.cacheHit ? "hit" : "miss",
      deviceState: snapshot.deviceLostCount > 0 ? "lost" : loaded ? "ready" : "unavailable",
      loaded,
      generating,
      contextTokens: snapshot.contextTokens,
      maxContextTokens: QWEN35_PRODUCT_CONTEXT_TOKENS,
      cpuBytes: snapshot.trackedCpuBytes,
      gpuBytes: snapshot.trackedGpuBytes,
    });
  };

  return Object.freeze({
    load,
    runPrompt,
    cancelPrompt: () => "coordinator" in options
      ? options.coordinator.cancel()
      : options.session.cancel(),
    dispose: () => "coordinator" in options
      ? options.coordinator.dispose()
      : options.session.dispose(),
    getState,
  });
};
