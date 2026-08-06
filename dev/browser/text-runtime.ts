import { QWEN35_PRODUCT_CONTEXT_TOKENS } from "../../src/qwen-chat-template.js";
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

export interface TextRuntimeSession {
  readonly state: Qwen35SessionState;
  load(options: Qwen35BrowserLoadOptions): Promise<void>;
  prefill(input: TextOrImageConversation): Promise<SequenceState>;
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
  state !== "idle" && state !== "loading" && state !== "disposed" && state !== "failed";

type TextRuntimeControllerOptions = {
  readonly onPromptStart?: (prompt: string) => void;
  readonly onText?: (text: string) => void;
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
      options.onPromptStart?.(parsed.prompt);
      if ("coordinator" in options) {
        await options.coordinator.replaceConversation([
          Object.freeze({ role: "user", content: parsed.prompt }),
        ]);
      } else {
        if (hasSequence) {
          await options.session.reset();
          hasSequence = false;
        }
        await options.session.prefill([
          Object.freeze({ role: "user", content: parsed.prompt }),
        ]);
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
      for await (const token of generated) {
        options.onText?.(token.text);
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
