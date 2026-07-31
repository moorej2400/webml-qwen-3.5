import { diagnosticError } from "./diagnostics.js";
import type { Qwen35Tokenizer } from "./qwen-tokenizer.js";

export const QWEN35_PRODUCT_CONTEXT_TOKENS = 16_384;
export const QWEN35_DEFAULT_VISUAL_TOKENS = 1_024;
const MAX_CHAT_MESSAGES = 1_024;
const MAX_CHAT_CONTENT_CODE_UNITS = 1_000_000;

const IM_START = "<|im_start|>";
const IM_END = "<|im_end|>";
const VISION_START = "<|vision_start|>";
const VISION_END = "<|vision_end|>";
const IMAGE_PAD = "<|image_pad|>";
const THINK_START = "<think>";
const THINK_END = "</think>";
const PYTHON_STRIP =
  /^[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+|[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+$/gu;

const RESERVED_TOKENS: readonly string[] = Object.freeze([
  "<|endoftext|>",
  IM_START,
  IM_END,
  "<|object_ref_start|>",
  "<|object_ref_end|>",
  "<|box_start|>",
  "<|box_end|>",
  "<|quad_start|>",
  "<|quad_end|>",
  VISION_START,
  VISION_END,
  "<|vision_pad|>",
  IMAGE_PAD,
  "<|video_pad|>",
  "<tool_call>",
  "</tool_call>",
  "<|fim_prefix|>",
  "<|fim_middle|>",
  "<|fim_suffix|>",
  "<|fim_pad|>",
  "<|repo_name|>",
  "<|file_sep|>",
  "<tool_response>",
  "</tool_response>",
  THINK_START,
  THINK_END,
]);

// Jinja's `trim` delegates to Python `str.strip()`. ECMAScript `trim()` both
// misses Python controls such as U+0085 and incorrectly removes U+FEFF.
function pythonStrip(value: string): string {
  return value.replace(PYTHON_STRIP, "");
}

export type Qwen35ChatRole = "system" | "user" | "assistant";

export type Qwen35ChatContentPart =
  | {
      readonly type: "text";
      readonly text: string;
    }
  | {
      /**
       * A typed placeholder only. Vision preprocessing replaces its one pad
       * token with projected visual tokens in a later runtime boundary.
       */
      readonly type: "image";
    };

export interface Qwen35ChatMessage {
  readonly role: Qwen35ChatRole;
  readonly content: string | readonly Qwen35ChatContentPart[] | null;
  readonly reasoningContent?: string;
}

export interface Qwen35ChatTemplateOptions {
  readonly addGenerationPrompt?: boolean;
  readonly enableThinking?: boolean;
  readonly addVisionId?: boolean;
}

export interface Qwen35ConversationOptions extends Qwen35ChatTemplateOptions {
  readonly visualTokensPerImage?: number;
  readonly reservedGenerationTokens?: number;
}

export interface Qwen35ImageMarker {
  readonly imageOrdinal: number;
  readonly messageIndex: number;
  readonly tokenIndex: number;
  readonly reservedVisualTokens: number;
}

export interface Qwen35ConversationReady {
  readonly ok: true;
  readonly rendered: string;
  readonly tokenIds: readonly number[];
  readonly imageMarkers: readonly Readonly<Qwen35ImageMarker>[];
  readonly promptTokenCount: number;
  readonly reservedGenerationTokens: number;
  readonly requiredTokenCount: number;
  readonly remainingContextTokens: number;
}

export interface Qwen35ConversationOverflow {
  readonly ok: false;
  readonly reason: "token-budget-exceeded";
  readonly contextLimit: 16_384;
  readonly promptTokenCount: number;
  readonly reservedGenerationTokens: number;
  readonly requiredTokenCount: number;
  readonly overflowTokens: number;
}

export type Qwen35ConversationAssembly =
  | Qwen35ConversationReady
  | Qwen35ConversationOverflow;

interface PendingImage {
  readonly imageOrdinal: number;
  readonly messageIndex: number;
}

interface RenderedChat {
  readonly text: string;
  readonly images: readonly PendingImage[];
}

interface RenderState {
  contentCodeUnits: number;
  imageCount: number;
  readonly images: PendingImage[];
}

function fail(code: string, message: string): never {
  throw diagnosticError(code, message);
}

function requireSafeContent(text: string, state: RenderState): void {
  if (typeof text !== "string") {
    fail("chat-content-invalid", "Chat content must be text");
  }
  state.contentCodeUnits += text.length;
  if (state.contentCodeUnits > MAX_CHAT_CONTENT_CODE_UNITS) {
    fail("chat-content-limit", "Chat content exceeds its configured limit");
  }
  if (RESERVED_TOKENS.some((token) => text.includes(token))) {
    fail("chat-reserved-token", "Chat content contains a reserved token");
  }
}

function renderContent(
  message: Qwen35ChatMessage,
  messageIndex: number,
  state: RenderState,
  addVisionId: boolean,
): string {
  const { content } = message;
  if (content === null) {
    return "";
  }
  if (typeof content === "string") {
    requireSafeContent(content, state);
    return content;
  }
  if (!Array.isArray(content)) {
    fail("chat-content-invalid", "Chat content has an unsupported type");
  }

  let output = "";
  let imagesInMessage = 0;
  for (const part of content) {
    if (typeof part !== "object" || part === null) {
      fail("chat-content-invalid", "Chat content part is invalid");
    }
    if (part.type === "text") {
      requireSafeContent(part.text, state);
      output += part.text;
      continue;
    }
    if (part.type !== "image") {
      fail("chat-content-invalid", "Chat content part is unsupported");
    }
    if (message.role !== "user") {
      fail("chat-image-role", "Images are supported only in user messages");
    }
    imagesInMessage += 1;
    if (imagesInMessage > 1) {
      fail("chat-image-count", "Only one image is supported per user turn");
    }
    const imageOrdinal = state.imageCount;
    state.imageCount += 1;
    state.images.push(Object.freeze({ imageOrdinal, messageIndex }));
    if (addVisionId) {
      output += `Picture ${imageOrdinal + 1}: `;
    }
    output += `${VISION_START}${IMAGE_PAD}${VISION_END}`;
  }
  return output;
}

function renderDetailed(
  messages: readonly Qwen35ChatMessage[],
  options: Qwen35ChatTemplateOptions,
): RenderedChat {
  if (!Array.isArray(messages) || messages.length === 0) {
    fail("chat-messages-empty", "At least one chat message is required");
  }
  if (messages.length > MAX_CHAT_MESSAGES) {
    fail("chat-message-limit", "Chat message count exceeds its configured limit");
  }
  for (const message of messages) {
    if (typeof message !== "object" || message === null) {
      fail("chat-message-invalid", "Chat message must be an object");
    }
  }
  const addGenerationPrompt = options.addGenerationPrompt ?? true;
  const enableThinking = options.enableThinking ?? true;
  const addVisionId = options.addVisionId ?? false;
  const state: RenderState = {
    contentCodeUnits: 0,
    imageCount: 0,
    images: [],
  };

  let lastUserIndex = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]!.role === "user") {
      lastUserIndex = index;
      break;
    }
  }
  if (lastUserIndex < 0) {
    fail("chat-user-missing", "Chat messages require a user query");
  }

  let output = "";
  if (messages[0]!.role === "system") {
    const systemContent = renderContent(
      messages[0]!,
      0,
      state,
      addVisionId,
    );
    const strippedSystemContent = pythonStrip(systemContent);
    output += `${IM_START}system\n${strippedSystemContent}${IM_END}\n`;
  }

  for (const [index, message] of messages.entries()) {
    if (
      message.role !== "system" &&
      message.role !== "user" &&
      message.role !== "assistant"
    ) {
      fail("chat-role-invalid", "Chat message role is unsupported");
    }
    if (message.role === "system") {
      if (index !== 0) {
        fail(
          "chat-system-position",
          "System message must be the first message",
        );
      }
      continue;
    }

    let content = pythonStrip(
      renderContent(message, index, state, addVisionId),
    );
    if (message.role === "user") {
      output += `${IM_START}user\n${content}${IM_END}\n`;
      continue;
    }

    let reasoning = "";
    if (message.reasoningContent !== undefined) {
      requireSafeContent(message.reasoningContent, state);
      reasoning = message.reasoningContent;
    }
    reasoning = pythonStrip(reasoning);
    if (index > lastUserIndex) {
      output +=
        `${IM_START}assistant\n${THINK_START}\n${reasoning}\n` +
        `${THINK_END}\n\n${content}`;
    } else {
      output += `${IM_START}assistant\n${content}`;
    }
    output += `${IM_END}\n`;
  }

  if (addGenerationPrompt) {
    output += `${IM_START}assistant\n`;
    output += enableThinking
      ? `${THINK_START}\n`
      : `${THINK_START}\n\n${THINK_END}\n\n`;
  }
  return Object.freeze({
    text: output,
    images: Object.freeze([...state.images]),
  });
}

/** Renders the supported portion of the pinned Qwen3.5 Jinja chat template. */
export function renderQwen35Chat(
  messages: readonly Qwen35ChatMessage[],
  options: Qwen35ChatTemplateOptions = {},
): string {
  return renderDetailed(messages, options).text;
}

function boundedInteger(
  value: number,
  label: string,
  allowZero: boolean,
): number {
  if (
    !Number.isSafeInteger(value) ||
    value < (allowZero ? 0 : 1) ||
    value > QWEN35_PRODUCT_CONTEXT_TOKENS
  ) {
    fail("chat-budget-option", `${label} is outside the product context bound`);
  }
  return value;
}

/**
 * Encodes a conversation and accounts for future visual-token replacement.
 *
 * Overflow is data, not truncation: callers receive the exact required count
 * and must ask the user to reduce content or the reserved generation budget.
 */
export function assembleQwen35Conversation(
  tokenizer: Qwen35Tokenizer,
  messages: readonly Qwen35ChatMessage[],
  options: Qwen35ConversationOptions = {},
): Qwen35ConversationAssembly {
  const visualTokensPerImage = boundedInteger(
    options.visualTokensPerImage ?? QWEN35_DEFAULT_VISUAL_TOKENS,
    "Visual token count",
    false,
  );
  const reservedGenerationTokens = boundedInteger(
    options.reservedGenerationTokens ?? 0,
    "Reserved generation token count",
    true,
  );
  const rendered = renderDetailed(messages, options);
  const tokenIds = tokenizer.encode(rendered.text, { addedTokens: "allow" });
  const imagePadId = tokenizer.addedTokenId(IMAGE_PAD);
  const pendingImageTokenIndices: number[] = [];
  if (imagePadId !== undefined) {
    for (const [index, id] of tokenIds.entries()) {
      if (id === imagePadId) {
        pendingImageTokenIndices.push(index);
      }
    }
  }
  if (pendingImageTokenIndices.length !== rendered.images.length) {
    fail(
      "chat-tokenizer-contract",
      "Tokenizer image marker contract does not match the chat template",
    );
  }

  const promptTokenCount =
    tokenIds.length +
    rendered.images.length * (visualTokensPerImage - 1);
  const requiredTokenCount =
    promptTokenCount + reservedGenerationTokens;
  if (requiredTokenCount > QWEN35_PRODUCT_CONTEXT_TOKENS) {
    return Object.freeze({
      ok: false as const,
      reason: "token-budget-exceeded" as const,
      contextLimit: QWEN35_PRODUCT_CONTEXT_TOKENS,
      promptTokenCount,
      reservedGenerationTokens,
      requiredTokenCount,
      overflowTokens:
        requiredTokenCount - QWEN35_PRODUCT_CONTEXT_TOKENS,
    });
  }

  const imageMarkers = rendered.images.map((image, index) =>
    Object.freeze({
      ...image,
      tokenIndex: pendingImageTokenIndices[index]!,
      reservedVisualTokens: visualTokensPerImage,
    }),
  );
  return Object.freeze({
    ok: true as const,
    rendered: rendered.text,
    tokenIds: Object.freeze(tokenIds),
    imageMarkers: Object.freeze(imageMarkers),
    promptTokenCount,
    reservedGenerationTokens,
    requiredTokenCount,
    remainingContextTokens:
      QWEN35_PRODUCT_CONTEXT_TOKENS - requiredTokenCount,
  });
}
