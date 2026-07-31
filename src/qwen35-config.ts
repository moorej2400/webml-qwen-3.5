import type { GgufMetadataValue } from "./gguf.js";

export type Qwen35MtpPolicy = "exclude-block-32";

export interface Qwen35Config {
  readonly architecture: "qwen35";
  readonly baseBlockCount: 32;
  readonly sourceBlockCount: 33;
  readonly mtpBlock: 32;
  readonly mtpPolicy: Qwen35MtpPolicy;
  readonly embeddingLength: 2_560;
  readonly feedForwardLength: 9_216;
  readonly vocabularySize: 248_320;
  /** Native GGUF capability metadata; this is not the selected product limit. */
  readonly sourceMaxContextLength: 262_144;
  /** Current product contract, not a claim about the browser's capability. */
  readonly productContextLength: number;
  readonly attentionHeadCount: 16;
  readonly keyValueHeadCount: 4;
  readonly headDimension: 256;
  readonly keyLength: 256;
  readonly valueLength: 256;
  readonly rmsNormEpsilon: 0.000_001;
  readonly fullAttentionInterval: 4;
  readonly fullAttentionLayers: readonly [3, 7, 11, 15, 19, 23, 27, 31];
  readonly linearAttentionLayerCount: 24;
  readonly fullAttentionLayerCount: 8;
  readonly ropeFrequencyBase: 10_000_000;
  readonly rotaryDimension: 64;
  readonly mropeSections: readonly [11, 11, 10, 0];
  readonly ssmConvKernel: 4;
  readonly ssmStateSize: 128;
  readonly ssmGroupCount: 16;
  readonly ssmTimeStepRank: 32;
  readonly ssmInnerSize: 4_096;
}

const FULL_ATTENTION_LAYERS =
  Object.freeze([3, 7, 11, 15, 19, 23, 27, 31]) as Qwen35Config["fullAttentionLayers"];
const MROPE_SECTIONS =
  Object.freeze([11, 11, 10, 0]) as Qwen35Config["mropeSections"];
export const QWEN35_PRODUCT_CONTEXT_CAP = 16_384;

function createConfig(productContextLength: number): Qwen35Config {
  return Object.freeze({
    architecture: "qwen35",
    baseBlockCount: 32,
    sourceBlockCount: 33,
    mtpBlock: 32,
    mtpPolicy: "exclude-block-32",
    embeddingLength: 2_560,
    feedForwardLength: 9_216,
    vocabularySize: 248_320,
    sourceMaxContextLength: 262_144,
    productContextLength,
    attentionHeadCount: 16,
    keyValueHeadCount: 4,
    headDimension: 256,
    keyLength: 256,
    valueLength: 256,
    rmsNormEpsilon: 0.000_001,
    fullAttentionInterval: 4,
    fullAttentionLayers: FULL_ATTENTION_LAYERS,
    linearAttentionLayerCount: 24,
    fullAttentionLayerCount: 8,
    ropeFrequencyBase: 10_000_000,
    rotaryDimension: 64,
    mropeSections: MROPE_SECTIONS,
    ssmConvKernel: 4,
    ssmStateSize: 128,
    ssmGroupCount: 16,
    ssmTimeStepRank: 32,
    ssmInnerSize: 4_096,
  });
}

export const QWEN35_4B_CONFIG = createConfig(QWEN35_PRODUCT_CONTEXT_CAP);

const REQUIRED_METADATA = Object.freeze({
  "general.architecture": "qwen35",
  "qwen35.block_count": 32,
  "qwen35.context_length": 262_144,
  "qwen35.embedding_length": 2_560,
  "qwen35.feed_forward_length": 9_216,
  "qwen35.attention.head_count": 16,
  "qwen35.attention.head_count_kv": 4,
  "qwen35.attention.key_length": 256,
  "qwen35.attention.value_length": 256,
  "qwen35.attention.layer_norm_rms_epsilon": 0.000_001,
  "qwen35.full_attention_interval": 4,
  "qwen35.rope.dimension_count": 64,
  "qwen35.rope.dimension_sections": MROPE_SECTIONS,
  "qwen35.rope.freq_base": 10_000_000,
  "qwen35.ssm.conv_kernel": 4,
  "qwen35.ssm.state_size": 128,
  "qwen35.ssm.group_count": 16,
  "qwen35.ssm.time_step_rank": 32,
  "qwen35.ssm.inner_size": 4_096,
} satisfies Readonly<Record<string, GgufMetadataValue>>);

function metadataEqual(actual: GgufMetadataValue | undefined, expected: GgufMetadataValue): boolean {
  if (Array.isArray(expected)) {
    return (
      Array.isArray(actual) &&
      actual.length === expected.length &&
      expected.every((item, index) => metadataEqual(actual[index], item))
    );
  }
  return actual === expected;
}

/**
 * Converts parsed GGUF metadata into the one supported language configuration.
 *
 * Shape-critical drift fails here so later planners never become a generic
 * dynamic tensor interpreter.
 */
export function validateQwen35Config(
  metadata: Readonly<Record<string, GgufMetadataValue>>,
  options: {
    readonly productContextLength?: number;
    readonly mtpPolicy?: Qwen35MtpPolicy;
  } = {},
): Qwen35Config {
  for (const [key, expected] of Object.entries(REQUIRED_METADATA)) {
    if (!metadataEqual(metadata[key], expected)) {
      throw new Error(`Qwen3.5 metadata ${key} does not match the pinned 4B contract`);
    }
  }
  if ((options.mtpPolicy ?? "exclude-block-32") !== "exclude-block-32") {
    throw new Error("Qwen3.5 MTP policy must exclude block 32");
  }
  const productContextLength =
    options.productContextLength ?? QWEN35_PRODUCT_CONTEXT_CAP;
  if (
    !Number.isSafeInteger(productContextLength) ||
    productContextLength < 1 ||
    productContextLength > QWEN35_PRODUCT_CONTEXT_CAP
  ) {
    throw new Error(
      `Qwen3.5 product context must be between 1 and ${QWEN35_PRODUCT_CONTEXT_CAP}`,
    );
  }
  return productContextLength === QWEN35_PRODUCT_CONTEXT_CAP
    ? QWEN35_4B_CONFIG
    : createConfig(productContextLength);
}
