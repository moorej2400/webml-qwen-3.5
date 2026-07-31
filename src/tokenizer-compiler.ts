import { createHash } from "node:crypto";
import { byteLevelStringToBytes } from "./byte-level.js";
import {
  MAX_COMPILED_ADDED_TOKENS,
  MAX_COMPILED_MERGE_COUNT,
  MAX_COMPILED_TOKEN_COUNT,
  QWEN35_MODEL_LOGIT_ROWS,
  TOKENIZER_ARTIFACT_VERSION,
  TOKENIZER_BINARY_HEADER_BYTES,
  TOKENIZER_BINARY_MAGIC,
} from "./tokenizer-binary.js";
export {
  deserializeCompiledTokenizer,
  QWEN35_MODEL_LOGIT_ROWS,
  TOKENIZER_ARTIFACT_VERSION,
} from "./tokenizer-binary.js";
export type { CompiledTokenizerTables } from "./tokenizer-binary.js";

export const MAX_TOKENIZER_SOURCE_BYTES = 12_807_982;
const MAX_TOKENIZER_CONFIG_BYTES = 1_048_576;
const MAX_TOKEN_UTF16_LENGTH = 65_535;
const MAGIC = new TextEncoder().encode(TOKENIZER_BINARY_MAGIC);
const SHA256 = /^[a-f0-9]{64}$/;
const REVISION = /^[a-f0-9]{40}$/;
const SPLIT_PATTERN =
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

export interface TokenizerSourceIdentity {
  readonly repository: string;
  readonly revision: string;
  readonly file: string;
  readonly size: number;
  readonly sha256: string;
}

export const PINNED_QWEN35_TOKENIZER: Readonly<TokenizerSourceIdentity> =
  Object.freeze({
    repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
    revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    file: "tokenizer.json",
    size: 12_807_982,
    sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
  });

export const PINNED_QWEN35_TOKENIZER_CONFIG = Object.freeze({
  repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
  revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
  file: "tokenizer_config.json",
  size: 16_710,
  sha256: "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
});
export const PINNED_QWEN35_CHAT_TEMPLATE_SHA256 =
  "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715";

export interface TokenizerPackageManifest {
  readonly format: "webml-qwen35-tokenizer";
  readonly version: 1;
  readonly runtimeAbi: "qwen35-tokenizer-v1";
  readonly source: Readonly<TokenizerSourceIdentity>;
  readonly chatTemplateSource: Readonly<TokenizerSourceIdentity>;
  readonly chatTemplateSha256: string;
  readonly normalization: "NFC";
  readonly preTokenizer: "qwen35-bytelevel-v1";
  readonly baseVocabSize: number;
  readonly tokenCount: number;
  /** The sampler must mask model rows at or above this tokenizer ceiling. */
  readonly decodableTokenCount: number;
  readonly modelLogitRows: 248_320;
  readonly undecodableLogitRows: number;
  readonly mergeCount: number;
  readonly addedTokenCount: number;
  readonly artifactByteLength: number;
  readonly artifactSha256: string;
}

export interface CompiledTokenizerPackage {
  readonly manifest: Readonly<TokenizerPackageManifest>;
  readonly binary: Uint8Array;
}

interface AddedTokenDefinition {
  id: number;
  content: string;
  single_word: boolean;
  lstrip: boolean;
  rstrip: boolean;
  normalized: boolean;
  special: boolean;
}

interface TokenizerDefinition {
  version: string;
  truncation: unknown;
  padding: unknown;
  added_tokens: AddedTokenDefinition[];
  normalizer: { type: string };
  pre_tokenizer: {
    type: string;
    pretokenizers: [
      {
        type: string;
        pattern: { Regex: string };
        behavior: string;
        invert: boolean;
      },
      {
        type: string;
        add_prefix_space: boolean;
        trim_offsets: boolean;
        use_regex: boolean;
      },
    ];
  };
  post_processor: {
    type: string;
    add_prefix_space: boolean;
    trim_offsets: boolean;
    use_regex: boolean;
  };
  decoder: {
    type: string;
    add_prefix_space: boolean;
    trim_offsets: boolean;
    use_regex: boolean;
  };
  model: {
    type: string;
    dropout: unknown;
    unk_token: unknown;
    continuing_subword_prefix: string;
    end_of_word_suffix: string;
    fuse_unk: boolean;
    byte_fallback: boolean;
    ignore_merges: boolean;
    vocab: Record<string, number>;
    merges: string[];
  };
}

function validateSourceIdentity(
  identity: TokenizerSourceIdentity,
  expectedFile: string,
  maxBytes: number,
  label: string,
): void {
  let repository: URL;
  try {
    repository = new URL(identity.repository);
  } catch {
    throw new Error(`${label} repository must be a valid URL`);
  }
  if (
    repository.protocol !== "https:" ||
    repository.username !== "" ||
    repository.password !== ""
  ) {
    throw new Error(`${label} repository must use credential-free HTTPS`);
  }
  if (!REVISION.test(identity.revision)) {
    throw new Error(`${label} revision must be an immutable commit`);
  }
  if (identity.file !== expectedFile) {
    throw new Error(`${label} file must be ${expectedFile}`);
  }
  if (
    !Number.isSafeInteger(identity.size) ||
    identity.size <= 0 ||
    identity.size > maxBytes
  ) {
    throw new Error(`${label} byte length exceeds the compiler bound`);
  }
  if (!SHA256.test(identity.sha256)) {
    throw new Error(`${label} SHA-256 is malformed`);
  }
}

async function readBoundedSource(
  chunks: AsyncIterable<Uint8Array>,
  identity: TokenizerSourceIdentity,
  expectedFile: string,
  maxBytes: number,
  label: string,
): Promise<Uint8Array> {
  validateSourceIdentity(identity, expectedFile, maxBytes, label);
  const output = new Uint8Array(identity.size);
  const hash = createHash("sha256");
  let offset = 0;
  for await (const chunk of chunks) {
    if (!(chunk instanceof Uint8Array)) {
      throw new Error(`${label} chunks must contain bytes`);
    }
    if (offset + chunk.byteLength > identity.size) {
      throw new Error(`${label} exceeds its declared byte length`);
    }
    output.set(chunk, offset);
    hash.update(chunk);
    offset += chunk.byteLength;
  }
  if (offset !== identity.size) {
    throw new Error(`${label} byte length does not match its identity`);
  }
  if (hash.digest("hex") !== identity.sha256) {
    throw new Error(`${label} SHA-256 does not match its identity`);
  }
  return output;
}

function requireObject(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function validateByteLevelStage(
  value: unknown,
  label: string,
): asserts value is TokenizerDefinition["decoder"] {
  const stage = requireObject(value, label);
  if (
    stage.type !== "ByteLevel" ||
    stage.add_prefix_space !== false ||
    stage.trim_offsets !== false ||
    stage.use_regex !== false
  ) {
    throw new Error(`${label} does not match the Qwen3.5 ByteLevel contract`);
  }
}

function parseDefinition(bytes: Uint8Array): TokenizerDefinition {
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    throw new Error("Tokenizer source must be valid UTF-8 JSON");
  }
  const definition = requireObject(value, "Tokenizer source");
  if (definition.version !== "1.0") {
    throw new Error("Tokenizer JSON version must be 1.0");
  }
  if (definition.truncation !== null || definition.padding !== null) {
    throw new Error("Tokenizer source must not configure truncation or padding");
  }

  const normalizer = requireObject(definition.normalizer, "Tokenizer normalizer");
  if (normalizer.type !== "NFC") {
    throw new Error("Tokenizer normalizer must be NFC");
  }

  const preTokenizer = requireObject(
    definition.pre_tokenizer,
    "Tokenizer pre-tokenizer",
  );
  if (
    preTokenizer.type !== "Sequence" ||
    !Array.isArray(preTokenizer.pretokenizers) ||
    preTokenizer.pretokenizers.length !== 2
  ) {
    throw new Error("Tokenizer pre-tokenizer must be the Qwen3.5 sequence");
  }
  const split = requireObject(
    preTokenizer.pretokenizers[0],
    "Tokenizer split pre-tokenizer",
  );
  const pattern = requireObject(split.pattern, "Tokenizer split pattern");
  if (
    split.type !== "Split" ||
    pattern.Regex !== SPLIT_PATTERN ||
    split.behavior !== "Isolated" ||
    split.invert !== false
  ) {
    throw new Error("Tokenizer split pattern does not match Qwen3.5");
  }
  validateByteLevelStage(
    preTokenizer.pretokenizers[1],
    "Tokenizer ByteLevel pre-tokenizer",
  );
  validateByteLevelStage(definition.post_processor, "Tokenizer post-processor");
  validateByteLevelStage(definition.decoder, "Tokenizer decoder");

  const model = requireObject(definition.model, "Tokenizer model");
  if (
    model.type !== "BPE" ||
    model.dropout !== null ||
    model.unk_token !== null ||
    model.continuing_subword_prefix !== "" ||
    model.end_of_word_suffix !== "" ||
    model.fuse_unk !== false ||
    model.byte_fallback !== false ||
    model.ignore_merges !== false
  ) {
    throw new Error("Tokenizer model does not match the Qwen3.5 BPE contract");
  }
  requireObject(model.vocab, "Tokenizer vocabulary");
  if (!Array.isArray(model.merges)) {
    throw new Error("Tokenizer merges must be an array");
  }
  if (!Array.isArray(definition.added_tokens)) {
    throw new Error("Tokenizer added tokens must be an array");
  }
  return value as TokenizerDefinition;
}

function verifyChatTemplateConfig(
  bytes: Uint8Array,
  expectedTemplateSha256: string,
): void {
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    throw new Error("Tokenizer config must be valid UTF-8 JSON");
  }
  const config = requireObject(value, "Tokenizer config");
  if (
    typeof config.chat_template !== "string" ||
    config.chat_template.length === 0 ||
    config.chat_template.length > 100_000 ||
    config.eos_token !== "<|im_end|>" ||
    config.pad_token !== "<|endoftext|>" ||
    config.model_max_length !== 262_144
  ) {
    throw new Error("Tokenizer config does not match the Qwen3.5 chat contract");
  }
  const templateHash = createHash("sha256")
    .update(config.chat_template)
    .digest("hex");
  if (templateHash !== expectedTemplateSha256) {
    throw new Error(
      "Tokenizer config chat template does not match its implementation pin",
    );
  }
}

function compileDefinition(definition: TokenizerDefinition): Uint8Array {
  const vocabularyEntries = Object.entries(definition.model.vocab);
  if (
    vocabularyEntries.length === 0 ||
    vocabularyEntries.length > MAX_COMPILED_TOKEN_COUNT
  ) {
    throw new Error("Tokenizer vocabulary count exceeds its bound");
  }
  if (definition.model.merges.length > MAX_COMPILED_MERGE_COUNT) {
    throw new Error("Tokenizer merge count exceeds its bound");
  }
  if (definition.added_tokens.length > MAX_COMPILED_ADDED_TOKENS) {
    throw new Error("Tokenizer added-token count exceeds its bound");
  }

  const baseTokens = new Array<string>(vocabularyEntries.length);
  for (const [token, id] of vocabularyEntries) {
    if (
      token.length === 0 ||
      token.length > MAX_TOKEN_UTF16_LENGTH ||
      !Number.isInteger(id) ||
      id < 0 ||
      id >= baseTokens.length ||
      baseTokens[id] !== undefined
    ) {
      throw new Error("Tokenizer vocabulary ids must be unique and dense");
    }
    baseTokens[id] = token;
  }
  if (baseTokens.some((token) => token === undefined)) {
    throw new Error("Tokenizer vocabulary ids must be unique and dense");
  }

  const added = [...definition.added_tokens].sort((left, right) => left.id - right.id);
  const seenAddedContent = new Set<string>();
  for (const [index, token] of added.entries()) {
    if (
      token.id !== baseTokens.length + index ||
      typeof token.content !== "string" ||
      token.content.length === 0 ||
      token.content.length > MAX_TOKEN_UTF16_LENGTH ||
      seenAddedContent.has(token.content)
    ) {
      throw new Error("Tokenizer added tokens must have unique contiguous ids");
    }
    if (
      token.single_word !== false ||
      token.lstrip !== false ||
      token.rstrip !== false ||
      token.normalized !== false ||
      typeof token.special !== "boolean"
    ) {
      throw new Error("Tokenizer added-token flags do not match Qwen3.5");
    }
    seenAddedContent.add(token.content);
  }

  const encoder = new TextEncoder();
  const tokenBytes = [
    ...baseTokens.map((token) => byteLevelStringToBytes(token)),
    ...added.map((token) => encoder.encode(token.content)),
  ];
  const tokenCount = tokenBytes.length;
  if (tokenCount > MAX_COMPILED_TOKEN_COUNT) {
    throw new Error("Tokenizer token count exceeds its bound");
  }
  let tokenDataLength = 0;
  for (const bytes of tokenBytes) {
    tokenDataLength += bytes.byteLength;
    if (!Number.isSafeInteger(tokenDataLength) || tokenDataLength > 0xffff_ffff) {
      throw new Error("Tokenizer token data exceeds the binary format");
    }
  }

  const mergeTriples = new Uint32Array(definition.model.merges.length * 3);
  for (const [rank, merge] of definition.model.merges.entries()) {
    if (typeof merge !== "string" || merge.length > MAX_TOKEN_UTF16_LENGTH) {
      throw new Error("Tokenizer merge entry is malformed");
    }
    const separator = merge.indexOf(" ");
    if (separator <= 0 || separator === merge.length - 1) {
      throw new Error("Tokenizer merge entry is malformed");
    }
    const left = merge.slice(0, separator);
    const right = merge.slice(separator + 1);
    const leftId = definition.model.vocab[left];
    const rightId = definition.model.vocab[right];
    const resultId = definition.model.vocab[left + right];
    if (leftId === undefined || rightId === undefined || resultId === undefined) {
      throw new Error("Tokenizer merge references an unknown vocabulary token");
    }
    mergeTriples[rank * 3] = leftId;
    mergeTriples[rank * 3 + 1] = rightId;
    mergeTriples[rank * 3 + 2] = resultId;
  }

  const offsetsBytes = (tokenCount + 1) * 4;
  const mergesBytes = mergeTriples.byteLength;
  const addedBytes = added.length * 8;
  const totalBytes =
    TOKENIZER_BINARY_HEADER_BYTES +
    offsetsBytes +
    tokenDataLength +
    mergesBytes +
    addedBytes;
  if (!Number.isSafeInteger(totalBytes) || totalBytes > 0xffff_ffff) {
    throw new Error("Tokenizer artifact exceeds the binary format");
  }

  const binary = new Uint8Array(totalBytes);
  binary.set(MAGIC, 0);
  const view = new DataView(binary.buffer);
  view.setUint32(8, TOKENIZER_ARTIFACT_VERSION, true);
  view.setUint32(12, tokenCount, true);
  view.setUint32(16, baseTokens.length, true);
  view.setUint32(20, definition.model.merges.length, true);
  view.setUint32(24, added.length, true);
  view.setUint32(28, tokenDataLength, true);

  let dataOffset = TOKENIZER_BINARY_HEADER_BYTES + offsetsBytes;
  let runningOffset = 0;
  for (const [id, bytes] of tokenBytes.entries()) {
    view.setUint32(
      TOKENIZER_BINARY_HEADER_BYTES + id * 4,
      runningOffset,
      true,
    );
    binary.set(bytes, dataOffset + runningOffset);
    runningOffset += bytes.byteLength;
  }
  view.setUint32(
    TOKENIZER_BINARY_HEADER_BYTES + tokenCount * 4,
    runningOffset,
    true,
  );

  dataOffset += tokenDataLength;
  for (const value of mergeTriples) {
    view.setUint32(dataOffset, value, true);
    dataOffset += 4;
  }
  for (const token of added) {
    view.setUint32(dataOffset, token.id, true);
    view.setUint32(dataOffset + 4, token.special ? 1 : 0, true);
    dataOffset += 8;
  }
  return binary;
}

function frozenIdentity(
  identity: TokenizerSourceIdentity,
): Readonly<TokenizerSourceIdentity> {
  return Object.freeze({ ...identity });
}

/**
 * Compiles the source only after its complete immutable identity is verified.
 *
 * The source buffer is capped at the pinned 12.8 MB size; the compiler never
 * clones, tees, or retains a second copy of the input byte stream.
 */
export async function compileTokenizerSource(
  tokenizerChunks: AsyncIterable<Uint8Array>,
  tokenizerIdentity: TokenizerSourceIdentity,
  configChunks: AsyncIterable<Uint8Array>,
  configIdentity: TokenizerSourceIdentity,
  expectedChatTemplateSha256: string,
): Promise<CompiledTokenizerPackage> {
  if (!SHA256.test(expectedChatTemplateSha256)) {
    throw new Error("Chat template implementation SHA-256 is malformed");
  }
  const tokenizerBytes = await readBoundedSource(
    tokenizerChunks,
    tokenizerIdentity,
    "tokenizer.json",
    MAX_TOKENIZER_SOURCE_BYTES,
    "Tokenizer source",
  );
  const configBytes = await readBoundedSource(
    configChunks,
    configIdentity,
    "tokenizer_config.json",
    MAX_TOKENIZER_CONFIG_BYTES,
    "Tokenizer config",
  );
  verifyChatTemplateConfig(configBytes, expectedChatTemplateSha256);
  const definition = parseDefinition(tokenizerBytes);
  const binary = compileDefinition(definition);
  const tokenCount =
    Object.keys(definition.model.vocab).length +
    definition.added_tokens.length;
  if (tokenCount > QWEN35_MODEL_LOGIT_ROWS) {
    throw new Error("Tokenizer token count exceeds model logit rows");
  }
  const manifest = Object.freeze({
    format: "webml-qwen35-tokenizer" as const,
    version: TOKENIZER_ARTIFACT_VERSION as 1,
    runtimeAbi: "qwen35-tokenizer-v1" as const,
    source: frozenIdentity(tokenizerIdentity),
    chatTemplateSource: frozenIdentity(configIdentity),
    chatTemplateSha256: expectedChatTemplateSha256,
    normalization: "NFC" as const,
    preTokenizer: "qwen35-bytelevel-v1" as const,
    baseVocabSize: Object.keys(definition.model.vocab).length,
    tokenCount,
    decodableTokenCount: tokenCount,
    modelLogitRows: QWEN35_MODEL_LOGIT_ROWS,
    undecodableLogitRows: QWEN35_MODEL_LOGIT_ROWS - tokenCount,
    mergeCount: definition.model.merges.length,
    addedTokenCount: definition.added_tokens.length,
    artifactByteLength: binary.byteLength,
    artifactSha256: createHash("sha256").update(binary).digest("hex"),
  });
  return Object.freeze({ manifest, binary });
}

export async function compileQwen35TokenizerSource(
  tokenizerChunks: AsyncIterable<Uint8Array>,
  configChunks: AsyncIterable<Uint8Array>,
): Promise<CompiledTokenizerPackage> {
  const compiled = await compileTokenizerSource(
    tokenizerChunks,
    PINNED_QWEN35_TOKENIZER,
    configChunks,
    PINNED_QWEN35_TOKENIZER_CONFIG,
    PINNED_QWEN35_CHAT_TEMPLATE_SHA256,
  );
  if (
    compiled.manifest.baseVocabSize !== 248_044 ||
    compiled.manifest.mergeCount !== 247_587 ||
    compiled.manifest.addedTokenCount !== 26 ||
    compiled.manifest.tokenCount !== 248_070
  ) {
    throw new Error("Pinned tokenizer counts do not match Qwen3.5");
  }
  return compiled;
}
