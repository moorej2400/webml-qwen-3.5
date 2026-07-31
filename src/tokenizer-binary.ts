import { IncrementalSha256 } from "./incremental-sha256.js";

export const TOKENIZER_ARTIFACT_VERSION = 1;
export const TOKENIZER_BINARY_MAGIC = "Q35TOK01";
export const TOKENIZER_BINARY_HEADER_BYTES = 32;
export const MAX_COMPILED_TOKEN_COUNT = 300_000;
export const MAX_COMPILED_MERGE_COUNT = 300_000;
export const MAX_COMPILED_ADDED_TOKENS = 1_024;
export const QWEN35_MODEL_LOGIT_ROWS = 248_320;
export const MAX_COMPILED_TOKENIZER_BYTES = 8 * 1024 * 1024;
const SHA256 = /^[a-f0-9]{64}$/;

export interface BrowserTokenizerArtifactIdentity {
  readonly byteLength: number;
  readonly sha256: string;
}

export interface CompiledTokenizerTables {
  readonly baseVocabSize: number;
  readonly tokenCount: number;
  readonly tokenOffsets: Uint32Array;
  readonly tokenBytes: Uint8Array;
  /** Packed triples of left token id, right token id, and result token id. */
  readonly merges: Uint32Array;
  readonly addedTokenIds: Uint32Array;
  /** Bit 0 marks a special token; every listed id is an added-token boundary. */
  readonly addedTokenFlags: Uint8Array;
}

/**
 * Authenticates a bounded browser byte stream before a parser sees its header.
 *
 * `declaredByteLength` must come from trusted package metadata or a validated
 * response header. A mismatch fails before allocating the artifact buffer or
 * asking the iterable for its first chunk.
 */
export async function readAuthenticatedTokenizerArtifact(
  chunks: AsyncIterable<Uint8Array>,
  identity: BrowserTokenizerArtifactIdentity,
  declaredByteLength: number,
): Promise<Uint8Array> {
  if (
    !Number.isSafeInteger(identity.byteLength) ||
    identity.byteLength <= 0 ||
    identity.byteLength > MAX_COMPILED_TOKENIZER_BYTES ||
    !SHA256.test(identity.sha256)
  ) {
    throw new Error("Tokenizer artifact identity is invalid");
  }
  if (declaredByteLength !== identity.byteLength) {
    throw new Error(
      "Tokenizer artifact declared byte length does not match its identity",
    );
  }

  const output = new Uint8Array(identity.byteLength);
  const hash = new IncrementalSha256();
  let offset = 0;
  for await (const chunk of chunks) {
    if (!(chunk instanceof Uint8Array)) {
      throw new Error("Tokenizer artifact stream must contain bytes");
    }
    if (offset + chunk.byteLength > identity.byteLength) {
      throw new Error(
        "Tokenizer artifact stream exceeds its authenticated byte length",
      );
    }
    output.set(chunk, offset);
    hash.update(chunk);
    offset += chunk.byteLength;
  }
  if (offset !== identity.byteLength) {
    throw new Error(
      "Tokenizer artifact stream ended before its authenticated byte length",
    );
  }
  if (hash.digestHex() !== identity.sha256) {
    throw new Error("Tokenizer artifact SHA-256 does not match its identity");
  }
  return output;
}

/**
 * Validates every byte range before exposing tokenizer tables to WebGPU code.
 *
 * This module is browser-safe by design. Node hashing and file conversion stay
 * in `tokenizer-compiler.ts`, outside the production module graph.
 */
export function deserializeCompiledTokenizer(
  binary: Uint8Array,
): CompiledTokenizerTables {
  if (
    !(binary instanceof Uint8Array) ||
    binary.byteLength < TOKENIZER_BINARY_HEADER_BYTES ||
    binary.byteLength > MAX_COMPILED_TOKENIZER_BYTES
  ) {
    throw new Error("Compiled tokenizer artifact is truncated");
  }
  const magic = new TextEncoder().encode(TOKENIZER_BINARY_MAGIC);
  for (let index = 0; index < magic.length; index += 1) {
    if (binary[index] !== magic[index]) {
      throw new Error("Compiled tokenizer artifact has an invalid magic value");
    }
  }
  const view = new DataView(
    binary.buffer,
    binary.byteOffset,
    binary.byteLength,
  );
  if (view.getUint32(8, true) !== TOKENIZER_ARTIFACT_VERSION) {
    throw new Error("Compiled tokenizer artifact version is unsupported");
  }
  const tokenCount = view.getUint32(12, true);
  const baseVocabSize = view.getUint32(16, true);
  const mergeCount = view.getUint32(20, true);
  const addedCount = view.getUint32(24, true);
  const tokenDataLength = view.getUint32(28, true);
  if (
    tokenCount === 0 ||
    tokenCount > MAX_COMPILED_TOKEN_COUNT ||
    baseVocabSize > tokenCount ||
    mergeCount > MAX_COMPILED_MERGE_COUNT ||
    addedCount > MAX_COMPILED_ADDED_TOKENS ||
    baseVocabSize + addedCount !== tokenCount
  ) {
    throw new Error("Compiled tokenizer artifact counts are invalid");
  }
  const offsetsBytes = (tokenCount + 1) * 4;
  const mergesBytes = mergeCount * 12;
  const addedBytes = addedCount * 8;
  const expectedLength =
    TOKENIZER_BINARY_HEADER_BYTES +
    offsetsBytes +
    tokenDataLength +
    mergesBytes +
    addedBytes;
  if (expectedLength !== binary.byteLength) {
    throw new Error("Compiled tokenizer artifact ranges are invalid");
  }

  const tokenOffsets = new Uint32Array(tokenCount + 1);
  let cursor = TOKENIZER_BINARY_HEADER_BYTES;
  let previous = 0;
  for (let index = 0; index <= tokenCount; index += 1) {
    const offset = view.getUint32(cursor, true);
    if (offset < previous || offset > tokenDataLength) {
      throw new Error("Compiled tokenizer token offsets are invalid");
    }
    tokenOffsets[index] = offset;
    previous = offset;
    cursor += 4;
  }
  if (tokenOffsets[tokenCount] !== tokenDataLength) {
    throw new Error("Compiled tokenizer token data is incomplete");
  }
  const tokenBytes = binary.subarray(cursor, cursor + tokenDataLength);
  cursor += tokenDataLength;

  const merges = new Uint32Array(mergeCount * 3);
  for (let index = 0; index < merges.length; index += 1) {
    const id = view.getUint32(cursor, true);
    if (id >= baseVocabSize) {
      throw new Error("Compiled tokenizer merge token id is invalid");
    }
    merges[index] = id;
    cursor += 4;
  }

  const addedTokenIds = new Uint32Array(addedCount);
  const addedTokenFlags = new Uint8Array(addedCount);
  for (let index = 0; index < addedCount; index += 1) {
    const id = view.getUint32(cursor, true);
    const flags = view.getUint32(cursor + 4, true);
    if (id !== baseVocabSize + index || flags > 1) {
      throw new Error("Compiled tokenizer added-token record is invalid");
    }
    addedTokenIds[index] = id;
    addedTokenFlags[index] = flags;
    cursor += 8;
  }
  return {
    baseVocabSize,
    tokenCount,
    tokenOffsets,
    tokenBytes,
    merges,
    addedTokenIds,
    addedTokenFlags,
  };
}
