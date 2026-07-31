import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import {
  deserializeAuthenticatedTokenizerArtifact,
  readAuthenticatedTokenizerArtifact,
  TOKENIZER_ARTIFACT_VERSION,
  TOKENIZER_BINARY_HEADER_BYTES,
  TOKENIZER_BINARY_MAGIC,
  type BrowserTokenizerArtifactIdentity,
} from "../src/tokenizer-binary.js";
import {
  loadPinnedQwen35Tokenizer,
  PINNED_QWEN35_COMPILED_TOKENIZER,
  Qwen35Tokenizer,
} from "../src/qwen-tokenizer.js";

async function* chunks(...values: Uint8Array[]): AsyncIterable<Uint8Array> {
  yield* values;
}

test("authenticates bounded browser bytes before returning ownership", async () => {
  const bytes = new TextEncoder().encode("authenticated fixture");
  const identity: BrowserTokenizerArtifactIdentity = {
    byteLength: bytes.byteLength,
    sha256: createHash("sha256").update(bytes).digest("hex"),
  };

  const authenticated = await readAuthenticatedTokenizerArtifact(
    chunks(bytes.subarray(0, 4), bytes.subarray(4)),
    identity,
    bytes.byteLength,
  );

  assert.deepEqual(authenticated, bytes);
  assert.equal(Object.isFrozen(authenticated), false);
});

test("rejects a declared size before allocating or consuming the stream", async () => {
  let consumed = false;
  async function* source(): AsyncIterable<Uint8Array> {
    consumed = true;
    yield new Uint8Array(1);
  }

  await assert.rejects(
    loadPinnedQwen35Tokenizer(
      source(),
      PINNED_QWEN35_COMPILED_TOKENIZER.byteLength - 1,
    ),
    /declared byte length/u,
  );
  assert.equal(consumed, false);
});

test("rejects wrong hashes and oversized chunks without reading later chunks", async () => {
  const expected = new TextEncoder().encode("expected");
  const identity: BrowserTokenizerArtifactIdentity = {
    byteLength: expected.byteLength,
    sha256: createHash("sha256").update(expected).digest("hex"),
  };
  let requestedSecondChunk = false;
  async function* oversized(): AsyncIterable<Uint8Array> {
    yield new Uint8Array(expected.byteLength + 1);
    requestedSecondChunk = true;
    yield new Uint8Array(1);
  }

  await assert.rejects(
    readAuthenticatedTokenizerArtifact(
      oversized(),
      identity,
      identity.byteLength,
    ),
    /exceeds its authenticated byte length/u,
  );
  assert.equal(requestedSecondChunk, false);
  await assert.rejects(
    readAuthenticatedTokenizerArtifact(
      chunks(new Uint8Array(expected.byteLength)),
      identity,
      identity.byteLength,
    ),
    /SHA-256/u,
  );
});

test("pins exact compiled counts and all 250 non-tokenizer logit rows", () => {
  assert.deepEqual(PINNED_QWEN35_COMPILED_TOKENIZER, {
    byteLength: 5_806_953,
    sha256: "7c0e92451601511a7396d8897f6f4ae16d4783197f6467e9d15e4bc6bc7197bd",
    baseVocabSize: 248_044,
    mergeCount: 247_587,
    addedTokenCount: 26,
    decodableTokenCount: 248_070,
    modelLogitRows: 248_320,
    maskedModelRows: 250,
  });
});

test("exposes unauthenticated construction only through explicit unsafe paths", () => {
  assert.equal("fromCompiledArtifact" in Qwen35Tokenizer, false);
  assert.equal("fromTables" in Qwen35Tokenizer, false);
  assert.equal(
    typeof Qwen35Tokenizer.fromUnsafeCompiledArtifactForTests,
    "function",
  );
  assert.equal(typeof Qwen35Tokenizer.fromUnsafeTablesForTests, "function");
});

test("authenticates and deserializes deterministic exact-count package bytes", async () => {
  const binary = exactCountBinaryFixture();
  const identity = {
    byteLength: binary.byteLength,
    sha256: "04057210f184b1e82039a66942b544048f6cf7f6e71c0816d4b2fcca0f1d81b1",
  };
  assert.equal(
    createHash("sha256").update(binary).digest("hex"),
    identity.sha256,
  );

  const tables = await deserializeAuthenticatedTokenizerArtifact(
    chunks(
      binary.subarray(0, 31),
      binary.subarray(31, 1_048_607),
      binary.subarray(1_048_607),
    ),
    identity,
    identity.byteLength,
  );

  assert.equal(tables.baseVocabSize, 248_044);
  assert.equal(tables.tokenCount, 248_070);
  assert.equal(tables.merges.length / 3, 247_587);
  assert.equal(tables.addedTokenIds.length, 26);
});

function exactCountBinaryFixture(): Uint8Array {
  const tokenCount = 248_070;
  const baseVocabSize = 248_044;
  const mergeCount = 247_587;
  const addedCount = 26;
  const totalBytes =
    TOKENIZER_BINARY_HEADER_BYTES +
    (tokenCount + 1) * 4 +
    mergeCount * 12 +
    addedCount * 8;
  const binary = new Uint8Array(totalBytes);
  binary.set(new TextEncoder().encode(TOKENIZER_BINARY_MAGIC));
  const view = new DataView(binary.buffer);
  view.setUint32(8, TOKENIZER_ARTIFACT_VERSION, true);
  view.setUint32(12, tokenCount, true);
  view.setUint32(16, baseVocabSize, true);
  view.setUint32(20, mergeCount, true);
  view.setUint32(24, addedCount, true);
  view.setUint32(28, 0, true);

  let cursor =
    TOKENIZER_BINARY_HEADER_BYTES +
    (tokenCount + 1) * 4 +
    mergeCount * 12;
  for (let index = 0; index < addedCount; index += 1) {
    view.setUint32(cursor, baseVocabSize + index, true);
    view.setUint32(cursor + 4, index % 2, true);
    cursor += 8;
  }
  return binary;
}
