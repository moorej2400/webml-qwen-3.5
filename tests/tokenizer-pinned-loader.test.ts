import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import {
  readAuthenticatedTokenizerArtifact,
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
