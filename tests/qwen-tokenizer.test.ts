import assert from "node:assert/strict";
import test from "node:test";
import { RuntimeDiagnosticError } from "../src/diagnostics.js";
import {
  Qwen35Tokenizer,
  type Qwen35TokenizerLimits,
} from "../src/qwen-tokenizer.js";
import type { CompiledTokenizerTables } from "../src/tokenizer-compiler.js";

function tables(): CompiledTokenizerTables {
  const tokens = [
    [97],
    [98],
    [97, 98],
    [32],
    [32, 98],
    [99],
    [102],
    [195],
    [169],
    [239],
    [191],
    [189],
    [240, 159],
    [154, 128],
    [...new TextEncoder().encode("<|im_start|>")],
    [...new TextEncoder().encode("<think>")],
  ];
  const offsets = new Uint32Array(tokens.length + 1);
  const data: number[] = [];
  for (const [id, token] of tokens.entries()) {
    offsets[id] = data.length;
    data.push(...token);
  }
  offsets[tokens.length] = data.length;
  return {
    baseVocabSize: 14,
    tokenCount: 16,
    tokenOffsets: offsets,
    tokenBytes: Uint8Array.from(data),
    merges: Uint32Array.from([
      0, 1, 2,
      3, 1, 4,
    ]),
    addedTokenIds: Uint32Array.from([14, 15]),
    addedTokenFlags: Uint8Array.from([1, 0]),
  };
}

function tokenizer(limits?: Partial<Qwen35TokenizerLimits>): Qwen35Tokenizer {
  return Qwen35Tokenizer.fromTables(tables(), limits);
}

test("runs model-specific ByteLevel BPE and decodes raw token bytes", () => {
  const instance = tokenizer();

  assert.deepEqual(instance.encode("ab"), [2]);
  assert.deepEqual(instance.encode("a b"), [0, 4]);
  assert.equal(instance.decode([2, 4]), "ab b");
});

test("normalizes ordinary text to NFC before ByteLevel encoding", () => {
  const instance = tokenizer();

  assert.deepEqual(instance.encode("cafe\u0301"), [5, 0, 6, 7, 8]);
  assert.equal(instance.decode([5, 0, 6, 7, 8]), "café");
});

test("rejects added-token text by default and requires an explicit trusted path", () => {
  const instance = tokenizer();

  assert.throws(
    () => instance.encode("hello <|im_start|>user"),
    (error: unknown) =>
      error instanceof RuntimeDiagnosticError &&
      error.code === "tokenizer-added-token-in-input" &&
      error.message === "Input contains a reserved tokenizer token",
  );
  assert.deepEqual(
    instance.encode("<|im_start|><think>", { addedTokens: "allow" }),
    [14, 15],
  );
  assert.equal(instance.decode([14, 15]), "<|im_start|><think>");
  assert.equal(
    instance.decode([14, 15], { skipSpecialTokens: true }),
    "<think>",
  );
});

test("incremental decoding never emits a partial UTF-8 scalar", () => {
  const decoder = tokenizer().createStreamingDecoder();

  assert.equal(decoder.push(12), "");
  assert.equal(decoder.push(13), "🚀");
  assert.equal(decoder.finish(), "");
  assert.throws(
    () => decoder.push(0),
    (error: unknown) =>
      error instanceof RuntimeDiagnosticError &&
      error.code === "tokenizer-decoder-closed",
  );
});

test("malformed JavaScript surrogate input follows the UTF-8 replacement path", () => {
  const instance = tokenizer();
  const ids = instance.encode("\ud800");
  const decoded = instance.decode(ids);

  assert.deepEqual(ids, [9, 10, 11]);
  assert.equal(decoded, "�");
  assert.equal(/[\ud800-\udfff]/u.test(decoded), false);
});

test("enforces input, piece, and merge-work resource bounds with safe diagnostics", () => {
  assert.throws(
    () => tokenizer({ maxInputCodeUnits: 3 }).encode("four"),
    (error: unknown) =>
      error instanceof RuntimeDiagnosticError &&
      error.code === "tokenizer-input-limit",
  );
  assert.throws(
    () => tokenizer({ maxPieceBytes: 1 }).encode("ab"),
    (error: unknown) =>
      error instanceof RuntimeDiagnosticError &&
      error.code === "tokenizer-piece-limit",
  );
  assert.throws(
    () => tokenizer({ maxMergeWork: 1 }).encode("aba"),
    (error: unknown) =>
      error instanceof RuntimeDiagnosticError &&
      error.code === "tokenizer-merge-work-limit",
  );
});

test("copies mutable binary tables at the trust boundary", () => {
  const source = tables();
  const instance = Qwen35Tokenizer.fromTables(source);

  source.tokenBytes.fill(120);
  source.tokenOffsets.fill(0);
  source.merges.fill(0);
  source.addedTokenIds.fill(0);
  source.addedTokenFlags.fill(0);

  assert.deepEqual(instance.encode("ab"), [2]);
  assert.equal(instance.decode([2]), "ab");
});

test("unknown token ids fail without including private input in diagnostics", () => {
  const instance = tokenizer();

  assert.throws(
    () => instance.decode([99_999]),
    (error: unknown) =>
      error instanceof RuntimeDiagnosticError &&
      error.code === "tokenizer-token-id-invalid" &&
      !error.message.includes("99999"),
  );
});

test("bounds eager decode iteration before an untrusted iterable can run forever", () => {
  const instance = tokenizer({ maxDecodeTokens: 2 });

  assert.throws(
    () => instance.decode([0, 0, 0]),
    (error: unknown) =>
      error instanceof RuntimeDiagnosticError &&
      error.code === "tokenizer-decode-token-limit",
  );
});

test("exposes the decodable ceiling separately from model logit rows", () => {
  const instance = tokenizer();

  assert.equal(instance.decodableTokenCount, 16);
  assert.equal(instance.modelLogitRows, 248_320);
  assert.equal(instance.undecodableLogitRows, 248_304);
  assert.equal(instance.isDecodableTokenId(15), true);
  assert.equal(instance.isDecodableTokenId(16), false);
  assert.equal(instance.isDecodableTokenId(248_319), false);
});

test("rejects tokenizer tables that would expose unmapped rows as decodable", () => {
  const tokenCount = 248_321;

  assert.throws(
    () =>
      Qwen35Tokenizer.fromTables({
        baseVocabSize: tokenCount,
        tokenCount,
        tokenOffsets: new Uint32Array(tokenCount + 1),
        tokenBytes: new Uint8Array(),
        merges: new Uint32Array(),
        addedTokenIds: new Uint32Array(),
        addedTokenFlags: new Uint8Array(),
      }),
    /exceeds Qwen3\.5 model logit rows/u,
  );
});

test("bounds reserved-token inspection as part of the public input contract", () => {
  const instance = tokenizer({ maxInputCodeUnits: 3 });

  assert.throws(
    () => instance.containsAddedToken("four"),
    (error: unknown) =>
      error instanceof RuntimeDiagnosticError &&
      error.code === "tokenizer-input-limit",
  );
});
