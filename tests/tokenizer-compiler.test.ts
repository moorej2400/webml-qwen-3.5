import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import {
  PINNED_QWEN35_TOKENIZER,
  TOKENIZER_ARTIFACT_VERSION,
  compileQwen35TokenizerSource,
  compileTokenizerSource,
  deserializeCompiledTokenizer,
  QWEN35_MODEL_LOGIT_ROWS,
  type TokenizerSourceIdentity,
} from "../src/tokenizer-compiler.js";

const SPLIT_PATTERN =
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

function miniTokenizerJson(): string {
  return JSON.stringify({
    version: "1.0",
    truncation: null,
    padding: null,
    added_tokens: [
      {
        id: 3,
        content: "<|im_start|>",
        single_word: false,
        lstrip: false,
        rstrip: false,
        normalized: false,
        special: true,
      },
    ],
    normalizer: { type: "NFC" },
    pre_tokenizer: {
      type: "Sequence",
      pretokenizers: [
        {
          type: "Split",
          pattern: { Regex: SPLIT_PATTERN },
          behavior: "Isolated",
          invert: false,
        },
        {
          type: "ByteLevel",
          add_prefix_space: false,
          trim_offsets: false,
          use_regex: false,
        },
      ],
    },
    post_processor: {
      type: "ByteLevel",
      add_prefix_space: false,
      trim_offsets: false,
      use_regex: false,
    },
    decoder: {
      type: "ByteLevel",
      add_prefix_space: false,
      trim_offsets: false,
      use_regex: false,
    },
    model: {
      type: "BPE",
      dropout: null,
      unk_token: null,
      continuing_subword_prefix: "",
      end_of_word_suffix: "",
      fuse_unk: false,
      byte_fallback: false,
      ignore_merges: false,
      vocab: { a: 0, b: 1, ab: 2 },
      merges: ["a b"],
    },
  });
}

function sourceIdentity(
  text: string,
  file = "tokenizer.json",
): TokenizerSourceIdentity {
  return {
    repository: "https://huggingface.co/example/public-model",
    revision: "0123456789abcdef0123456789abcdef01234567",
    file,
    size: new TextEncoder().encode(text).byteLength,
    sha256: createHash("sha256").update(text).digest("hex"),
  };
}

async function* chunked(text: string): AsyncIterable<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  yield bytes.subarray(0, 19);
  yield bytes.subarray(19);
}

const miniConfig = JSON.stringify({
  chat_template: "mini-template-v1",
  eos_token: "<|im_end|>",
  pad_token: "<|endoftext|>",
  model_max_length: 262_144,
});
const miniTemplateSha256 = createHash("sha256")
  .update("mini-template-v1")
  .digest("hex");

function compileMini(json: string) {
  return compileTokenizerSource(
    chunked(json),
    sourceIdentity(json),
    chunked(miniConfig),
    sourceIdentity(miniConfig, "tokenizer_config.json"),
    miniTemplateSha256,
  );
}

test("compiles deterministic compact tokenizer tables from bounded chunks", async () => {
  const json = miniTokenizerJson();

  const first = await compileMini(json);
  const second = await compileMini(json);

  assert.deepEqual(first.binary, second.binary);
  assert.deepEqual(first.manifest, second.manifest);
  assert.equal(first.manifest.format, "webml-qwen35-tokenizer");
  assert.equal(first.manifest.version, TOKENIZER_ARTIFACT_VERSION);
  assert.equal(first.manifest.baseVocabSize, 3);
  assert.equal(first.manifest.tokenCount, 4);
  assert.equal(first.manifest.mergeCount, 1);
  assert.equal(first.manifest.addedTokenCount, 1);
  assert.equal(first.manifest.decodableTokenCount, 4);
  assert.equal(first.manifest.modelLogitRows, QWEN35_MODEL_LOGIT_ROWS);
  assert.equal(first.manifest.undecodableLogitRows, 248_316);
  assert.equal(first.manifest.chatTemplateSha256, miniTemplateSha256);
  assert.ok(Object.isFrozen(first.manifest));
  assert.ok(Object.isFrozen(first.manifest.source));
  assert.ok(Object.isFrozen(first.manifest.chatTemplateSource));

  const tables = deserializeCompiledTokenizer(first.binary);
  assert.equal(tables.baseVocabSize, 3);
  assert.equal(tables.tokenCount, 4);
  assert.deepEqual(Array.from(tables.merges), [0, 1, 2]);
  assert.deepEqual(Array.from(tables.addedTokenIds), [3]);
});

test("rejects source bytes that do not match the immutable identity", async () => {
  const json = miniTokenizerJson();
  const expected = sourceIdentity(json);

  await assert.rejects(
    compileTokenizerSource(
      chunked(`${json} `),
      { ...expected, size: expected.size + 1 },
      chunked(miniConfig),
      sourceIdentity(miniConfig, "tokenizer_config.json"),
      miniTemplateSha256,
    ),
    (error: unknown) =>
      error instanceof Error &&
      error.message === "Tokenizer source SHA-256 does not match its identity",
  );
});

test("stops reading when source bytes exceed the declared bounded size", async () => {
  const json = miniTokenizerJson();
  const expected = sourceIdentity(json);
  let requestedSecondChunk = false;
  async function* oversized(): AsyncIterable<Uint8Array> {
    yield new Uint8Array(expected.size + 1);
    requestedSecondChunk = true;
    yield new Uint8Array(1);
  }

  await assert.rejects(
    compileTokenizerSource(
      oversized(),
      expected,
      chunked(miniConfig),
      sourceIdentity(miniConfig, "tokenizer_config.json"),
      miniTemplateSha256,
    ),
    (error: unknown) =>
      error instanceof Error &&
      error.message === "Tokenizer source exceeds its declared byte length",
  );
  assert.equal(requestedSecondChunk, false);
});

test("rejects tokenizer schemas that differ from the pinned model contract", async () => {
  const parsed = JSON.parse(miniTokenizerJson()) as {
    normalizer: { type: string };
  };
  parsed.normalizer.type = "NFD";
  const json = JSON.stringify(parsed);

  await assert.rejects(
    compileMini(json),
    (error: unknown) =>
      error instanceof Error &&
      error.message === "Tokenizer normalizer must be NFC",
  );
});

test("the pinned compiler rejects any artifact other than the exact public source", async () => {
  const json = miniTokenizerJson();

  await assert.rejects(
    compileQwen35TokenizerSource(chunked(json), chunked(miniConfig)),
    (error: unknown) =>
      error instanceof Error &&
      error.message ===
        "Tokenizer source byte length does not match its identity",
  );
  assert.deepEqual(PINNED_QWEN35_TOKENIZER, {
    repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
    revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    file: "tokenizer.json",
    size: 12_807_982,
    sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
  });
});

test("does not claim chat-template provenance until config bytes are verified", async () => {
  const json = miniTokenizerJson();
  const badConfig = `${miniConfig} `;

  await assert.rejects(
    compileTokenizerSource(
      chunked(json),
      sourceIdentity(json),
      chunked(badConfig),
      {
        ...sourceIdentity(miniConfig, "tokenizer_config.json"),
        size: new TextEncoder().encode(badConfig).byteLength,
      },
      miniTemplateSha256,
    ),
    (error: unknown) =>
      error instanceof Error &&
      error.message ===
        "Tokenizer config SHA-256 does not match its identity",
  );
});

test("rejects a verified config whose chat template differs from compiled behavior", async () => {
  const json = miniTokenizerJson();
  const changedConfig = JSON.stringify({
    ...JSON.parse(miniConfig),
    chat_template: "different-template",
  });

  await assert.rejects(
    compileTokenizerSource(
      chunked(json),
      sourceIdentity(json),
      chunked(changedConfig),
      sourceIdentity(changedConfig, "tokenizer_config.json"),
      miniTemplateSha256,
    ),
    (error: unknown) =>
      error instanceof Error &&
      error.message ===
        "Tokenizer config chat template does not match its implementation pin",
  );
});
