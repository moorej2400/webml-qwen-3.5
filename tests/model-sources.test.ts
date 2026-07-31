import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  MAX_EXCLUDED_TENSORS,
  MAX_SHARDS,
  MAX_TENSOR_SEGMENTS,
} from "../src/manifest.js";
import { PINNED_LANGUAGE_BLOCK_POLICY } from "../src/tensor-policy.js";

test("pins the public source identities and inspected language inventory", async () => {
  const descriptor = JSON.parse(
    await readFile(new URL("../model-sources.json", import.meta.url), "utf8"),
  );

  assert.deepEqual(descriptor.official, {
    repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
    revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
  });
  assert.deepEqual(descriptor.tokenizer, {
    repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
    revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    file: "tokenizer.json",
    size: 12807982,
    sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
  });
  assert.deepEqual(descriptor.language, {
    repository: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF",
    revision: "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
    file: "Qwen_Qwen3.5-4B-Q3_K_L.gguf",
    size: 2665441248,
    sha256: "41c3f1bf47e477693dab332e73347c7138d5e9fbfe74c6d2eaba590be1f3d20a",
    dataOffset: 10969056,
    alignment: 32,
    blockCount: 33,
    excludedBlock: 32,
    tensorTypeCounts: {
      F32: 232,
      Q8_0: 20,
      Q3_K: 112,
      Q4_K: 8,
      Q5_K: 64,
      Q6_K: 5,
    },
  });
  assert.deepEqual(descriptor.vision, {
    repository: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF",
    revision: "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
    file: "mmproj-Qwen_Qwen3.5-4B-bf16.gguf",
    size: 675569216,
    sha256: "463f39bd1c291c1186c319a8c90ff8640aafa678b14cbee2232d695113dfbb66",
  });
  assert.deepEqual(PINNED_LANGUAGE_BLOCK_POLICY, {
    blockCount: descriptor.language.blockCount,
    baseBlockCount: descriptor.language.excludedBlock,
    excludedBlock: descriptor.language.excludedBlock,
  });
  const tensorCount = Object.values<number>(
    descriptor.language.tensorTypeCounts,
  ).reduce((total, count) => total + count, 0);
  const shardCount = Math.ceil(
    descriptor.language.size / (128 * 1024 * 1024),
  );
  assert.ok(shardCount < MAX_SHARDS);
  assert.ok(tensorCount + shardCount < MAX_TENSOR_SEGMENTS);
  assert.ok(tensorCount < MAX_EXCLUDED_TENSORS);
});
