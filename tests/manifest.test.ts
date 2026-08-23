import assert from "node:assert/strict";
import test from "node:test";

import {
  stringifyManifest,
  validateModelPackageManifest,
  type ModelPackageManifest,
} from "../src/manifest.js";

const SHA_A = "a".repeat(64);
const SHA_B = "b".repeat(64);
const REVISION = "1".repeat(40);

function validManifest(): ModelPackageManifest {
  return {
    format: "webml-qwen-package",
    version: 1,
    packageKind: "language",
    source: {
      repository: "example/model",
      revision: REVISION,
      file: "model.gguf",
      size: "110",
      sha256: SHA_A,
    },
    runtime: { abi: "qwen35-webgpu-v2" },
    tokenizer: {
      repository: "example/tokenizer",
      revision: "2".repeat(40),
      file: "tokenizer.json",
      size: "512",
      sha256: SHA_B,
    },
    tensorLayout: [
      {
        name: "blk.0.attn_q.weight",
        shape: ["256"],
        ggmlType: 11,
        storageType: "q3-k-112",
        shard: 0,
        shardOffset: "0",
        tensorOffset: "0",
        length: "112",
        quantization: { blockElements: 256, blockBytes: 112 },
      },
    ],
    shards: [
      {
        url: "shards/model-00000.bin",
        offset: "0",
        length: "112",
        sha256: SHA_A,
      },
    ],
    excludedTensors: [
      { name: "mtp.output.weight", reason: "excluded-by-mtp-name-policy-v1" },
    ],
  };
}

test("accepts an immutable versioned language package manifest", () => {
  const manifest = validManifest();
  assert.equal(validateModelPackageManifest(manifest), manifest);
});

test("requires processor identity for vision packages", () => {
  const manifest = validManifest();
  manifest.packageKind = "vision";

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /processor.*vision/i,
  );

  manifest.processor = {
    repository: "example/processor",
    revision: "3".repeat(40),
    file: "processor_config.json",
    size: "256",
    sha256: SHA_A,
  };
  assert.throws(
    () => validateModelPackageManifest(manifest),
    /processor settings.*required/i,
  );
  manifest.processorSettings = {
    processorClass: "Qwen3VLProcessor",
    imageProcessorType: "Qwen2VLImageProcessorFast",
    patchSize: 16,
    temporalPatchSize: 2,
    mergeSize: 2,
    shortestEdge: 65_536,
    longestEdge: 16_777_216,
    imageMean: [0.5, 0.5, 0.5],
    imageStd: [0.5, 0.5, 0.5],
  };
  assert.doesNotThrow(() => validateModelPackageManifest(manifest));
});

test("rejects invalid vision preprocessing settings and language settings", () => {
  const vision = validManifest();
  vision.packageKind = "vision";
  vision.processor = {
    repository: "example/processor",
    revision: "3".repeat(40),
    file: "processor_config.json",
    size: "256",
    sha256: SHA_A,
  };
  vision.processorSettings = {
    processorClass: "Qwen3VLProcessor",
    imageProcessorType: "Qwen2VLImageProcessorFast",
    patchSize: 16,
    temporalPatchSize: 2,
    mergeSize: 2,
    shortestEdge: 65_536,
    longestEdge: 16_777_216,
    imageMean: [0.5, 0.5, 0.5],
    imageStd: [0.5, 0.5, 0.5],
  };
  vision.processorSettings.shortestEdge = 16_777_217;
  assert.throws(
    () => validateModelPackageManifest(vision),
    /shortest edge.*longest edge/i,
  );
  vision.processorSettings.shortestEdge = 65_536;
  vision.processorSettings.imageMean = [0.5, Number.POSITIVE_INFINITY, 0.5];
  assert.throws(
    () => validateModelPackageManifest(vision),
    /finite channel values/i,
  );

  const language = validManifest();
  language.processorSettings = vision.processorSettings;
  assert.throws(
    () => validateModelPackageManifest(language),
    /only valid for a vision package/i,
  );
});

test("rejects mutable source and tokenizer revisions", () => {
  const sourceMutable = validManifest();
  sourceMutable.source.revision = "main";
  assert.throws(
    () => validateModelPackageManifest(sourceMutable),
    /immutable.*revision/i,
  );

  const tokenizerMutable = validManifest();
  tokenizerMutable.tokenizer.revision = "refs/heads/main";
  assert.throws(
    () => validateModelPackageManifest(tokenizerMutable),
    /immutable.*revision/i,
  );
});

test("rejects malformed hashes and unsafe integer strings", () => {
  const badHash = validManifest();
  badHash.shards[0]!.sha256 = "not-a-hash";
  assert.throws(() => validateModelPackageManifest(badHash), /sha-?256/i);

  const badSize = validManifest();
  badSize.source.size = "1e6";
  assert.throws(() => validateModelPackageManifest(badSize), /size/i);
});

test("rejects tensor ranges outside their shard", () => {
  const manifest = validManifest();
  manifest.tensorLayout[0]!.shardOffset = "64";

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /tensor.*shard bounds/i,
  );
});

test("requires explicit WebGPU tensor mappings to start on u32 boundaries", () => {
  const manifest = validManifest();
  manifest.shards[0]!.length = "114";
  manifest.tensorLayout[0]!.shardOffset = "2";

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /shardOffset.*u32 aligned/i,
  );
});

test("rejects duplicate tensor names unless segments have distinct offsets", () => {
  const manifest = validManifest();
  manifest.tensorLayout.push({ ...manifest.tensorLayout[0]! });

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /duplicate tensor segment/i,
  );
});

test("serializes deterministic manifest JSON independent of object insertion order", () => {
  const manifest = validManifest();
  const reordered = {
    excludedTensors: manifest.excludedTensors,
    shards: manifest.shards,
    tensorLayout: manifest.tensorLayout,
    tokenizer: manifest.tokenizer,
    runtime: manifest.runtime,
    source: manifest.source,
    packageKind: manifest.packageKind,
    version: manifest.version,
    format: manifest.format,
  } as ModelPackageManifest;

  assert.equal(stringifyManifest(manifest), stringifyManifest(reordered));
  assert.match(stringifyManifest(manifest), /^\{\n  "excludedTensors"/);
  assert.ok(stringifyManifest(manifest).endsWith("\n"));
});

test("rejects storage formats that contradict the GGML tensor type", () => {
  const manifest = validManifest();
  manifest.tensorLayout[0]!.ggmlType = 0;

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /q3-k-112.*Q3_K/i,
  );
});

test("requires every segment of one tensor to have consistent attributes", () => {
  const manifest = validManifest();
  manifest.shards[0]!.length = "224";
  manifest.tensorLayout[0]!.shape = ["512"];
  manifest.tensorLayout.push({
    ...manifest.tensorLayout[0]!,
    shape: ["256"],
    shardOffset: "112",
    tensorOffset: "112",
  });

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /tensor.*consistent.*attributes/i,
  );
});

test("requires contiguous tensor segments with exact shape-derived coverage", () => {
  const manifest = validManifest();
  manifest.shards[0]!.length = "336";
  manifest.tensorLayout[0]!.shape = ["512"];
  manifest.tensorLayout.push({
    ...manifest.tensorLayout[0]!,
    shardOffset: "224",
    tensorOffset: "224",
  });

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /tensor.*contiguous/i,
  );

  const incomplete = validManifest();
  incomplete.tensorLayout[0]!.shape = ["512"];
  assert.throws(
    () => validateModelPackageManifest(incomplete),
    /tensor.*exact.*length/i,
  );
});

test("rejects overlapping tensor byte mappings inside a shard", () => {
  const manifest = validManifest();
  manifest.tensorLayout.push({
    name: "output_norm.weight",
    shape: ["1"],
    ggmlType: 0,
    storageType: "f32",
    shard: 0,
    shardOffset: "0",
    tensorOffset: "0",
    length: "4",
  });

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /overlapping.*shard.*mapping/i,
  );
});

test("rejects raw tensor segments that split native storage blocks", () => {
  const manifest = validManifest();
  manifest.shards[0]!.length = "18";
  manifest.tensorLayout[0] = {
    name: "blk.0.weight",
    shape: ["32"],
    ggmlType: 2,
    storageType: "raw",
    shard: 0,
    shardOffset: "0",
    tensorOffset: "0",
    length: "9",
  };
  manifest.tensorLayout.push({
    ...manifest.tensorLayout[0],
    shardOffset: "9",
    tensorOffset: "9",
  });

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /native storage block alignment/i,
  );
});

test("compares segment quantization attributes by value, not key order", () => {
  const manifest = validManifest();
  manifest.shards[0]!.length = "224";
  manifest.tensorLayout[0]!.shape = ["512"];
  manifest.tensorLayout.push({
    ...manifest.tensorLayout[0]!,
    shardOffset: "112",
    tensorOffset: "112",
    quantization: { blockBytes: 112, blockElements: 256 },
  });

  assert.doesNotThrow(() => validateModelPackageManifest(manifest));
});

test("rejects quantized tensors whose contiguous row is a partial block", () => {
  const manifest = validManifest();
  manifest.tensorLayout[0]!.shape = ["1", "256"];

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /contiguous row dimension.*complete.*block/i,
  );
});

test("bounds decimal digits before parsing byte counts", () => {
  const manifest = validManifest();
  manifest.source.size = "1".repeat(21);

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /size.*decimal digit bound/i,
  );
});

test("requires a non-empty tensor layout with rank from one through four", () => {
  const empty = validManifest();
  empty.tensorLayout = [];
  assert.throws(
    () => validateModelPackageManifest(empty),
    /tensorLayout.*non-empty/i,
  );

  const excessiveRank = validManifest();
  excessiveRank.tensorLayout[0]!.shape = ["256", "1", "1", "1", "1"];
  assert.throws(
    () => validateModelPackageManifest(excessiveRank),
    /tensor rank.*1.*4/i,
  );
});

test("bounds manifest array counts before walking their entries", () => {
  const manifest = validManifest();
  manifest.shards = Array.from({ length: 4_097 }, () => ({
    ...manifest.shards[0]!,
  }));

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /shard count.*bound/i,
  );
});

test("bounds manifest string byte lengths", () => {
  const manifest = validManifest();
  manifest.runtime.abi = "a".repeat(65_536);

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /runtime ABI.*byte length/i,
  );
});

test("rejects malformed HTTPS shard URLs with WHATWG parsing", () => {
  const manifest = validManifest();
  manifest.shards[0]!.url = "https://[invalid";

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /shard URL.*valid WHATWG URL/i,
  );
});

test("rejects encoded traversal in relative and HTTPS shard paths", () => {
  const relative = validManifest();
  relative.shards[0]!.url = "shards/%2e%2e/secret.bin";
  assert.throws(
    () => validateModelPackageManifest(relative),
    /shard URL.*encoded traversal/i,
  );

  const absolute = validManifest();
  absolute.shards[0]!.url = "https://example.invalid/%2e%2e/secret.bin";
  assert.throws(
    () => validateModelPackageManifest(absolute),
    /shard URL.*encoded traversal/i,
  );
});

test("enforces the version-one MTP exclusion name and reason policy", () => {
  const invalidName = validManifest();
  invalidName.excludedTensors[0]!.name = "attempt.weight";
  assert.throws(
    () => validateModelPackageManifest(invalidName),
    /excluded tensor.*MTP.*name policy/i,
  );

  const invalidReason = validManifest();
  invalidReason.excludedTensors[0]!.reason = "manual exclusion";
  assert.throws(
    () => validateModelPackageManifest(invalidReason),
    /excluded tensor.*reason.*version 1/i,
  );
});

test("accepts pinned block 32 exclusions and rejects block-like substrings", () => {
  const manifest = validManifest();
  manifest.excludedTensors = [
    { name: "blk.32.attn_q.weight", reason: "excluded-by-mtp-name-policy-v1" },
    { name: "blk.32.ffn_up.weight", reason: "excluded-by-mtp-name-policy-v1" },
    {
      name: "blk.32.post_attention_layernorm.weight",
      reason: "excluded-by-mtp-name-policy-v1",
    },
  ];
  assert.doesNotThrow(() => validateModelPackageManifest(manifest));

  for (const name of [
    "blk.31.attn_q.weight",
    "blk.320.attn_q.weight",
    "xblk.32.attn_q.weight",
    "blk.32ish.weight",
  ]) {
    const invalid = validManifest();
    invalid.excludedTensors[0]!.name = name;
    assert.throws(
      () => validateModelPackageManifest(invalid),
      /does not match.*MTP.*policy/i,
    );
  }
});

test("requires included and excluded tensor names to be disjoint", () => {
  const manifest = validManifest();
  manifest.tensorLayout[0]!.name = "model.mtp.output.weight";
  manifest.excludedTensors[0]!.name = "model.mtp.output.weight";

  assert.throws(
    () => validateModelPackageManifest(manifest),
    /included and excluded.*disjoint/i,
  );
});
