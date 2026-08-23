import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";

import {
  MAX_STREAM_READ_BYTES,
  MTP_EXCLUSION_REASON,
  createManifestFromPlan,
  executeConversionPlan,
  planConversion,
  type ConversionPlan,
  type RandomAccessWriter,
} from "../src/converter.js";
import { GgmlType, type ParsedGguf } from "../src/gguf.js";
import {
  MAX_EXCLUDED_TENSORS,
  MAX_SHARDS,
  MAX_TENSOR_SEGMENTS,
  stringifyManifest,
  type ImmutableArtifactIdentity,
} from "../src/manifest.js";
import { NATIVE_Q3_K_BLOCK_BYTES } from "../src/q3k.js";
import {
  repackNativeQ3KFusedBrowser,
  repackNativeQ6KBrowser,
} from "../src/browser-quant.js";
import {
  NATIVE_Q6_K_BLOCK_BYTES,
  NATIVE_Q8_0_BLOCK_BYTES,
} from "../src/mixed-quant.js";
import { WEBGPU_LANGUAGE_TENSOR_LAYOUTS } from "../src/tensor-policy.js";
import { memoryReader } from "./fixture-utils.js";

const REVISION = "1".repeat(40);
const SHA_A = "a".repeat(64);

function fixtureGguf(tensors = mixedTensors()): ParsedGguf {
  return {
    version: 3,
    metadata: {},
    alignment: 32,
    dataOffset: 1024n,
    tensors,
  };
}

function mixedTensors(): ParsedGguf["tensors"] {
  return [
    {
      name: "blk.0.attn_q.weight",
      dimensions: [512n],
      type: GgmlType.Q3_K,
      offset: 0n,
    },
    {
      name: "output_norm.weight",
      dimensions: [16n],
      type: GgmlType.F32,
      offset: 224n,
    },
    {
      name: "model.mtp.output.weight",
      dimensions: [256n],
      type: GgmlType.Q3_K,
      offset: 288n,
    },
  ];
}

test("inventories mixed source types and records explicit MTP exclusions", () => {
  const plan = planConversion(fixtureGguf(), {
    maxShardBytes: 384n,
    tensorAlignment: 16,
  });

  assert.deepEqual(plan.inventory, [
    { ggmlType: GgmlType.F32, tensorCount: 1, sourceBytes: 64n },
    { ggmlType: GgmlType.Q3_K, tensorCount: 2, sourceBytes: 330n },
  ]);
  assert.deepEqual(plan.excludedTensors, [
    {
      name: "model.mtp.output.weight",
      reason: MTP_EXCLUSION_REASON,
    },
  ]);
  assert.ok(plan.segments.every((segment) => !segment.tensorName.includes(".mtp.")));
});

test("plans and manifests every pinned language tensor layout explicitly", () => {
  const tensors: ParsedGguf["tensors"] = [
    { name: "a.f32", dimensions: [4n], type: GgmlType.F32, offset: 0n },
    { name: "b.q8", dimensions: [32n], type: GgmlType.Q8_0, offset: 32n },
    { name: "c.q3", dimensions: [256n], type: GgmlType.Q3_K, offset: 96n },
    { name: "d.q4", dimensions: [256n], type: GgmlType.Q4_K, offset: 224n },
    { name: "e.q5", dimensions: [256n], type: GgmlType.Q5_K, offset: 384n },
    { name: "f.q6", dimensions: [256n], type: GgmlType.Q6_K, offset: 576n },
  ];
  const plan = planConversion(fixtureGguf(tensors), {
    maxShardBytes: 4096n,
    tensorAlignment: 16,
  });

  assert.deepEqual(
    plan.segments.map(({ ggmlType, transform, sourceBlockBytes, outputBlockBytes }) => ({
      ggmlType,
      transform,
      sourceBlockBytes,
      outputBlockBytes,
    })),
    [
      { ggmlType: GgmlType.F32, transform: "copy", sourceBlockBytes: 4, outputBlockBytes: 4 },
      { ggmlType: GgmlType.Q8_0, transform: "q8-0-34-to-36", sourceBlockBytes: 34, outputBlockBytes: 36 },
      { ggmlType: GgmlType.Q3_K, transform: "q3-k-110-to-fused-f32-192", sourceBlockBytes: 110, outputBlockBytes: 192 },
      { ggmlType: GgmlType.Q4_K, transform: "q4-k-144-to-fused-f32-192", sourceBlockBytes: 144, outputBlockBytes: 192 },
      { ggmlType: GgmlType.Q5_K, transform: "q5-k-176-to-fused-f32-224", sourceBlockBytes: 176, outputBlockBytes: 224 },
      { ggmlType: GgmlType.Q6_K, transform: "q6-k-210-to-fused-f32-256", sourceBlockBytes: 210, outputBlockBytes: 256 },
    ],
  );

  const identity: ImmutableArtifactIdentity = {
    repository: "Qwen/Qwen3.5-4B",
    revision: REVISION,
    file: "model.gguf",
    size: "4096",
    sha256: SHA_A,
  };
  const manifest = createManifestFromPlan(plan, {
    packageKind: "language",
    source: identity,
    runtimeAbi: "qwen35-webgpu-v2",
    tokenizer: { ...identity, file: "tokenizer.json" },
    shards: [{ url: "shards/model-00000.bin", sha256: SHA_A }],
  });
  assert.deepEqual(
    manifest.tensorLayout.map(({ storageType, quantization }) => ({
      storageType,
      quantization: quantization ?? null,
    })),
    [
      { storageType: "f32", quantization: null },
      { storageType: "q8-0-36", quantization: { blockElements: 32, blockBytes: 36 } },
      { storageType: "q3-k-fused-f32-192", quantization: { blockElements: 256, blockBytes: 192 } },
      { storageType: "q4-k-fused-f32-192", quantization: { blockElements: 256, blockBytes: 192 } },
      { storageType: "q5-k-fused-f32-224", quantization: { blockElements: 256, blockBytes: 224 } },
      { storageType: "q6-k-fused-f32-256", quantization: { blockElements: 256, blockBytes: 256 } },
    ],
  );
});

test("executes padded Q8_0 and browser Q6_K conversion", async () => {
  const tensors: ParsedGguf["tensors"] = [
    { name: "a.q8", dimensions: [32n], type: GgmlType.Q8_0, offset: 0n },
    { name: "b.q6", dimensions: [256n], type: GgmlType.Q6_K, offset: 64n },
  ];
  const parsed = { ...fixtureGguf(tensors), dataOffset: 0n };
  const plan = planConversion(parsed, {
    maxShardBytes: 512n,
    tensorAlignment: 4,
  });
  const source = new Uint8Array(64 + NATIVE_Q6_K_BLOCK_BYTES);
  source.set(
    Uint8Array.from({ length: NATIVE_Q8_0_BLOCK_BYTES }, (_, index) => index),
    0,
  );
  source.set(
    Uint8Array.from({ length: NATIVE_Q6_K_BLOCK_BYTES }, (_, index) => 255 - index),
    64,
  );
  const output = new Uint8Array(Number(plan.shards[0]!.length));
  await executeConversionPlan(
    plan,
    memoryReader(source),
    [{ async write(offset, bytes) { output.set(bytes, Number(offset)); } }],
  );
  const [q8, q6] = plan.segments;
  const q8Bytes = output.subarray(Number(q8!.shardOffset), Number(q8!.shardOffset + q8!.outputLength));
  assert.deepEqual(q8Bytes.subarray(0, 2), source.subarray(0, 2));
  assert.deepEqual(q8Bytes.subarray(2, 4), Uint8Array.of(0, 0));
  assert.deepEqual(q8Bytes.subarray(4), source.subarray(2, 34));
  const q6Bytes = output.subarray(Number(q6!.shardOffset), Number(q6!.shardOffset + q6!.outputLength));
  assert.deepEqual(q6Bytes, repackNativeQ6KBrowser(source.subarray(64, 274)));
});

test("matches only complete MTP and nextn name segments", () => {
  const parsed = fixtureGguf([
    {
      name: "attempt.weight",
      dimensions: [16n],
      type: GgmlType.F32,
      offset: 0n,
    },
    {
      name: "model.nextness.weight",
      dimensions: [16n],
      type: GgmlType.F32,
      offset: 64n,
    },
    {
      name: "mtp.output.weight",
      dimensions: [16n],
      type: GgmlType.F32,
      offset: 128n,
    },
    {
      name: "model.nextn.output.weight",
      dimensions: [16n],
      type: GgmlType.F32,
      offset: 192n,
    },
  ]);

  const plan = planConversion(parsed, {
    maxShardBytes: 128n,
    tensorAlignment: 16,
  });
  assert.deepEqual(
    plan.excludedTensors.map(({ name }) => name),
    ["model.nextn.output.weight", "mtp.output.weight"],
  );
  assert.deepEqual(
    plan.segments.map(({ tensorName }) => tensorName),
    ["attempt.weight", "model.nextness.weight"],
  );
});

test("excludes the pinned block 32 MTP tensors without matching similar names", () => {
  const parsed = fixtureGguf([
    { name: "blk.31.attn_q.weight", dimensions: [256n], type: GgmlType.Q3_K, offset: 0n },
    { name: "blk.32.attn_q.weight", dimensions: [256n], type: GgmlType.Q3_K, offset: 128n },
    { name: "blk.32.ffn_up.weight", dimensions: [256n], type: GgmlType.Q3_K, offset: 256n },
    { name: "blk.32.post_attention_layernorm.weight", dimensions: [16n], type: GgmlType.F32, offset: 384n },
    { name: "blk.320.attn_q.weight", dimensions: [256n], type: GgmlType.Q3_K, offset: 448n },
    { name: "xblk.32.attn_q.weight", dimensions: [256n], type: GgmlType.Q3_K, offset: 576n },
    { name: "blk.32ish.weight", dimensions: [16n], type: GgmlType.F32, offset: 704n },
    { name: "attempt.weight", dimensions: [16n], type: GgmlType.F32, offset: 768n },
  ]);

  const plan = planConversion(parsed, {
    maxShardBytes: 4096n,
    tensorAlignment: 16,
  });

  assert.deepEqual(plan.excludedTensors, [
    { name: "blk.32.attn_q.weight", reason: MTP_EXCLUSION_REASON },
    { name: "blk.32.ffn_up.weight", reason: MTP_EXCLUSION_REASON },
    {
      name: "blk.32.post_attention_layernorm.weight",
      reason: MTP_EXCLUSION_REASON,
    },
  ]);
  assert.deepEqual(
    plan.segments.map((segment) => segment.tensorName),
    [
      "attempt.weight",
      "blk.31.attn_q.weight",
      "blk.320.attn_q.weight",
      "blk.32ish.weight",
      "xblk.32.attn_q.weight",
    ],
  );
});

test("packs aligned shards without splitting Q3_K blocks", () => {
  const plan = planConversion(fixtureGguf(), {
    maxShardBytes: 384n,
    tensorAlignment: 16,
  });

  assert.deepEqual(
    plan.shards.map((shard) => shard.length),
    [384n, 64n],
  );
  assert.deepEqual(
    plan.segments.map((segment) => ({
      tensor: segment.tensorName,
      shard: segment.shard,
      shardOffset: segment.shardOffset,
      sourceLength: segment.sourceLength,
      outputLength: segment.outputLength,
      blocks: segment.blockCount,
    })),
    [
      {
        tensor: "blk.0.attn_q.weight",
        shard: 0,
        shardOffset: 0n,
        sourceLength: 220n,
        outputLength: 384n,
        blocks: 2n,
      },
      {
        tensor: "output_norm.weight",
        shard: 1,
        shardOffset: 0n,
        sourceLength: 64n,
        outputLength: 64n,
        blocks: 16n,
      },
    ],
  );
});

test("segments every language layout only at complete contiguous row boundaries", () => {
  for (const policy of WEBGPU_LANGUAGE_TENSOR_LAYOUTS) {
    const rowBlocks = 2;
    const rows = 5;
    const rowBytes = BigInt(policy.outputBlockBytes * rowBlocks);
    const plan = planConversion(
      fixtureGguf([
        {
          name: `matrix.${policy.storageType}`,
          dimensions: [BigInt(policy.blockElements * rowBlocks), BigInt(rows)],
          type: policy.ggmlType,
          offset: 0n,
        },
      ]),
      {
        // The extra bytes deliberately do not fit another complete row.
        maxShardBytes:
          rowBytes * 2n + BigInt(policy.outputBlockBytes) + 3n,
        tensorAlignment: 16,
      },
    );

    assert.equal(plan.segments.length, 3, policy.storageType);
    assert.ok(
      plan.segments.every(
        (segment) =>
          segment.outputLength % rowBytes === 0n &&
          segment.tensorOffset % rowBytes === 0n &&
          segment.blockCount % BigInt(rowBlocks) === 0n,
      ),
      policy.storageType,
    );
  }
});

test("keeps full-vocabulary-style matrix segments on complete rows", () => {
  const columns = 2560n;
  const rows = 248_320n;
  const rowBytes = (columns / 256n) * 192n;
  const plan = planConversion(
    fixtureGguf([
      {
        name: "output.weight",
        dimensions: [columns, rows],
        type: GgmlType.Q3_K,
        offset: 0n,
      },
    ]),
    {
      maxShardBytes: 128n * 1024n * 1024n + 37n,
      tensorAlignment: 256,
    },
  );

  assert.ok(plan.segments.length > 1);
  assert.ok(
    plan.segments.every(
      (segment) =>
        segment.outputLength % rowBytes === 0n &&
        segment.tensorOffset % rowBytes === 0n,
    ),
  );
});

test("rejects a shard limit that cannot hold one complete matrix row", () => {
  assert.throws(
    () =>
      planConversion(
        fixtureGguf([
          {
            name: "matrix.weight",
            dimensions: [512n, 2n],
            type: GgmlType.Q3_K,
            offset: 0n,
          },
        ]),
        { maxShardBytes: 223n, tensorAlignment: 16 },
      ),
    /cannot hold one complete.*row/i,
  );
});

test("rejects plan collections before appending past shared manifest bounds", () => {
  const tensor = (
    name: string,
    dimensions: readonly bigint[] = [1n],
  ): ParsedGguf["tensors"][number] => ({
    name,
    dimensions,
    type: GgmlType.F32,
    offset: 0n,
  });

  assert.throws(
    () =>
      planConversion(
        fixtureGguf([
          tensor("matrix.shared-limit.weight", [1n, BigInt(MAX_SHARDS + 1)]),
        ]),
        {
          maxShardBytes: 4n,
          tensorAlignment: 4,
        },
      ),
    /shard count.*bound/i,
  );

  assert.throws(
    () =>
      planConversion(fixtureGguf([tensor("matrix.weight", [1n, 3n])]), {
        maxShardBytes: 4n,
        tensorAlignment: 4,
        planningLimits: {
          maxShards: 2,
          maxTensorSegments: MAX_TENSOR_SEGMENTS,
          maxExcludedTensors: MAX_EXCLUDED_TENSORS,
        },
      }),
    /shard count.*bound/i,
  );

  assert.throws(
    () =>
      planConversion(
        fixtureGguf([
          tensor("a.weight"),
          tensor("b.weight"),
          tensor("c.weight"),
        ]),
        {
          maxShardBytes: 64n,
          tensorAlignment: 4,
          planningLimits: {
            maxShards: MAX_SHARDS,
            maxTensorSegments: 2,
            maxExcludedTensors: MAX_EXCLUDED_TENSORS,
          },
        },
      ),
    /tensor segment count.*bound/i,
  );

  assert.throws(
    () =>
      planConversion(
        fixtureGguf([
          tensor("mtp.a.weight"),
          tensor("mtp.b.weight"),
          tensor("mtp.c.weight"),
        ]),
        {
          maxShardBytes: 64n,
          tensorAlignment: 4,
          planningLimits: {
            maxShards: MAX_SHARDS,
            maxTensorSegments: MAX_TENSOR_SEGMENTS,
            maxExcludedTensors: 2,
          },
        },
      ),
    /excluded tensor count.*bound/i,
  );

  assert.throws(
    () =>
      planConversion(fixtureGguf([tensor("a.weight")]), {
        maxShardBytes: 64n,
        tensorAlignment: 4,
        planningLimits: {
          maxShards: MAX_SHARDS + 1,
          maxTensorSegments: MAX_TENSOR_SEGMENTS,
          maxExcludedTensors: MAX_EXCLUDED_TENSORS,
        },
      }),
    /planning limit.*manifest bound/i,
  );
});

test("planning is deterministic regardless of tensor directory order", () => {
  const forward = planConversion(fixtureGguf(), {
    maxShardBytes: 384n,
    tensorAlignment: 16,
  });
  const reversed = planConversion(fixtureGguf([...mixedTensors()].reverse()), {
    maxShardBytes: 384n,
    tensorAlignment: 16,
  });

  assert.deepEqual(reversed, forward);
});

test("streams Q3_K repacking through bounded reader and writer abstractions", async () => {
  const parsed = fixtureGguf([mixedTensors()[0]!]);
  const plan = planConversion(parsed, {
    maxShardBytes: 384n,
    tensorAlignment: 16,
  });
  const source = new Uint8Array(1024 + NATIVE_Q3_K_BLOCK_BYTES * 2);
  for (let index = 1024; index < source.length; index += 1) {
    source[index] = index & 0xff;
  }
  const reader = memoryReader(source);
  const output = new Uint8Array(384);
  const writer: RandomAccessWriter = {
    async write(offset, bytes) {
      output.set(bytes, Number(offset));
    },
  };

  await executeConversionPlan(plan, reader, [writer], {
    maxBlocksPerRead: 1,
  });

  assert.equal(reader.reads.length, 2);
  assert.ok(
    reader.reads.every(({ length }) => length === NATIVE_Q3_K_BLOCK_BYTES),
  );
  assert.deepEqual(
    output,
    repackNativeQ3KFusedBrowser(
      source.subarray(1024, 1024 + NATIVE_Q3_K_BLOCK_BYTES * 2),
    ),
  );
});

test("emits a valid deterministic manifest from planned shards", () => {
  const plan = planConversion(fixtureGguf(), {
    maxShardBytes: 384n,
    tensorAlignment: 16,
  });
  const source: ImmutableArtifactIdentity = {
    repository: "example/model",
    revision: REVISION,
    file: "model.gguf",
    size: "4096",
    sha256: SHA_A,
  };
  const manifest = createManifestFromPlan(plan, {
    packageKind: "language",
    source,
    runtimeAbi: "qwen35-webgpu-v2",
    tokenizer: { ...source, file: "tokenizer.json" },
    shards: [
      { url: "shards/model-00000.bin", sha256: SHA_A },
      { url: "shards/model-00001.bin", sha256: SHA_A },
    ],
  });

  const first = stringifyManifest(manifest);
  const second = stringifyManifest(
    createManifestFromPlan(plan, {
      packageKind: "language",
      source,
      runtimeAbi: "qwen35-webgpu-v2",
      tokenizer: { ...source, file: "tokenizer.json" },
      shards: [
        { url: "shards/model-00000.bin", sha256: SHA_A },
        { url: "shards/model-00001.bin", sha256: SHA_A },
      ],
    }),
  );

  assert.equal(first, second);
  assert.equal(manifest.tensorLayout[0]!.storageType, "q3-k-fused-f32-192");
  assert.equal(manifest.shards[0]!.offset, "0");
});

test("rejects unsupported tensor types and shards that cannot hold one row", () => {
  const unsupported = fixtureGguf([
    {
      name: "bad.weight",
      dimensions: [32n],
      type: 33 as GgmlType,
      offset: 0n,
    },
  ]);
  assert.throws(
    () =>
      planConversion(unsupported, {
        maxShardBytes: 224n,
        tensorAlignment: 16,
      }),
    /unsupported.*type/i,
  );

  assert.throws(
    () =>
      planConversion(fixtureGguf([mixedTensors()[0]!]), {
        maxShardBytes: 111n,
        tensorAlignment: 16,
      }),
    /cannot hold.*row/i,
  );
});

test("rejects an oversized tensor alignment before planning padding", () => {
  assert.throws(
    () =>
      planConversion(fixtureGguf([mixedTensors()[1]!]), {
        maxShardBytes: 2n ** 40n,
        tensorAlignment: 2 ** 30,
      }),
    /alignment.*bound/i,
  );
});

test("starts a new shard when alignment crosses a non-aligned shard limit", () => {
  const script = `
    import { planConversion } from "./src/converter.js";
    import { GgmlType } from "./src/gguf.js";
    const plan = planConversion({
      version: 3,
      metadata: {},
      alignment: 8,
      dataOffset: 0n,
      tensors: [
        { name: "a.weight", dimensions: [113n], type: GgmlType.I8, offset: 0n },
        { name: "b.weight", dimensions: [1n], type: GgmlType.F32, offset: 120n },
      ],
    }, { maxShardBytes: 113n, tensorAlignment: 16 });
    if (plan.shards.length !== 2 || plan.segments.some((segment) => segment.blockCount <= 0n)) {
      process.exit(2);
    }
  `;
  const result = spawnSync(
    process.execPath,
    ["--import", "tsx", "--input-type=module", "--eval", script],
    { cwd: process.cwd(), timeout: 500 },
  );

  assert.equal(
    result.error,
    undefined,
    `planner did not terminate: ${result.error?.message}`,
  );
  assert.equal(result.status, 0, result.stderr.toString());
});

test("rejects quantized tensors whose contiguous row is a partial block", () => {
  const malformed = fixtureGguf([
    {
      name: "blk.0.attn_q.weight",
      dimensions: [1n, 256n],
      type: GgmlType.Q3_K,
      offset: 0n,
    },
  ]);

  assert.throws(
    () =>
      planConversion(malformed, {
        maxShardBytes: 224n,
        tensorAlignment: 16,
      }),
    /contiguous row dimension.*complete.*block/i,
  );
});

test("rejects huge maxBlocksPerRead before calling the source reader", async () => {
  const plan = planConversion(fixtureGguf([mixedTensors()[0]!]), {
    maxShardBytes: 384n,
    tensorAlignment: 16,
  });
  let reads = 0;

  await assert.rejects(
    executeConversionPlan(
      plan,
      {
        size: 2_000n,
        async read() {
          reads += 1;
          return new Uint8Array();
        },
      },
      [{ async write() {} }],
      { maxBlocksPerRead: Number.MAX_SAFE_INTEGER },
    ),
    /maxBlocksPerRead.*streaming read ceiling/i,
  );
  assert.equal(reads, 0);
});

test("accepts a chunk exactly at the streaming read ceiling", async () => {
  const blockCount = BigInt(MAX_STREAM_READ_BYTES);
  const plan: ConversionPlan = {
    inventory: [],
    excludedTensors: [],
    shards: [{ index: 0, length: blockCount }],
    segments: [
      {
        tensorName: "fixture.bytes",
        dimensions: [blockCount],
        ggmlType: GgmlType.I8,
        transform: "copy",
        shard: 0,
        shardOffset: 0n,
        tensorOffset: 0n,
        sourceOffset: 0n,
        sourceLength: blockCount,
        outputLength: blockCount,
        blockCount,
        sourceBlockBytes: 1,
        outputBlockBytes: 1,
      },
    ],
  };
  let requested = 0;

  await executeConversionPlan(
    plan,
    {
      size: blockCount,
      async read(_offset, length) {
        requested = length;
        return new Uint8Array(length);
      },
    },
    [{ async write() {} }],
    { maxBlocksPerRead: MAX_STREAM_READ_BYTES },
  );

  assert.equal(requested, MAX_STREAM_READ_BYTES);
});
