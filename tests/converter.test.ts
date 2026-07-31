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
import { stringifyManifest, type ImmutableArtifactIdentity } from "../src/manifest.js";
import { NATIVE_Q3_K_BLOCK_BYTES } from "../src/q3k.js";
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
    maxShardBytes: 224n,
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

test("does not exclude names that only contain the letters mtp", () => {
  const parsed = fixtureGguf([
    {
      name: "attempt.weight",
      dimensions: [16n],
      type: GgmlType.F32,
      offset: 0n,
    },
  ]);

  assert.equal(
    planConversion(parsed, {
      maxShardBytes: 128n,
      tensorAlignment: 16,
    }).excludedTensors.length,
    0,
  );
});

test("packs aligned shards without splitting Q3_K blocks", () => {
  const plan = planConversion(fixtureGguf(), {
    maxShardBytes: 224n,
    tensorAlignment: 16,
  });

  assert.deepEqual(
    plan.shards.map((shard) => shard.length),
    [224n, 64n],
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
        outputLength: 224n,
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

test("planning is deterministic regardless of tensor directory order", () => {
  const forward = planConversion(fixtureGguf(), {
    maxShardBytes: 224n,
    tensorAlignment: 16,
  });
  const reversed = planConversion(fixtureGguf([...mixedTensors()].reverse()), {
    maxShardBytes: 224n,
    tensorAlignment: 16,
  });

  assert.deepEqual(reversed, forward);
});

test("streams Q3_K repacking through bounded reader and writer abstractions", async () => {
  const parsed = fixtureGguf([mixedTensors()[0]!]);
  const plan = planConversion(parsed, {
    maxShardBytes: 224n,
    tensorAlignment: 16,
  });
  const source = new Uint8Array(1024 + NATIVE_Q3_K_BLOCK_BYTES * 2);
  for (let index = 1024; index < source.length; index += 1) {
    source[index] = index & 0xff;
  }
  const reader = memoryReader(source);
  const output = new Uint8Array(224);
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
  assert.equal(output[2], 0);
  assert.equal(output[3], 0);
  assert.deepEqual(output.slice(80, 112), source.slice(1024, 1056));
});

test("emits a valid deterministic manifest from planned shards", () => {
  const plan = planConversion(fixtureGguf(), {
    maxShardBytes: 224n,
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
    runtimeAbi: "qwen35-webgpu-v1",
    tokenizer: { ...source, file: "tokenizer.json" },
    shards: [
      { url: "shards/model-00000.bin", sha256: SHA_A },
      { url: "shards/model-00001.bin", sha256: "b".repeat(64) },
    ],
  });

  const first = stringifyManifest(manifest);
  const second = stringifyManifest(
    createManifestFromPlan(plan, {
      packageKind: "language",
      source,
      runtimeAbi: "qwen35-webgpu-v1",
      tokenizer: { ...source, file: "tokenizer.json" },
      shards: [
        { url: "shards/model-00000.bin", sha256: SHA_A },
        { url: "shards/model-00001.bin", sha256: "b".repeat(64) },
      ],
    }),
  );

  assert.equal(first, second);
  assert.equal(manifest.tensorLayout[0]!.storageType, "q3-k-112");
  assert.equal(manifest.shards[1]!.offset, "224");
});

test("rejects unsupported tensor types and shard sizes that cannot hold one block", () => {
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
    /cannot hold.*block/i,
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
    maxShardBytes: 224n,
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
