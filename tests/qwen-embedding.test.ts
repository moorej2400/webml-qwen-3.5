import assert from "node:assert/strict";
import test from "node:test";

import { GgmlType } from "../src/gguf.js";
import {
  PACKED_EMBEDDING_KERNELS,
  embeddingCpu,
  planPackedEmbeddingRow,
  qwenEmbeddingRegistryDefinitions,
} from "../src/qwen-embedding.js";
import { KernelRegistry } from "../src/kernel-registry.js";
import {
  dequantizeQ6KBlock,
  repackNativeQ6K,
  unpackNativeQ6KBlock,
} from "../src/mixed-quant.js";

function setHalf(bytes: Uint8Array, offset: number, bits = 0x3c00): void {
  new DataView(bytes.buffer, bytes.byteOffset).setUint16(offset, bits, true);
}

test("decodes only the selected packed embedding row", () => {
  const native = Uint8Array.from({ length: 210 }, (_, index) => (index * 31 + 7) & 255);
  setHalf(native, 208);
  const row = repackNativeQ6K(native);
  const packed = new Uint8Array(row.length * 2);
  packed.set(row);
  packed.set(row, row.length);

  assert.deepEqual(
    embeddingCpu("q6-k-212", packed, 1, {
      vocabSize: 2,
      embeddingLength: 256,
    }),
    dequantizeQ6KBlock(unpackNativeQ6KBlock(native)),
  );
});

test("checks token bounds and packed table extent", () => {
  assert.throws(
    () =>
      embeddingCpu("f32", new Uint8Array(16), 2, {
        vocabSize: 2,
        embeddingLength: 2,
      }),
    /token.*bounds/i,
  );
  assert.throws(
    () =>
      embeddingCpu("f32", new Uint8Array(12), 1, {
        vocabSize: 2,
        embeddingLength: 2,
      }),
    /table.*extent/i,
  );
});

test("plans a direct packed row read without expanding the token table", () => {
  assert.deepEqual(
    planPackedEmbeddingRow({
      ggmlType: GgmlType.Q6_K,
      storageType: "q6-k-212",
      tokenId: 17,
      vocabSize: 248_320,
      embeddingLength: 2_560,
      maxComputeWorkgroupsPerDimension: 65_535,
    }),
    {
      storageType: "q6-k-212",
      packedByteOffset: 36_040,
      packedByteLength: 2_120,
      outputElements: 2_560,
      workgroups: { x: 10, y: 1, z: 1 },
    },
  );
});

test("rejects shader u32 overflow and device dispatch overflow", () => {
  assert.throws(
    () =>
      planPackedEmbeddingRow({
        ggmlType: GgmlType.F32,
        storageType: "f32",
        tokenId: 0,
        vocabSize: 1,
        embeddingLength: 2 ** 32,
        maxComputeWorkgroupsPerDimension: 65_535,
      }),
    /embedding length.*u32/i,
  );
  assert.throws(
    () =>
      planPackedEmbeddingRow({
        ggmlType: GgmlType.F32,
        storageType: "f32",
        tokenId: 0,
        vocabSize: 1,
        embeddingLength: 512,
        maxComputeWorkgroupsPerDimension: 1,
      }),
    /workgroups.*device limit/i,
  );
  assert.throws(
    () =>
      planPackedEmbeddingRow({
        ggmlType: GgmlType.F32,
        storageType: "f32",
        tokenId: 0,
        vocabSize: 1,
        embeddingLength: 256,
        maxComputeWorkgroupsPerDimension: 2 ** 32,
      }),
    /device limit.*u32/i,
  );
});

test("defines six direct packed embedding kernels and phase registry entries", () => {
  assert.deepEqual(
    PACKED_EMBEDDING_KERNELS.map((kernel) => kernel.storageType),
    ["f32", "q8-0-36", "q3-k-112", "q4-k-144", "q5-k-176", "q6-k-212"],
  );
  for (const kernel of PACKED_EMBEDDING_KERNELS) {
    assert.match(kernel.source, /array<u32>/);
    assert.match(kernel.source, /row_word_offset/);
    assert.match(kernel.source, /weight_value/);
    assert.match(kernel.source, /fn main/);
    assert.doesNotMatch(kernel.source, /array<f16>|mat(2|3|4)x/);
  }

  const registry = new KernelRegistry();
  const definitions = qwenEmbeddingRegistryDefinitions({
    phase: "prefill",
    profile: "portable-f32",
  });
  definitions.forEach((definition) => registry.register(definition));
  assert.equal(
    registry.select({
      key: {
        operation: "embedding-row",
        layout: "q6-k-212",
        phase: "prefill",
        profile: "portable-f32",
      },
      fallbackProfiles: [],
    }).kernel.id,
    "q6-k-212-embedding-row-prefill-portable-f32",
  );
});
