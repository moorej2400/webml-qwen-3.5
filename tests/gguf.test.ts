import assert from "node:assert/strict";
import test from "node:test";

import {
  GgmlType,
  GgufMetadataType,
  parseGguf,
  type RandomAccessReader,
} from "../src/gguf.js";
import { BinaryWriter, memoryReader } from "./fixture-utils.js";

function validFixture(): Uint8Array {
  const writer = new BinaryWriter()
    .bytesFrom([0x47, 0x47, 0x55, 0x46])
    .u32(3)
    .u64(2)
    .u64(4)
    .string("general.alignment")
    .u32(GgufMetadataType.Uint32)
    .u32(32)
    .string("general.architecture")
    .u32(GgufMetadataType.String)
    .string("qwen35")
    .string("fixture.enabled")
    .u32(GgufMetadataType.Bool)
    .u8(1)
    .string("fixture.values")
    .u32(GgufMetadataType.Array)
    .u32(GgufMetadataType.Uint16)
    .u64(3)
    .u16(3)
    .u16(5)
    .u16(8)
    .string("blk.0.attn_q.weight")
    .u32(2)
    .u64(256)
    .u64(4)
    .u32(GgmlType.Q3_K)
    .u64(0)
    .string("output_norm.weight")
    .u32(1)
    .u64(8)
    .u32(GgmlType.F32)
    .u64(448);

  writer.pad(32).bytesFrom(new Uint8Array(480));
  return writer.build();
}

test("parses GGUF v3 metadata and tensor directory through random access", async () => {
  const reader = memoryReader(validFixture());

  const parsed = await parseGguf(reader);

  assert.equal(parsed.version, 3);
  assert.equal(parsed.alignment, 32);
  assert.equal(parsed.metadata["general.architecture"], "qwen35");
  assert.deepEqual(parsed.metadata["fixture.values"], [3, 5, 8]);
  assert.deepEqual(parsed.tensors[0], {
    name: "blk.0.attn_q.weight",
    dimensions: [256n, 4n],
    type: GgmlType.Q3_K,
    offset: 0n,
  });
  assert.equal(parsed.dataOffset % 32n, 0n);
  assert.ok(reader.reads.length > 10);
  assert.ok(reader.reads.every(({ length }) => length < validFixture().length));
});

test("rejects invalid magic and unsupported versions", async () => {
  const badMagic = validFixture();
  badMagic[0] = 0;
  await assert.rejects(parseGguf(memoryReader(badMagic)), /magic/i);

  const badVersion = validFixture();
  new DataView(badVersion.buffer).setUint32(4, 2, true);
  await assert.rejects(parseGguf(memoryReader(badVersion)), /version/i);
});

test("fails closed on truncated input", async () => {
  const bytes = validFixture().slice(0, 70);
  await assert.rejects(parseGguf(memoryReader(bytes)), /truncated/i);
});

test("enforces count and string bounds before allocation", async () => {
  const excessiveCount = new BinaryWriter()
    .bytesFrom([0x47, 0x47, 0x55, 0x46])
    .u32(3)
    .u64(1_000_001)
    .u64(0)
    .build();
  await assert.rejects(parseGguf(memoryReader(excessiveCount)), /tensor count/i);

  const excessiveString = new BinaryWriter()
    .bytesFrom([0x47, 0x47, 0x55, 0x46])
    .u32(3)
    .u64(0)
    .u64(1)
    .u64(16_777_217)
    .build();
  await assert.rejects(parseGguf(memoryReader(excessiveString)), /string length/i);
});

test("rejects unsupported metadata arrays, invalid tensor rank, and unaligned offsets", async () => {
  const nestedArray = new BinaryWriter()
    .bytesFrom([0x47, 0x47, 0x55, 0x46])
    .u32(3)
    .u64(0)
    .u64(1)
    .string("bad")
    .u32(GgufMetadataType.Array)
    .u32(GgufMetadataType.Array)
    .u64(0)
    .build();
  await assert.rejects(parseGguf(memoryReader(nestedArray)), /array element type/i);

  const rankZero = new BinaryWriter()
    .bytesFrom([0x47, 0x47, 0x55, 0x46])
    .u32(3)
    .u64(1)
    .u64(0)
    .string("bad")
    .u32(0)
    .u32(GgmlType.F32)
    .u64(0)
    .build();
  await assert.rejects(parseGguf(memoryReader(rankZero)), /rank/i);

  const unaligned = validFixture();
  const marker = new TextEncoder().encode("output_norm.weight");
  const markerIndex = unaligned.findIndex((_, index) =>
    marker.every((byte, offset) => unaligned[index + offset] === byte),
  );
  const offsetField = markerIndex + marker.length + 4 + 8 + 4;
  new DataView(unaligned.buffer).setBigUint64(offsetField, 1n, true);
  await assert.rejects(parseGguf(memoryReader(unaligned)), /alignment/i);
});

test("rejects readers that return fewer bytes than requested", async () => {
  const bytes = validFixture();
  const reader: RandomAccessReader = {
    size: BigInt(bytes.length),
    async read(offset, length) {
      return bytes.slice(Number(offset), Number(offset) + Math.max(0, length - 1));
    },
  };

  await assert.rejects(parseGguf(reader), /truncated/i);
});
