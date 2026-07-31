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
  await assert.rejects(
    parseGguf(memoryReader(excessiveString)),
    /metadata key length/i,
  );
});

test("rejects invalid tensor rank and unaligned offsets", async () => {
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

function nestedArrayFixture(): Uint8Array {
  return new BinaryWriter()
    .bytesFrom([0x47, 0x47, 0x55, 0x46])
    .u32(3)
    .u64(0)
    .u64(1)
    .string("fixture.nested")
    .u32(GgufMetadataType.Array)
    .u32(GgufMetadataType.Array)
    .u64(2)
    .u32(GgufMetadataType.Uint8)
    .u64(2)
    .u8(1)
    .u8(2)
    .u32(GgufMetadataType.Uint8)
    .u64(1)
    .u8(3)
    .pad(32)
    .build();
}

test("parses valid nested GGUF metadata arrays recursively", async () => {
  const parsed = await parseGguf(memoryReader(nestedArrayFixture()));

  assert.deepEqual(parsed.metadata["fixture.nested"], [[1, 2], [3]]);
});

test("bounds nested metadata array depth and aggregate elements", async () => {
  await assert.rejects(
    parseGguf(memoryReader(nestedArrayFixture()), {
      maxMetadataArrayDepth: 1,
    }),
    /array depth/i,
  );
  await assert.rejects(
    parseGguf(memoryReader(nestedArrayFixture()), {
      maxMetadataArrayElements: 4,
    }),
    /aggregate array element/i,
  );
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

test("requires every tensor payload extent to fit inside the reader", async () => {
  const truncatedPayload = validFixture().slice(0, -1);

  await assert.rejects(
    parseGguf(memoryReader(truncatedPayload)),
    /tensor.*extent.*file size/i,
  );
});

test("rejects tensor shapes that do not contain complete quant blocks", async () => {
  const partialBlock = validFixture();
  const marker = new TextEncoder().encode("blk.0.attn_q.weight");
  const markerIndex = partialBlock.findIndex((_, index) =>
    marker.every((byte, offset) => partialBlock[index + offset] === byte),
  );
  const firstDimension = markerIndex + marker.length + 4;
  new DataView(partialBlock.buffer).setBigUint64(firstDimension, 257n, true);

  await assert.rejects(
    parseGguf(memoryReader(partialBlock)),
    /complete.*quantization block/i,
  );
});

function setGeneralAlignment(bytes: Uint8Array, alignment: number): void {
  const marker = new TextEncoder().encode("general.alignment");
  const markerIndex = bytes.findIndex((_, index) =>
    marker.every((byte, offset) => bytes[index + offset] === byte),
  );
  const valueOffset = markerIndex + marker.length + 4;
  new DataView(bytes.buffer).setUint32(valueOffset, alignment, true);
}

test("accepts GGUF alignment that is a bounded multiple of eight", async () => {
  const fixture = new Uint8Array(validFixture().length + 64);
  fixture.set(validFixture());
  setGeneralAlignment(fixture, 24);
  const tensorMarker = new TextEncoder().encode("output_norm.weight");
  const tensorIndex = fixture.findIndex((_, index) =>
    tensorMarker.every((byte, offset) => fixture[index + offset] === byte),
  );
  const offsetField = tensorIndex + tensorMarker.length + 4 + 8 + 4;
  new DataView(fixture.buffer).setBigUint64(offsetField, 456n, true);

  const parsed = await parseGguf(memoryReader(fixture));

  assert.equal(parsed.alignment, 24);
  assert.equal(parsed.dataOffset % 24n, 0n);
});

test("rejects GGUF alignment smaller than eight", async () => {
  const fixture = validFixture();
  setGeneralAlignment(fixture, 4);

  await assert.rejects(
    parseGguf(memoryReader(fixture)),
    /alignment.*multiple of 8/i,
  );
});

function metadataKeyFixture(key: string): Uint8Array {
  return new BinaryWriter()
    .bytesFrom([0x47, 0x47, 0x55, 0x46])
    .u32(3)
    .u64(0)
    .u64(1)
    .string(key)
    .u32(GgufMetadataType.Uint8)
    .u8(1)
    .pad(32)
    .build();
}

function tensorNameFixture(name: string): Uint8Array {
  return new BinaryWriter()
    .bytesFrom([0x47, 0x47, 0x55, 0x46])
    .u32(3)
    .u64(1)
    .u64(0)
    .string(name)
    .u32(1)
    .u64(1)
    .u32(GgmlType.F32)
    .u64(0)
    .pad(32)
    .bytesFrom(new Uint8Array(4))
    .build();
}

test("requires ASCII hierarchical GGUF metadata keys", async () => {
  await assert.rejects(
    parseGguf(memoryReader(metadataKeyFixture("fixture..invalid"))),
    /metadata key.*hierarchical ASCII/i,
  );
  await assert.rejects(
    parseGguf(memoryReader(metadataKeyFixture("fixture.é"))),
    /metadata key.*hierarchical ASCII/i,
  );
});

test("bounds GGUF metadata keys at 65535 encoded bytes", async () => {
  await assert.rejects(
    parseGguf(memoryReader(metadataKeyFixture("a".repeat(65_536)))),
    /metadata key length.*65535/i,
  );
});

test("bounds GGUF tensor names at 64 encoded bytes", async () => {
  await assert.rejects(
    parseGguf(memoryReader(tensorNameFixture("t".repeat(65)))),
    /tensor name length.*64/i,
  );
});

test("decodes signed bytes from readers that return subarray views", async () => {
  const fixture = new BinaryWriter()
    .bytesFrom([0x47, 0x47, 0x55, 0x46])
    .u32(3)
    .u64(0)
    .u64(1)
    .string("fixture.signed")
    .u32(GgufMetadataType.Int8)
    .u8(0xf9)
    .pad(32)
    .build();
  const prefixLength = 8;
  const backing = new Uint8Array(prefixLength + fixture.length + 8);
  backing.fill(0x55);
  backing.set(fixture, prefixLength);
  const reader: RandomAccessReader = {
    size: BigInt(fixture.length),
    async read(offset, length) {
      const start = prefixLength + Number(offset);
      return backing.subarray(start, start + length);
    },
  };

  const parsed = await parseGguf(reader);

  assert.equal(parsed.metadata["fixture.signed"], -7);
});

test("rejects quantized tensors whose contiguous row is a partial block", async () => {
  const malformed = validFixture();
  const marker = new TextEncoder().encode("blk.0.attn_q.weight");
  const markerIndex = malformed.findIndex((_, index) =>
    marker.every((byte, offset) => malformed[index + offset] === byte),
  );
  const firstDimension = markerIndex + marker.length + 4;
  const dimensions = new DataView(malformed.buffer);
  dimensions.setBigUint64(firstDimension, 1n, true);
  dimensions.setBigUint64(firstDimension + 8, 256n, true);

  await assert.rejects(
    parseGguf(memoryReader(malformed)),
    /contiguous row dimension.*complete.*block/i,
  );
});
