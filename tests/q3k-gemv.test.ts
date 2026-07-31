import assert from "node:assert/strict";
import test from "node:test";

import {
  NATIVE_Q3_K_BLOCK_BYTES,
  packWebGpuQ3KBlock,
  unpackNativeQ3KBlock,
} from "../src/q3k.js";
import {
  Q3K_GEMV_ABI,
  Q3K_GEMV_WGSL,
  planQ3KGemvDispatch,
  q3kGemvCpu,
} from "../src/q3k-gemv.js";

function uniformBlock(scaleByte = 0x11): Uint8Array {
  const native = new Uint8Array(NATIVE_Q3_K_BLOCK_BYTES);
  native.fill(0xff, 0, 32);
  native.fill(0xe4, 32, 96);
  native.fill(scaleByte, 96, 104);
  native.fill(0xaa, 104, 108);
  native[108] = 0;
  native[109] = 0x3c;
  return packWebGpuQ3KBlock(unpackNativeQ3KBlock(native));
}

test("executes deterministic packed matrix rows through the CPU reference", () => {
  const packed = new Uint8Array(112 * 2);
  packed.set(uniformBlock(), 0);
  packed.set(uniformBlock(0x22), 112);
  const activation = Float32Array.from(
    { length: 256 },
    (_, index) => (index % 7) - 3,
  );
  const decodedPattern = [0, 1, 2, 3].flatMap((value) =>
    Array(32).fill(value),
  );
  const oneRow = [...decodedPattern, ...decodedPattern].reduce(
    (sum, weight, index) => sum + weight * activation[index]!,
    0,
  );

  assert.deepEqual(
    q3kGemvCpu(packed, activation, { rows: 2, columns: 256 }),
    Float32Array.of(oneRow, oneRow * 2),
  );
});

test("plans one correctness invocation per row with aligned u32 offsets", () => {
  assert.deepEqual(
    planQ3KGemvDispatch({
      rows: 7,
      columns: 512,
      packedByteOffset: 224,
    }),
    {
      workgroups: { x: 7, y: 1, z: 1 },
      uniforms: {
        rows: 7,
        columns: 512,
        blocksPerRow: 2,
        weightWordOffset: 56,
      },
    },
  );
});

test("uses a two-dimensional dispatch for the full vocabulary row count", () => {
  assert.deepEqual(
    planQ3KGemvDispatch({
      rows: 248_320,
      columns: 2560,
    }).workgroups,
    { x: 65_535, y: 4, z: 1 },
  );
  assert.match(Q3K_GEMV_WGSL, /@builtin\(num_workgroups\)/);
});

test("rejects invalid rows, block shapes, offsets, and packed extents", () => {
  assert.throws(
    () => planQ3KGemvDispatch({ rows: 0, columns: 256 }),
    /rows/i,
  );
  assert.throws(
    () => planQ3KGemvDispatch({ rows: 1, columns: 255 }),
    /multiple of 256/i,
  );
  assert.throws(
    () =>
      planQ3KGemvDispatch({
        rows: 1,
        columns: 256,
        packedByteOffset: 2,
      }),
    /u32 aligned/i,
  );
  assert.throws(
    () =>
      planQ3KGemvDispatch({
        rows: 1,
        columns: 256,
        packedByteOffset: 4,
      }),
    /112-byte block/i,
  );
  assert.throws(
    () =>
      q3kGemvCpu(new Uint8Array(111), new Float32Array(256), {
        rows: 1,
        columns: 256,
      }),
    /packed weights/i,
  );
  assert.throws(
    () =>
      q3kGemvCpu(uniformBlock(), new Float32Array(255), {
        rows: 1,
        columns: 256,
      }),
    /activation length/i,
  );
});

test("publishes the packed shader ABI and structurally complete WGSL", () => {
  assert.deepEqual(Q3K_GEMV_ABI, {
    wordsPerBlock: 28,
    valuesPerBlock: 256,
    workgroupSize: 1,
    bindings: {
      packedWeights: 0,
      activation: 1,
      output: 2,
      uniforms: 3,
    },
  });
  assert.match(Q3K_GEMV_WGSL, /array<u32>/);
  assert.match(Q3K_GEMV_WGSL, /WORDS_PER_BLOCK\s*:\s*u32\s*=\s*28u/);
  assert.match(Q3K_GEMV_WGSL, /VALUES_PER_BLOCK\s*:\s*u32\s*=\s*256u/);
  assert.match(Q3K_GEMV_WGSL, /unpack2x16float/);
  assert.match(Q3K_GEMV_WGSL, /@workgroup_size\(1\)/);
  assert.doesNotMatch(Q3K_GEMV_WGSL, /array<f16>|mat(2|3|4)x/);
});
