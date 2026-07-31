import assert from "node:assert/strict";
import test from "node:test";

import {
  NATIVE_Q3_K_BLOCK_BYTES,
  Q3_K_ELEMENTS_PER_BLOCK,
  WEBGPU_Q3_K_BLOCK_BYTES,
  dequantizeQ3KBlock,
  packNativeQ3KBlock,
  packWebGpuQ3KBlock,
  repackNativeQ3K,
  repackWebGpuQ3K,
  unpackNativeQ3KBlock,
  unpackWebGpuQ3KBlock,
} from "../src/q3k.js";

function deterministicNativeBlock(seed: number): Uint8Array {
  const block = Uint8Array.from(
    { length: NATIVE_Q3_K_BLOCK_BYTES },
    (_, index) => (index * 73 + seed) & 0xff,
  );
  block[108] = 0;
  block[109] = 0x3c;
  return block;
}

test("unpacks and packs the exact native 110-byte field order", () => {
  const native = deterministicNativeBlock(19);
  const block = unpackNativeQ3KBlock(native);

  assert.deepEqual(block.hmask, native.slice(0, 32));
  assert.deepEqual(block.qs, native.slice(32, 96));
  assert.deepEqual(block.scales, native.slice(96, 108));
  assert.equal(block.deltaBits, 0x3c00);
  assert.deepEqual(packNativeQ3KBlock(block), native);
});

test("uses the aligned 112-byte WebGPU field layout", () => {
  const block = unpackNativeQ3KBlock(deterministicNativeBlock(41));

  const packed = packWebGpuQ3KBlock(block);

  assert.equal(packed.length, WEBGPU_Q3_K_BLOCK_BYTES);
  assert.deepEqual(packed.slice(0, 2), Uint8Array.of(0, 0x3c));
  assert.deepEqual(packed.slice(2, 4), Uint8Array.of(0, 0));
  assert.deepEqual(packed.slice(4, 16), block.scales);
  assert.deepEqual(packed.slice(16, 80), block.qs);
  assert.deepEqual(packed.slice(80, 112), block.hmask);
  assert.deepEqual(unpackWebGpuQ3KBlock(packed), block);
});

test("repacking multiple blocks is byte-exact in both directions", () => {
  const native = new Uint8Array(NATIVE_Q3_K_BLOCK_BYTES * 2);
  native.set(deterministicNativeBlock(7));
  native.set(deterministicNativeBlock(113), NATIVE_Q3_K_BLOCK_BYTES);

  const webGpu = repackNativeQ3K(native);

  assert.equal(webGpu.length, WEBGPU_Q3_K_BLOCK_BYTES * 2);
  assert.deepEqual(repackWebGpuQ3K(webGpu), native);
});

test("rejects partial native and WebGPU superblocks", () => {
  assert.throws(
    () => repackNativeQ3K(new Uint8Array(NATIVE_Q3_K_BLOCK_BYTES + 1)),
    /whole.*110-byte/i,
  );
  assert.throws(
    () => repackWebGpuQ3K(new Uint8Array(WEBGPU_Q3_K_BLOCK_BYTES - 1)),
    /whole.*112-byte/i,
  );
});

test("dequantizes a deterministic Q3_K reference vector", () => {
  const native = new Uint8Array(NATIVE_Q3_K_BLOCK_BYTES);
  native.fill(0xff, 0, 32);
  native.fill(0xe4, 32, 96);
  native.fill(0x11, 96, 104);
  native.fill(0xaa, 104, 108);
  native[108] = 0;
  native[109] = 0x3c;

  const values = dequantizeQ3KBlock(unpackNativeQ3KBlock(native));
  const expected = new Float32Array(Q3_K_ELEMENTS_PER_BLOCK);
  for (let half = 0; half < 2; half += 1) {
    for (let quant = 0; quant < 4; quant += 1) {
      expected.fill(quant, half * 128 + quant * 32, half * 128 + (quant + 1) * 32);
    }
  }

  assert.deepEqual(values, expected);
});

test("dequantization applies the packed high-bit sign plane", () => {
  const native = new Uint8Array(NATIVE_Q3_K_BLOCK_BYTES);
  native.fill(0xe4, 32, 96);
  native.fill(0x11, 96, 104);
  native.fill(0xaa, 104, 108);
  native[108] = 0;
  native[109] = 0x3c;

  const values = dequantizeQ3KBlock(unpackNativeQ3KBlock(native));

  assert.deepEqual(
    Array.from(values.slice(0, 128)),
    [-4, -3, -2, -1].flatMap((value) => Array(32).fill(value)),
  );
});

test("matches an independent decode for all distinct packed scale positions", () => {
  // Encode the 6-bit scale fields directly so this fixture does not share the
  // production unpacker's indexing logic.
  const signedScales = [
    -32, -28, -24, -20, -16, -12, -8, -4,
    1, 5, 9, 13, 17, 21, 25, 31,
  ];
  const packedScales = new Uint8Array(12);
  for (let index = 0; index < signedScales.length; index += 1) {
    const encoded = signedScales[index]! + 32;
    if (index < 8) {
      packedScales[index] = encoded & 0x0f;
    } else {
      packedScales[index - 8] |= (encoded & 0x0f) << 4;
    }
    packedScales[8 + (index % 4)]! |=
      (encoded >>> 4) << (2 * Math.floor(index / 4));
  }

  const native = new Uint8Array(NATIVE_Q3_K_BLOCK_BYTES);
  const hmask = Uint8Array.from(
    { length: 32 },
    (_, index) => (index * 19 + 7) & 0xff,
  );
  const qs = Uint8Array.from(
    { length: 64 },
    (_, index) => (index * 37 + 11) & 0xff,
  );
  native.set(hmask, 0);
  native.set(qs, 32);
  native.set(packedScales, 96);
  native[108] = 0;
  native[109] = 0x3c;

  const expected = new Float32Array(Q3_K_ELEMENTS_PER_BLOCK);
  for (let index = 0; index < expected.length; index += 1) {
    const group = Math.floor(index / 128);
    const withinGroup = index % 128;
    const subgroup = Math.floor(withinGroup / 16);
    const plane = Math.floor(subgroup / 2);
    const half = subgroup % 2;
    const lane = index % 16;
    const qIndex = group * 32 + half * 16 + lane;
    const low = (qs[qIndex]! >>> (plane * 2)) & 0x03;
    const highMask = 1 << (group * 4 + plane);
    const high = (hmask[half * 16 + lane]! & highMask) === 0 ? 4 : 0;
    expected[index] = signedScales[Math.floor(index / 16)]! * (low - high);
  }

  assert.deepEqual(
    dequantizeQ3KBlock(unpackNativeQ3KBlock(native)),
    expected,
  );
});
