import assert from "node:assert/strict";
import test from "node:test";

import {
  NATIVE_Q4_K_BLOCK_BYTES,
  NATIVE_Q5_K_BLOCK_BYTES,
  NATIVE_Q6_K_BLOCK_BYTES,
  NATIVE_Q8_0_BLOCK_BYTES,
  WEBGPU_Q4_K_BLOCK_BYTES,
  WEBGPU_Q5_K_BLOCK_BYTES,
  WEBGPU_Q6_K_BLOCK_BYTES,
  WEBGPU_Q8_0_BLOCK_BYTES,
  dequantizeQ4KBlock,
  dequantizeQ5KBlock,
  dequantizeQ6KBlock,
  dequantizeQ8_0Block,
  packWebGpuQ4KBlock,
  packWebGpuQ5KBlock,
  packWebGpuQ6KBlock,
  packWebGpuQ8_0Block,
  repackNativeQ4K,
  repackNativeQ5K,
  repackNativeQ6K,
  repackNativeQ8_0,
  repackWebGpuQ4K,
  repackWebGpuQ5K,
  repackWebGpuQ6K,
  repackWebGpuQ8_0,
  unpackNativeQ4KBlock,
  unpackNativeQ5KBlock,
  unpackNativeQ6KBlock,
  unpackNativeQ8_0Block,
  unpackWebGpuQ4KBlock,
  unpackWebGpuQ5KBlock,
  unpackWebGpuQ6KBlock,
  unpackWebGpuQ8_0Block,
} from "../src/mixed-quant.js";

function half(view: DataView, offset: number, value: number): void {
  const known = new Map([
    [0.5, 0x3800],
    [1, 0x3c00],
    [2, 0x4000],
  ]);
  view.setUint16(offset, known.get(value)!, true);
}

function independentHalf(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 31;
  const fraction = bits & 1023;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) {
    return fraction === 0 ? sign * Infinity : Number.NaN;
  }
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

function scaleMin(scales: Uint8Array, group: number): [number, number] {
  if (group < 4) {
    return [scales[group]! & 63, scales[group + 4]! & 63];
  }
  return [
    (scales[group + 4]! & 15) | ((scales[group - 4]! >>> 6) << 4),
    (scales[group + 4]! >>> 4) | ((scales[group]! >>> 6) << 4),
  ];
}

function independentQ4(native: Uint8Array): Float32Array {
  const view = new DataView(native.buffer, native.byteOffset);
  const d = independentHalf(view.getUint16(0, true));
  const dmin = independentHalf(view.getUint16(2, true));
  const scales = native.subarray(4, 16);
  const qs = native.subarray(16);
  const output = new Float32Array(256);
  for (let chunk = 0; chunk < 4; chunk += 1) {
    const [s0, m0] = scaleMin(scales, chunk * 2);
    const [s1, m1] = scaleMin(scales, chunk * 2 + 1);
    for (let lane = 0; lane < 32; lane += 1) {
      const packed = qs[chunk * 32 + lane]!;
      output[chunk * 64 + lane] = d * s0 * (packed & 15) - dmin * m0;
      output[chunk * 64 + lane + 32] =
        d * s1 * (packed >>> 4) - dmin * m1;
    }
  }
  return output;
}

function independentQ5(native: Uint8Array): Float32Array {
  const delta = independentHalf(
    new DataView(native.buffer, native.byteOffset).getUint16(0, true),
  );
  const qh = native.subarray(16, 48);
  const lowOnly = new Uint8Array(NATIVE_Q4_K_BLOCK_BYTES);
  lowOnly.set(native.subarray(0, 16), 0);
  lowOnly.set(native.subarray(48), 16);
  const output = independentQ4(lowOnly);
  const scales = native.subarray(4, 16);
  for (let chunk = 0; chunk < 4; chunk += 1) {
    const [s0] = scaleMin(scales, chunk * 2);
    const [s1] = scaleMin(scales, chunk * 2 + 1);
    const lowMask = 1 << (chunk * 2);
    const highMask = 1 << (chunk * 2 + 1);
    for (let lane = 0; lane < 32; lane += 1) {
      const high = qh[lane]!;
      if ((high & lowMask) !== 0) {
        output[chunk * 64 + lane]! += delta * 16 * s0;
      }
      if ((high & highMask) !== 0) {
        output[chunk * 64 + lane + 32]! += delta * 16 * s1;
      }
    }
  }
  return output;
}

function independentQ6(native: Uint8Array): Float32Array {
  const delta = independentHalf(
    new DataView(native.buffer, native.byteOffset).getUint16(208, true),
  );
  const ql = native.subarray(0, 128);
  const qh = native.subarray(128, 192);
  const scales = new Int8Array(
    native.buffer,
    native.byteOffset + 192,
    16,
  );
  const output = new Float32Array(256);
  for (let halfIndex = 0; halfIndex < 2; halfIndex += 1) {
    for (let lane = 0; lane < 32; lane += 1) {
      const low0 = ql[halfIndex * 64 + lane]!;
      const low1 = ql[halfIndex * 64 + lane + 32]!;
      const high = qh[halfIndex * 32 + lane]!;
      const base = halfIndex * 128;
      const scaleBase = halfIndex * 8 + Math.floor(lane / 16);
      output[base + lane] =
        delta *
        scales[scaleBase]! *
        (((low0 & 15) | ((high & 3) << 4)) - 32);
      output[base + lane + 32] =
        delta *
        scales[scaleBase + 2]! *
        (((low1 & 15) | (((high >>> 2) & 3) << 4)) - 32);
      output[base + lane + 64] =
        delta *
        scales[scaleBase + 4]! *
        (((low0 >>> 4) | (((high >>> 4) & 3) << 4)) - 32);
      output[base + lane + 96] =
        delta *
        scales[scaleBase + 6]! *
        (((low1 >>> 4) | (((high >>> 6) & 3) << 4)) - 32);
    }
  }
  return output;
}

function deterministicNative(length: number, multiplier: number): Uint8Array {
  return Uint8Array.from(
    { length },
    (_, index) => (index * multiplier + 17) & 255,
  );
}

test("uses exact llama.cpp native field order and WebGPU block bytes", () => {
  const q4 = deterministicNative(NATIVE_Q4_K_BLOCK_BYTES, 29);
  half(new DataView(q4.buffer), 0, 1);
  half(new DataView(q4.buffer), 2, 2);
  const q5 = deterministicNative(NATIVE_Q5_K_BLOCK_BYTES, 31);
  half(new DataView(q5.buffer), 0, 1);
  half(new DataView(q5.buffer), 2, 2);
  const q6 = deterministicNative(NATIVE_Q6_K_BLOCK_BYTES, 37);
  half(new DataView(q6.buffer), 208, 1);
  const q8 = deterministicNative(NATIVE_Q8_0_BLOCK_BYTES, 41);
  half(new DataView(q8.buffer), 0, 0.5);

  assert.equal(packWebGpuQ4KBlock(unpackNativeQ4KBlock(q4)).length, 144);
  assert.deepEqual(packWebGpuQ4KBlock(unpackNativeQ4KBlock(q4)), q4);
  assert.equal(packWebGpuQ5KBlock(unpackNativeQ5KBlock(q5)).length, 176);
  assert.deepEqual(packWebGpuQ5KBlock(unpackNativeQ5KBlock(q5)), q5);

  const q6Packed = packWebGpuQ6KBlock(unpackNativeQ6KBlock(q6));
  assert.equal(q6Packed.length, WEBGPU_Q6_K_BLOCK_BYTES);
  assert.deepEqual(q6Packed.subarray(0, 210), q6);
  assert.deepEqual(q6Packed.subarray(210), Uint8Array.of(0, 0));

  const q8Packed = packWebGpuQ8_0Block(unpackNativeQ8_0Block(q8));
  assert.equal(q8Packed.length, WEBGPU_Q8_0_BLOCK_BYTES);
  assert.deepEqual(q8Packed.subarray(0, 2), q8.subarray(0, 2));
  assert.deepEqual(q8Packed.subarray(2, 4), Uint8Array.of(0, 0));
  assert.deepEqual(q8Packed.subarray(4), q8.subarray(2));

  assert.equal(WEBGPU_Q4_K_BLOCK_BYTES, 144);
  assert.equal(WEBGPU_Q5_K_BLOCK_BYTES, 176);
  assert.equal(WEBGPU_Q6_K_BLOCK_BYTES, 212);
  assert.equal(WEBGPU_Q8_0_BLOCK_BYTES, 36);
});

test("repacking is exact and rejects partial native or WebGPU blocks", () => {
  const cases = [
    [NATIVE_Q4_K_BLOCK_BYTES, WEBGPU_Q4_K_BLOCK_BYTES, repackNativeQ4K, repackWebGpuQ4K],
    [NATIVE_Q5_K_BLOCK_BYTES, WEBGPU_Q5_K_BLOCK_BYTES, repackNativeQ5K, repackWebGpuQ5K],
    [NATIVE_Q6_K_BLOCK_BYTES, WEBGPU_Q6_K_BLOCK_BYTES, repackNativeQ6K, repackWebGpuQ6K],
    [NATIVE_Q8_0_BLOCK_BYTES, WEBGPU_Q8_0_BLOCK_BYTES, repackNativeQ8_0, repackWebGpuQ8_0],
  ] as const;
  for (const [nativeBytes, webGpuBytes, toGpu, toNative] of cases) {
    const native = deterministicNative(nativeBytes * 2, nativeBytes);
    const packed = toGpu(native);
    assert.equal(packed.length, webGpuBytes * 2);
    assert.deepEqual(toNative(packed), native);
    assert.throws(() => toGpu(new Uint8Array(nativeBytes + 1)), /whole.*blocks/i);
    assert.throws(() => toNative(new Uint8Array(webGpuBytes - 1)), /whole.*blocks/i);
  }
});

test("repacking accepts bounded Node Buffer views with larger backing stores", () => {
  const cases = [
    [NATIVE_Q6_K_BLOCK_BYTES, repackNativeQ6K, repackWebGpuQ6K],
    [NATIVE_Q8_0_BLOCK_BYTES, repackNativeQ8_0, repackWebGpuQ8_0],
  ] as const;
  for (const [nativeBytes, toGpu, toNative] of cases) {
    const expected = deterministicNative(nativeBytes * 2, nativeBytes + 7);
    const allocation = Buffer.alloc(expected.byteLength + 64, 0xa5);
    expected.forEach((value, index) => {
      allocation[index + 31] = value;
    });
    const boundedView = allocation.subarray(31, 31 + expected.byteLength);
    assert.deepEqual(toNative(toGpu(boundedView)), expected);
  }
});

test("matches independent Q4_K and Q5_K vectors with scale, min, and high bits", () => {
  const q4 = deterministicNative(NATIVE_Q4_K_BLOCK_BYTES, 43);
  half(new DataView(q4.buffer), 0, 1);
  half(new DataView(q4.buffer), 2, 2);
  assert.deepEqual(dequantizeQ4KBlock(unpackNativeQ4KBlock(q4)), independentQ4(q4));

  const q5 = deterministicNative(NATIVE_Q5_K_BLOCK_BYTES, 47);
  half(new DataView(q5.buffer), 0, 0.5);
  half(new DataView(q5.buffer), 2, 2);
  assert.deepEqual(dequantizeQ5KBlock(unpackNativeQ5KBlock(q5)), independentQ5(q5));
});

test("matches independent Q6_K sign planes and Q8_0 signed bytes", () => {
  const q6 = deterministicNative(NATIVE_Q6_K_BLOCK_BYTES, 53);
  half(new DataView(q6.buffer), 208, 2);
  assert.deepEqual(dequantizeQ6KBlock(unpackNativeQ6KBlock(q6)), independentQ6(q6));

  const q8 = deterministicNative(NATIVE_Q8_0_BLOCK_BYTES, 59);
  half(new DataView(q8.buffer), 0, 0.5);
  const expectedQ8 = Float32Array.from(
    new Int8Array(q8.buffer, q8.byteOffset + 2, 32),
    (value) => value * 0.5,
  );
  assert.deepEqual(dequantizeQ8_0Block(unpackNativeQ8_0Block(q8)), expectedQ8);
});

test("unpacks all WebGPU layouts and validates zero padding", () => {
  const q4 = packWebGpuQ4KBlock(
    unpackNativeQ4KBlock(deterministicNative(144, 61)),
  );
  const q5 = packWebGpuQ5KBlock(
    unpackNativeQ5KBlock(deterministicNative(176, 67)),
  );
  const q6Native = deterministicNative(210, 71);
  const q6 = packWebGpuQ6KBlock(unpackNativeQ6KBlock(q6Native));
  const q8 = packWebGpuQ8_0Block(
    unpackNativeQ8_0Block(deterministicNative(34, 73)),
  );

  assert.doesNotThrow(() => unpackWebGpuQ4KBlock(q4));
  assert.doesNotThrow(() => unpackWebGpuQ5KBlock(q5));
  assert.doesNotThrow(() => unpackWebGpuQ6KBlock(q6));
  assert.doesNotThrow(() => unpackWebGpuQ8_0Block(q8));
  q6[211] = 1;
  q8[2] = 1;
  assert.throws(() => unpackWebGpuQ6KBlock(q6), /padding/i);
  assert.throws(() => unpackWebGpuQ8_0Block(q8), /padding/i);
});
