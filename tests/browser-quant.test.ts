import assert from "node:assert/strict";
import test from "node:test";

import {
  BROWSER_Q3_K_BLOCK_BYTES,
  BROWSER_Q3_FUSED_K_BLOCK_BYTES,
  BROWSER_Q4_K_BLOCK_BYTES,
  BROWSER_Q5_K_BLOCK_BYTES,
  BROWSER_Q6_K_BLOCK_BYTES,
  dequantizeBrowserQ3KBlock,
  dequantizeBrowserQ3KFusedBlock,
  dequantizeBrowserQ4KBlock,
  dequantizeBrowserQ5KBlock,
  dequantizeBrowserQ6KBlock,
  repackNativeQ3KBrowser,
  repackNativeQ3KFusedBrowser,
  repackNativeQ4KBrowser,
  repackNativeQ5KBrowser,
  repackNativeQ6KBrowser,
} from "../src/browser-quant.js";
import {
  dequantizeQ4KBlock,
  dequantizeQ5KBlock,
  dequantizeQ6KBlock,
  unpackNativeQ4KBlock,
  unpackNativeQ5KBlock,
  unpackNativeQ6KBlock,
} from "../src/mixed-quant.js";
import {
  dequantizeQ3KBlock,
  unpackNativeQ3KBlock,
} from "../src/q3k.js";

function native(length: number, multiplier: number): Uint8Array {
  return Uint8Array.from(
    { length },
    (_, index) => (index * multiplier + 23) & 255,
  );
}

function setHalf(bytes: Uint8Array, offset: number, bits: number): void {
  new DataView(bytes.buffer, bytes.byteOffset).setUint16(offset, bits, true);
}

function maxDifference(left: Float32Array, right: Float32Array): number {
  assert.equal(left.length, right.length);
  let maximum = 0;
  for (let index = 0; index < left.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(left[index]! - right[index]!));
  }
  return maximum;
}

test("repackages Q3 into exact aligned nibble values", () => {
  const nativeBytes = native(110 * 2, 19);
  setHalf(nativeBytes, 108, 0x3555);
  setHalf(nativeBytes, 218, 0x3155);
  const packed = repackNativeQ3KBrowser(nativeBytes);
  assert.equal(packed.byteLength, BROWSER_Q3_K_BLOCK_BYTES * 2);
  for (let block = 0; block < 2; block += 1) {
    const expected = dequantizeQ3KBlock(
      unpackNativeQ3KBlock(nativeBytes.subarray(block * 110, block * 110 + 110)),
    );
    const actual = dequantizeBrowserQ3KBlock(
      packed.subarray(
        block * BROWSER_Q3_K_BLOCK_BYTES,
        (block + 1) * BROWSER_Q3_K_BLOCK_BYTES,
      ),
    );
    assert.ok(maxDifference(expected, actual) <= 1e-7);
  }
});

test("repackages Q3 with fused exact FP32 factors and aligned quant words", () => {
  const nativeBytes = native(110, 19);
  setHalf(nativeBytes, 108, 0x3555);
  const packed = repackNativeQ3KFusedBrowser(nativeBytes);
  assert.equal(packed.byteLength, BROWSER_Q3_FUSED_K_BLOCK_BYTES);
  const expected = dequantizeQ3KBlock(unpackNativeQ3KBlock(nativeBytes));
  const actual = dequantizeBrowserQ3KFusedBlock(packed);
  assert.ok(maxDifference(expected, actual) === 0);
});

test("repackages Q5 into lane-major quant values with fused exact FP32 factors", () => {
  const nativeBytes = native(176, 29);
  setHalf(nativeBytes, 0, 0x3555);
  setHalf(nativeBytes, 2, 0x3155);
  const packed = repackNativeQ5KBrowser(nativeBytes);
  assert.equal(packed.byteLength, BROWSER_Q5_K_BLOCK_BYTES);
  const expected = dequantizeQ5KBlock(unpackNativeQ5KBlock(nativeBytes));
  const actual = dequantizeBrowserQ5KBlock(packed);
  assert.ok(maxDifference(expected, actual) === 0);
});

test("repackages Q4 into lane-major quant values with fused exact FP32 factors", () => {
  const nativeBytes = native(144, 23);
  setHalf(nativeBytes, 0, 0x3555);
  setHalf(nativeBytes, 2, 0x3155);
  const packed = repackNativeQ4KBrowser(nativeBytes);
  assert.equal(packed.byteLength, BROWSER_Q4_K_BLOCK_BYTES);
  const expected = dequantizeQ4KBlock(unpackNativeQ4KBlock(nativeBytes));
  const actual = dequantizeBrowserQ4KBlock(packed);
  assert.ok(maxDifference(expected, actual) === 0);
});

test("repackages Q6 into lane-major quant values with fused exact FP32 factors", () => {
  const nativeBytes = native(210, 31);
  setHalf(nativeBytes, 208, 0x3155);
  const packed = repackNativeQ6KBrowser(nativeBytes);
  assert.equal(packed.byteLength, BROWSER_Q6_K_BLOCK_BYTES);
  const expected = dequantizeQ6KBlock(unpackNativeQ6KBlock(nativeBytes));
  const actual = dequantizeBrowserQ6KBlock(packed);
  assert.ok(maxDifference(expected, actual) === 0);
});

test("rejects partial browser quant blocks", () => {
  assert.throws(() => repackNativeQ3KBrowser(new Uint8Array(111)), /whole 110-byte/i);
  assert.throws(() => repackNativeQ3KFusedBrowser(new Uint8Array(111)), /whole 110-byte/i);
  assert.throws(() => repackNativeQ4KBrowser(new Uint8Array(145)), /whole 144-byte/i);
  assert.throws(() => repackNativeQ5KBrowser(new Uint8Array(177)), /whole 176-byte/i);
  assert.throws(() => repackNativeQ6KBrowser(new Uint8Array(211)), /whole 210-byte/i);
});
