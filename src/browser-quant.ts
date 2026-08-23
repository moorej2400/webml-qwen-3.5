import {
  NATIVE_Q4_K_BLOCK_BYTES,
  NATIVE_Q5_K_BLOCK_BYTES,
  NATIVE_Q6_K_BLOCK_BYTES,
  unpackNativeQ4KBlock,
  unpackNativeQ5KBlock,
  unpackNativeQ6KBlock,
} from "./mixed-quant.js";
import {
  NATIVE_Q3_K_BLOCK_BYTES,
  unpackNativeQ3KBlock,
} from "./q3k.js";

export const BROWSER_Q3_K_BLOCK_BYTES = 148;
export const BROWSER_Q3_FUSED_K_BLOCK_BYTES = 192;
export const BROWSER_Q4_K_BLOCK_BYTES = 192;
export const BROWSER_Q5_K_BLOCK_BYTES = 224;
export const BROWSER_Q6_K_BLOCK_BYTES = 256;
const K_QUANT_ELEMENTS = 256;

function requireWholeBlocks(
  bytes: Uint8Array,
  blockBytes: number,
): number {
  if (bytes.byteLength % blockBytes !== 0) {
    throw new Error(`Browser quant input must contain whole ${blockBytes}-byte blocks`);
  }
  return bytes.byteLength / blockBytes;
}

function float16ToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

function q3Scales(packed: Uint8Array): Int8Array {
  const scales = new Int8Array(16);
  for (let index = 0; index < scales.length; index += 1) {
    const low = index < 8
      ? packed[index]! & 0x0f
      : packed[index - 8]! >>> 4;
    const high = (
      packed[8 + (index % 4)]! >>> (2 * Math.floor(index / 4))
    ) & 0x03;
    scales[index] = (low | (high << 4)) - 32;
  }
  return scales;
}

function q5Component(
  scales: Uint8Array,
  group: number,
  minimum: boolean,
): number {
  if (group < 4) {
    return scales[minimum ? group + 4 : group]! & 0x3f;
  }
  if (!minimum) {
    return (scales[group + 4]! & 0x0f) |
      ((scales[group - 4]! >>> 6) << 4);
  }
  return (scales[group + 4]! >>> 4) |
    ((scales[group]! >>> 6) << 4);
}

/**
 * Q3 keeps its original delta and integer scales. Only its irregular three-bit
 * planes become consecutive nibbles so one GPU lane can decode eight values
 * with one word load. This preserves every quantized value exactly.
 */
export function repackNativeQ3KBrowser(bytes: Uint8Array): Uint8Array {
  const blockCount = requireWholeBlocks(bytes, NATIVE_Q3_K_BLOCK_BYTES);
  const output = new Uint8Array(blockCount * BROWSER_Q3_K_BLOCK_BYTES);
  for (let blockIndex = 0; blockIndex < blockCount; blockIndex += 1) {
    const sourceOffset = blockIndex * NATIVE_Q3_K_BLOCK_BYTES;
    const targetOffset = blockIndex * BROWSER_Q3_K_BLOCK_BYTES;
    const block = unpackNativeQ3KBlock(
      bytes.subarray(sourceOffset, sourceOffset + NATIVE_Q3_K_BLOCK_BYTES),
    );
    new DataView(output.buffer).setUint16(targetOffset, block.deltaBits, true);
    const scales = q3Scales(block.scales);
    output.set(new Uint8Array(scales.buffer), targetOffset + 4);
    for (let element = 0; element < K_QUANT_ELEMENTS; element += 1) {
      const group = Math.floor(element / 128);
      const withinGroup = element % 128;
      const subgroup = Math.floor(withinGroup / 16);
      const plane = Math.floor(subgroup / 2);
      const half = subgroup % 2;
      const lane = element % 16;
      const low = (block.qs[group * 32 + half * 16 + lane]! >>> (plane * 2)) & 3;
      const high = (block.hmask[half * 16 + lane]! & (1 << (group * 4 + plane))) === 0
        ? 4
        : 0;
      const nibble = low - high + 4;
      const outputIndex = targetOffset + 20 + Math.floor(element / 2);
      output[outputIndex] = output[outputIndex]! |
        (nibble << ((element % 2) * 4));
    }
  }
  return output;
}

/** Q3 with exact FP32 scale products and a 16-byte-aligned quant plane. */
export function repackNativeQ3KFusedBrowser(bytes: Uint8Array): Uint8Array {
  const blockCount = requireWholeBlocks(bytes, NATIVE_Q3_K_BLOCK_BYTES);
  const output = new Uint8Array(blockCount * BROWSER_Q3_FUSED_K_BLOCK_BYTES);
  const view = new DataView(output.buffer);
  for (let blockIndex = 0; blockIndex < blockCount; blockIndex += 1) {
    const sourceOffset = blockIndex * NATIVE_Q3_K_BLOCK_BYTES;
    const targetOffset = blockIndex * BROWSER_Q3_FUSED_K_BLOCK_BYTES;
    const block = unpackNativeQ3KBlock(
      bytes.subarray(sourceOffset, sourceOffset + NATIVE_Q3_K_BLOCK_BYTES),
    );
    const delta = float16ToNumber(block.deltaBits);
    const scales = q3Scales(block.scales);
    for (let index = 0; index < scales.length; index += 1) {
      view.setFloat32(
        targetOffset + index * 4,
        delta * scales[index]!,
        true,
      );
    }
    for (let element = 0; element < K_QUANT_ELEMENTS; element += 1) {
      const group = Math.floor(element / 128);
      const withinGroup = element % 128;
      const subgroup = Math.floor(withinGroup / 16);
      const plane = Math.floor(subgroup / 2);
      const half = subgroup % 2;
      const lane = element % 16;
      const low = (block.qs[group * 32 + half * 16 + lane]! >>> (plane * 2)) & 3;
      const high = (block.hmask[half * 16 + lane]! & (1 << (group * 4 + plane))) === 0
        ? 4
        : 0;
      const nibble = low - high + 4;
      const outputIndex = targetOffset + 64 + Math.floor(element / 2);
      output[outputIndex] = output[outputIndex]! |
        (nibble << ((element % 2) * 4));
    }
  }
  return output;
}

/**
 * Q4 stores lane-major nibbles and exact FP32 multiplier and bias terms.
 */
export function repackNativeQ4KBrowser(bytes: Uint8Array): Uint8Array {
  const blockCount = requireWholeBlocks(bytes, NATIVE_Q4_K_BLOCK_BYTES);
  const output = new Uint8Array(blockCount * BROWSER_Q4_K_BLOCK_BYTES);
  const view = new DataView(output.buffer);
  for (let blockIndex = 0; blockIndex < blockCount; blockIndex += 1) {
    const sourceOffset = blockIndex * NATIVE_Q4_K_BLOCK_BYTES;
    const targetOffset = blockIndex * BROWSER_Q4_K_BLOCK_BYTES;
    const block = unpackNativeQ4KBlock(
      bytes.subarray(sourceOffset, sourceOffset + NATIVE_Q4_K_BLOCK_BYTES),
    );
    const delta = float16ToNumber(block.deltaBits);
    const minimum = float16ToNumber(block.minBits);
    for (let group = 0; group < 8; group += 1) {
      view.setFloat32(
        targetOffset + group * 4,
        delta * q5Component(block.scales, group, false),
        true,
      );
      view.setFloat32(
        targetOffset + 32 + group * 4,
        minimum * q5Component(block.scales, group, true),
        true,
      );
    }
    for (let lane = 0; lane < 32; lane += 1) {
      for (let group = 0; group < 8; group += 1) {
        const chunk = Math.floor(group / 2);
        const packed = block.qs[chunk * 32 + lane]!;
        const quant = (group & 1) === 0 ? packed & 0x0f : packed >>> 4;
        const outputIndex = targetOffset + 64 + lane * 4 + Math.floor(group / 2);
        output[outputIndex] = output[outputIndex]! |
          (quant << ((group % 2) * 4));
      }
    }
  }
  return output;
}

/**
 * Q5 stores eight lane-major quant values and exact FP32 multiplier and bias.
 */
export function repackNativeQ5KBrowser(bytes: Uint8Array): Uint8Array {
  const blockCount = requireWholeBlocks(bytes, NATIVE_Q5_K_BLOCK_BYTES);
  const output = new Uint8Array(blockCount * BROWSER_Q5_K_BLOCK_BYTES);
  const view = new DataView(output.buffer);
  for (let blockIndex = 0; blockIndex < blockCount; blockIndex += 1) {
    const sourceOffset = blockIndex * NATIVE_Q5_K_BLOCK_BYTES;
    const targetOffset = blockIndex * BROWSER_Q5_K_BLOCK_BYTES;
    const block = unpackNativeQ5KBlock(
      bytes.subarray(sourceOffset, sourceOffset + NATIVE_Q5_K_BLOCK_BYTES),
    );
    const delta = float16ToNumber(block.deltaBits);
    const minimum = float16ToNumber(block.minBits);
    for (let group = 0; group < 8; group += 1) {
      view.setFloat32(
        targetOffset + group * 4,
        delta * q5Component(block.scales, group, false),
        true,
      );
      view.setFloat32(
        targetOffset + 32 + group * 4,
        minimum * q5Component(block.scales, group, true),
        true,
      );
    }
    for (let lane = 0; lane < 32; lane += 1) {
      const high = block.qh[lane]!;
      output[targetOffset + 64 + lane] = high;
      for (let group = 0; group < 8; group += 1) {
        const chunk = Math.floor(group / 2);
        const packed = block.qs[chunk * 32 + lane]!;
        const low = (group & 1) === 0 ? packed & 0x0f : packed >>> 4;
        const outputIndex = targetOffset + 96 + lane * 4 + Math.floor(group / 2);
        output[outputIndex] = output[outputIndex]! |
          (low << ((group % 2) * 4));
      }
    }
  }
  return output;
}

/** Q6 transposes quant planes and stores exact FP32 delta-scale products. */
export function repackNativeQ6KBrowser(bytes: Uint8Array): Uint8Array {
  const blockCount = requireWholeBlocks(bytes, NATIVE_Q6_K_BLOCK_BYTES);
  const output = new Uint8Array(blockCount * BROWSER_Q6_K_BLOCK_BYTES);
  const view = new DataView(output.buffer);
  for (let blockIndex = 0; blockIndex < blockCount; blockIndex += 1) {
    const sourceOffset = blockIndex * NATIVE_Q6_K_BLOCK_BYTES;
    const targetOffset = blockIndex * BROWSER_Q6_K_BLOCK_BYTES;
    const block = unpackNativeQ6KBlock(
      bytes.subarray(sourceOffset, sourceOffset + NATIVE_Q6_K_BLOCK_BYTES),
    );
    const delta = float16ToNumber(block.deltaBits);
    for (let scaleIndex = 0; scaleIndex < 16; scaleIndex += 1) {
      const half = scaleIndex % 2;
      const group = Math.floor(scaleIndex / 2);
      view.setFloat32(
        targetOffset + (half * 8 + group) * 4,
        delta * block.scales[scaleIndex]!,
        true,
      );
    }
    for (let lane = 0; lane < 32; lane += 1) {
      for (let group = 0; group < 8; group += 1) {
        const half = Math.floor(group / 4);
        const section = group % 4;
        const lowByte = block.ql[
          half * 64 + lane + ((section & 1) === 0 ? 0 : 32)
        ]!;
        const low = section < 2 ? lowByte & 0x0f : lowByte >>> 4;
        const high = (block.qh[half * 32 + lane]! >>> (section * 2)) & 3;
        const lowOutputIndex = targetOffset + 64 + lane * 4 + Math.floor(group / 2);
        output[lowOutputIndex] = output[lowOutputIndex]! |
          (low << ((group % 2) * 4));
        const highOutputIndex = targetOffset + 192 + lane * 2 + Math.floor(group / 4);
        output[highOutputIndex] = output[highOutputIndex]! |
          (high << ((group % 4) * 2));
      }
    }
  }
  return output;
}

function requireBlock(bytes: Uint8Array, expected: number): void {
  if (bytes.byteLength !== expected) {
    throw new Error(`Browser quant block must be exactly ${expected} bytes`);
  }
}

export function dequantizeBrowserQ3KBlock(bytes: Uint8Array): Float32Array {
  requireBlock(bytes, BROWSER_Q3_K_BLOCK_BYTES);
  const delta = float16ToNumber(new DataView(bytes.buffer, bytes.byteOffset).getUint16(0, true));
  const output = new Float32Array(K_QUANT_ELEMENTS);
  for (let element = 0; element < output.length; element += 1) {
    const packed = bytes[20 + Math.floor(element / 2)]!;
    const quant = ((packed >>> ((element % 2) * 4)) & 0x0f) - 4;
    const rawScale = bytes[4 + Math.floor(element / 16)]!;
    const scale = rawScale >= 128 ? rawScale - 256 : rawScale;
    output[element] = delta * scale * quant;
  }
  return output;
}

export function dequantizeBrowserQ3KFusedBlock(bytes: Uint8Array): Float32Array {
  requireBlock(bytes, BROWSER_Q3_FUSED_K_BLOCK_BYTES);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const output = new Float32Array(K_QUANT_ELEMENTS);
  for (let element = 0; element < output.length; element += 1) {
    const packed = bytes[64 + Math.floor(element / 2)]!;
    const quant = ((packed >>> ((element % 2) * 4)) & 0x0f) - 4;
    const factor = view.getFloat32(Math.floor(element / 16) * 4, true);
    output[element] = factor * quant;
  }
  return output;
}

export function dequantizeBrowserQ4KBlock(bytes: Uint8Array): Float32Array {
  requireBlock(bytes, BROWSER_Q4_K_BLOCK_BYTES);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const output = new Float32Array(K_QUANT_ELEMENTS);
  for (let group = 0; group < 8; group += 1) {
    const multiplier = view.getFloat32(group * 4, true);
    const bias = view.getFloat32(32 + group * 4, true);
    for (let lane = 0; lane < 32; lane += 1) {
      const packed = bytes[64 + lane * 4 + Math.floor(group / 2)]!;
      const quant = (packed >>> ((group % 2) * 4)) & 0x0f;
      output[group * 32 + lane] = multiplier * quant - bias;
    }
  }
  return output;
}

export function dequantizeBrowserQ5KBlock(bytes: Uint8Array): Float32Array {
  requireBlock(bytes, BROWSER_Q5_K_BLOCK_BYTES);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const output = new Float32Array(K_QUANT_ELEMENTS);
  for (let group = 0; group < 8; group += 1) {
    const multiplier = view.getFloat32(group * 4, true);
    const bias = view.getFloat32(32 + group * 4, true);
    for (let lane = 0; lane < 32; lane += 1) {
      const packed = bytes[96 + lane * 4 + Math.floor(group / 2)]!;
      const low = (packed >>> ((group % 2) * 4)) & 0x0f;
      const high = (bytes[64 + lane]! >>> group) & 1;
      output[group * 32 + lane] = multiplier * (low + high * 16) - bias;
    }
  }
  return output;
}

export function dequantizeBrowserQ6KBlock(bytes: Uint8Array): Float32Array {
  requireBlock(bytes, BROWSER_Q6_K_BLOCK_BYTES);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const output = new Float32Array(K_QUANT_ELEMENTS);
  for (let group = 0; group < 8; group += 1) {
    for (let lane = 0; lane < 32; lane += 1) {
      const half = Math.floor(lane / 16);
      const multiplier = view.getFloat32((half * 8 + group) * 4, true);
      const packed = bytes[64 + lane * 4 + Math.floor(group / 2)]!;
      const low = (packed >>> ((group % 2) * 4)) & 0x0f;
      const high = (
        bytes[192 + lane * 2 + Math.floor(group / 4)]! >>>
        ((group % 4) * 2)
      ) & 3;
      output[group * 32 + lane] = multiplier * (low + high * 16 - 32);
    }
  }
  return output;
}
