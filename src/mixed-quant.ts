/**
 * Adapts native ggml quant blocks into u32-addressable browser layouts.
 *
 * Pack and unpack operations preserve integer planes and binary16 scale bits;
 * dequantization exists only as an independent correctness reference.
 */
export const K_QUANT_ELEMENTS_PER_BLOCK = 256;
export const Q8_0_ELEMENTS_PER_BLOCK = 32;

export const NATIVE_Q4_K_BLOCK_BYTES = 144;
export const NATIVE_Q5_K_BLOCK_BYTES = 176;
export const NATIVE_Q6_K_BLOCK_BYTES = 210;
export const NATIVE_Q8_0_BLOCK_BYTES = 34;

export const WEBGPU_Q4_K_BLOCK_BYTES = 144;
export const WEBGPU_Q5_K_BLOCK_BYTES = 176;
export const WEBGPU_Q6_K_BLOCK_BYTES = 212;
export const WEBGPU_Q8_0_BLOCK_BYTES = 36;

interface KScaleMinBlock {
  readonly deltaBits: number;
  readonly minBits: number;
  readonly scales: Uint8Array;
  readonly qs: Uint8Array;
}

export interface Q4KBlock extends KScaleMinBlock {}

export interface Q5KBlock extends KScaleMinBlock {
  readonly qh: Uint8Array;
}

export interface Q6KBlock {
  readonly ql: Uint8Array;
  readonly qh: Uint8Array;
  readonly scales: Int8Array;
  readonly deltaBits: number;
}

export interface Q8_0Block {
  readonly deltaBits: number;
  readonly qs: Int8Array;
}

function requireBytes(
  bytes: Uint8Array,
  expected: number,
  label: string,
): void {
  if (bytes.byteLength !== expected) {
    throw new Error(`${label} must be exactly ${expected} bytes`);
  }
}

function requireU16(value: number, label: string): void {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff) {
    throw new Error(`${label} must be an unsigned 16-bit integer`);
  }
}

function requireArray(
  value: Uint8Array | Int8Array,
  length: number,
  label: string,
): void {
  if (value.byteLength !== length) {
    throw new Error(`${label} must contain exactly ${length} bytes`);
  }
}

function view(bytes: Uint8Array): DataView {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function float16ToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : Number.NaN;
  }
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

function scaleAndMin(
  scales: Uint8Array,
  group: number,
): readonly [number, number] {
  if (group < 4) {
    return [scales[group]! & 0x3f, scales[group + 4]! & 0x3f];
  }
  return [
    (scales[group + 4]! & 0x0f) |
      ((scales[group - 4]! >>> 6) << 4),
    (scales[group + 4]! >>> 4) | ((scales[group]! >>> 6) << 4),
  ];
}

export function unpackNativeQ4KBlock(bytes: Uint8Array): Q4KBlock {
  requireBytes(bytes, NATIVE_Q4_K_BLOCK_BYTES, "native Q4_K block");
  return {
    deltaBits: view(bytes).getUint16(0, true),
    minBits: view(bytes).getUint16(2, true),
    scales: bytes.slice(4, 16),
    qs: bytes.slice(16, 144),
  };
}

function requireQ4K(block: Q4KBlock): void {
  requireU16(block.deltaBits, "Q4_K deltaBits");
  requireU16(block.minBits, "Q4_K minBits");
  requireArray(block.scales, 12, "Q4_K scales");
  requireArray(block.qs, 128, "Q4_K low nibbles");
}

export function packWebGpuQ4KBlock(block: Q4KBlock): Uint8Array {
  requireQ4K(block);
  const bytes = new Uint8Array(WEBGPU_Q4_K_BLOCK_BYTES);
  const output = view(bytes);
  output.setUint16(0, block.deltaBits, true);
  output.setUint16(2, block.minBits, true);
  bytes.set(block.scales, 4);
  bytes.set(block.qs, 16);
  return bytes;
}

export function unpackWebGpuQ4KBlock(bytes: Uint8Array): Q4KBlock {
  requireBytes(bytes, WEBGPU_Q4_K_BLOCK_BYTES, "WebGPU Q4_K block");
  return unpackNativeQ4KBlock(bytes);
}

export function unpackNativeQ5KBlock(bytes: Uint8Array): Q5KBlock {
  requireBytes(bytes, NATIVE_Q5_K_BLOCK_BYTES, "native Q5_K block");
  return {
    deltaBits: view(bytes).getUint16(0, true),
    minBits: view(bytes).getUint16(2, true),
    scales: bytes.slice(4, 16),
    qh: bytes.slice(16, 48),
    qs: bytes.slice(48, 176),
  };
}

function requireQ5K(block: Q5KBlock): void {
  requireU16(block.deltaBits, "Q5_K deltaBits");
  requireU16(block.minBits, "Q5_K minBits");
  requireArray(block.scales, 12, "Q5_K scales");
  requireArray(block.qh, 32, "Q5_K high bits");
  requireArray(block.qs, 128, "Q5_K low nibbles");
}

export function packWebGpuQ5KBlock(block: Q5KBlock): Uint8Array {
  requireQ5K(block);
  const bytes = new Uint8Array(WEBGPU_Q5_K_BLOCK_BYTES);
  const output = view(bytes);
  output.setUint16(0, block.deltaBits, true);
  output.setUint16(2, block.minBits, true);
  bytes.set(block.scales, 4);
  bytes.set(block.qh, 16);
  bytes.set(block.qs, 48);
  return bytes;
}

export function unpackWebGpuQ5KBlock(bytes: Uint8Array): Q5KBlock {
  requireBytes(bytes, WEBGPU_Q5_K_BLOCK_BYTES, "WebGPU Q5_K block");
  return unpackNativeQ5KBlock(bytes);
}

export function unpackNativeQ6KBlock(bytes: Uint8Array): Q6KBlock {
  requireBytes(bytes, NATIVE_Q6_K_BLOCK_BYTES, "native Q6_K block");
  return {
    ql: bytes.slice(0, 128),
    qh: bytes.slice(128, 192),
    scales: Int8Array.from(bytes.subarray(192, 208)),
    deltaBits: view(bytes).getUint16(208, true),
  };
}

function requireQ6K(block: Q6KBlock): void {
  requireArray(block.ql, 128, "Q6_K low nibbles");
  requireArray(block.qh, 64, "Q6_K high bits");
  requireArray(block.scales, 16, "Q6_K scales");
  requireU16(block.deltaBits, "Q6_K deltaBits");
}

/**
 * The native Q6_K fields already start on u32 boundaries. The browser layout
 * only appends two zero bytes so adjacent 210-byte blocks cannot misalign them.
 */
export function packWebGpuQ6KBlock(block: Q6KBlock): Uint8Array {
  requireQ6K(block);
  const bytes = new Uint8Array(WEBGPU_Q6_K_BLOCK_BYTES);
  bytes.set(block.ql, 0);
  bytes.set(block.qh, 128);
  bytes.set(
    new Uint8Array(
      block.scales.buffer,
      block.scales.byteOffset,
      block.scales.byteLength,
    ),
    192,
  );
  view(bytes).setUint16(208, block.deltaBits, true);
  return bytes;
}

export function unpackWebGpuQ6KBlock(bytes: Uint8Array): Q6KBlock {
  requireBytes(bytes, WEBGPU_Q6_K_BLOCK_BYTES, "WebGPU Q6_K block");
  if (bytes[210] !== 0 || bytes[211] !== 0) {
    throw new Error("WebGPU Q6_K padding bytes must be zero");
  }
  return unpackNativeQ6KBlock(bytes.subarray(0, NATIVE_Q6_K_BLOCK_BYTES));
}

export function unpackNativeQ8_0Block(bytes: Uint8Array): Q8_0Block {
  requireBytes(bytes, NATIVE_Q8_0_BLOCK_BYTES, "native Q8_0 block");
  return {
    deltaBits: view(bytes).getUint16(0, true),
    qs: Int8Array.from(bytes.subarray(2, 34)),
  };
}

function requireQ8_0(block: Q8_0Block): void {
  requireU16(block.deltaBits, "Q8_0 deltaBits");
  requireArray(block.qs, 32, "Q8_0 signed quants");
}

/**
 * Q8_0 moves its signed byte array from native offset 2 to offset 4. This
 * keeps both the scale word and quant bytes u32-addressable in WGSL.
 */
export function packWebGpuQ8_0Block(block: Q8_0Block): Uint8Array {
  requireQ8_0(block);
  const bytes = new Uint8Array(WEBGPU_Q8_0_BLOCK_BYTES);
  view(bytes).setUint16(0, block.deltaBits, true);
  bytes.set(
    new Uint8Array(block.qs.buffer, block.qs.byteOffset, block.qs.byteLength),
    4,
  );
  return bytes;
}

export function unpackWebGpuQ8_0Block(bytes: Uint8Array): Q8_0Block {
  requireBytes(bytes, WEBGPU_Q8_0_BLOCK_BYTES, "WebGPU Q8_0 block");
  if (bytes[2] !== 0 || bytes[3] !== 0) {
    throw new Error("WebGPU Q8_0 padding bytes must be zero");
  }
  return {
    deltaBits: view(bytes).getUint16(0, true),
    qs: Int8Array.from(bytes.subarray(4, 36)),
  };
}

function repackBlocks<T>(
  bytes: Uint8Array,
  sourceBytes: number,
  outputBytes: number,
  label: string,
  unpack: (block: Uint8Array) => T,
  pack: (block: T) => Uint8Array,
): Uint8Array {
  if (bytes.byteLength % sourceBytes !== 0) {
    throw new Error(`${label} input must contain whole ${sourceBytes}-byte blocks`);
  }
  const count = bytes.byteLength / sourceBytes;
  const output = new Uint8Array(count * outputBytes);
  for (let index = 0; index < count; index += 1) {
    output.set(
      pack(bytesToBlock(bytes, index, sourceBytes, unpack)),
      index * outputBytes,
    );
  }
  return output;
}

function bytesToBlock<T>(
  bytes: Uint8Array,
  index: number,
  blockBytes: number,
  unpack: (block: Uint8Array) => T,
): T {
  const offset = index * blockBytes;
  return unpack(bytes.subarray(offset, offset + blockBytes));
}

export function repackNativeQ4K(bytes: Uint8Array): Uint8Array {
  return repackBlocks(
    bytes,
    NATIVE_Q4_K_BLOCK_BYTES,
    WEBGPU_Q4_K_BLOCK_BYTES,
    "Q4_K native",
    unpackNativeQ4KBlock,
    packWebGpuQ4KBlock,
  );
}

export function repackWebGpuQ4K(bytes: Uint8Array): Uint8Array {
  return repackBlocks(
    bytes,
    WEBGPU_Q4_K_BLOCK_BYTES,
    NATIVE_Q4_K_BLOCK_BYTES,
    "Q4_K WebGPU",
    unpackWebGpuQ4KBlock,
    packWebGpuQ4KBlock,
  );
}

export function repackNativeQ5K(bytes: Uint8Array): Uint8Array {
  return repackBlocks(
    bytes,
    NATIVE_Q5_K_BLOCK_BYTES,
    WEBGPU_Q5_K_BLOCK_BYTES,
    "Q5_K native",
    unpackNativeQ5KBlock,
    packWebGpuQ5KBlock,
  );
}

export function repackWebGpuQ5K(bytes: Uint8Array): Uint8Array {
  return repackBlocks(
    bytes,
    WEBGPU_Q5_K_BLOCK_BYTES,
    NATIVE_Q5_K_BLOCK_BYTES,
    "Q5_K WebGPU",
    unpackWebGpuQ5KBlock,
    packWebGpuQ5KBlock,
  );
}

export function repackNativeQ6K(bytes: Uint8Array): Uint8Array {
  return repackBlocks(
    bytes,
    NATIVE_Q6_K_BLOCK_BYTES,
    WEBGPU_Q6_K_BLOCK_BYTES,
    "Q6_K native",
    unpackNativeQ6KBlock,
    packWebGpuQ6KBlock,
  );
}

function packNativeQ6KBlock(block: Q6KBlock): Uint8Array {
  return packWebGpuQ6KBlock(block).subarray(0, NATIVE_Q6_K_BLOCK_BYTES);
}

export function repackWebGpuQ6K(bytes: Uint8Array): Uint8Array {
  return repackBlocks(
    bytes,
    WEBGPU_Q6_K_BLOCK_BYTES,
    NATIVE_Q6_K_BLOCK_BYTES,
    "Q6_K WebGPU",
    unpackWebGpuQ6KBlock,
    packNativeQ6KBlock,
  );
}

export function repackNativeQ8_0(bytes: Uint8Array): Uint8Array {
  return repackBlocks(
    bytes,
    NATIVE_Q8_0_BLOCK_BYTES,
    WEBGPU_Q8_0_BLOCK_BYTES,
    "Q8_0 native",
    unpackNativeQ8_0Block,
    packWebGpuQ8_0Block,
  );
}

function packNativeQ8_0Block(block: Q8_0Block): Uint8Array {
  requireQ8_0(block);
  const bytes = new Uint8Array(NATIVE_Q8_0_BLOCK_BYTES);
  view(bytes).setUint16(0, block.deltaBits, true);
  bytes.set(
    new Uint8Array(block.qs.buffer, block.qs.byteOffset, block.qs.byteLength),
    2,
  );
  return bytes;
}

export function repackWebGpuQ8_0(bytes: Uint8Array): Uint8Array {
  return repackBlocks(
    bytes,
    WEBGPU_Q8_0_BLOCK_BYTES,
    NATIVE_Q8_0_BLOCK_BYTES,
    "Q8_0 WebGPU",
    unpackWebGpuQ8_0Block,
    packNativeQ8_0Block,
  );
}

export function dequantizeQ4KBlock(block: Q4KBlock): Float32Array {
  requireQ4K(block);
  const output = new Float32Array(K_QUANT_ELEMENTS_PER_BLOCK);
  const delta = float16ToNumber(block.deltaBits);
  const minDelta = float16ToNumber(block.minBits);
  for (let chunk = 0; chunk < 4; chunk += 1) {
    const [scale0, min0] = scaleAndMin(block.scales, chunk * 2);
    const [scale1, min1] = scaleAndMin(block.scales, chunk * 2 + 1);
    for (let lane = 0; lane < 32; lane += 1) {
      const quant = block.qs[chunk * 32 + lane]!;
      output[chunk * 64 + lane] =
        delta * scale0 * (quant & 0x0f) - minDelta * min0;
      output[chunk * 64 + lane + 32] =
        delta * scale1 * (quant >>> 4) - minDelta * min1;
    }
  }
  return output;
}

export function dequantizeQ5KBlock(block: Q5KBlock): Float32Array {
  requireQ5K(block);
  const output = new Float32Array(K_QUANT_ELEMENTS_PER_BLOCK);
  const delta = float16ToNumber(block.deltaBits);
  const minDelta = float16ToNumber(block.minBits);
  for (let chunk = 0; chunk < 4; chunk += 1) {
    const [scale0, min0] = scaleAndMin(block.scales, chunk * 2);
    const [scale1, min1] = scaleAndMin(block.scales, chunk * 2 + 1);
    const lowMask = 1 << (chunk * 2);
    const highMask = 1 << (chunk * 2 + 1);
    for (let lane = 0; lane < 32; lane += 1) {
      const quant = block.qs[chunk * 32 + lane]!;
      const high = block.qh[lane]!;
      const value0 = (quant & 0x0f) + ((high & lowMask) === 0 ? 0 : 16);
      const value1 = (quant >>> 4) + ((high & highMask) === 0 ? 0 : 16);
      output[chunk * 64 + lane] =
        delta * scale0 * value0 - minDelta * min0;
      output[chunk * 64 + lane + 32] =
        delta * scale1 * value1 - minDelta * min1;
    }
  }
  return output;
}

export function dequantizeQ6KBlock(block: Q6KBlock): Float32Array {
  requireQ6K(block);
  const output = new Float32Array(K_QUANT_ELEMENTS_PER_BLOCK);
  const delta = float16ToNumber(block.deltaBits);
  for (let half = 0; half < 2; half += 1) {
    const base = half * 128;
    for (let lane = 0; lane < 32; lane += 1) {
      const ql0 = block.ql[half * 64 + lane]!;
      const ql1 = block.ql[half * 64 + lane + 32]!;
      const qh = block.qh[half * 32 + lane]!;
      const scale = half * 8 + Math.floor(lane / 16);
      output[base + lane] =
        delta *
        block.scales[scale]! *
        (((ql0 & 15) | ((qh & 3) << 4)) - 32);
      output[base + lane + 32] =
        delta *
        block.scales[scale + 2]! *
        (((ql1 & 15) | (((qh >>> 2) & 3) << 4)) - 32);
      output[base + lane + 64] =
        delta *
        block.scales[scale + 4]! *
        (((ql0 >>> 4) | (((qh >>> 4) & 3) << 4)) - 32);
      output[base + lane + 96] =
        delta *
        block.scales[scale + 6]! *
        (((ql1 >>> 4) | (((qh >>> 6) & 3) << 4)) - 32);
    }
  }
  return output;
}

export function dequantizeQ8_0Block(block: Q8_0Block): Float32Array {
  requireQ8_0(block);
  const delta = float16ToNumber(block.deltaBits);
  return Float32Array.from(block.qs, (quant) => delta * quant);
}
