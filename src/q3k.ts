export const Q3_K_ELEMENTS_PER_BLOCK = 256;
export const NATIVE_Q3_K_BLOCK_BYTES = 110;
export const WEBGPU_Q3_K_BLOCK_BYTES = 112;

export interface Q3KBlock {
  readonly hmask: Uint8Array;
  readonly qs: Uint8Array;
  readonly scales: Uint8Array;
  /** Raw IEEE 754 binary16 bits; repacking never rounds through JavaScript. */
  readonly deltaBits: number;
}

function requireLength(
  bytes: Uint8Array,
  expected: number,
  label: string,
): void {
  if (bytes.byteLength !== expected) {
    throw new Error(`${label} must be exactly ${expected} bytes`);
  }
}

function requireBlock(block: Q3KBlock): void {
  requireLength(block.hmask, 32, "Q3_K high-bit mask");
  requireLength(block.qs, 64, "Q3_K low-bit plane");
  requireLength(block.scales, 12, "Q3_K packed scales");
  if (
    !Number.isInteger(block.deltaBits) ||
    block.deltaBits < 0 ||
    block.deltaBits > 0xffff
  ) {
    throw new Error("Q3_K deltaBits must be an unsigned 16-bit integer");
  }
}

export function unpackNativeQ3KBlock(bytes: Uint8Array): Q3KBlock {
  requireLength(bytes, NATIVE_Q3_K_BLOCK_BYTES, "native Q3_K block");
  return {
    hmask: bytes.slice(0, 32),
    qs: bytes.slice(32, 96),
    scales: bytes.slice(96, 108),
    deltaBits: new DataView(
      bytes.buffer,
      bytes.byteOffset + 108,
      2,
    ).getUint16(0, true),
  };
}

export function packNativeQ3KBlock(block: Q3KBlock): Uint8Array {
  requireBlock(block);
  const bytes = new Uint8Array(NATIVE_Q3_K_BLOCK_BYTES);
  bytes.set(block.hmask, 0);
  bytes.set(block.qs, 32);
  bytes.set(block.scales, 96);
  new DataView(bytes.buffer).setUint16(108, block.deltaBits, true);
  return bytes;
}

/**
 * The WebGPU form places every byte array on a four-byte boundary:
 * delta/padding, scales, low-bit plane, then high-bit plane.
 */
export function packWebGpuQ3KBlock(block: Q3KBlock): Uint8Array {
  requireBlock(block);
  const bytes = new Uint8Array(WEBGPU_Q3_K_BLOCK_BYTES);
  new DataView(bytes.buffer).setUint16(0, block.deltaBits, true);
  bytes.set(block.scales, 4);
  bytes.set(block.qs, 16);
  bytes.set(block.hmask, 80);
  return bytes;
}

export function unpackWebGpuQ3KBlock(bytes: Uint8Array): Q3KBlock {
  requireLength(bytes, WEBGPU_Q3_K_BLOCK_BYTES, "WebGPU Q3_K block");
  if (bytes[2] !== 0 || bytes[3] !== 0) {
    throw new Error("WebGPU Q3_K padding bytes must be zero");
  }
  return {
    deltaBits: new DataView(
      bytes.buffer,
      bytes.byteOffset,
      2,
    ).getUint16(0, true),
    scales: bytes.slice(4, 16),
    qs: bytes.slice(16, 80),
    hmask: bytes.slice(80, 112),
  };
}

export function repackNativeQ3K(nativeBytes: Uint8Array): Uint8Array {
  if (nativeBytes.byteLength % NATIVE_Q3_K_BLOCK_BYTES !== 0) {
    throw new Error("Q3_K input must contain whole 110-byte native blocks");
  }
  const blockCount = nativeBytes.byteLength / NATIVE_Q3_K_BLOCK_BYTES;
  const output = new Uint8Array(blockCount * WEBGPU_Q3_K_BLOCK_BYTES);
  for (let index = 0; index < blockCount; index += 1) {
    const start = index * NATIVE_Q3_K_BLOCK_BYTES;
    output.set(
      packWebGpuQ3KBlock(
        unpackNativeQ3KBlock(
          nativeBytes.subarray(start, start + NATIVE_Q3_K_BLOCK_BYTES),
        ),
      ),
      index * WEBGPU_Q3_K_BLOCK_BYTES,
    );
  }
  return output;
}

export function repackWebGpuQ3K(webGpuBytes: Uint8Array): Uint8Array {
  if (webGpuBytes.byteLength % WEBGPU_Q3_K_BLOCK_BYTES !== 0) {
    throw new Error("Q3_K input must contain whole 112-byte WebGPU blocks");
  }
  const blockCount = webGpuBytes.byteLength / WEBGPU_Q3_K_BLOCK_BYTES;
  const output = new Uint8Array(blockCount * NATIVE_Q3_K_BLOCK_BYTES);
  for (let index = 0; index < blockCount; index += 1) {
    const start = index * WEBGPU_Q3_K_BLOCK_BYTES;
    output.set(
      packNativeQ3KBlock(
        unpackWebGpuQ3KBlock(
          webGpuBytes.subarray(start, start + WEBGPU_Q3_K_BLOCK_BYTES),
        ),
      ),
      index * NATIVE_Q3_K_BLOCK_BYTES,
    );
  }
  return output;
}

function float16ToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) {
    return sign * fraction * 2 ** -24;
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : Number.NaN;
  }
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

function unpackScales(scales: Uint8Array): Int8Array {
  const unpacked = new Int8Array(16);
  for (let index = 0; index < 16; index += 1) {
    const low =
      index < 8
        ? scales[index]! & 0x0f
        : scales[index - 8]! >>> 4;
    const high =
      ((scales[8 + (index % 4)]! >>> (2 * Math.floor(index / 4))) & 0x03) <<
      4;
    unpacked[index] = (low | high) - 32;
  }
  return unpacked;
}

/**
 * CPU reference decode for fixture validation only. Runtime kernels consume the
 * packed 112-byte form directly and do not expand full tensors to FP16.
 */
export function dequantizeQ3KBlock(block: Q3KBlock): Float32Array {
  requireBlock(block);
  const values = new Float32Array(Q3_K_ELEMENTS_PER_BLOCK);
  const scales = unpackScales(block.scales);
  const superBlockDelta = float16ToNumber(block.deltaBits);
  let mask = 1;
  let scaleIndex = 0;
  let outputIndex = 0;

  for (let group = 0; group < 2; group += 1) {
    const qBase = group * 32;
    let shift = 0;
    for (let plane = 0; plane < 4; plane += 1) {
      for (let half = 0; half < 2; half += 1) {
        const delta = superBlockDelta * scales[scaleIndex++]!;
        const qHalfBase = qBase + half * 16;
        // The two 128-value groups reuse the 32 mask bytes; `mask` selects
        // their distinct bit planes, so the byte index must not follow qBase.
        const hmaskHalfBase = half * 16;
        for (let lane = 0; lane < 16; lane += 1) {
          const low = (block.qs[qHalfBase + lane]! >>> shift) & 0x03;
          const high =
            (block.hmask[hmaskHalfBase + lane]! & mask) === 0 ? 4 : 0;
          values[outputIndex++] = delta * (low - high);
        }
      }
      shift += 2;
      mask <<= 1;
    }
  }
  return values;
}
