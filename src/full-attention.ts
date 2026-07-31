import {
  attentionOutputGateCpu,
  partialMropeCpu,
  qkRmsNormPerHeadCpu,
} from "./qwen-primitives.js";

const QUERY_HEAD_COUNT = 16;
const KV_HEAD_COUNT = 4;
const HEAD_DIMENSION = 256;
const QUERY_GATE_RECORD_WIDTH = HEAD_DIMENSION * 2;
const QUERY_SCALE = Math.fround(1 / 16);
const NORM_EPSILON = Math.fround(1e-6);

export interface FullAttentionToken {
  /**
   * GGUF `attn_q` conversion keeps Qwen3.5's per-head
   * `[q[256], gate[256]]` records instead of splitting the full tensors.
   */
  readonly queryGate: Float32Array;
  readonly key: Float32Array;
  readonly value: Float32Array;
  readonly queryNormWeight: Float32Array;
  readonly keyNormWeight: Float32Array;
  readonly positions: readonly [number, number, number];
}

export interface SplitQueryGate {
  readonly query: Float32Array;
  readonly gate: Float32Array;
}

function requireLength(
  values: Float32Array,
  expected: number,
  label: string,
): void {
  if (values.length !== expected) {
    throw new Error(`${label} shape mismatch`);
  }
}

export function splitQwen35QueryGateProjection(
  input: Float32Array,
): SplitQueryGate {
  requireLength(
    input,
    QUERY_HEAD_COUNT * QUERY_GATE_RECORD_WIDTH,
    "Q/gate projection",
  );
  const query = new Float32Array(QUERY_HEAD_COUNT * HEAD_DIMENSION);
  const gate = new Float32Array(query.length);
  for (let head = 0; head < QUERY_HEAD_COUNT; head += 1) {
    const source = head * QUERY_GATE_RECORD_WIDTH;
    const destination = head * HEAD_DIMENSION;
    query.set(input.subarray(source, source + HEAD_DIMENSION), destination);
    gate.set(
      input.subarray(
        source + HEAD_DIMENSION,
        source + QUERY_GATE_RECORD_WIDTH,
      ),
      destination,
    );
  }
  return Object.freeze({ query, gate });
}

export function qwen35GqaKvHead(queryHead: number): number {
  if (
    !Number.isSafeInteger(queryHead) ||
    queryHead < 0 ||
    queryHead >= QUERY_HEAD_COUNT
  ) {
    throw new Error("Qwen3.5 query head is out of range");
  }
  return Math.floor(queryHead / (QUERY_HEAD_COUNT / KV_HEAD_COUNT));
}

export class FullAttentionCpuCache {
  readonly #capacity: number;
  readonly #keys: Uint16Array;
  readonly #values: Uint16Array;
  #position = 0;

  constructor(capacity: number) {
    if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > 16_384) {
      throw new Error("Full-attention cache capacity is invalid");
    }
    this.#capacity = capacity;
    this.#keys = new Uint16Array(capacity * KV_HEAD_COUNT * HEAD_DIMENSION);
    this.#values = new Uint16Array(this.#keys.length);
  }

  get capacity(): number {
    return this.#capacity;
  }

  get position(): number {
    return this.#position;
  }

  append(key: Float32Array, value: Float32Array): void {
    if (this.#position >= this.#capacity) {
      throw new Error("Full-attention cache capacity exceeded");
    }
    requireLength(key, KV_HEAD_COUNT * HEAD_DIMENSION, "Attention key");
    requireLength(value, KV_HEAD_COUNT * HEAD_DIMENSION, "Attention value");
    const offset = this.#position * KV_HEAD_COUNT * HEAD_DIMENSION;
    for (let lane = 0; lane < key.length; lane += 1) {
      this.#keys[offset + lane] = float16Bits(key[lane]!);
      this.#values[offset + lane] = float16Bits(value[lane]!);
    }
    this.#position += 1;
  }

  key(token: number, head: number, lane: number): number {
    this.#requireAddress(token, head, lane);
    return float16Value(
      this.#keys[
        (token * KV_HEAD_COUNT + head) * HEAD_DIMENSION + lane
      ]!,
    );
  }

  value(token: number, head: number, lane: number): number {
    this.#requireAddress(token, head, lane);
    return float16Value(
      this.#values[
        (token * KV_HEAD_COUNT + head) * HEAD_DIMENSION + lane
      ]!,
    );
  }

  reset(): void {
    this.#position = 0;
  }

  #requireAddress(token: number, head: number, lane: number): void {
    if (
      !Number.isSafeInteger(token) ||
      token < 0 ||
      token >= this.#position ||
      !Number.isSafeInteger(head) ||
      head < 0 ||
      head >= KV_HEAD_COUNT ||
      !Number.isSafeInteger(lane) ||
      lane < 0 ||
      lane >= HEAD_DIMENSION
    ) {
      throw new Error("Full-attention cache address is invalid");
    }
  }
}

function normalizedAndRotated(
  values: Float32Array,
  weight: Float32Array,
  headCount: number,
  positions: readonly [number, number, number],
): Float32Array {
  const normalized = qkRmsNormPerHeadCpu(values, weight, {
    headCount,
    headDimension: HEAD_DIMENSION,
    epsilon: NORM_EPSILON,
  });
  return partialMropeCpu(normalized, {
    headCount,
    headDimension: HEAD_DIMENSION,
    rotaryDimension: 64,
    sections: [11, 11, 10],
    positions,
    theta: 10_000_000,
  });
}

function validateFullAttentionToken(token: FullAttentionToken): void {
  requireLength(
    token.queryGate,
    QUERY_HEAD_COUNT * QUERY_GATE_RECORD_WIDTH,
    "Q/gate projection",
  );
  requireLength(token.key, KV_HEAD_COUNT * HEAD_DIMENSION, "Attention key");
  requireLength(token.value, KV_HEAD_COUNT * HEAD_DIMENSION, "Attention value");
  requireLength(token.queryNormWeight, HEAD_DIMENSION, "Q norm weight");
  requireLength(token.keyNormWeight, HEAD_DIMENSION, "K norm weight");
  if (
    token.positions.some(
      (position) =>
        !Number.isSafeInteger(position) || position < 0,
    )
  ) {
    throw new Error("Attention M-RoPE positions are invalid");
  }
}

function onlineAttentionHead(
  query: Float32Array,
  tokenCount: number,
  keyAt: (token: number, lane: number) => number,
  valueAt: (token: number, lane: number) => number,
): Float32Array {
  const accumulator = new Float32Array(HEAD_DIMENSION);
  let runningMaximum = Number.NEGATIVE_INFINITY;
  let runningDenominator = Math.fround(0);
  for (let contextToken = 0; contextToken < tokenCount; contextToken += 1) {
    let dot = Math.fround(0);
    for (let lane = 0; lane < HEAD_DIMENSION; lane += 1) {
      dot = Math.fround(
        dot +
          Math.fround(query[lane]! * keyAt(contextToken, lane)),
      );
    }
    const score = Math.fround(dot * QUERY_SCALE);
    const nextMaximum = Math.max(runningMaximum, score);
    const oldScale =
      runningMaximum === Number.NEGATIVE_INFINITY
        ? Math.fround(0)
        : Math.fround(
            Math.exp(Math.fround(runningMaximum - nextMaximum)),
          );
    const tokenScale = Math.fround(
      Math.exp(Math.fround(score - nextMaximum)),
    );
    runningDenominator = Math.fround(
      Math.fround(runningDenominator * oldScale) + tokenScale,
    );
    for (let lane = 0; lane < HEAD_DIMENSION; lane += 1) {
      accumulator[lane] = Math.fround(
        Math.fround(accumulator[lane]! * oldScale) +
          Math.fround(
            tokenScale * Math.fround(valueAt(contextToken, lane)),
          ),
      );
    }
    runningMaximum = nextMaximum;
  }
  return Float32Array.from(accumulator, (value) =>
    Math.fround(value / runningDenominator),
  );
}

export function qwen35OnlineAttentionHeadCpu(
  query: Float32Array,
  keys: Float32Array,
  values: Float32Array,
): Float32Array {
  requireLength(query, HEAD_DIMENSION, "Attention query head");
  if (
    keys.length === 0 ||
    keys.length % HEAD_DIMENSION !== 0 ||
    values.length !== keys.length
  ) {
    throw new Error("Online attention K/V shape mismatch");
  }
  const tokenCount = keys.length / HEAD_DIMENSION;
  return onlineAttentionHead(
    query,
    tokenCount,
    (token, lane) => keys[token * HEAD_DIMENSION + lane]!,
    (token, lane) => values[token * HEAD_DIMENSION + lane]!,
  );
}

/**
 * Computes one causal token with a running max, denominator, and value
 * accumulator. Production kernels use the same recurrence and never allocate
 * a context-length score matrix.
 */
export function fullAttentionDecodeCpu(
  cache: FullAttentionCpuCache,
  token: FullAttentionToken,
): Float32Array {
  if (cache.position >= cache.capacity) {
    throw new Error("Full-attention cache capacity exceeded");
  }
  validateFullAttentionToken(token);

  const split = splitQwen35QueryGateProjection(token.queryGate);
  const query = normalizedAndRotated(
    split.query,
    token.queryNormWeight,
    QUERY_HEAD_COUNT,
    token.positions,
  );
  const key = normalizedAndRotated(
    token.key,
    token.keyNormWeight,
    KV_HEAD_COUNT,
    token.positions,
  );
  // The prepare shader publishes K/V as binary16 before online attention reads
  // the current row. Quantize this row now so the CPU oracle models both the
  // current token and all historical rows with the production representation.
  const quantizedKey = quantizeFloat16ArrayCpu(key);
  const quantizedValue = quantizeFloat16ArrayCpu(token.value);
  const attention = new Float32Array(QUERY_HEAD_COUNT * HEAD_DIMENSION);
  const lastToken = cache.position;

  for (let queryHead = 0; queryHead < QUERY_HEAD_COUNT; queryHead += 1) {
    const kvHead = qwen35GqaKvHead(queryHead);
    const queryOffset = queryHead * HEAD_DIMENSION;
    const headOutput = onlineAttentionHead(
      query.subarray(queryOffset, queryOffset + HEAD_DIMENSION),
      lastToken + 1,
      (contextToken, lane) =>
        contextToken === lastToken
          ? quantizedKey[kvHead * HEAD_DIMENSION + lane]!
          : cache.key(contextToken, kvHead, lane),
      (contextToken, lane) =>
        contextToken === lastToken
          ? quantizedValue[kvHead * HEAD_DIMENSION + lane]!
          : cache.value(contextToken, kvHead, lane),
    );
    attention.set(headOutput, queryOffset);
  }

  cache.append(quantizedKey, quantizedValue);
  return attentionOutputGateCpu(attention, split.gate);
}

export function fullAttentionPrefillCpu(
  cache: FullAttentionCpuCache,
  tokens: readonly FullAttentionToken[],
): readonly Float32Array[] {
  if (tokens.length > cache.capacity - cache.position) {
    throw new Error("Full-attention prefill exceeds cache capacity");
  }
  // Validate the complete batch before the first decode. This keeps a bad
  // later token from leaving a valid-looking but partially advanced cache.
  for (const token of tokens) {
    validateFullAttentionToken(token);
  }
  return Object.freeze(
    tokens.map((token) => fullAttentionDecodeCpu(cache, token)),
  );
}

const FLOAT32_VIEW = new Float32Array(1);
const UINT32_VIEW = new Uint32Array(FLOAT32_VIEW.buffer);

function float16Bits(value: number): number {
  FLOAT32_VIEW[0] = value;
  const word = UINT32_VIEW[0]!;
  const sign = (word >>> 16) & 0x8000;
  const mantissa = word & 0x007f_ffff;
  const exponent = (word >> 23) & 0xff;

  if (exponent === 0xff) {
    // JavaScript preserves a NaN sign but not its payload. Use the canonical
    // quiet payload required by DataView.setFloat16 and keep infinities exact.
    return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  }
  if (exponent > 142) {
    return sign | 0x7c00;
  }
  if (exponent >= 113) {
    // Add one less than half an f16 ULP, then add the retained low bit. This
    // makes exact halfway cases round to the even retained significand.
    const rounded =
      mantissa + 0x0fff + ((mantissa >>> 13) & 1);
    return (
      sign |
      (((exponent - 112) << 10) + (rounded >>> 13))
    );
  }
  if (exponent < 102) {
    return sign;
  }

  const significand = mantissa | 0x0080_0000;
  const shift = 126 - exponent;
  const divisor = 2 ** shift;
  const quotient = Math.floor(significand / divisor);
  const remainder = significand - quotient * divisor;
  const halfway = divisor / 2;
  const rounded =
    remainder > halfway ||
    (remainder === halfway && (quotient & 1) !== 0)
      ? quotient + 1
      : quotient;
  return sign | rounded;
}

function float16Value(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0x1f) {
    return mantissa === 0
      ? sign * Number.POSITIVE_INFINITY
      : Number.NaN;
  }
  if (exponent === 0) {
    return Math.fround(sign * mantissa * 2 ** -24);
  }
  return Math.fround(
    sign * (1 + mantissa / 1_024) * 2 ** (exponent - 15),
  );
}

function quantizeFloat16ArrayCpu(values: Float32Array): Float32Array {
  return Float32Array.from(values, (value) =>
    float16Value(float16Bits(value)),
  );
}

export function packFloat16PairCpu(first: number, second: number): number {
  return (float16Bits(first) | (float16Bits(second) << 16)) >>> 0;
}

export function unpackFloat16PairCpu(
  packed: number,
): readonly [number, number] {
  if (!Number.isSafeInteger(packed) || packed < 0 || packed > 0xffff_ffff) {
    throw new Error("Packed FP16 pair must be a u32");
  }
  return Object.freeze([
    float16Value(packed & 0xffff),
    float16Value(packed >>> 16),
  ]);
}

/**
 * Mirrors the prepare shader's one-token packed cache publication. Validation
 * completes before the first write so an invalid position cannot corrupt a
 * prior cache row or its suffix sentinel.
 */
export function writeFullAttentionKvCpu(
  packedKeys: Uint32Array,
  packedValues: Uint32Array,
  position: number,
  capacity: number,
  key: Float32Array,
  value: Float32Array,
): void {
  if (
    !Number.isSafeInteger(capacity) ||
    capacity < 1 ||
    capacity > 16_384
  ) {
    throw new Error("Full-attention packed K/V capacity is invalid");
  }
  if (
    !Number.isSafeInteger(position) ||
    position < 0 ||
    position >= capacity
  ) {
    throw new Error("Full-attention packed K/V position is invalid");
  }
  requireLength(key, KV_HEAD_COUNT * HEAD_DIMENSION, "Packed attention key");
  requireLength(value, KV_HEAD_COUNT * HEAD_DIMENSION, "Packed attention value");
  const requiredWords = capacity * KV_HEAD_COUNT * HEAD_DIMENSION / 2;
  if (
    packedKeys.length < requiredWords ||
    packedValues.length < requiredWords
  ) {
    throw new Error("Full-attention packed K/V storage is too small");
  }
  const wordOffset = position * KV_HEAD_COUNT * HEAD_DIMENSION / 2;
  for (let scalar = 0; scalar < key.length; scalar += 2) {
    const word = wordOffset + scalar / 2;
    packedKeys[word] = packFloat16PairCpu(key[scalar]!, key[scalar + 1]!);
    packedValues[word] = packFloat16PairCpu(
      value[scalar]!,
      value[scalar + 1]!,
    );
  }
}
