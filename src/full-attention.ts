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
  readonly #keys: Float32Array;
  readonly #values: Float32Array;
  #position = 0;

  constructor(capacity: number) {
    if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > 16_384) {
      throw new Error("Full-attention cache capacity is invalid");
    }
    this.#capacity = capacity;
    this.#keys = new Float32Array(capacity * KV_HEAD_COUNT * HEAD_DIMENSION);
    this.#values = new Float32Array(this.#keys.length);
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
    this.#keys.set(key, offset);
    this.#values.set(value, offset);
    this.#position += 1;
  }

  key(token: number, head: number, lane: number): number {
    this.#requireAddress(token, head, lane);
    return this.#keys[
      (token * KV_HEAD_COUNT + head) * HEAD_DIMENSION + lane
    ]!;
  }

  value(token: number, head: number, lane: number): number {
    this.#requireAddress(token, head, lane);
    return this.#values[
      (token * KV_HEAD_COUNT + head) * HEAD_DIMENSION + lane
    ]!;
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
  requireLength(
    token.queryGate,
    QUERY_HEAD_COUNT * QUERY_GATE_RECORD_WIDTH,
    "Q/gate projection",
  );
  requireLength(token.key, KV_HEAD_COUNT * HEAD_DIMENSION, "Attention key");
  requireLength(token.value, KV_HEAD_COUNT * HEAD_DIMENSION, "Attention value");
  requireLength(token.queryNormWeight, HEAD_DIMENSION, "Q norm weight");
  requireLength(token.keyNormWeight, HEAD_DIMENSION, "K norm weight");

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
          ? key[kvHead * HEAD_DIMENSION + lane]!
          : cache.key(contextToken, kvHead, lane),
      (contextToken, lane) =>
        contextToken === lastToken
          ? token.value[kvHead * HEAD_DIMENSION + lane]!
          : cache.value(contextToken, kvHead, lane),
    );
    attention.set(headOutput, queryOffset);
  }

  cache.append(key, token.value);
  return attentionOutputGateCpu(attention, split.gate);
}

export function fullAttentionPrefillCpu(
  cache: FullAttentionCpuCache,
  tokens: readonly FullAttentionToken[],
): readonly Float32Array[] {
  if (tokens.length > cache.capacity - cache.position) {
    throw new Error("Full-attention prefill exceeds cache capacity");
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
  let bits = (word >> 16) & 0x8000;
  let mantissa = (word >> 12) & 0x07ff;
  const exponent = (word >> 23) & 0xff;
  if (exponent < 103) {
    return bits;
  }
  if (exponent > 142) {
    bits |= 0x7c00;
    bits |= exponent === 255 && (word & 0x007f_ffff) !== 0 ? 1 : 0;
    return bits;
  }
  if (exponent < 113) {
    mantissa |= 0x0800;
    bits |=
      (mantissa >> (114 - exponent)) +
      ((mantissa >> (113 - exponent)) & 1);
    return bits;
  }
  bits |= ((exponent - 112) << 10) | (mantissa >> 1);
  bits += mantissa & 1;
  return bits;
}

export function packFloat16PairCpu(first: number, second: number): number {
  return (float16Bits(first) | (float16Bits(second) << 16)) >>> 0;
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
