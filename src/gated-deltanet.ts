const QK_HEAD_COUNT = 16;
const VALUE_HEAD_COUNT = 32;
const HEAD_DIMENSION = 128;
const Q_WIDTH = QK_HEAD_COUNT * HEAD_DIMENSION;
const K_WIDTH = Q_WIDTH;
const V_WIDTH = VALUE_HEAD_COUNT * HEAD_DIMENSION;
const QKV_WIDTH = Q_WIDTH + K_WIDTH + V_WIDTH;
const CONV_TAPS = 4;
const NORM_EPSILON = Math.fround(1e-6);
const QUERY_SCALE = Math.fround(1 / Math.sqrt(HEAD_DIMENSION));

export interface DeltaNetQkv {
  readonly query: Float32Array;
  readonly key: Float32Array;
  readonly value: Float32Array;
}

export interface GatedDeltaNetToken {
  readonly qkv: Float32Array;
  /**
   * GGUF `ssm_conv1d.weight` is converted to channel-major
   * `[tap0 oldest, tap1, tap2, tap3 current]` records.
   */
  readonly convWeight: Float32Array;
  readonly beta: Float32Array;
  readonly a: Float32Array;
  readonly dt: Float32Array;
  /** Converted GGUF values are already negative; do not negate them again. */
  readonly ssmA: Float32Array;
  readonly z: Float32Array;
  /** Direct multiplicative `ssm_norm` weights from the GGUF tensor. */
  readonly normWeight: Float32Array;
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

function sigmoid(value: number): number {
  const rounded = Math.fround(value);
  if (value >= 0) {
    const exponential = Math.fround(Math.exp(Math.fround(-rounded)));
    return Math.fround(
      1 / Math.fround(Math.fround(1) + exponential),
    );
  }
  const exponential = Math.fround(Math.exp(rounded));
  return Math.fround(
    exponential / Math.fround(Math.fround(1) + exponential),
  );
}

function silu(value: number): number {
  return Math.fround(value * sigmoid(value));
}

function softplus(value: number): number {
  const rounded = Math.fround(value);
  const exponential = Math.fround(
    Math.exp(Math.fround(-Math.fround(Math.abs(rounded)))),
  );
  const logarithm = Math.fround(Math.log1p(exponential));
  return Math.fround(Math.fround(Math.max(rounded, 0)) + logarithm);
}

export function qwen35DeltaNetParametersCpu(
  betaInput: number,
  a: number,
  dt: number,
  ssmA: number,
): { readonly beta: number; readonly decay: number } {
  if (
    !Number.isFinite(betaInput) ||
    !Number.isFinite(a) ||
    !Number.isFinite(dt) ||
    !Number.isFinite(ssmA) ||
    ssmA >= 0
  ) {
    throw new Error("DeltaNet ssm_a and parameter values are invalid");
  }
  const sum = Math.fround(Math.fround(a) + Math.fround(dt));
  const gate = Math.fround(Math.fround(ssmA) * softplus(sum));
  return Object.freeze({
    beta: sigmoid(betaInput),
    decay: Math.fround(Math.exp(gate)),
  });
}

export function splitQwen35DeltaNetQkv(input: Float32Array): DeltaNetQkv {
  requireLength(input, QKV_WIDTH, "DeltaNet QKV projection");
  return Object.freeze({
    query: input.slice(0, Q_WIDTH),
    key: input.slice(Q_WIDTH, Q_WIDTH + K_WIDTH),
    value: input.slice(Q_WIDTH + K_WIDTH),
  });
}

export function qwen35DeltaNetQkHead(valueHead: number): number {
  if (
    !Number.isSafeInteger(valueHead) ||
    valueHead < 0 ||
    valueHead >= VALUE_HEAD_COUNT
  ) {
    throw new Error("DeltaNet value head is out of range");
  }
  return valueHead % QK_HEAD_COUNT;
}

export class DeltaNetConvCpuState {
  readonly #values = new Float32Array(QKV_WIDTH * CONV_TAPS);

  shiftAndInsert(input: Float32Array): void {
    requireLength(input, QKV_WIDTH, "DeltaNet raw QKV");
    for (let channel = 0; channel < QKV_WIDTH; channel += 1) {
      const base = channel * CONV_TAPS;
      this.#values[base] = this.#values[base + 1]!;
      this.#values[base + 1] = this.#values[base + 2]!;
      this.#values[base + 2] = this.#values[base + 3]!;
      this.#values[base + 3] = input[channel]!;
    }
  }

  value(channel: number, tap: number): number {
    return this.#values[channel * CONV_TAPS + tap]!;
  }

  reset(): void {
    this.#values.fill(0);
  }
}

export function deltaNetConvStepCpu(
  state: DeltaNetConvCpuState,
  input: Float32Array,
  weights: Float32Array,
): Float32Array {
  requireLength(input, QKV_WIDTH, "DeltaNet raw QKV");
  requireLength(weights, QKV_WIDTH * CONV_TAPS, "DeltaNet convolution weight");
  state.shiftAndInsert(input);
  const output = new Float32Array(QKV_WIDTH);
  for (let channel = 0; channel < QKV_WIDTH; channel += 1) {
    let sum = Math.fround(0);
    const base = channel * CONV_TAPS;
    for (let tap = 0; tap < CONV_TAPS; tap += 1) {
      sum = Math.fround(
        sum +
          Math.fround(state.value(channel, tap) * weights[base + tap]!),
      );
    }
    output[channel] = silu(sum);
  }
  return output;
}

function normalizedHead(
  values: Float32Array,
  offset: number,
  query: boolean,
): Float32Array {
  let sumSquares = Math.fround(0);
  for (let lane = 0; lane < HEAD_DIMENSION; lane += 1) {
    const value = values[offset + lane]!;
    sumSquares = Math.fround(
      sumSquares + Math.fround(value * value),
    );
  }
  const inverseLength = Math.fround(
    1 / Math.sqrt(Math.fround(sumSquares + NORM_EPSILON)),
  );
  const scale = query
    ? Math.fround(inverseLength * QUERY_SCALE)
    : inverseLength;
  return Float32Array.from(
    { length: HEAD_DIMENSION },
    (_, lane) => Math.fround(values[offset + lane]! * scale),
  );
}

/**
 * Executes the exact update order on one FP32 `[key, value]` state matrix.
 * The input Q/K vectors are already L2-normalized by the caller.
 */
export function deltaNetRecurrentHeadStepCpu(
  state: Float32Array,
  query: Float32Array,
  key: Float32Array,
  value: Float32Array,
  beta: number,
  decay: number,
): Float32Array {
  requireLength(state, HEAD_DIMENSION * HEAD_DIMENSION, "DeltaNet recurrent state");
  requireLength(query, HEAD_DIMENSION, "DeltaNet query head");
  requireLength(key, HEAD_DIMENSION, "DeltaNet key head");
  requireLength(value, HEAD_DIMENSION, "DeltaNet value head");
  if (
    !Number.isFinite(beta) ||
    beta < 0 ||
    beta > 1 ||
    !Number.isFinite(decay) ||
    decay < 0
  ) {
    throw new Error("DeltaNet recurrent parameters are invalid");
  }

  const memory = new Float32Array(HEAD_DIMENSION);
  for (let keyLane = 0; keyLane < HEAD_DIMENSION; keyLane += 1) {
    const keyValue = key[keyLane]!;
    const row = keyLane * HEAD_DIMENSION;
    for (let valueLane = 0; valueLane < HEAD_DIMENSION; valueLane += 1) {
      const index = row + valueLane;
      const decayed = Math.fround(state[index]! * decay);
      state[index] = decayed;
      memory[valueLane] = Math.fround(
        memory[valueLane]! + Math.fround(keyValue * decayed),
      );
    }
  }

  const delta = Float32Array.from(
    value,
    (target, lane) =>
      Math.fround(beta * Math.fround(target - memory[lane]!)),
  );
  const output = new Float32Array(HEAD_DIMENSION);
  for (let keyLane = 0; keyLane < HEAD_DIMENSION; keyLane += 1) {
    const row = keyLane * HEAD_DIMENSION;
    const keyValue = key[keyLane]!;
    const queryValue = query[keyLane]!;
    for (let valueLane = 0; valueLane < HEAD_DIMENSION; valueLane += 1) {
      const index = row + valueLane;
      const updated = Math.fround(
        state[index]! + Math.fround(keyValue * delta[valueLane]!),
      );
      state[index] = updated;
      output[valueLane] = Math.fround(
        output[valueLane]! + Math.fround(queryValue * updated),
      );
    }
  }
  return output;
}

function rmsNormHead(
  values: Float32Array,
  weight: Float32Array,
): Float32Array {
  let sum = Math.fround(0);
  for (const value of values) {
    sum = Math.fround(sum + Math.fround(value * value));
  }
  const inverseRms = Math.fround(
    1 /
      Math.sqrt(
        Math.fround(
          Math.fround(sum / HEAD_DIMENSION) + NORM_EPSILON,
        ),
      ),
  );
  return Float32Array.from(values, (value, lane) =>
    Math.fround(Math.fround(value * inverseRms) * weight[lane]!),
  );
}

export class GatedDeltaNetCpuState {
  readonly #capacity: number;
  readonly #conv = new DeltaNetConvCpuState();
  readonly #recurrent = new Float32Array(
    VALUE_HEAD_COUNT * HEAD_DIMENSION * HEAD_DIMENSION,
  );
  #position = 0;

  constructor(capacity: number) {
    if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > 16_384) {
      throw new Error("DeltaNet state capacity is invalid");
    }
    this.#capacity = capacity;
  }

  get capacity(): number {
    return this.#capacity;
  }

  get position(): number {
    return this.#position;
  }

  conv(): DeltaNetConvCpuState {
    return this.#conv;
  }

  recurrentHead(head: number): Float32Array {
    if (
      !Number.isSafeInteger(head) ||
      head < 0 ||
      head >= VALUE_HEAD_COUNT
    ) {
      throw new Error("DeltaNet recurrent head is out of range");
    }
    const start = head * HEAD_DIMENSION * HEAD_DIMENSION;
    return this.#recurrent.subarray(
      start,
      start + HEAD_DIMENSION * HEAD_DIMENSION,
    );
  }

  seedRecurrent(
    head: number,
    keyLane: number,
    valueLane: number,
    value: number,
  ): void {
    if (
      !Number.isSafeInteger(keyLane) ||
      keyLane < 0 ||
      keyLane >= HEAD_DIMENSION ||
      !Number.isSafeInteger(valueLane) ||
      valueLane < 0 ||
      valueLane >= HEAD_DIMENSION ||
      !Number.isFinite(value)
    ) {
      throw new Error("DeltaNet recurrent seed is invalid");
    }
    this.recurrentHead(head)[keyLane * HEAD_DIMENSION + valueLane] = value;
  }

  commitToken(): void {
    if (this.#position >= this.#capacity) {
      throw new Error("DeltaNet state capacity exceeded");
    }
    this.#position += 1;
  }

  reset(): void {
    this.#conv.reset();
    this.#recurrent.fill(0);
    this.#position = 0;
  }
}

function validateToken(token: GatedDeltaNetToken): void {
  requireLength(token.qkv, QKV_WIDTH, "DeltaNet QKV projection");
  requireLength(
    token.convWeight,
    QKV_WIDTH * CONV_TAPS,
    "DeltaNet convolution weight",
  );
  for (const [values, label] of [
    [token.beta, "DeltaNet beta"],
    [token.a, "DeltaNet a"],
    [token.dt, "DeltaNet dt"],
    [token.ssmA, "DeltaNet ssm_a"],
  ] as const) {
    requireLength(values, VALUE_HEAD_COUNT, label);
  }
  requireLength(token.z, V_WIDTH, "DeltaNet z");
  requireLength(token.normWeight, HEAD_DIMENSION, "DeltaNet norm weight");
  for (const [values, label] of [
    [token.qkv, "DeltaNet QKV projection"],
    [token.convWeight, "DeltaNet convolution weight"],
    [token.beta, "DeltaNet beta"],
    [token.a, "DeltaNet a"],
    [token.dt, "DeltaNet dt"],
    [token.z, "DeltaNet z"],
    [token.normWeight, "DeltaNet norm weight"],
  ] as const) {
    if (!values.every(Number.isFinite)) {
      throw new Error(`${label} values must be finite`);
    }
  }
  if (
    !token.ssmA.every((value) => Number.isFinite(value) && value < 0)
  ) {
    throw new Error("DeltaNet ssm_a values must be finite and negative");
  }
}

export function gatedDeltaNetDecodeCpu(
  state: GatedDeltaNetCpuState,
  token: GatedDeltaNetToken,
): Float32Array {
  if (state.position >= state.capacity) {
    throw new Error("DeltaNet state capacity exceeded");
  }
  validateToken(token);
  const parameters = Array.from({ length: VALUE_HEAD_COUNT }, (_, head) =>
    qwen35DeltaNetParametersCpu(
      token.beta[head]!,
      token.a[head]!,
      token.dt[head]!,
      token.ssmA[head]!,
    ),
  );
  const convolved = deltaNetConvStepCpu(
    state.conv(),
    token.qkv,
    token.convWeight,
  );
  const split = splitQwen35DeltaNetQkv(convolved);
  const output = new Float32Array(V_WIDTH);

  for (let valueHead = 0; valueHead < VALUE_HEAD_COUNT; valueHead += 1) {
    const qkHead = qwen35DeltaNetQkHead(valueHead);
    const query = normalizedHead(
      split.query,
      qkHead * HEAD_DIMENSION,
      true,
    );
    const key = normalizedHead(
      split.key,
      qkHead * HEAD_DIMENSION,
      false,
    );
    const value = split.value.subarray(
      valueHead * HEAD_DIMENSION,
      (valueHead + 1) * HEAD_DIMENSION,
    );
    const { beta, decay } = parameters[valueHead]!;
    const recurrentOutput = deltaNetRecurrentHeadStepCpu(
      state.recurrentHead(valueHead),
      query,
      key,
      value,
      beta,
      decay,
    );
    const normalized = rmsNormHead(recurrentOutput, token.normWeight);
    const offset = valueHead * HEAD_DIMENSION;
    for (let lane = 0; lane < HEAD_DIMENSION; lane += 1) {
      output[offset + lane] = Math.fround(
        normalized[lane]! * silu(token.z[offset + lane]!),
      );
    }
  }
  state.commitToken();
  return output;
}

export function gatedDeltaNetPrefillCpu(
  state: GatedDeltaNetCpuState,
  tokens: readonly GatedDeltaNetToken[],
): readonly Float32Array[] {
  if (tokens.length > state.capacity - state.position) {
    throw new Error("DeltaNet prefill exceeds state capacity");
  }
  // Decode mutates convolution and recurrent state. Validate the full batch
  // first so a malformed later token cannot leave a partial prefix committed.
  for (const token of tokens) {
    validateToken(token);
  }
  return Object.freeze(
    tokens.map((token) => gatedDeltaNetDecodeCpu(state, token)),
  );
}

export function planSerialDeltaNetPrefill(
  tokenCount: number,
): readonly number[] {
  if (!Number.isSafeInteger(tokenCount) || tokenCount < 1 || tokenCount > 16_384) {
    throw new Error("DeltaNet prefill token count is invalid");
  }
  // This explicit serial plan is the correctness oracle. A future bounded
  // parallel scan must prove parity before it can replace these token steps.
  return Object.freeze(
    Array.from({ length: tokenCount }, (_, token) => token),
  );
}
