import assert from "node:assert/strict";
import test from "node:test";

async function loadDeltaNetModule(): Promise<Record<string, unknown>> {
  return import("../src/gated-deltanet.js").catch(() => ({}));
}

test("splits the convolved QKV projection at 2048, 2048, and 4096", async () => {
  const module = await loadDeltaNetModule();
  assert.equal(typeof module.splitQwen35DeltaNetQkv, "function");
  const input = Float32Array.from({ length: 8_192 }, (_, index) => index);

  const split = (
    module.splitQwen35DeltaNetQkv as (input: Float32Array) => {
      readonly query: Float32Array;
      readonly key: Float32Array;
      readonly value: Float32Array;
    }
  )(input);

  assert.equal(split.query.length, 2_048);
  assert.equal(split.key.length, 2_048);
  assert.equal(split.value.length, 4_096);
  assert.equal(split.query[2_047], 2_047);
  assert.equal(split.key[0], 2_048);
  assert.equal(split.key[2_047], 4_095);
  assert.equal(split.value[0], 4_096);
  assert.equal(split.value[4_095], 8_191);
});

test("moves an impulse from convolution tap 3 through tap 0", async () => {
  const module = await loadDeltaNetModule();
  assert.equal(typeof module.DeltaNetConvCpuState, "function");
  assert.equal(typeof module.deltaNetConvStepCpu, "function");
  const State = module.DeltaNetConvCpuState as new () => object;
  const step = module.deltaNetConvStepCpu as (
    state: object,
    input: Float32Array,
    weights: Float32Array,
  ) => Float32Array;
  const state = new State();
  const weights = new Float32Array(8_192 * 4);
  weights.set([1, 2, 3, 4], 0);
  const impulse = new Float32Array(8_192);
  impulse[0] = 1;
  const zero = new Float32Array(8_192);

  const observed = [
    step(state, impulse, weights)[0]!,
    step(state, zero, weights)[0]!,
    step(state, zero, weights)[0]!,
    step(state, zero, weights)[0]!,
  ];
  const silu = (value: number) => value / (1 + Math.exp(-value));

  assert.deepEqual(
    observed.map((value) => Math.round(value * 1e5)),
    [silu(4), silu(3), silu(2), silu(1)].map((value) =>
      Math.round(value * 1e5),
    ),
  );
});

test("maps tiled value heads to Q/K heads with h modulo 16", async () => {
  const module = await loadDeltaNetModule();
  assert.equal(typeof module.qwen35DeltaNetQkHead, "function");
  const map = module.qwen35DeltaNetQkHead as (head: number) => number;

  assert.deepEqual(
    Array.from({ length: 32 }, (_, head) => map(head)),
    [...Array.from({ length: 16 }, (_, head) => head), ...Array.from({ length: 16 }, (_, head) => head)],
  );
  assert.throws(() => map(32), /value head/i);
});

test("decays, reads, applies beta delta, updates, then queries state", async () => {
  const module = await loadDeltaNetModule();
  assert.equal(typeof module.deltaNetRecurrentHeadStepCpu, "function");
  const state = new Float32Array(128 * 128);
  state[0] = 2;
  const query = new Float32Array(128);
  const key = new Float32Array(128);
  const value = new Float32Array(128);
  query[0] = 1;
  key[0] = 1;
  value[0] = 10;

  const output = (
    module.deltaNetRecurrentHeadStepCpu as (
      state: Float32Array,
      query: Float32Array,
      key: Float32Array,
      value: Float32Array,
      beta: number,
      decay: number,
    ) => Float32Array
  )(state, query, key, value, 0.25, 0.5);

  // decay 2 -> 1; delta=.25*(10-1)=2.25; update -> 3.25; query -> 3.25
  assert.ok(Math.abs(output[0]! - 3.25) < 1e-6);
  assert.ok(Math.abs(state[0]! - 3.25) < 1e-6);
});

test("matches explicit f32 beta and decay parameter stages", async () => {
  const module = await loadDeltaNetModule();
  assert.equal(typeof module.qwen35DeltaNetParametersCpu, "function");
  const result = (
    module.qwen35DeltaNetParametersCpu as (
      betaInput: number,
      a: number,
      dt: number,
      ssmA: number,
    ) => { readonly beta: number; readonly decay: number }
  )(
    -13.173408508300781,
    7.812345504760742,
    -7.611111164093018,
    -0.7312344908714294,
  );

  assert.equal(result.beta, 0.0000019004679643330746);
  assert.equal(Number.isFinite(result.decay), true);
});

test("rejects invalid parameters before mutating convolution state", async () => {
  const module = await loadDeltaNetModule();
  const State = module.GatedDeltaNetCpuState as new (capacity: number) => object;
  const decode = module.gatedDeltaNetDecodeCpu as (
    state: object,
    token: DeltaNetToken,
  ) => Float32Array;
  const expectedState = new State(1);
  const actualState = new State(1);
  const valid = makeDeltaNetToken(0);
  const invalid = makeDeltaNetToken(0);
  invalid.ssmA[0] = 0.25;

  assert.throws(() => decode(actualState, invalid), /ssm_a/i);
  assert.deepEqual(decode(actualState, valid), decode(expectedState, valid));

  const nanState = new State(1);
  const nanToken = makeDeltaNetToken(0);
  nanToken.qkv[0] = Number.NaN;
  assert.throws(() => decode(nanState, nanToken), /finite/i);
});

test("serial DeltaNet prefill equals repeated decode from zero and nonzero state", async () => {
  const module = await loadDeltaNetModule();
  assert.equal(typeof module.GatedDeltaNetCpuState, "function");
  assert.equal(typeof module.gatedDeltaNetDecodeCpu, "function");
  assert.equal(typeof module.gatedDeltaNetPrefillCpu, "function");
  const State = module.GatedDeltaNetCpuState as new (
    capacity: number,
  ) => {
    seedRecurrent(
      head: number,
      keyLane: number,
      valueLane: number,
      value: number,
    ): void;
  };
  const decode = module.gatedDeltaNetDecodeCpu as (
    state: object,
    token: DeltaNetToken,
  ) => Float32Array;
  const prefill = module.gatedDeltaNetPrefillCpu as (
    state: object,
    tokens: readonly DeltaNetToken[],
  ) => readonly Float32Array[];

  const tokens = [makeDeltaNetToken(0), makeDeltaNetToken(1)];
  for (const seed of [0, 0.75]) {
    const decodeState = new State(2);
    const prefillState = new State(2);
    if (seed !== 0) {
      decodeState.seedRecurrent(7, 3, 5, seed);
      prefillState.seedRecurrent(7, 3, 5, seed);
    }

    const expected = tokens.map((token) => decode(decodeState, token));
    const actual = prefill(prefillState, tokens);
    assert.deepEqual(actual, expected);
  }
});

test("DeltaNet prefill validates every token before changing state", async () => {
  const module = await loadDeltaNetModule();
  const State = module.GatedDeltaNetCpuState as new (
    capacity: number,
  ) => {
    readonly position: number;
  };
  const decode = module.gatedDeltaNetDecodeCpu as (
    state: object,
    token: DeltaNetToken,
  ) => Float32Array;
  const prefill = module.gatedDeltaNetPrefillCpu as (
    state: object,
    tokens: readonly DeltaNetToken[],
  ) => readonly Float32Array[];
  const valid = makeDeltaNetToken(0);
  const invalid = makeDeltaNetToken(1);
  invalid.ssmA[7] = 0;
  const actualState = new State(2);
  const expectedState = new State(1);

  assert.throws(() => prefill(actualState, [valid, invalid]), /ssm_a/i);
  assert.equal(actualState.position, 0);
  assert.deepEqual(
    decode(actualState, valid),
    decode(expectedState, valid),
  );
});

test("keeps the correctness-first prefill plan serial at critical boundaries", async () => {
  const module = await loadDeltaNetModule();
  assert.equal(typeof module.planSerialDeltaNetPrefill, "function");
  const plan = module.planSerialDeltaNetPrefill as (
    length: number,
  ) => readonly number[];

  for (const length of [1, 2, 4, 63, 64, 65]) {
    assert.deepEqual(plan(length), Array.from({ length }, (_, index) => index));
  }
});

interface DeltaNetToken {
  readonly qkv: Float32Array;
  readonly convWeight: Float32Array;
  readonly beta: Float32Array;
  readonly a: Float32Array;
  readonly dt: Float32Array;
  readonly ssmA: Float32Array;
  readonly z: Float32Array;
  readonly normWeight: Float32Array;
}

function makeDeltaNetToken(token: number): DeltaNetToken {
  const qkv = new Float32Array(8_192);
  for (let head = 0; head < 16; head += 1) {
    qkv[head * 128 + ((head + token) % 128)] = 1;
    qkv[2_048 + head * 128 + ((head * 3 + token) % 128)] = 1;
  }
  for (let head = 0; head < 32; head += 1) {
    qkv[4_096 + head * 128 + ((head * 5 + token) % 128)] =
      (head + token + 1) / 32;
  }
  const convWeight = new Float32Array(8_192 * 4);
  for (let channel = 0; channel < 8_192; channel += 1) {
    convWeight[channel * 4 + 3] = 1;
  }
  return {
    qkv,
    convWeight,
    beta: new Float32Array(32).fill(0),
    a: new Float32Array(32).fill(0.25),
    dt: new Float32Array(32).fill(-0.1),
    ssmA: new Float32Array(32).fill(-0.5),
    z: new Float32Array(4_096).fill(1),
    normWeight: new Float32Array(128).fill(1),
  };
}
