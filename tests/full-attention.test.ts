import assert from "node:assert/strict";
import test from "node:test";

async function loadAttentionModule(): Promise<Record<string, unknown>> {
  return import("../src/full-attention.js").catch(() => ({}));
}

function makeProjection(
  qValue: (head: number, lane: number) => number,
  gateValue: (head: number, lane: number) => number,
): Float32Array {
  const projection = new Float32Array(16 * 512);
  for (let head = 0; head < 16; head += 1) {
    const record = head * 512;
    for (let lane = 0; lane < 256; lane += 1) {
      projection[record + lane] = qValue(head, lane);
      projection[record + 256 + lane] = gateValue(head, lane);
    }
  }
  return projection;
}

test("splits Q and gate as interleaved per-head records", async () => {
  const module = await loadAttentionModule();
  assert.equal(typeof module.splitQwen35QueryGateProjection, "function");
  const projection = makeProjection(
    (head, lane) => head * 1_000 + lane,
    (head, lane) => 100_000 + head * 1_000 + lane,
  );

  const { query, gate } = (
    module.splitQwen35QueryGateProjection as (input: Float32Array) => {
      readonly query: Float32Array;
      readonly gate: Float32Array;
    }
  )(projection);

  assert.equal(query.length, 4_096);
  assert.equal(gate.length, 4_096);
  assert.equal(query[0], 0);
  assert.equal(query[256], 1_000);
  assert.equal(query[4_095], 15_255);
  assert.equal(gate[0], 100_000);
  assert.equal(gate[256], 101_000);
  assert.equal(gate[4_095], 115_255);
});

test("maps four query heads to each grouped K/V head", async () => {
  const module = await loadAttentionModule();
  assert.equal(typeof module.qwen35GqaKvHead, "function");
  const map = module.qwen35GqaKvHead as (head: number) => number;
  assert.deepEqual(
    Array.from({ length: 16 }, (_, head) => map(head)),
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
  );
  assert.throws(() => map(-1), /query head/i);
  assert.throws(() => map(16), /query head/i);
});

test("rounds every online-softmax stage to shader f32 semantics", async () => {
  const module = await loadAttentionModule();
  assert.equal(typeof module.qwen35OnlineAttentionHeadCpu, "function");
  const query = new Float32Array(256);
  const keys = new Float32Array(2 * 256);
  const values = new Float32Array(2 * 256);
  query[0] = 16;
  keys[0] = 47.80127716064453;
  keys[256] = 46.7545280456543;
  values[0] = 4_015.98779296875;
  values[256] = 35_675.51171875;

  const output = (
    module.qwen35OnlineAttentionHeadCpu as (
      query: Float32Array,
      keys: Float32Array,
      values: Float32Array,
    ) => Float32Array
  )(query, keys, values);

  assert.equal(output[0], 12_242.7099609375);
  // Leaving Math.exp in f64 produces 12242.7109375 for this fixture.
  assert.notEqual(output[0], 12_242.7109375);
});

test("uses stable online attention for extreme finite scores", async () => {
  const module = await loadAttentionModule();
  assert.equal(typeof module.FullAttentionCpuCache, "function");
  assert.equal(typeof module.fullAttentionDecodeCpu, "function");
  const Cache = module.FullAttentionCpuCache as new (
    capacity: number,
  ) => {
    readonly position: number;
  };
  const decode = module.fullAttentionDecodeCpu as (
    state: InstanceType<typeof Cache>,
    token: {
      readonly queryGate: Float32Array;
      readonly key: Float32Array;
      readonly value: Float32Array;
      readonly queryNormWeight: Float32Array;
      readonly keyNormWeight: Float32Array;
      readonly positions: readonly [number, number, number];
    },
  ) => Float32Array;
  const state = new Cache(2);
  const weights = new Float32Array(256).fill(1_000);
  const firstKey = new Float32Array(4 * 256).fill(-1);
  const secondKey = new Float32Array(4 * 256).fill(1);
  const firstValue = new Float32Array(4 * 256).fill(-7);
  const secondValue = new Float32Array(4 * 256).fill(9);
  const gateOpen = 80;
  const queryGate = makeProjection(() => 1, () => gateOpen);

  decode(state, {
    queryGate,
    key: firstKey,
    value: firstValue,
    queryNormWeight: weights,
    keyNormWeight: weights,
    positions: [0, 0, 0],
  });
  const output = decode(state, {
    queryGate,
    key: secondKey,
    value: secondValue,
    queryNormWeight: weights,
    keyNormWeight: weights,
    positions: [0, 0, 0],
  });

  assert.equal(output.length, 4_096);
  assert.equal(output.every(Number.isFinite), true);
  assert.ok(output[0]! > 8.99);
  assert.equal(state.position, 2);
  assert.throws(
    () =>
      decode(state, {
        queryGate,
        key: secondKey,
        value: secondValue,
        queryNormWeight: weights,
        keyNormWeight: weights,
        positions: [0, 0, 0],
      }),
    /capacity/i,
  );
});

test("serial prefill equals repeated one-token decode", async () => {
  const module = await loadAttentionModule();
  assert.equal(typeof module.fullAttentionPrefillCpu, "function");
  const Cache = module.FullAttentionCpuCache as new (capacity: number) => object;
  const decode = module.fullAttentionDecodeCpu as (
    state: object,
    token: AttentionToken,
  ) => Float32Array;
  const prefill = module.fullAttentionPrefillCpu as (
    state: object,
    tokens: readonly AttentionToken[],
  ) => readonly Float32Array[];
  const weights = new Float32Array(256).fill(1);
  const tokens: AttentionToken[] = Array.from({ length: 4 }, (_, token) => ({
    queryGate: makeProjection(
      (head, lane) => (head + lane + token + 1) / 512,
      () => 0,
    ),
    key: Float32Array.from(
      { length: 4 * 256 },
      (_, index) => ((index + token) % 17) / 17,
    ),
    value: Float32Array.from(
      { length: 4 * 256 },
      (_, index) => ((index * 3 + token) % 19) / 19,
    ),
    queryNormWeight: weights,
    keyNormWeight: weights,
    positions: [token, token, token],
  }));
  const decodeState = new Cache(4);
  const prefillState = new Cache(4);

  const expected = tokens.map((token) => decode(decodeState, token));
  const actual = prefill(prefillState, tokens);

  assert.equal(actual.length, expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    assert.deepEqual(actual[index], expected[index]);
  }
});

test("prefill validates every token before changing the cache", async () => {
  const module = await loadAttentionModule();
  const Cache = module.FullAttentionCpuCache as new (
    capacity: number,
  ) => {
    readonly position: number;
  };
  const decode = module.fullAttentionDecodeCpu as (
    state: object,
    token: AttentionToken,
  ) => Float32Array;
  const prefill = module.fullAttentionPrefillCpu as (
    state: object,
    tokens: readonly AttentionToken[],
  ) => readonly Float32Array[];
  const valid = makeAttentionToken(1 / 3);
  const invalid = {
    ...makeAttentionToken(2 / 3),
    key: new Float32Array(1),
  };
  const actualState = new Cache(2);
  const expectedState = new Cache(1);

  assert.throws(() => prefill(actualState, [valid, invalid]), /key/i);
  assert.equal(actualState.position, 0);
  assert.deepEqual(
    decode(actualState, valid),
    decode(expectedState, valid),
  );
});

test("quantizes current and historical K/V to production FP16", async () => {
  const module = await loadAttentionModule();
  const Cache = module.FullAttentionCpuCache as new (
    capacity: number,
  ) => {
    readonly position: number;
    value(token: number, head: number, lane: number): number;
  };
  const decode = module.fullAttentionDecodeCpu as (
    state: InstanceType<typeof Cache>,
    token: AttentionToken,
  ) => Float32Array;
  const state = new Cache(2);
  const first = makeAttentionToken(1 / 3);
  const second = makeAttentionToken(2 / 3);

  decode(state, first);
  assert.equal(state.value(0, 0, 0), nativeFloat16(1 / 3));
  const output = decode(state, second);
  const expected = Math.fround(
    Math.fround(
      nativeFloat16(1 / 3) + nativeFloat16(2 / 3),
    ) / 2,
  );

  assert.equal(output[0], expected);
  assert.notEqual(output[0], Math.fround((1 / 3 + 2 / 3) / 2));
});

test("packs FP16 pairs and writes only the selected current-token K/V row", async () => {
  const module = await loadAttentionModule();
  assert.equal(typeof module.packFloat16PairCpu, "function");
  assert.equal(typeof module.writeFullAttentionKvCpu, "function");
  const pack = module.packFloat16PairCpu as (
    first: number,
    second: number,
  ) => number;
  assert.equal(pack(1, -2), 0xc0003c00);
  assert.equal(pack(58_832, 0) & 0xffff, 0x7b2e);
  assert.equal(pack(2 ** -25, 0) & 0xffff, 0);
  assert.equal(pack(2 ** -25 + 2 ** -48, 0) & 0xffff, 1);
  assert.equal(pack(3 * 2 ** -25, 0) & 0xffff, 2);
  assert.equal(pack(65_504, 0) & 0xffff, 0x7bff);
  assert.equal(pack(65_520, 0) & 0xffff, 0x7c00);
  assert.equal(pack(Number.POSITIVE_INFINITY, 0) & 0xffff, 0x7c00);
  assert.equal(pack(Number.NEGATIVE_INFINITY, 0) & 0xffff, 0xfc00);
  assert.equal(pack(Number.NaN, 0) & 0xffff, 0x7e00);

  // Every positive finite binary16 midpoint is exactly representable in f32.
  // This checks ties-to-even across subnormal and normal boundaries; the
  // explicit 65,520 assertion above covers the finite-to-infinity threshold.
  for (let lower = 0; lower <= 0x7bfe; lower += 1) {
    const midpoint = (nativeFloat16Bits(lower) + nativeFloat16Bits(lower + 1)) / 2;
    assert.equal(
      pack(midpoint, 0) & 0xffff,
      nativeFloat16BitPattern(midpoint),
      `positive midpoint after 0x${lower.toString(16)}`,
    );
    assert.equal(
      pack(-midpoint, 0) & 0xffff,
      nativeFloat16BitPattern(-midpoint),
      `negative midpoint after 0x${lower.toString(16)}`,
    );
  }

  const sentinel = 0xdeadbeef;
  const packedKeys = new Uint32Array(2 * 512 + 3).fill(sentinel);
  const packedValues = new Uint32Array(2 * 512 + 3).fill(sentinel);
  const key = Float32Array.from(
    { length: 1_024 },
    (_, index) => (index % 7) - 3,
  );
  const value = Float32Array.from(
    { length: 1_024 },
    (_, index) => (index % 11) / 4,
  );
  (
    module.writeFullAttentionKvCpu as (
      packedKeys: Uint32Array,
      packedValues: Uint32Array,
      position: number,
      capacity: number,
      key: Float32Array,
      value: Float32Array,
    ) => void
  )(packedKeys, packedValues, 1, 2, key, value);

  assert.equal(packedKeys[0], sentinel);
  assert.equal(packedValues[0], sentinel);
  assert.equal(packedKeys[512], pack(key[0]!, key[1]!));
  assert.equal(packedValues[512], pack(value[0]!, value[1]!));
  assert.equal(packedKeys[1_023], pack(key[1_022]!, key[1_023]!));
  assert.equal(packedValues[1_023], pack(value[1_022]!, value[1_023]!));
  assert.deepEqual(Array.from(packedKeys.subarray(1_024)), [
    sentinel,
    sentinel,
    sentinel,
  ]);
  assert.deepEqual(Array.from(packedValues.subarray(1_024)), [
    sentinel,
    sentinel,
    sentinel,
  ]);
});

interface AttentionToken {
  readonly queryGate: Float32Array;
  readonly key: Float32Array;
  readonly value: Float32Array;
  readonly queryNormWeight: Float32Array;
  readonly keyNormWeight: Float32Array;
  readonly positions: readonly [number, number, number];
}

const FLOAT16_REFERENCE_BUFFER = new ArrayBuffer(2);
const FLOAT16_REFERENCE_VIEW = new DataView(FLOAT16_REFERENCE_BUFFER);

function nativeFloat16BitPattern(value: number): number {
  FLOAT16_REFERENCE_VIEW.setFloat16(0, value, true);
  return FLOAT16_REFERENCE_VIEW.getUint16(0, true);
}

function nativeFloat16(value: number): number {
  FLOAT16_REFERENCE_VIEW.setFloat16(0, value, true);
  return FLOAT16_REFERENCE_VIEW.getFloat16(0, true);
}

function nativeFloat16Bits(bits: number): number {
  FLOAT16_REFERENCE_VIEW.setUint16(0, bits, true);
  return FLOAT16_REFERENCE_VIEW.getFloat16(0, true);
}

function makeAttentionToken(value: number): AttentionToken {
  return {
    queryGate: makeProjection(() => 0, () => 80),
    key: new Float32Array(4 * 256).fill(value),
    value: new Float32Array(4 * 256).fill(value),
    queryNormWeight: new Float32Array(256).fill(1),
    keyNormWeight: new Float32Array(256).fill(1),
    positions: [0, 0, 0],
  };
}
