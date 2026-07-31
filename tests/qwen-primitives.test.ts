import assert from "node:assert/strict";
import test from "node:test";

import {
  QWEN_PRIMITIVE_KERNELS,
  attentionOutputGateCpu,
  fusedSwiGluCpu,
  mropeFrequencyOwners,
  partialMropeCpu,
  planPrimitiveDispatch,
  qkRmsNormPerHeadCpu,
  residualAddCpu,
  rmsNormCpu,
  siluCpu,
  stableTiledTopK,
  qwenPrimitiveRegistryDefinitions,
} from "../src/qwen-primitives.js";
import { KernelRegistry } from "../src/kernel-registry.js";
import { PINNED_QWEN35_GGUF_FIXTURE } from "./fixtures/qwen35-4b-q3-k-l-sanitized.js";

function close(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance = 1e-5): void {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(
      Math.abs(actual[index]! - expected[index]!) <= tolerance,
      `${index}: ${actual[index]} != ${expected[index]}`,
    );
  }
}

test("computes deterministic CPU references for zero and large finite values", () => {
  close(rmsNormCpu(Float32Array.of(0, 0), Float32Array.of(2, 3), 1e-6), [0, 0]);
  close(
    rmsNormCpu(Float32Array.of(3, 4), Float32Array.of(1, 2), 0),
    [3 / Math.sqrt(12.5), 8 / Math.sqrt(12.5)],
  );
  assert.deepEqual(
    residualAddCpu(Float32Array.of(1e20, -2, 0), Float32Array.of(-1e20, 5, -0)),
    Float32Array.of(0, 3, 0),
  );
  close(
    fusedSwiGluCpu(Float32Array.of(0, 1, -100), Float32Array.of(5, 3, 7)),
    [0, siluCpu(1) * 3, siluCpu(-100) * 7],
    1e-7,
  );
  close(
    attentionOutputGateCpu(Float32Array.of(2, -4), Float32Array.of(0, 100)),
    [1, -4],
  );
});

test("uses multiplicative GGUF norm weights after llama.cpp adds one", () => {
  assert.deepEqual(
    PINNED_QWEN35_GGUF_FIXTURE.normOracle.ggufSemantics,
    "multiplicative",
  );
  close(
    rmsNormCpu(
      Float32Array.of(3, 4),
      Float32Array.from(PINNED_QWEN35_GGUF_FIXTURE.normOracle.ggufWeight),
      0,
    ),
    [3 / Math.sqrt(12.5), 3 / Math.sqrt(12.5)],
  );
  const source = QWEN_PRIMITIVE_KERNELS.find(
    (kernel) => kernel.operation === "rms-norm",
  )!.source;
  assert.match(source, /inverse_rms \* weights/);
  assert.doesNotMatch(source, /1\\.0f?\\s*\\+\\s*weights/);
});

test("normalizes Q and K independently per head with FP32 accumulation", () => {
  close(
    qkRmsNormPerHeadCpu(
      Float32Array.of(3, 4, 0, 5),
      Float32Array.of(1, 2),
      { headCount: 2, headDimension: 2, epsilon: 0 },
    ),
    [
      3 / Math.sqrt(12.5),
      8 / Math.sqrt(12.5),
      0,
      10 / Math.sqrt(12.5),
    ],
  );
  assert.throws(
    () =>
      qkRmsNormPerHeadCpu(Float32Array.of(1, 2, 3), Float32Array.of(1, 1), {
        headCount: 2,
        headDimension: 2,
        epsilon: 1e-6,
      }),
    /shape/i,
  );
});

test("uses the exact interleaved M-RoPE owners and split-half rotation", () => {
  assert.deepEqual(
    mropeFrequencyOwners([11, 11, 10]),
    [
      0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
      1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
    ],
  );
  const input = Float32Array.from({ length: 256 }, (_, index) => index + 1);
  const output = partialMropeCpu(input, {
    headCount: 1,
    headDimension: 256,
    rotaryDimension: 64,
    sections: [11, 11, 10],
    positions: [0, 1, 2],
    theta: 1,
  });

  assert.equal(output[0], input[0]);
  assert.equal(output[32], input[32]);
  close(
    [output[1]!, output[33]!],
    [
      input[1]! * Math.cos(1) - input[33]! * Math.sin(1),
      input[33]! * Math.cos(1) + input[1]! * Math.sin(1),
    ],
  );
  close(
    [output[2]!, output[34]!],
    [
      input[2]! * Math.cos(2) - input[34]! * Math.sin(2),
      input[34]! * Math.cos(2) + input[2]! * Math.sin(2),
    ],
  );
  assert.deepEqual(output.slice(64), input.slice(64));
  assert.throws(
    () =>
      partialMropeCpu(input, {
        headCount: 1,
        headDimension: 256,
        rotaryDimension: 64,
        sections: [11, 10, 10],
        positions: [0, 1, 2],
        theta: 10_000,
      }),
    /sections.*rotary/i,
  );
});

test("matches WGSL f32 M-RoPE arithmetic at the product context boundary", () => {
  const input = Float32Array.from(
    { length: 256 },
    (_, index) => Math.fround(((index * 19) % 31 - 15) / 7),
  );
  const actual = partialMropeCpu(input, {
    headCount: 1,
    headDimension: 256,
    rotaryDimension: 64,
    sections: [11, 11, 10],
    positions: [16_384, 16_383, 16_382],
    theta: 10_000_000,
  });
  const exponent = Math.fround(Math.fround(2) / Math.fround(64));
  const divisor = Math.fround(Math.pow(Math.fround(10_000_000), exponent));
  const angle = Math.fround(Math.fround(16_383) / divisor);
  const tau = Math.fround(2 * Math.PI);
  const turns = Math.floor(Math.fround(angle / tau));
  const reducedAngle = Math.fround(
    angle - Math.fround(Math.fround(turns) * tau),
  );
  const cosine = Math.fround(Math.cos(reducedAngle));
  const sine = Math.fround(Math.sin(reducedAngle));
  assert.equal(
    actual[1],
    Math.fround(
      Math.fround(input[1]! * cosine) -
        Math.fround(input[33]! * sine),
    ),
  );
  assert.equal(
    actual[33],
    Math.fround(
      Math.fround(input[33]! * cosine) +
        Math.fround(input[1]! * sine),
    ),
  );
});

test("merges tiled top-k with stable ties and explicit non-finite handling", () => {
  assert.deepEqual(
    stableTiledTopK(
      [
        { startIndex: 0, scores: Float32Array.of(1, Number.NaN, 5, 5) },
        {
          startIndex: 4,
          scores: Float32Array.of(Number.POSITIVE_INFINITY, 5, -2, Number.NEGATIVE_INFINITY),
        },
      ],
      4,
    ),
    [
      { index: 2, score: 5 },
      { index: 3, score: 5 },
      { index: 5, score: 5 },
      { index: 0, score: 1 },
    ],
  );
  assert.throws(() => stableTiledTopK([], 0), /positive/i);
  assert.deepEqual(
    stableTiledTopK(
      [{ startIndex: 7, scores: Float32Array.of(Number.NaN, 3, Infinity) }],
      4,
    ),
    [{ index: 8, score: 3 }],
  );
});

test("defines typed WGSL ABIs, dispatches, and phase-profile registry entries", () => {
  assert.deepEqual(
    QWEN_PRIMITIVE_KERNELS.map((kernel) => kernel.operation),
    [
      "rms-norm",
      "residual-add",
      "silu",
      "swiglu",
      "attention-output-gate",
      "qk-rms-norm",
      "partial-mrope",
      "top-k-merge",
    ],
  );
  for (const kernel of QWEN_PRIMITIVE_KERNELS) {
    assert.equal(kernel.profile, "portable-f32");
    assert.match(kernel.source, /@compute/);
    assert.match(kernel.source, /@group\(0\) @binding\(0\)/);
    assert.doesNotMatch(kernel.source, /f16|mat(2|3|4)x/);
  }
  assert.match(
    QWEN_PRIMITIVE_KERNELS.find((kernel) => kernel.operation === "rms-norm")!.source,
    /sum.*f32/,
  );
  const mropeKernel = QWEN_PRIMITIVE_KERNELS.find(
    (kernel) => kernel.operation === "partial-mrope",
  )!;
  assert.match(mropeKernel.source, /sections|positions/);
  assert.match(mropeKernel.source, /lane >= params\.rotary_dimension/);
  assert.match(
    mropeKernel.source,
    /output_values\[index\] = input_values\[index\]/,
  );
  assert.equal(mropeKernel.abi.outputCoverage, "all-elements");
  assert.equal(mropeKernel.abi.coordinateCount, 3);
  assert.match(
    QWEN_PRIMITIVE_KERNELS.find((kernel) => kernel.operation === "top-k-merge")!.source,
    /bitcast<u32>.*0x7f800000u/,
  );
  const topKKernel = QWEN_PRIMITIVE_KERNELS.find(
    (kernel) => kernel.operation === "top-k-merge",
  )!;
  assert.equal(topKKernel.abi.bindings.outputValidCount, 3);
  assert.equal(topKKernel.abi.bindings.uniforms, 4);
  assert.match(topKKernel.source, /if \(!found\) \{ break; \}/);
  assert.match(topKKernel.source, /output_valid_count\[0\] = valid_count/);

  assert.deepEqual(
    planPrimitiveDispatch({ operation: "residual-add", elementCount: 257 }).workgroups,
    { x: 2, y: 1, z: 1 },
  );
  assert.throws(
    () => planPrimitiveDispatch({ operation: "qk-rms-norm", elementCount: 3, headDimension: 2 }),
    /head/i,
  );
  assert.deepEqual(
    planPrimitiveDispatch({
      operation: "partial-mrope",
      elementCount: 256,
      headDimension: 256,
      rotaryDimension: 64,
    }).workgroups,
    { x: 1, y: 1, z: 1 },
  );

  const registry = new KernelRegistry();
  const definitions = qwenPrimitiveRegistryDefinitions({
    phase: "decode",
    profile: "portable-f32",
  });
  definitions.forEach((definition) => registry.register(definition));
  assert.equal(
    registry.select({
      key: {
        operation: "partial-mrope",
        layout: "f32",
        phase: "decode",
        profile: "portable-f32",
      },
      fallbackProfiles: [],
    }).kernel.id,
    "partial-mrope-decode-portable-f32",
  );
});
