import assert from "node:assert/strict";
import test from "node:test";

import {
  QWEN_PRIMITIVE_KERNELS,
  attentionOutputGateCpu,
  fusedSwiGluCpu,
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

test("applies explicit M-RoPE position sections and preserves non-rotary values", () => {
  const input = Float32Array.from({ length: 10 }, (_, index) => index + 1);
  const output = partialMropeCpu(input, {
    headCount: 1,
    headDimension: 10,
    rotaryDimension: 8,
    sections: [1, 1, 2, 0],
    positions: [0, 1, 2, 99],
    theta: 10_000,
  });

  close(output.slice(0, 2), input.slice(0, 2));
  assert.notDeepEqual(output.slice(2, 4), input.slice(2, 4));
  assert.notDeepEqual(output.slice(4, 8), input.slice(4, 8));
  assert.deepEqual(output.slice(8), input.slice(8));
  assert.throws(
    () =>
      partialMropeCpu(input, {
        headCount: 1,
        headDimension: 10,
        rotaryDimension: 8,
        sections: [1, 1, 1, 0],
        positions: [0, 1, 2, 3],
        theta: 10_000,
      }),
    /sections.*rotary/i,
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
  assert.match(
    QWEN_PRIMITIVE_KERNELS.find((kernel) => kernel.operation === "partial-mrope")!.source,
    /sections|positions/,
  );
  assert.match(
    QWEN_PRIMITIVE_KERNELS.find((kernel) => kernel.operation === "top-k-merge")!.source,
    /bitcast<u32>.*0x7f800000u/,
  );

  assert.deepEqual(
    planPrimitiveDispatch({ operation: "residual-add", elementCount: 257 }).workgroups,
    { x: 2, y: 1, z: 1 },
  );
  assert.throws(
    () => planPrimitiveDispatch({ operation: "qk-rms-norm", elementCount: 3, headDimension: 2 }),
    /head/i,
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
