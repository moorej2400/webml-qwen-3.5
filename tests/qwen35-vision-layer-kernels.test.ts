import assert from "node:assert/strict";
import test from "node:test";

import {
  QWEN35_VISION_LAYER_KERNELS,
  visionLayerNormCpu,
  visionLinearBf16Cpu,
  visionOnlineAttentionCpu,
  planQwen35VisionLayerDispatches,
  visionTanhGeluCpu,
  visionTransformerLayerCpu,
} from "../src/qwen35-vision-layer-kernels.js";

function close(actual: readonly number[], expected: readonly number[], tolerance = 1e-5): void {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    assert.ok(Math.abs(actual[index]! - expected[index]!) <= tolerance, `lane ${index}`);
  }
}

test("uses Qwen vision FP32 layer norm, packed BF16 row-major linear, and tanh GELU", () => {
  assert.deepEqual(
    [...visionLayerNormCpu({
      input: Float32Array.of(1, 3, 2, 4), tokenCount: 2, hiddenSize: 2,
      weight: Float32Array.of(1, 1), bias: Float32Array.of(0, 0),
    })],
    [-0.9999995231628418, 0.9999995231628418, -0.9999995231628418, 0.9999995231628418],
  );
  assert.deepEqual(
    [...visionLinearBf16Cpu({
      input: Float32Array.of(1, 2), tokenCount: 1, inputWidth: 2, outputWidth: 2,
      weight: Uint16Array.of(0x3f80, 0x4000, 0x4040, 0x4080), bias: Float32Array.of(0.5, -0.5),
    })],
    [5.5, 10.5],
  );
  assert.deepEqual([...visionTanhGeluCpu(Float32Array.of(-1, 0, 1))], [
    -0.15880800783634186, 0, 0.8411920070648193,
  ]);
});

test("uses online non-causal attention without crossing image segments", () => {
  assert.deepEqual(
    [...visionOnlineAttentionCpu({
      query: Float32Array.of(1, 0, 1, 0, 1, 0, 1, 0),
      key: Float32Array.of(1, 0, 0, 1, 100, 0, 0, 100),
      value: Float32Array.of(1, 2, 3, 4, 100, 200, 300, 400),
      tokenCount: 4, headCount: 1, headDimension: 2, segmentOffsets: Uint32Array.of(0, 2, 4),
    })],
    [1.6604769229888916, 2.6604769229888916, 1.6604769229888916, 2.6604769229888916, 100, 200, 100, 200],
  );
});

test("defines only fixed Qwen vision layer kernels", () => {
  assert.deepEqual(QWEN35_VISION_LAYER_KERNELS.map((kernel) => kernel.key.operation), [
    "vision-layernorm", "vision-bf16-linear", "vision-qkv-bf16-linear", "vision-online-attention", "vision-tanh-gelu", "vision-residual-add",
  ]);
  const attention = QWEN35_VISION_LAYER_KERNELS.find((kernel) => kernel.key.operation === "vision-online-attention")!;
  const linear = QWEN35_VISION_LAYER_KERNELS.find((kernel) => kernel.key.operation === "vision-bf16-linear")!;
  assert.match(attention.source, /var<workgroup> has_key: u32/u);
  assert.doesNotMatch(attention.source, /var<workgroup> has_key: bool/u);
  assert.match(attention.source, /workgroupBarrier\(\)/u);
  assert.match(linear.source, /packed_weight\[\(row_base \+ column\) \/ 2u\]/u);
});

test("requires authenticated layer weights before it plans the fixed execution order", () => {
  const buffer = { destroy() {} };
  const storage = (byteLength: number) => ({ buffer, byteLength });
  assert.throws(() => planQwen35VisionLayerDispatches({
    layer: 0,
    staged: { layer: 0, shards: [], tensors: [], async destroy() {} },
    workspace: {
      hidden: storage(4 * 1_024 * 4), normalized: storage(4 * 1_024 * 4), qkv: storage(4 * 3_072 * 4),
      attention: storage(4 * 1_024 * 4), mlp: storage(4 * 4_096 * 4), rope: storage(4 * 64 * 4),
      segmentOffsets: storage(8), uniforms: Array.from({ length: 10 }, () => storage(16)),
    },
    tokenCount: 4,
    segmentCount: 1,
    limits: { minStorageBufferOffsetAlignment: 256, minUniformBufferOffsetAlignment: 256, maxStorageBufferBindingSize: 16 * 1024 * 1024, maxUniformBufferBindingSize: 256, maxComputeWorkgroupsPerDimension: 64 },
  }), { code: "vision-stage-group-unauthenticated" });
});

test("rejects reduced layers whose head dimension cannot use Qwen 2D RoPE", () => {
  assert.throws(() => visionTransformerLayerCpu({
    input: Float32Array.of(1, 2, 3, 4, 5, 6), tokenCount: 1, hiddenSize: 6, headCount: 2,
    qkvWeight: new Uint16Array(108), qkvBias: new Float32Array(18),
    attentionOutputWeight: new Uint16Array(36), attentionOutputBias: new Float32Array(6),
    preAttentionWeight: new Float32Array(6), preAttentionBias: new Float32Array(6),
    preMlpWeight: new Float32Array(6), preMlpBias: new Float32Array(6),
    mlpUpWeight: new Uint16Array(12), mlpUpBias: new Float32Array(2),
    mlpDownWeight: new Uint16Array(12), mlpDownBias: new Float32Array(6), feedForwardSize: 2,
    rope: new Float32Array(6), segmentOffsets: Uint32Array.of(0, 1),
  }), { code: "vision-layer-invalid" });
});

test("matches the independent nonzero reduced layer oracle through QKV, RoPE, attention, and residual", () => {
  const qkvWeight = new Uint16Array(4 * 12);
  for (let lane = 0; lane < 4; lane += 1) {
    qkvWeight[lane * 4 + lane] = 0x3f80;
    qkvWeight[(4 + lane) * 4 + lane] = 0x3f80;
    qkvWeight[(8 + lane) * 4 + lane] = 0x3f80;
  }
  const identity = new Uint16Array(16);
  for (let lane = 0; lane < 4; lane += 1) identity[lane * 4 + lane] = 0x3f80;
  close([...visionTransformerLayerCpu({
    input: Float32Array.of(1, 2, 3, 4, 1, 3, 7, 15), tokenCount: 2, hiddenSize: 4, headCount: 1,
    qkvWeight, qkvBias: new Float32Array(12), attentionOutputWeight: identity, attentionOutputBias: new Float32Array(4),
    preAttentionWeight: new Float32Array([1, 1, 1, 1]), preAttentionBias: new Float32Array(4),
    preMlpWeight: new Float32Array([1, 1, 1, 1]), preMlpBias: new Float32Array(4),
    mlpUpWeight: new Uint16Array(16), mlpUpBias: new Float32Array(4), mlpDownWeight: new Uint16Array(16), mlpDownBias: new Float32Array(4),
    feedForwardSize: 4, rope: Float32Array.of(1, 0, 1, 0, 1, 0, Math.cos(1), Math.sin(1)), segmentOffsets: Uint32Array.of(0, 2),
  })], [
    -0.2373714, 1.4849409, 3.3303757, 5.4220552,
    -0.1300241, 2.4150922, 7.2100883, 16.5048447,
  ]);
});
