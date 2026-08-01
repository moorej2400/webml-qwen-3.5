import assert from "node:assert/strict";
import test from "node:test";

import { KernelRegistry } from "../src/kernel-registry.js";
import {
  QWEN35_VISION_FOUNDATION_KERNELS,
  registerQwen35VisionFoundationKernels,
  planQwen35VisionBootstrapFoundationDispatches,
  planQwen35VisionLayerRopeApplyDispatch,
  visionAddLearnedPositionCpu,
  visionApply2dRopeCpu,
  visionPatchConv3dReferenceCpu,
  visionPrepare2dRopeCpu,
} from "../src/qwen35-vision-foundation-kernels.js";
import type { Qwen35VisionGpuStagedGroup, Qwen35VisionGpuTensorView } from "../src/qwen35-vision-gpu-staging.js";

test("defines only bounded fixed-architecture patch, position, and 2D RoPE kernels", () => {
  assert.deepEqual(
    QWEN35_VISION_FOUNDATION_KERNELS.map((kernel) => kernel.key.operation),
    ["vision-patch-conv3d", "vision-add-learned-position", "vision-prepare-2d-rope", "vision-apply-2d-rope"],
  );
  for (const kernel of QWEN35_VISION_FOUNDATION_KERNELS) {
    assert.match(kernel.source, /@compute/u);
    assert.match(kernel.source, /patch_count/u);
    assert.match(kernel.source, /return;/u);
  }
  assert.match(QWEN35_VISION_FOUNDATION_KERNELS[3]!.source, /let q_left = query\[base \+ lane\]; let q_right = query\[base \+ pair\];/u);
  const registry = new KernelRegistry();
  registerQwen35VisionFoundationKernels(registry);
  assert.equal(
    registry.select({
      key: { operation: "vision-patch-conv3d", layout: "f32", phase: "vision", profile: "portable-f32" },
      fallbackProfiles: [],
    }).kernel.id,
    "qwen35-vision-patch-conv3d-f32",
  );
});

test("matches the fixed packed temporal-channel patch order and align-corners position order", () => {
  const patches = Float32Array.of(1, 2, 3, 4, 5, 6);
  const temporalZero = Float32Array.of(1, 100, 10_000);
  const temporalOne = Float32Array.of(10, 1_000, 100_000);
  const embedding = visionPatchConv3dReferenceCpu({
    patches,
    patchCount: 1,
    hiddenSize: 1,
    patchSize: 1,
    temporalPatchSize: 2,
    weightsTemporalZero: temporalZero,
    weightsTemporalOne: temporalOne,
    bias: Float32Array.of(0.5),
  });
  assert.deepEqual([...embedding], [654_321.5]);

  const positioned = visionAddLearnedPositionCpu({
    embeddings: Float32Array.of(1, 2, 3, 4),
    gridHeight: 2,
    gridWidth: 2,
    hiddenSize: 1,
    tableHeight: 2,
    tableWidth: 2,
    table: Float32Array.of(10, 20, 30, 40),
  });
  assert.deepEqual([...positioned], [11, 22, 33, 44]);
});

test("prepares merge-block coordinates and rotates Q/K split halves in f32", () => {
  const rope = visionPrepare2dRopeCpu({ gridHeight: 2, gridWidth: 2, headDimension: 4 });
  assert.deepEqual([...rope.coordinates], [0, 0, 0, 1, 1, 0, 1, 1]);
  assert.deepEqual(
    [...visionPrepare2dRopeCpu({ gridHeight: 4, gridWidth: 4, headDimension: 4 }).coordinates],
    [0, 0, 0, 1, 1, 0, 1, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 0, 2, 1, 3, 0, 3, 1, 2, 2, 2, 3, 3, 2, 3, 3],
  );
  const rotated = visionApply2dRopeCpu({
    query: Float32Array.of(1, 2, 3, 4),
    key: Float32Array.of(5, 6, 7, 8),
    rope: rope.values.subarray(0, 4),
    patchCount: 1,
    headCount: 1,
    headDimension: 4,
  });
  assert.deepEqual([...rotated.query], [1, 2, 3, 4]);
  assert.deepEqual([...rotated.key], [5, 6, 7, 8]);
  assert.equal(visionPrepare2dRopeCpu({ gridHeight: 10, gridWidth: 1600, headDimension: 4 }).coordinates.length, 32_000);
});

test("requires an authenticated staged bootstrap before binding direct tensor segments", () => {
  const buffer = { destroy() {} };
  const view = (name: string, byteLength: number, segments = 1): Qwen35VisionGpuTensorView => ({
    name, shape: [], precision: "f32", storageType: "f32",
    orientation: { kind: "element-contiguous", contiguousDimension: "element" },
    segments: Array.from({ length: segments }, (_, index) => ({
      shard: index, buffer, bufferOffset: 0, tensorOffset: index === 0 ? 0 : 1_048_576,
      byteLength: index === 0 && segments === 2 ? 1_048_576 : byteLength - (segments === 2 ? 1_048_576 : 0),
    })),
  });
  const bootstrap = {
    layer: "bootstrap",
    shards: [],
    tensors: [
      view("v.patch_embd.weight", 3_145_728), view("v.patch_embd.weight.1", 3_145_728), view("v.patch_embd.bias", 4_096),
      view("v.position_embd.weight", 9_437_184, 2),
    ],
    async destroy() {},
  } as unknown as Qwen35VisionGpuStagedGroup;
  const storage = (byteLength: number) => ({ buffer, byteLength });
  const bootstrapWorkspace = {
    patches: storage(4 * 1_536 * 4), embeddings: storage(4 * 1_024 * 4), rope: storage(4 * 64 * 4),
    patchUniform: storage(16), positionUniform: storage(16), ropeUniform: storage(16),
  };
  const ropeWorkspace = {
    rope: storage(4 * 64 * 4),
    query: storage(4 * 16 * 64 * 4), key: storage(4 * 16 * 64 * 4),
    applyUniform: storage(16),
  };
  assert.throws(
    () => planQwen35VisionBootstrapFoundationDispatches({
      bootstrap, workspace: bootstrapWorkspace, gridHeight: 2, gridWidth: 2,
      limits: { minStorageBufferOffsetAlignment: 256, minUniformBufferOffsetAlignment: 256, maxStorageBufferBindingSize: 16 * 1024 * 1024, maxUniformBufferBindingSize: 256, maxComputeWorkgroupsPerDimension: 64 },
    }),
    { code: "vision-stage-group-unauthenticated" },
  );
  assert.deepEqual(
    planQwen35VisionLayerRopeApplyDispatch({
      workspace: ropeWorkspace, patchCount: 4,
      limits: { minStorageBufferOffsetAlignment: 256, minUniformBufferOffsetAlignment: 256, maxStorageBufferBindingSize: 16 * 1024 * 1024, maxUniformBufferBindingSize: 256, maxComputeWorkgroupsPerDimension: 64 },
    }).workgroups,
    { x: 1, y: 4, z: 16 },
  );
});
