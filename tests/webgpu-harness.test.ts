import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { validateParity } from "../tools/webgpu-parity.mjs";

test("rejects non-finite CPU and GPU parity values", () => {
  validateParity("finite", Float32Array.of(1), Float32Array.of(1.0001));
  for (const [expected, actual] of [
    [Float32Array.of(Number.NaN), Float32Array.of(1)],
    [Float32Array.of(1), Float32Array.of(Number.NaN)],
    [Float32Array.of(Number.POSITIVE_INFINITY), Float32Array.of(1)],
    [Float32Array.of(1), Float32Array.of(Number.NEGATIVE_INFINITY)],
  ]) {
    assert.throws(
      () => validateParity("non-finite", expected, actual),
      /non-finite/i,
    );
  }
});

test("ships a deterministic browser execution and CPU parity harness", async () => {
  const html = await readFile(
    new URL("../tools/webgpu-kernel-harness.html", import.meta.url),
    "utf8",
  );
  const script = await readFile(
    new URL("../tools/webgpu-kernel-harness.mjs", import.meta.url),
    "utf8",
  );

  assert.match(html, /webgpu-kernel-harness\.mjs/);
  assert.match(html, /Executes all pinned language kernels/);
  assert.doesNotMatch(html, /Compiles all pinned language/);
  assert.match(script, /LANGUAGE_GEMV_KERNELS/);
  assert.match(script, /repackNativeQ3KFusedBrowser/);
  assert.match(script, /q6-k-fused-f32-256/);
  assert.match(script, /QWEN_PRIMITIVE_KERNELS/);
  assert.match(script, /PACKED_EMBEDDING_KERNELS/);
  assert.match(script, /QWEN35_VISION_FOUNDATION_KERNELS/);
  assert.match(script, /QWEN35_VISION_LAYER_KERNELS/);
  assert.match(script, /for \(const kernel of LANGUAGE_GEMV_KERNELS\)/);
  assert.match(script, /for \(const kernel of QWEN_PRIMITIVE_KERNELS\)/);
  assert.match(script, /for \(const kernel of PACKED_EMBEDDING_KERNELS\)/);
  assert.doesNotMatch(script, /compilePrimitive/);
  assert.match(script, /results\.push\(await runKernel\(device, kernel\)\)/);
  assert.match(script, /runPrimitive/);
  assert.match(script, /runEmbedding/);
  assert.match(script, /runVisionFoundationKernel/);
  assert.match(script, /runVisionLayerKernel/);
  assert.match(script, /visionLayerNormCpu/);
  assert.match(script, /visionLinearBf16Cpu/);
  assert.match(script, /visionOnlineAttentionCpu/);
  assert.match(script, /visionTanhGeluCpu/);
  assert.match(script, /bf16Fixture/);
  assert.match(script, /const tokenCount = qkv \? 2 : 1/);
  assert.match(script, /planar QKV fixture did not differ from token-major QKV/);
  assert.match(script, /for \(const kernel of QWEN35_VISION_LAYER_KERNELS\)/);
  assert.match(script, /QWEN35_VISION_MERGER_KERNELS/);
  assert.match(script, /visionExactGeluCpu/);
  assert.match(script, /visionPatchConv3dCpu/);
  assert.doesNotMatch(script, /visionPatchConv3dReferenceCpu/);
  assert.match(script, /visionPrepare2dRopeCpu/);
  assert.match(script, /const convWeightIndex/);
  assert.match(script, /for \(let channel = 0; channel < 3; channel \+= 1\)/);
  assert.match(script, /for \(const hidden of \[0, 1\]\)/);
  assert.match(script, /gridHeight: 4, gridWidth: 4/);
  assert.match(script, /const split = \(31 \* 48 \+ 16\) \* 1024/);
  assert.match(script, /uniform\(\[16, 4, 4, split\]\)/);
  assert.match(script, /gridHeight: 10, gridWidth: 1_600/);
  assert.match(script, /uniform\(\[16_000, 10, 1_600, 0\]\)/);
  assert.match(script, /nonIdentityRope = prepared\.values\.subarray\(\(16_000 - 1\) \* 64, 16_000 \* 64\)/);
  assert.match(script, /partialMropeCpu/);
  assert.match(script, /stableTiledTopK/);
  assert.match(script, /embeddingCpu/);
  assert.match(script, /positions:\s*\[16_384/);
  assert.match(script, /validCount/);
  assert.match(script, /getCompilationInfo/);
  assert.match(script, /gemvCpu/);
  assert.match(script, /dispatchWorkgroups/);
  assert.match(script, /packedByteOffset\s*=\s*32/);
  assert.match(script, /weightWordOffset/);
  assert.match(script, /blocksPerRow\s*=\s*2/);
  assert.match(script, /rowIndex/);
  assert.match(script, /blockIndex/);
  assert.match(script, /outputRowOffset\s*=\s*[1-9]/);
  assert.match(script, /sentinel/);
  assert.match(script, /maxWorkgroupsPerDimension\s*:\s*2/);
  assert.match(script, /workgroups\.y\s*!==\s*2/);
  assert.match(script, /Math\.imul/);
  assert.match(script, /validateParity/);
  assert.doesNotMatch(script, /Math\.random/);
});
