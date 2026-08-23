import assert from "node:assert/strict";
import test from "node:test";

import * as runtime from "../src/index.js";
import type { Qwen35WeightResidencyPolicy } from "../src/index.js";

const publicResidencyPolicy: Qwen35WeightResidencyPolicy = "auto";
void publicResidencyPolicy;

test("exports mixed tensor conversion and GEMV APIs from the package entrypoint", () => {
  assert.equal(typeof runtime.planConversion, "function");
  assert.equal(typeof runtime.validateModelPackageManifest, "function");
  assert.equal(typeof runtime.repackNativeQ6K, "function");
  assert.equal(typeof runtime.planGemvDispatch, "function");
  assert.equal(runtime.LANGUAGE_GEMV_KERNELS.length, 21);
  assert.equal(
    new Set(runtime.LANGUAGE_GEMV_KERNELS.map(({ id }) => id)).size,
    runtime.LANGUAGE_GEMV_KERNELS.length,
  );
  assert.equal(typeof runtime.createQwen35HybridState, "function");
  assert.equal(typeof runtime.fullAttentionDecodeCpu, "function");
  assert.equal(typeof runtime.gatedDeltaNetDecodeCpu, "function");
  assert.equal(runtime.QWEN35_HYBRID_KERNELS.length, 7);
  assert.equal(
    new Set(runtime.QWEN35_HYBRID_KERNELS.map(({ id }) => id)).size,
    runtime.QWEN35_HYBRID_KERNELS.length,
  );
  assert.equal(typeof runtime.Qwen35Tokenizer, "function");
  assert.equal(typeof runtime.loadPinnedQwen35Tokenizer, "function");
  assert.equal(typeof runtime.renderQwen35Chat, "function");
  assert.equal(typeof runtime.Qwen35Session, "function");
  assert.equal(typeof runtime.createQwen35UniformArena, "function");
  assert.equal(typeof runtime.planQwen35DeltaNetLayerDispatch, "function");
  assert.equal(typeof runtime.assembleQwen35TiledLogitsCommands, "function");
  assert.equal(typeof runtime.planQwen35FullAttentionLayerGeometry, "function");
  assert.equal(typeof runtime.planQwen35FullAttentionLayerDispatch, "function");
  assert.equal(typeof runtime.planQwen35FinalNormDispatch, "function");
  assert.equal(typeof runtime.createQwen35AllocationClearer, "function");
  assert.equal(typeof runtime.planQwen35GreedyUniformGeometry, "function");
  assert.equal(typeof runtime.createQwen35GreedyTextDriver, "function");
  assert.equal(
    typeof runtime.createQwen35GreedyExecutionDriverFactory,
    "function",
  );
  assert.equal(typeof runtime.createQwen35VisionProgram, "function");
  assert.equal("streamQwen35CachedWeights" in runtime, false);
  assert.equal("compileTokenizerSource" in runtime, false);
});
