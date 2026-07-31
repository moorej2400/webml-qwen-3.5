import assert from "node:assert/strict";
import test from "node:test";

import * as runtime from "../src/index.js";

test("exports mixed tensor conversion and GEMV APIs from the package entrypoint", () => {
  assert.equal(typeof runtime.planConversion, "function");
  assert.equal(typeof runtime.validateModelPackageManifest, "function");
  assert.equal(typeof runtime.repackNativeQ6K, "function");
  assert.equal(typeof runtime.planGemvDispatch, "function");
  assert.equal(runtime.LANGUAGE_GEMV_KERNELS.length, 6);
  assert.equal(typeof runtime.createQwen35HybridState, "function");
  assert.equal(typeof runtime.fullAttentionDecodeCpu, "function");
  assert.equal(typeof runtime.gatedDeltaNetDecodeCpu, "function");
  assert.equal(runtime.QWEN35_HYBRID_KERNELS.length, 6);
  assert.equal(typeof runtime.Qwen35Tokenizer, "function");
  assert.equal(typeof runtime.loadPinnedQwen35Tokenizer, "function");
  assert.equal(typeof runtime.renderQwen35Chat, "function");
  assert.equal(typeof runtime.Qwen35Session, "function");
  assert.equal("streamQwen35CachedWeights" in runtime, false);
  assert.equal("compileTokenizerSource" in runtime, false);
});
