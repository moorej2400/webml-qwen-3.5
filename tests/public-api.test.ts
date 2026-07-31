import assert from "node:assert/strict";
import test from "node:test";

import * as runtime from "../src/index.js";

test("exports mixed tensor conversion and GEMV APIs from the package entrypoint", () => {
  assert.equal(typeof runtime.planConversion, "function");
  assert.equal(typeof runtime.validateModelPackageManifest, "function");
  assert.equal(typeof runtime.repackNativeQ6K, "function");
  assert.equal(typeof runtime.planGemvDispatch, "function");
  assert.equal(runtime.LANGUAGE_GEMV_KERNELS.length, 6);
});
