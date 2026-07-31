import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("ships a deterministic browser compile and CPU parity harness", async () => {
  const html = await readFile(
    new URL("../tools/webgpu-kernel-harness.html", import.meta.url),
    "utf8",
  );
  const script = await readFile(
    new URL("../tools/webgpu-kernel-harness.mjs", import.meta.url),
    "utf8",
  );

  assert.match(html, /webgpu-kernel-harness\.mjs/);
  assert.match(script, /LANGUAGE_GEMV_KERNELS/);
  assert.match(script, /getCompilationInfo/);
  assert.match(script, /gemvCpu/);
  assert.match(script, /dispatchWorkgroups/);
  assert.match(script, /Math\.imul/);
  assert.match(script, /tolerance/);
  assert.doesNotMatch(script, /Math\.random/);
});
