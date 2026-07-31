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
  assert.match(script, /for \(const kernel of LANGUAGE_GEMV_KERNELS\)/);
  assert.match(script, /results\.push\(await runKernel\(device, kernel\)\)/);
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
