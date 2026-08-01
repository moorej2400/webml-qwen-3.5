import assert from "node:assert/strict";
import test from "node:test";
import {
  createQwen35VisionProjectedOutput,
  executeQwen35VisionMerger,
  visionExactGeluCpu,
  visionMergerCpu,
  visionMergerReferenceCpu,
} from "../src/qwen35-vision-merger-kernels.js";
import { visionMerger as pinnedVisionMerger } from "../dev/reference/qwen35-vision-cpu-oracle.js";

test("treats four merge-block-major rows as one logical merger row without a shuffle copy", () => {
  const output = visionMergerCpu({
    hidden: Float32Array.from({ length: 4 * 1_024 }, (_, index) => (index % 17) / 17),
    patchCount: 4,
    postWeight: new Float32Array(1_024).fill(1), postBias: new Float32Array(1_024),
    inputWeight: new Uint16Array(4_096 * 4_096), inputBias: new Float32Array(4_096),
    outputWeight: new Uint16Array(4_096 * 2_560), outputBias: new Float32Array(2_560),
  });
  assert.equal(output.length, 2_560);
  assert.throws(() => visionMergerCpu({ hidden: new Float32Array(3 * 1_024), patchCount: 3, postWeight: new Float32Array(1_024), postBias: new Float32Array(1_024), inputWeight: new Uint16Array(4_096 * 4_096), inputBias: new Float32Array(4_096), outputWeight: new Uint16Array(4_096 * 2_560), outputBias: new Float32Array(2_560) }), /patch count/);
});

test("uses erf GELU rather than the transformer tanh approximation", () => {
  const output = visionExactGeluCpu(new Float32Array([-3, -1, 0, 1, 3]));
  assert.deepEqual(Array.from(output).map((value) => Number(value.toFixed(5))), [-0.00405, -0.15866, 0, 0.84134, 2.99595]);
});

test("matches the pinned reduced PyTorch merger oracle through post-LN, logical four-row view, both linears, and erf GELU", () => {
  const hiddenSize = 3; const merged = 12; const outputSize = 5;
  const identity = new Uint16Array(merged * merged); for (let index = 0; index < merged; index += 1) identity[index * merged + index] = 0x3f80;
  const outputWeight = new Uint16Array(merged * outputSize); for (let index = 0; index < outputSize; index += 1) outputWeight[index * merged + index] = 0x3f80;
  const common = { input: new Float32Array([0, 1, 4, 2, 7, 3, 9, 1, 5, 4, 6, 12]), patchCount: 4, hiddenSize, mergeSize: 2, normalizationWeight: new Float32Array(3).fill(1), normalizationBias: new Float32Array(3), inputWeight: identity, inputBias: new Float32Array(merged), intermediateSize: merged, outputWeight, outputBias: new Float32Array(outputSize), outputSize };
  const expected = pinnedVisionMerger(common);
  const actual = visionMergerReferenceCpu({ hidden: common.input, patchCount: 4, hiddenSize, intermediateSize: merged, outputSize, postWeight: common.normalizationWeight, postBias: common.normalizationBias, inputWeight: identity, inputBias: common.inputBias, outputWeight, outputBias: common.outputBias });
  // Checked-in PyTorch fixture: this guards the full operation sequence even
  // if the development oracle changes independently.
  const pytorchExpected = [-0.16022668778896332, -0.13627846539020538, 1.2562536001205444, -0.1641198843717575, 1.2742189168930054];
  assert.equal(actual.length, outputSize);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(Math.abs(actual[index]! - expected[index]!) < 5e-5, `oracle lane ${index}`);
    assert.ok(Math.abs(actual[index]! - pytorchExpected[index]!) < 5e-5, `PyTorch lane ${index}`);
  }
});

test("allocates a one-range projected output up to the 4096 visual-token contract", async () => {
  const requests: unknown[] = []; let retired = 0; let destroyed = 0;
  const output = await createQwen35VisionProjectedOutput({ arena: { async allocate(request) { requests.push(request); return { logicalBytes: request.byteLength, allocatedBytes: request.byteLength, shards: [{ buffer: { destroy() {} }, logicalByteOffset: 0n, logicalByteLength: request.byteLength, allocatedByteLength: request.byteLength }], destroy() { destroyed += 1; } }; } }, queue: { async onSubmittedWorkDone() { retired += 1; } }, visualTokenCount: 4_096, maxStorageBufferBindingSize: 4_096 * 2_560 * 4, allocationId: "projected" });
  assert.equal(output.storage.byteLength, 4_096 * 2_560 * 4);
  await output.dispose(); await output.dispose();
  assert.equal(requests.length, 1); assert.equal(retired, 1); assert.equal(destroyed, 1);
});

test("rejects malformed projected-output limits and malformed one-range allocations", async () => {
  const queue = { async onSubmittedWorkDone() {} };
  const arena = { async allocate(request: { readonly byteLength: bigint }) { return { logicalBytes: request.byteLength, allocatedBytes: request.byteLength, shards: [{ buffer: { destroy() {} }, logicalByteOffset: 0n, logicalByteLength: request.byteLength - 4n, allocatedByteLength: request.byteLength - 4n }], destroy() {} }; } };
  await assert.rejects(createQwen35VisionProjectedOutput({ arena, queue, visualTokenCount: 1, maxStorageBufferBindingSize: 0, allocationId: "projected" }), { code: "vision-projected-output-invalid" });
  await assert.rejects(createQwen35VisionProjectedOutput({ arena, queue, visualTokenCount: 1, maxStorageBufferBindingSize: 10_240, allocationId: "   " }), { code: "vision-projected-output-invalid" });
  await assert.rejects(createQwen35VisionProjectedOutput({ arena, queue, visualTokenCount: 1, maxStorageBufferBindingSize: 10_240, allocationId: "projected" }), { code: "vision-projected-output-invalid" });
});

test("merger execution cancels safely before dispatch and after GPU retirement", async () => {
  for (const point of ["before", "after"] as const) {
    const controller = new AbortController(); const events: string[] = [];
    if (point === "before") controller.abort();
    const projected = { storage: { buffer: {}, byteLength: 4 }, async dispose() { events.push("projected:dispose"); } };
    const plans = Array.from({ length: 4 }, () => ({ kernel: { id: "test", source: "x", entryPoint: "main" }, bindings: [], workgroups: { x: 1, y: 1, z: 1 }, uniformWords: [1, 0, 0, 0] as const }));
    await assert.rejects(executeQwen35VisionMerger({ plans, uniforms: Array.from({ length: 4 }, () => ({ update() { events.push("uniform"); } })), projected, signal: controller.signal, gpu: { async dispatchBatch() { events.push("dispatch"); }, async submittedWorkDone() { events.push("retire"); if (point === "after") controller.abort(); }, async dispose() { events.push("gpu:dispose"); } } }), { name: "AbortError" });
    assert.equal(events.includes("projected:dispose"), true);
    assert.equal(events.includes("gpu:dispose"), false);
    assert.equal(point === "before" ? !events.includes("dispatch") : events.includes("retire"), true);
  }
});

test("merger execution disposes GPU after dispatch or retirement failure even when the error resembles cancellation", async () => {
  for (const failure of ["dispatch", "retirement"] as const) {
    const events: string[] = [];
    const plans = Array.from({ length: 4 }, () => ({ kernel: { id: "test", source: "x", entryPoint: "main" }, bindings: [], workgroups: { x: 1, y: 1, z: 1 }, uniformWords: [1, 0, 0, 0] as const }));
    await assert.rejects(executeQwen35VisionMerger({ plans, uniforms: Array.from({ length: 4 }, () => ({ update() {} })), projected: { storage: { buffer: {}, byteLength: 4 }, async dispose() { events.push("projected:dispose"); } }, gpu: { async dispatchBatch() { events.push("dispatch"); if (failure === "dispatch") throw Object.assign(new Error("dispatch failure"), { name: "AbortError" }); }, async submittedWorkDone() { events.push("retire"); if (failure === "retirement") throw Object.assign(new Error("retirement failure"), { name: "AbortError" }); }, async dispose() { events.push("gpu:dispose"); } } }), /failure/);
    assert.equal(events.includes("gpu:dispose"), true);
    assert.equal(events.includes("projected:dispose"), true);
  }
});
