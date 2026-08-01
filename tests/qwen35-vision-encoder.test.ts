import assert from "node:assert/strict";
import test from "node:test";
import { Qwen35VisionEncoder } from "../src/qwen35-vision-encoder.js";

test("runs bootstrap, foundation, streamed layers, and merger in fixed order", async () => {
  const events: string[] = [];
  const projected = { tokenCount: 1, storage: { buffer: {}, byteLength: 10_240 }, async dispose() { events.push("projected:dispose"); } };
  const bootstrap = { async destroy() { events.push("bootstrap:destroy"); } };
  const encoder = new Qwen35VisionEncoder({
    async stageBootstrap() { events.push("bootstrap:stage"); return bootstrap; },
    planFoundation(staged, input) { assert.equal(staged, bootstrap); assert.deepEqual(input, { gridHeight: 2, gridWidth: 2 }); return ["foundation:patch", "foundation:position", "foundation:rope"]; },
    async runFoundation(plans) { events.push(...plans); },
    async runLayers(input) { assert.equal(input.bootstrap, bootstrap); events.push("layers"); },
    async createProjectedOutput(input) { assert.deepEqual(input, { visualTokenCount: 1 }); events.push("projected:create"); return projected; },
    planMerger(staged, output) { assert.equal(staged, bootstrap); assert.equal(output, projected); events.push("merger:plan"); return ["merger"]; },
    async runMerger(plans, output) { assert.deepEqual(plans, ["merger"]); assert.equal(output, projected); events.push("merger"); return output; },
  });

  const result = await encoder.encode({ gridHeight: 2, gridWidth: 2 });
  assert.equal(result, projected);
  assert.deepEqual(events, ["bootstrap:stage", "foundation:patch", "foundation:position", "foundation:rope", "layers", "projected:create", "merger:plan", "merger"]);
  await encoder.dispose();
  assert.deepEqual(events.slice(-1), ["bootstrap:destroy"]);
  await projected.dispose();
  assert.deepEqual(events.slice(-2), ["bootstrap:destroy", "projected:dispose"]);
});

test("releases owned bootstrap and projected output when cancellation occurs before merger", async () => {
  const events: string[] = [];
  const controller = new AbortController();
  const projected = { tokenCount: 1, storage: { buffer: {}, byteLength: 10_240 }, async dispose() { events.push("projected:dispose"); } };
  const bootstrap = { async destroy() { events.push("bootstrap:destroy"); } };
  const encoder = new Qwen35VisionEncoder({
    async stageBootstrap() { return bootstrap; },
    planFoundation() { return ["foundation"]; },
    async runFoundation() { controller.abort(); },
    async runLayers() { throw new Error("layers must not run"); },
    async createProjectedOutput() { events.push("projected:create"); return projected; },
    planMerger() { return ["merger"]; },
    async runMerger() { return projected; },
  });

  await assert.rejects(encoder.encode({ gridHeight: 2, gridWidth: 2, signal: controller.signal }), { name: "AbortError" });
  assert.deepEqual(events, ["bootstrap:destroy"]);
  await encoder.dispose();
  assert.deepEqual(events, ["bootstrap:destroy"]);
});

test("preserves the encoding failure when owned cleanup also fails", async () => {
  const bootstrap = { async destroy() { throw new Error("cleanup failure"); } };
  const encoder = new Qwen35VisionEncoder({
    async stageBootstrap() { return bootstrap; },
    planFoundation() { return ["foundation"]; },
    async runFoundation() {},
    async runLayers() { throw new Error("layer failure"); },
    async createProjectedOutput() { throw new Error("must not create output"); },
    planMerger() { return ["merger"]; },
    async runMerger() { throw new Error("must not run merger"); },
  });

  await assert.rejects(encoder.encode({ gridHeight: 2, gridWidth: 2 }), /layer failure/);
});
