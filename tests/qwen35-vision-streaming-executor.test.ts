import assert from "node:assert/strict";
import test from "node:test";
import { AllocationLedger } from "../src/allocation-ledger.js";
import {
  Qwen35VisionStreamingExecutor,
  createQwen35VisionStreamingOwnedActivationWorkspace,
  type Qwen35VisionStreamingDependencies,
} from "../src/qwen35-vision-streaming-executor.js";
import type { GpuAllocation, GpuAllocationRequest } from "../src/gpu-arena.js";

type Event = string;

function fixture(options: {
  readonly cancelAt?: "pre-stage" | "post-stage" | "between-layers";
  readonly failAt?: "dispatch" | "retirement" | "stage" | "destroy";
  readonly failDestroyToo?: boolean;
  readonly throwNow?: boolean;
  readonly disposeGpu?: boolean;
} = {}): { readonly executor: Qwen35VisionStreamingExecutor; readonly events: Event[]; readonly ledger: AllocationLedger; readonly controller: AbortController } {
  const events: Event[] = [];
  const ledger = new AllocationLedger(1_000_000n);
  const activation = ledger.reserve({ id: "vision-activations", category: "activation", bytes: 100n });
  const uniforms = ledger.reserve({ id: "vision-uniforms", category: "activation", bytes: 100n });
  const uniformBuffer = {};
  let active = 0;
  let maxActive = 0;
  let clock = 0;
  let destroyAttempts = 0;
  const controller = new AbortController();
  const dependencies: Qwen35VisionStreamingDependencies = {
    limits: { minStorageBufferOffsetAlignment: 4, minUniformBufferOffsetAlignment: 4, maxStorageBufferBindingSize: 1_000_000, maxUniformBufferBindingSize: 16, maxComputeWorkgroupsPerDimension: 16_384 },
    resources: {
      workspace: { hidden: {}, normalized: {}, qkv: {}, attention: {}, mlp: {}, rope: {}, segmentOffsets: {} },
      preparedTokenCount: 2,
      preparedSegmentCount: 1,
      uniformSlots: Array.from({ length: 10 }, (_, index) => ({ index, binding: { buffer: uniformBuffer, byteOffset: index * 16, byteLength: 16 }, update(words: Uint32Array) { events.push(`uniform:${index}:${[...words].join(",")}`); } })),
      async dispose() { events.push("resources:dispose"); ledger.release(uniforms); ledger.release(activation); },
    },
    async stageLayer(layer) {
      events.push(`stage:${layer}`);
      if (options.failAt === "stage" && layer === 0) throw new Error("stage failure");
      active += 1; maxActive = Math.max(maxActive, active);
      if (options.cancelAt === "post-stage" && layer === 0) controller.abort();
      return {
        layer,
        shards: [],
        tensors: [],
        async destroy() {
          events.push(`destroy:${layer}`);
          active -= 1;
          destroyAttempts += 1;
          if ((options.failAt === "destroy" || options.failDestroyToo && destroyAttempts === 1) && layer === 0) throw new Error("destroy failure");
        },
      };
    },
    planLayer(input) {
      return Array.from({ length: 11 }, (_, stage) => ({
        kernel: { id: `kernel-${stage}`, source: "@compute @workgroup_size(1) fn main() {}", entryPoint: "main" },
        bindings: [], workgroups: { x: 1, y: 1, z: 1 }, uniformWords: [input.tokenCount, input.layer, stage, 0] as const,
      }));
    },
    gpu: {
      async dispatchBatch(plans) {
        events.push(`dispatch:${plans[0]!.uniformWords[1]}:${plans.length}`);
        if (options.failAt === "dispatch") throw new Error("dispatch failure");
      },
      async submittedWorkDone() {
        events.push("retire");
        if (options.failAt === "retirement") throw new Error("retirement failure");
      },
      async dispose() { events.push("gpu:dispose"); },
    },
    disposeGpu: options.disposeGpu,
    now: () => { if (options.throwNow) throw new Error("clock failure"); return ++clock; },
    onLayerComplete(layer) {
      if (options.cancelAt === "between-layers" && layer === 0) controller.abort();
    },
  };
  const executor = new Qwen35VisionStreamingExecutor(dependencies);
  return { executor, events, ledger, controller };
}

test("streams the fixed 24-layer order with one group resident and exact uniform uploads", async () => {
  const { executor, events, ledger, controller } = fixture();
  const result = await executor.run({ tokenCount: 2, segmentCount: 1, signal: controller.signal });
  assert.deepEqual(result.layerOrder, Array.from({ length: 24 }, (_, layer) => layer));
  assert.equal(events.filter((event) => event.startsWith("dispatch:")).length, 24);
  assert.equal(events.filter((event) => event.startsWith("uniform:")).length, 24 * 11);
  assert.equal(executor.getMetrics().dispatchCount, 24 * 11);
  assert.equal(executor.getMetrics().maxResidentLayerGroups, 1);
  assert.equal(executor.getMetrics().layers.length, 24);
  for (const metric of executor.getMetrics().layers) {
    assert.equal(metric.layer >= 0 && metric.layer < 24, true);
    assert.equal(Object.values(metric).every((value) => typeof value !== "number" || Number.isFinite(value)), true);
  }
  for (let layer = 0; layer < 24; layer += 1) {
    const dispatch = events.indexOf(`dispatch:${layer}:11`);
    const retire = events.indexOf("retire", dispatch);
    const destroy = events.indexOf(`destroy:${layer}`);
    assert.equal(events.indexOf(`stage:${layer}`) < dispatch, true);
    assert.equal(dispatch < retire && retire < destroy, true);
    if (layer < 23) assert.equal(destroy < events.indexOf(`stage:${layer + 1}`), true);
  }
  await executor.dispose();
  await executor.dispose();
  assert.equal(ledger.snapshot().allocationCount, 0);
});

test("releases vision resources without disposing a borrowed language GPU executor", async () => {
  const { executor, events, ledger, controller } = fixture({ disposeGpu: false });
  await executor.run({ tokenCount: 2, segmentCount: 1, signal: controller.signal });
  await executor.dispose();
  assert.equal(events.includes("gpu:dispose"), false);
  assert.equal(ledger.snapshot().allocationCount, 0);
});

test("cancellation before staging, after staging, and between layers releases all owned resources", async () => {
  for (const cancelAt of ["pre-stage", "post-stage", "between-layers"] as const) {
    const { executor, events, ledger, controller } = fixture({ cancelAt });
    if (cancelAt === "pre-stage") controller.abort();
    await assert.rejects(executor.run({ tokenCount: 2, segmentCount: 1, signal: controller.signal }), (error: unknown) => (error as { name?: string }).name === "AbortError");
    assert.equal(events.filter((event) => event.startsWith("destroy:")).length, cancelAt === "pre-stage" ? 0 : cancelAt === "post-stage" ? 1 : 1);
    assert.equal(ledger.snapshot().allocationCount, 0);
    assert.equal(executor.getMetrics().state, "cancelled");
  }
});

test("keeps the primary execution error when cleanup also fails", async () => {
  for (const failAt of ["dispatch", "retirement", "stage", "destroy"] as const) {
    const { executor, ledger } = fixture({ failAt });
    await assert.rejects(executor.run({ tokenCount: 2, segmentCount: 1 }), /failure/);
    assert.equal(ledger.snapshot().allocationCount, 0);
    assert.equal(executor.getMetrics().state, "failed");
  }
});

test("keeps dispatch failure when the staged layer cleanup also fails", async () => {
  const { executor, ledger } = fixture({ failAt: "dispatch", failDestroyToo: true });
  await assert.rejects(executor.run({ tokenCount: 2, segmentCount: 1 }), /dispatch failure/);
  assert.equal(ledger.snapshot().allocationCount, 0);
});

test("fences a failed submission before it releases staged weights", async () => {
  for (const failAt of ["dispatch", "retirement"] as const) {
    const { executor, events, ledger } = fixture({ failAt });
    await assert.rejects(executor.run({ tokenCount: 2, segmentCount: 1 }), /failure/);
    assert.equal(events.indexOf(`dispatch:0:11`) < events.indexOf("gpu:dispose"), true);
    assert.equal(events.indexOf("gpu:dispose") < events.indexOf("destroy:0"), true);
    assert.equal(events.indexOf("destroy:0") < events.indexOf("resources:dispose"), true);
    assert.equal(ledger.snapshot().allocationCount, 0);
  }
});

test("does not report a failed staged cleanup as released", async () => {
  const { executor, ledger } = fixture({ failAt: "destroy" });
  await assert.rejects(executor.run({ tokenCount: 2, segmentCount: 1 }), /destroy failure/);
  assert.equal(executor.getMetrics().unresolvedLayerGroups, 1);
  assert.equal(executor.getMetrics().completedLayers, 0);
  assert.equal(ledger.snapshot().allocationCount, 0);
  await assert.rejects(executor.dispose(), /destroy failure/);
  assert.equal(executor.getMetrics().unresolvedLayerGroups, 1);
});

test("rejects an input that does not match the uploaded segment workspace and ignores throwing timing", async () => {
  const { executor } = fixture();
  await assert.rejects(executor.run({ tokenCount: 2, segmentCount: 2 }), /input is invalid/);
  await executor.dispose();
  const timed = fixture({ throwNow: true });
  await timed.executor.run({ tokenCount: 2, segmentCount: 1 });
  await timed.executor.dispose();
});

test("external dispose waits for active layer cleanup and prevents a second run", async () => {
  const { executor, events, ledger } = fixture();
  const running = executor.run({ tokenCount: 2, segmentCount: 1 });
  const disposing = executor.dispose();
  await assert.rejects(executor.run({ tokenCount: 2, segmentCount: 1 }), /not ready/);
  await running;
  await disposing;
  assert.equal(events.indexOf("destroy:23") < events.indexOf("resources:dispose"), true);
  assert.equal(executor.getMetrics().state, "disposed");
  assert.equal(ledger.snapshot().allocationCount, 0);
});

test("allocates one exact bindable activation range per vision workspace resource and rolls back in reverse", async () => {
  const requests: GpuAllocationRequest[] = [];
  const destroyed: string[] = [];
  const arena = {
    async allocate(request: GpuAllocationRequest): Promise<GpuAllocation> {
      requests.push(request);
      const buffer = {};
      return { logicalBytes: request.byteLength, allocatedBytes: request.byteLength, shards: [{ buffer: { destroy() {} }, logicalByteOffset: 0n, logicalByteLength: request.byteLength, allocatedByteLength: request.byteLength }], destroy() { destroyed.push(request.id); } };
    },
  };
  const uploads: Uint32Array[] = [];
  const workspace = await createQwen35VisionStreamingOwnedActivationWorkspace({ arena, queue: { writeBuffer(_buffer, _offset, data, dataOffset, size) { uploads.push(new Uint32Array(data.slice(dataOffset, dataOffset + size))); }, async onSubmittedWorkDone() {} }, tokenCount: 2, segmentOffsets: new Uint32Array([0, 2]), minStorageBufferOffsetAlignment: 256, maxStorageBufferBindingSize: 100_000_000, allocationIdPrefix: "test" });
  assert.deepEqual(requests.map((request) => request.id), ["test-hidden", "test-normalized", "test-qkv", "test-attention", "test-mlp", "test-rope", "test-segments"]);
  assert.equal(workspace.hidden.byteLength, 2 * 1_024 * 4);
  assert.equal(workspace.qkv.byteLength, 2 * 1_024 * 3 * 4);
  assert.equal(workspace.mlp.byteLength, 2 * 4_096 * 4);
  assert.equal(requests.at(-1)!.alignment, 4);
  assert.equal(requests.at(-1)!.requiredShardQuantumBytes, 8n);
  assert.deepEqual(uploads, [new Uint32Array([0, 2])]);
  await workspace.dispose();
  assert.deepEqual(destroyed, [...requests].reverse().map((request) => request.id));

  const rollback: string[] = [];
  let count = 0;
  await assert.rejects(createQwen35VisionStreamingOwnedActivationWorkspace({ arena: { async allocate(request) { count += 1; if (count === 3) throw new Error("allocation failed"); return { logicalBytes: request.byteLength, allocatedBytes: request.byteLength, shards: [{ buffer: { destroy() {} }, logicalByteOffset: 0n, logicalByteLength: request.byteLength, allocatedByteLength: request.byteLength }], destroy() { rollback.push(request.id); } }; } }, queue: { writeBuffer() {}, async onSubmittedWorkDone() {} }, tokenCount: 2, segmentOffsets: new Uint32Array([0, 2]), minStorageBufferOffsetAlignment: 4, maxStorageBufferBindingSize: 100_000_000, allocationIdPrefix: "rollback" }), /allocation failed/);
  assert.deepEqual(rollback, ["rollback-normalized", "rollback-hidden"]);

  const uploadRollback: string[] = [];
  await assert.rejects(createQwen35VisionStreamingOwnedActivationWorkspace({ arena: { async allocate(request) { return { logicalBytes: request.byteLength, allocatedBytes: request.byteLength, shards: [{ buffer: { destroy() {} }, logicalByteOffset: 0n, logicalByteLength: request.byteLength, allocatedByteLength: request.byteLength }], destroy() { uploadRollback.push(request.id); } }; } }, queue: { writeBuffer() { throw new Error("upload"); }, async onSubmittedWorkDone() {} }, tokenCount: 2, segmentOffsets: new Uint32Array([0, 2]), minStorageBufferOffsetAlignment: 4, maxStorageBufferBindingSize: 100_000_000, allocationIdPrefix: "upload" }), /segment upload failed/);
  assert.deepEqual(uploadRollback, ["upload-segments", "upload-rope", "upload-mlp", "upload-attention", "upload-qkv", "upload-normalized", "upload-hidden"]);
});
