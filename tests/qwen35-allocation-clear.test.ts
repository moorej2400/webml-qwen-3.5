import assert from "node:assert/strict";
import test from "node:test";

import type { GpuAllocation, GpuBufferLike } from "../src/gpu-arena.js";
import {
  createQwen35AllocationClearer,
} from "../src/qwen35-allocation-clear.js";
import { createQwen35PerformanceCounters } from "../src/qwen35-performance.js";
import type {
  Qwen35WebGpuBuffer,
  Qwen35WebGpuDevice,
} from "../src/qwen35-webgpu-executor.js";

interface FakeBuffer extends GpuBufferLike {
  readonly id: string;
}

function allocation(
  shards: readonly {
    readonly buffer: FakeBuffer;
    readonly logicalByteOffset: bigint;
    readonly logicalByteLength: bigint;
    readonly allocatedByteLength: bigint;
  }[],
  logicalBytes: bigint,
  allocatedBytes: bigint,
): GpuAllocation {
  return {
    shards,
    logicalBytes,
    allocatedBytes,
    destroy() {},
  };
}

function fakeDevice(options: {
  readonly submitError?: boolean;
  readonly validationError?: boolean;
  readonly retirementError?: boolean;
} = {}) {
  const events: string[] = [];
  const clears: Array<{
    readonly buffer: Qwen35WebGpuBuffer;
    readonly offset: number | undefined;
    readonly size: number | undefined;
  }> = [];
  const device: Qwen35WebGpuDevice = {
    limits: {
      minStorageBufferOffsetAlignment: 256,
      minUniformBufferOffsetAlignment: 256,
      maxStorageBufferBindingSize: 1_073_741_824,
      maxUniformBufferBindingSize: 65_536,
      maxComputeWorkgroupsPerDimension: 65_535,
    },
    queue: {
      writeBuffer() {},
      submit() {
        events.push("submit");
        if (options.submitError === true) {
          throw new Error("private submission detail");
        }
      },
      async onSubmittedWorkDone() {
        events.push("retired");
        if (options.retirementError === true) {
          throw new Error("private retirement detail");
        }
      },
    },
    pushErrorScope(filter) {
      events.push(`push:${filter}`);
    },
    async popErrorScope() {
      events.push("pop");
      return options.validationError === true
        ? { message: "private validation detail" }
        : null;
    },
    createCommandEncoder() {
      events.push("encoder");
      return {
        clearBuffer(buffer, offset, size) {
          clears.push({ buffer, offset, size });
        },
        beginComputePass() {
          throw new Error("not used");
        },
        copyBufferToBuffer() {
          throw new Error("not used");
        },
        finish() {
          events.push("finish");
          return {};
        },
      };
    },
    createShaderModule() {
      throw new Error("not used");
    },
    createComputePipelineAsync() {
      throw new Error("not used");
    },
    createBindGroup() {
      throw new Error("not used");
    },
    createBuffer() {
      throw new Error("not used");
    },
  };
  return { device, events, clears };
}

test("clears every allocation shard in one submission and waits for retirement", async () => {
  const { device, events, clears } = fakeDevice();
  const first: FakeBuffer = { id: "first", destroy() {} };
  const second: FakeBuffer = { id: "second", destroy() {} };
  const clearer = createQwen35AllocationClearer(device);

  await clearer.clearAllocation(allocation([
    {
      buffer: first,
      logicalByteOffset: 0n,
      logicalByteLength: 32n,
      allocatedByteLength: 32n,
    },
    {
      buffer: second,
      logicalByteOffset: 32n,
      logicalByteLength: 12n,
      allocatedByteLength: 16n,
    },
  ], 44n, 48n));

  assert.deepEqual(clears, [
    { buffer: first, offset: 0, size: 32 },
    { buffer: second, offset: 0, size: 16 },
  ]);
  assert.deepEqual(events, [
    "push:validation",
    "encoder",
    "finish",
    "submit",
    "pop",
    "retired",
  ]);
});

test("accounts allocation clear submissions and retirements", async () => {
  const { device } = fakeDevice();
  const counters = createQwen35PerformanceCounters();
  const buffer: FakeBuffer = { id: "buffer", destroy() {} };
  const clearer = createQwen35AllocationClearer(device, counters);

  await clearer.clearAllocation(allocation([{
    buffer,
    logicalByteOffset: 0n,
    logicalByteLength: 16n,
    allocatedByteLength: 16n,
  }], 16n, 16n));

  assert.equal(counters.snapshot().queueSubmissionCount, 1);
  assert.equal(counters.snapshot().queueRetirementCount, 1);
});

test("rejects malformed logical coverage and clear alignment before submission", async () => {
  const { device, events } = fakeDevice();
  const buffer: FakeBuffer = { id: "buffer", destroy() {} };
  const clearer = createQwen35AllocationClearer(device);

  await assert.rejects(clearer.clearAllocation(allocation([
    {
      buffer,
      logicalByteOffset: 4n,
      logicalByteLength: 8n,
      allocatedByteLength: 10n,
    },
  ], 8n, 10n)), {
    code: "webgpu-allocation-clear-invalid",
    message: "Qwen3.5 GPU allocation clear metadata is invalid",
  });
  assert.deepEqual(events, []);
});

test("fails closed after a scoped validation error and still retires submitted work", async () => {
  const { device, events } = fakeDevice({ validationError: true });
  const buffer: FakeBuffer = { id: "buffer", destroy() {} };
  const target = allocation([{
    buffer,
    logicalByteOffset: 0n,
    logicalByteLength: 16n,
    allocatedByteLength: 16n,
  }], 16n, 16n);
  const clearer = createQwen35AllocationClearer(device);

  await assert.rejects(clearer.clearAllocation(target), {
    code: "webgpu-allocation-clear-validation-failed",
    message: "Qwen3.5 GPU allocation clear validation failed",
  });
  assert.equal(events.includes("retired"), true);
  await assert.rejects(clearer.clearAllocation(target), {
    code: "webgpu-allocation-clear-poisoned",
  });
});

test("sanitizes queue submission failures and fails closed", async () => {
  const { device } = fakeDevice({ submitError: true });
  const buffer: FakeBuffer = { id: "buffer", destroy() {} };
  const target = allocation([{
    buffer,
    logicalByteOffset: 0n,
    logicalByteLength: 16n,
    allocatedByteLength: 16n,
  }], 16n, 16n);
  const clearer = createQwen35AllocationClearer(device);

  await assert.rejects(clearer.clearAllocation(target), {
    code: "webgpu-allocation-clear-submission-failed",
    message: "Qwen3.5 GPU allocation clear submission failed",
  });
  await assert.rejects(clearer.clearAllocation(target), {
    code: "webgpu-allocation-clear-poisoned",
  });
});

test("sanitizes queue retirement failures and fails closed", async () => {
  const { device } = fakeDevice({ retirementError: true });
  const buffer: FakeBuffer = { id: "buffer", destroy() {} };
  const target = allocation([{
    buffer,
    logicalByteOffset: 0n,
    logicalByteLength: 16n,
    allocatedByteLength: 16n,
  }], 16n, 16n);
  const clearer = createQwen35AllocationClearer(device);

  await assert.rejects(clearer.clearAllocation(target), {
    code: "webgpu-allocation-clear-retirement-failed",
    message: "Qwen3.5 GPU allocation clear retirement failed",
  });
  await assert.rejects(clearer.clearAllocation(target), {
    code: "webgpu-allocation-clear-poisoned",
  });
});
