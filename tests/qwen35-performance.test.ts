import assert from "node:assert/strict";
import test from "node:test";

import {
  assertQwen35PerformanceBudget,
  createQwen35PerformanceCounters,
  createQwen35PerformanceWriteQueue,
  diffQwen35PerformanceSnapshots,
} from "../src/qwen35-performance.js";

test("performance counters expose bounded GPU and I/O metrics", () => {
  const counters = createQwen35PerformanceCounters();
  counters.recordGpuUpload(64);
  counters.recordDispatch(3);
  counters.recordQueueSubmission();
  counters.recordQueueRetirement(2);
  counters.recordGpuReadback();
  counters.setGpuBreakdown({
    permanentGpuBytes: 100,
    transientGpuBytes: 20,
    stateGpuBytes: 30,
  });

  assert.deepEqual(counters.snapshot(), {
    permanentGpuBytes: 100,
    transientGpuBytes: 20,
    stateGpuBytes: 30,
    diskReadBytes: 0,
    gpuUploadBytes: 64,
    dispatchCount: 3,
    queueSubmissionCount: 1,
    queueRetirementCount: 2,
    gpuReadbackCount: 1,
  });
});

test("performance write queue counts successful writes and retirements", async () => {
  let writes = 0;
  let retirements = 0;
  const counters = createQwen35PerformanceCounters();
  const queue = createQwen35PerformanceWriteQueue({
    queue: {
      writeBuffer(_buffer, _offset, _data, _dataOffset, size) {
        writes += size;
      },
      async onSubmittedWorkDone() {
        retirements += 1;
      },
    },
    counters,
  });

  queue.writeBuffer({}, 0, new Uint8Array(8), 2, 4);
  await queue.onSubmittedWorkDone();

  assert.equal(writes, 4);
  assert.equal(retirements, 1);
  assert.equal(counters.snapshot().gpuUploadBytes, 4);
  assert.equal(counters.snapshot().queueRetirementCount, 1);
});

test("performance budgets reject regressions per generated token", () => {
  assert.doesNotThrow(() => assertQwen35PerformanceBudget({
    metrics: {
      permanentGpuBytes: 1,
      transientGpuBytes: 2,
      stateGpuBytes: 3,
      diskReadBytes: 20,
      gpuUploadBytes: 10,
      dispatchCount: 8,
      queueSubmissionCount: 4,
      queueRetirementCount: 2,
      gpuReadbackCount: 1,
    },
    generatedTokens: 2,
    budget: {
      maxDiskReadBytesPerToken: 10,
      maxGpuUploadBytesPerToken: 5,
      maxDispatchesPerToken: 4,
      maxQueueSubmissionsPerToken: 2,
      maxQueueRetirementsPerToken: 1,
      maxGpuReadbacksPerToken: 1,
    },
  }));

  assert.throws(() => assertQwen35PerformanceBudget({
    metrics: {
      permanentGpuBytes: 1,
      transientGpuBytes: 2,
      stateGpuBytes: 3,
      diskReadBytes: 21,
      gpuUploadBytes: 10,
      dispatchCount: 8,
      queueSubmissionCount: 4,
      queueRetirementCount: 2,
      gpuReadbackCount: 1,
    },
    generatedTokens: 2,
    budget: { maxDiskReadBytesPerToken: 10 },
  }), { code: "performance-regression" });
});

test("performance budgets can subtract load and prefill baselines", () => {
  const baseline = {
    permanentGpuBytes: 100,
    transientGpuBytes: 20,
    stateGpuBytes: 30,
    diskReadBytes: 1_000,
    gpuUploadBytes: 2_000,
    dispatchCount: 50,
    queueSubmissionCount: 25,
    queueRetirementCount: 10,
    gpuReadbackCount: 1,
  } as const;
  const after = {
    ...baseline,
    diskReadBytes: 1_020,
    gpuUploadBytes: 2_010,
    dispatchCount: 58,
    queueSubmissionCount: 29,
    queueRetirementCount: 12,
    gpuReadbackCount: 3,
  } as const;

  assert.deepEqual(diffQwen35PerformanceSnapshots(after, baseline), {
    permanentGpuBytes: 0,
    transientGpuBytes: 0,
    stateGpuBytes: 0,
    diskReadBytes: 20,
    gpuUploadBytes: 10,
    dispatchCount: 8,
    queueSubmissionCount: 4,
    queueRetirementCount: 2,
    gpuReadbackCount: 2,
  });
  assert.doesNotThrow(() => assertQwen35PerformanceBudget({
    metrics: after,
    baseline,
    generatedTokens: 2,
    budget: {
      maxDiskReadBytesPerToken: 10,
      maxGpuUploadBytesPerToken: 5,
      maxDispatchesPerToken: 4,
      maxQueueSubmissionsPerToken: 2,
      maxQueueRetirementsPerToken: 1,
      maxGpuReadbacksPerToken: 1,
    },
  }));
  assert.throws(() => diffQwen35PerformanceSnapshots(baseline, after), {
    code: "performance-baseline-invalid",
  });
});
