import { diagnosticError } from "./diagnostics.js";

/** Public-safe counters for one loaded runtime. No prompt or output text is stored. */
export interface Qwen35PerformanceSnapshot {
  readonly permanentGpuBytes: number;
  readonly transientGpuBytes: number;
  readonly stateGpuBytes: number;
  readonly diskReadBytes: number;
  readonly gpuUploadBytes: number;
  readonly dispatchCount: number;
  readonly queueSubmissionCount: number;
  readonly queueRetirementCount: number;
  readonly gpuReadbackCount: number;
}

export interface Qwen35PerformanceCounters {
  recordGpuUpload(bytes: number): void;
  recordDispatch(count?: number): void;
  recordQueueSubmission(count?: number): void;
  recordQueueRetirement(count?: number): void;
  recordGpuReadback(count?: number): void;
  recordDiskRead(bytes: number): void;
  setGpuBreakdown(input: {
    readonly permanentGpuBytes: number;
    readonly transientGpuBytes: number;
    readonly stateGpuBytes: number;
  }): void;
  snapshot(): Qwen35PerformanceSnapshot;
}

export interface Qwen35PerformanceWriteQueue {
  writeBuffer(
    buffer: object,
    bufferOffset: number,
    data: ArrayBuffer | Uint8Array<ArrayBufferLike>,
    dataOffset?: number,
    size?: number,
  ): void;
  onSubmittedWorkDone(): Promise<void>;
}

export interface Qwen35PerformanceBudget {
  readonly maxDiskReadBytesPerToken?: number;
  readonly maxGpuUploadBytesPerToken?: number;
  readonly maxDispatchesPerToken?: number;
  readonly maxQueueSubmissionsPerToken?: number;
  readonly maxQueueRetirementsPerToken?: number;
  readonly maxGpuReadbacksPerToken?: number;
}

const ZERO_PERFORMANCE_SNAPSHOT: Qwen35PerformanceSnapshot = Object.freeze({
  permanentGpuBytes: 0,
  transientGpuBytes: 0,
  stateGpuBytes: 0,
  diskReadBytes: 0,
  gpuUploadBytes: 0,
  dispatchCount: 0,
  queueSubmissionCount: 0,
  queueRetirementCount: 0,
  gpuReadbackCount: 0,
});

const COUNTER_FIELDS = Object.freeze([
  "permanentGpuBytes",
  "transientGpuBytes",
  "stateGpuBytes",
  "diskReadBytes",
  "gpuUploadBytes",
  "dispatchCount",
  "queueSubmissionCount",
  "queueRetirementCount",
  "gpuReadbackCount",
] as const);

type CounterField = (typeof COUNTER_FIELDS)[number];

function requireNonnegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw diagnosticError(
      "performance-metric-invalid",
      `Qwen3.5 ${label} is invalid`,
    );
  }
}

function addCounter(
  counters: Record<CounterField, number>,
  field: CounterField,
  amount: number,
): void {
  requireNonnegativeSafeInteger(amount, field);
  const next = counters[field] + amount;
  requireNonnegativeSafeInteger(next, field);
  counters[field] = next;
}

function snapshotOf(
  counters: Readonly<Record<CounterField, number>>,
): Qwen35PerformanceSnapshot {
  return Object.freeze({
    permanentGpuBytes: counters.permanentGpuBytes,
    transientGpuBytes: counters.transientGpuBytes,
    stateGpuBytes: counters.stateGpuBytes,
    diskReadBytes: counters.diskReadBytes,
    gpuUploadBytes: counters.gpuUploadBytes,
    dispatchCount: counters.dispatchCount,
    queueSubmissionCount: counters.queueSubmissionCount,
    queueRetirementCount: counters.queueRetirementCount,
    gpuReadbackCount: counters.gpuReadbackCount,
  });
}

/** Returns operation deltas without mixing model-load traffic into decode budgets. */
export function diffQwen35PerformanceSnapshots(
  after: Qwen35PerformanceSnapshot,
  before: Qwen35PerformanceSnapshot = ZERO_PERFORMANCE_SNAPSHOT,
): Qwen35PerformanceSnapshot {
  const values = Object.fromEntries(
    COUNTER_FIELDS.map((field) => {
      requireNonnegativeSafeInteger(after[field], field);
      requireNonnegativeSafeInteger(before[field], field);
      if (after[field] < before[field]) {
        throw diagnosticError(
          "performance-baseline-invalid",
          "Qwen3.5 performance baseline is newer than the measured snapshot",
        );
      }
      return [field, after[field] - before[field]];
    }),
  ) as Record<CounterField, number>;
  return snapshotOf(values);
}

export function createQwen35PerformanceCounters(): Qwen35PerformanceCounters {
  const counters: Record<CounterField, number> = {
    permanentGpuBytes: 0,
    transientGpuBytes: 0,
    stateGpuBytes: 0,
    diskReadBytes: 0,
    gpuUploadBytes: 0,
    dispatchCount: 0,
    queueSubmissionCount: 0,
    queueRetirementCount: 0,
    gpuReadbackCount: 0,
  };
  return Object.freeze({
    recordGpuUpload(bytes = 1): void {
      addCounter(counters, "gpuUploadBytes", bytes);
    },
    recordDispatch(count = 1): void {
      addCounter(counters, "dispatchCount", count);
    },
    recordQueueSubmission(count = 1): void {
      addCounter(counters, "queueSubmissionCount", count);
    },
    recordQueueRetirement(count = 1): void {
      addCounter(counters, "queueRetirementCount", count);
    },
    recordGpuReadback(count = 1): void {
      addCounter(counters, "gpuReadbackCount", count);
    },
    recordDiskRead(bytes: number): void {
      addCounter(counters, "diskReadBytes", bytes);
    },
    setGpuBreakdown(input: {
      readonly permanentGpuBytes: number;
      readonly transientGpuBytes: number;
      readonly stateGpuBytes: number;
    }): void {
      requireNonnegativeSafeInteger(input.permanentGpuBytes, "permanent GPU bytes");
      requireNonnegativeSafeInteger(input.transientGpuBytes, "transient GPU bytes");
      requireNonnegativeSafeInteger(input.stateGpuBytes, "state GPU bytes");
      counters.permanentGpuBytes = input.permanentGpuBytes;
      counters.transientGpuBytes = input.transientGpuBytes;
      counters.stateGpuBytes = input.stateGpuBytes;
    },
    snapshot(): Qwen35PerformanceSnapshot {
      return snapshotOf(counters);
    },
  });
}

/** Counts writes and queue retirement without changing WebGPU ownership. */
export function createQwen35PerformanceWriteQueue(input: {
  readonly queue: Qwen35PerformanceWriteQueue;
  readonly counters: Qwen35PerformanceCounters;
}): Qwen35PerformanceWriteQueue {
  return Object.freeze({
    writeBuffer(
      buffer: object,
      bufferOffset: number,
      data: ArrayBuffer | Uint8Array<ArrayBufferLike>,
      dataOffset = 0,
      size = data.byteLength - dataOffset,
    ): void {
      input.queue.writeBuffer(buffer, bufferOffset, data, dataOffset, size);
      input.counters.recordGpuUpload(size);
    },
    async onSubmittedWorkDone(): Promise<void> {
      await input.queue.onSubmittedWorkDone();
      input.counters.recordQueueRetirement();
    },
  });
}

function budgetValue(value: number | undefined, label: string): number | undefined {
  if (value === undefined) return undefined;
  requireNonnegativeSafeInteger(value, label);
  return value;
}

/** Fails only when an explicit per-token regression budget is exceeded. */
export function assertQwen35PerformanceBudget(input: {
  readonly metrics: Qwen35PerformanceSnapshot;
  /** Optional snapshot captured after load/prefill and before the measured run. */
  readonly baseline?: Qwen35PerformanceSnapshot;
  readonly generatedTokens: number;
  readonly budget: Qwen35PerformanceBudget;
}): void {
  requireNonnegativeSafeInteger(input.generatedTokens, "generated token count");
  if (input.generatedTokens === 0) return;
  const metrics = diffQwen35PerformanceSnapshots(input.metrics, input.baseline);
  const checks: readonly [keyof Qwen35PerformanceBudget, number][] = [
    ["maxDiskReadBytesPerToken", metrics.diskReadBytes],
    ["maxGpuUploadBytesPerToken", metrics.gpuUploadBytes],
    ["maxDispatchesPerToken", metrics.dispatchCount],
    ["maxQueueSubmissionsPerToken", metrics.queueSubmissionCount],
    ["maxQueueRetirementsPerToken", metrics.queueRetirementCount],
    ["maxGpuReadbacksPerToken", metrics.gpuReadbackCount],
  ];
  for (const [budgetKey, total] of checks) {
    const budget = budgetValue(input.budget[budgetKey], budgetKey);
    if (budget !== undefined && total / input.generatedTokens > budget) {
      throw diagnosticError(
        "performance-regression",
        "Qwen3.5 runtime performance budget was exceeded",
      );
    }
  }
}
