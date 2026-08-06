import { diagnosticError } from "./diagnostics.js";
import type { GpuAllocation } from "./gpu-arena.js";
import type {
  Qwen35WebGpuBuffer,
  Qwen35WebGpuDevice,
} from "./qwen35-webgpu-executor.js";
import type { Qwen35PerformanceCounters } from "./qwen35-performance.js";

interface ClearRange {
  readonly buffer: Qwen35WebGpuBuffer;
  readonly size: number;
}

export interface Qwen35AllocationClearer {
  clearAllocation(allocation: GpuAllocation): Promise<void>;
}

function invalidMetadata(): Error {
  return diagnosticError(
    "webgpu-allocation-clear-invalid",
    "Qwen3.5 GPU allocation clear metadata is invalid",
  );
}

function planClearRanges(allocation: GpuAllocation): readonly ClearRange[] {
  if (
    typeof allocation !== "object" ||
    allocation === null ||
    !Array.isArray(allocation.shards) ||
    allocation.shards.length === 0 ||
    typeof allocation.logicalBytes !== "bigint" ||
    allocation.logicalBytes <= 0n ||
    typeof allocation.allocatedBytes !== "bigint" ||
    allocation.allocatedBytes < allocation.logicalBytes
  ) {
    throw invalidMetadata();
  }

  let logicalCursor = 0n;
  let allocatedTotal = 0n;
  const ranges: ClearRange[] = [];
  for (const shard of allocation.shards) {
    if (
      typeof shard !== "object" ||
      shard === null ||
      typeof shard.buffer !== "object" ||
      shard.buffer === null ||
      typeof shard.logicalByteOffset !== "bigint" ||
      shard.logicalByteOffset !== logicalCursor ||
      shard.logicalByteOffset % 4n !== 0n ||
      typeof shard.logicalByteLength !== "bigint" ||
      shard.logicalByteLength <= 0n ||
      typeof shard.allocatedByteLength !== "bigint" ||
      shard.allocatedByteLength < shard.logicalByteLength ||
      shard.allocatedByteLength % 4n !== 0n ||
      shard.allocatedByteLength > BigInt(Number.MAX_SAFE_INTEGER)
    ) {
      throw invalidMetadata();
    }
    logicalCursor += shard.logicalByteLength;
    allocatedTotal += shard.allocatedByteLength;
    ranges.push(Object.freeze({
      buffer: shard.buffer,
      size: Number(shard.allocatedByteLength),
    }));
  }
  if (
    logicalCursor !== allocation.logicalBytes ||
    allocatedTotal !== allocation.allocatedBytes
  ) {
    throw invalidMetadata();
  }
  return Object.freeze(ranges);
}

class BoundQwen35AllocationClearer implements Qwen35AllocationClearer {
  readonly #device: Qwen35WebGpuDevice;
  readonly #performanceCounters: Qwen35PerformanceCounters | undefined;
  #poisoned = false;

  constructor(
    device: Qwen35WebGpuDevice,
    performanceCounters?: Qwen35PerformanceCounters,
  ) {
    this.#device = device;
    this.#performanceCounters = performanceCounters;
  }

  async clearAllocation(allocation: GpuAllocation): Promise<void> {
    if (this.#poisoned) {
      throw diagnosticError(
        "webgpu-allocation-clear-poisoned",
        "Qwen3.5 GPU allocation clearer must be replaced",
      );
    }
    const ranges = planClearRanges(allocation);

    try {
      this.#device.pushErrorScope("validation");
    } catch {
      this.#poisoned = true;
      throw diagnosticError(
        "webgpu-allocation-clear-validation-failed",
        "Qwen3.5 GPU allocation clear validation failed",
      );
    }

    try {
      const encoder = this.#device.createCommandEncoder({
        label: "qwen35-allocation-clear",
      });
      for (const range of ranges) {
        encoder.clearBuffer(range.buffer, 0, range.size);
      }
      this.#device.queue.submit([encoder.finish()]);
      this.#performanceCounters?.recordQueueSubmission();
    } catch {
      this.#poisoned = true;
      try {
        await this.#device.popErrorScope();
      } catch {
        // Keep the stable submission diagnostic. This instance stays poisoned.
      }
      throw diagnosticError(
        "webgpu-allocation-clear-submission-failed",
        "Qwen3.5 GPU allocation clear submission failed",
      );
    }

    let validationFailed = false;
    try {
      validationFailed = await this.#device.popErrorScope() !== null;
    } catch {
      validationFailed = true;
    }

    let retirementFailed = false;
    try {
      await this.#device.queue.onSubmittedWorkDone();
      this.#performanceCounters?.recordQueueRetirement();
    } catch {
      retirementFailed = true;
    }

    if (validationFailed) {
      this.#poisoned = true;
      throw diagnosticError(
        "webgpu-allocation-clear-validation-failed",
        "Qwen3.5 GPU allocation clear validation failed",
      );
    }
    if (retirementFailed) {
      this.#poisoned = true;
      throw diagnosticError(
        "webgpu-allocation-clear-retirement-failed",
        "Qwen3.5 GPU allocation clear retirement failed",
      );
    }
  }
}

/**
 * Binds clearing to the loader-owned device without taking allocation ownership.
 * GpuAllocation does not record usage flags, so allocation owners must request
 * COPY_DST. This boundary validates all coverage and alignment metadata it has.
 */
export function createQwen35AllocationClearer(
  device: Qwen35WebGpuDevice,
  performanceCounters?: Qwen35PerformanceCounters,
): Qwen35AllocationClearer {
  return new BoundQwen35AllocationClearer(device, performanceCounters);
}
