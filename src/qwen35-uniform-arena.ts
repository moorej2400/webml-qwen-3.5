import { diagnosticError } from "./diagnostics.js";
import type {
  GpuAllocation,
  GpuAllocationRequest,
} from "./gpu-arena.js";
import type { Qwen35ForwardBufferSlice } from "./qwen35-forward-dispatch.js";

const GPU_BUFFER_USAGE_COPY_DST = 0x0008;
const GPU_BUFFER_USAGE_UNIFORM = 0x0040;

export interface Qwen35UniformArenaAllocator {
  allocate(request: GpuAllocationRequest): Promise<GpuAllocation>;
}

export interface Qwen35UniformWriteQueue {
  writeBuffer(
    buffer: object,
    bufferOffset: number,
    data: ArrayBuffer,
    dataOffset: number,
    size: number,
  ): void;
  onSubmittedWorkDone(): Promise<void>;
}

export interface Qwen35UniformSlot {
  readonly index: number;
  readonly wordCount: number;
  readonly binding: Qwen35ForwardBufferSlice;
  update(values: Uint32Array<ArrayBuffer>): void;
}

export interface CreateQwen35UniformArenaOptions {
  readonly arena: Qwen35UniformArenaAllocator;
  readonly queue: Qwen35UniformWriteQueue;
  /** Stable allocation identity when more than one arena shares a ledger. */
  readonly allocationId?: string;
  readonly slotCount: number;
  readonly slotWordCapacity: number;
  readonly minUniformBufferOffsetAlignment: number;
  readonly maxUniformBufferBindingSize: number;
}

function positiveSafeInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function powerOfTwo(value: number): boolean {
  if (!positiveSafeInteger(value)) return false;
  const bits = BigInt(value);
  return (bits & (bits - 1n)) === 0n;
}

function requirePlan(options: CreateQwen35UniformArenaOptions): {
  readonly stride: number;
  readonly totalBytes: number;
} {
  const payloadBytes = options.slotWordCapacity * 4;
  if (
    !positiveSafeInteger(options.slotCount) ||
    !positiveSafeInteger(options.slotWordCapacity) ||
    !powerOfTwo(options.minUniformBufferOffsetAlignment) ||
    !positiveSafeInteger(options.maxUniformBufferBindingSize) ||
    !Number.isSafeInteger(payloadBytes) ||
    payloadBytes > options.maxUniformBufferBindingSize
  ) {
    throw diagnosticError(
      "uniform-arena-plan-invalid",
      "Qwen3.5 uniform arena requirements are invalid",
    );
  }
  const alignment = options.minUniformBufferOffsetAlignment;
  const stride = Math.ceil(payloadBytes / alignment) * alignment;
  const totalBytes = stride * options.slotCount;
  if (!Number.isSafeInteger(totalBytes) || totalBytes <= 0) {
    throw diagnosticError(
      "uniform-arena-plan-invalid",
      "Qwen3.5 uniform arena requirements are invalid",
    );
  }
  return { stride, totalBytes };
}

export interface Qwen35UniformArena {
  readonly slotCount: number;
  slot(index: number, wordCount: number): Qwen35UniformSlot;
  dispose(): Promise<void>;
}

/** Ownership is created only by the validated factory below. */
class OwnedQwen35UniformArena implements Qwen35UniformArena {
  readonly #allocation: GpuAllocation;
  readonly #queue: Qwen35UniformWriteQueue;
  readonly #buffer: object;
  readonly #slotCount: number;
  readonly #slotWordCapacity: number;
  readonly #stride: number;
  readonly #slots = new Map<number, Qwen35UniformSlot>();
  readonly #lastValues = new Map<number, Uint32Array<ArrayBuffer>>();
  #disposed = false;
  #poisoned = false;
  #disposePromise: Promise<void> | null = null;

  constructor(input: {
    readonly allocation: GpuAllocation;
    readonly queue: Qwen35UniformWriteQueue;
    readonly buffer: object;
    readonly slotCount: number;
    readonly slotWordCapacity: number;
    readonly stride: number;
  }) {
    this.#allocation = input.allocation;
    this.#queue = input.queue;
    this.#buffer = input.buffer;
    this.#slotCount = input.slotCount;
    this.#slotWordCapacity = input.slotWordCapacity;
    this.#stride = input.stride;
  }

  get slotCount(): number {
    return this.#slotCount;
  }

  slot(index: number, wordCount: number): Qwen35UniformSlot {
    this.#assertUsable();
    if (
      !Number.isSafeInteger(index) ||
      index < 0 ||
      index >= this.#slotCount
    ) {
      throw diagnosticError(
        "uniform-arena-slot-invalid",
        "Qwen3.5 uniform slot is outside the allocated arena",
      );
    }
    if (
      !Number.isSafeInteger(wordCount) ||
      wordCount < 1 ||
      wordCount > this.#slotWordCapacity
    ) {
      throw diagnosticError(
        "uniform-arena-word-count-invalid",
        "Qwen3.5 uniform slot word count is invalid",
      );
    }
    const existing = this.#slots.get(index);
    if (existing !== undefined) {
      if (existing.wordCount !== wordCount) {
        throw diagnosticError(
          "uniform-arena-word-count-changed",
          "Qwen3.5 uniform slot word count changed",
        );
      }
      return existing;
    }
    const binding = Object.freeze({
      buffer: this.#buffer,
      offset: index * this.#stride,
      byteLength: wordCount * 4,
    });
    const created: Qwen35UniformSlot = Object.freeze({
      index,
      wordCount,
      binding,
      update: (values: Uint32Array<ArrayBuffer>) => {
        this.#update(index, wordCount, values);
      },
    });
    this.#slots.set(index, created);
    return created;
  }

  dispose(): Promise<void> {
    if (this.#disposePromise !== null) return this.#disposePromise;
    this.#disposed = true;
    this.#disposePromise = this.#cleanup();
    return this.#disposePromise;
  }

  #assertUsable(): void {
    if (this.#disposed) {
      throw diagnosticError(
        "uniform-arena-disposed",
        "Qwen3.5 uniform arena is disposed",
      );
    }
    if (this.#poisoned) {
      throw diagnosticError(
        "uniform-arena-poisoned",
        "Qwen3.5 uniform arena is poisoned and must be disposed",
      );
    }
  }

  #update(
    index: number,
    wordCount: number,
    values: Uint32Array<ArrayBuffer>,
  ): void {
    this.#assertUsable();
    if (values.length !== wordCount) {
      throw diagnosticError(
        "uniform-arena-update-size-invalid",
        "Qwen3.5 uniform update size is invalid",
      );
    }
    const previous = this.#lastValues.get(index);
    if (
      previous !== undefined &&
      previous.length === values.length &&
      previous.every((value, word) => value === values[word])
    ) {
      return;
    }
    try {
      this.#queue.writeBuffer(
        this.#buffer,
        index * this.#stride,
        values.buffer,
        values.byteOffset,
        values.byteLength,
      );
      // Copy only after a successful queue write. A failed upload poisons the
      // arena and must never make a later caller believe the GPU has new data.
      this.#lastValues.set(index, values.slice());
    } catch {
      this.#poisoned = true;
      throw diagnosticError(
        "uniform-arena-upload-failed",
        "Qwen3.5 uniform upload failed",
      );
    }
  }

  async #cleanup(): Promise<void> {
    let failed = false;
    try {
      await this.#queue.onSubmittedWorkDone();
    } catch {
      failed = true;
    }
    try {
      this.#allocation.destroy();
    } catch {
      failed = true;
    }
    this.#slots.clear();
    this.#lastValues.clear();
    if (failed) {
      throw diagnosticError(
        "uniform-arena-cleanup-failed",
        "Qwen3.5 uniform arena cleanup did not complete",
      );
    }
  }
}

export async function createQwen35UniformArena(
  options: CreateQwen35UniformArenaOptions,
): Promise<Qwen35UniformArena> {
  const plan = requirePlan(options);
  let allocation: GpuAllocation;
  try {
    allocation = await options.arena.allocate({
      id: options.allocationId ?? "qwen35-uniform-arena",
      category: "scratch",
      byteLength: BigInt(plan.totalBytes),
      usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
      alignment: options.minUniformBufferOffsetAlignment,
      // Uniform offsets must share one buffer so a slot index is stable.
      requiredShardQuantumBytes: BigInt(plan.totalBytes),
    });
  } catch {
    throw diagnosticError(
      "uniform-arena-allocation-failed",
      "Qwen3.5 uniform arena allocation failed",
    );
  }
  const shard = allocation.shards[0];
  if (
    allocation.shards.length !== 1 ||
    allocation.logicalBytes !== BigInt(plan.totalBytes) ||
    shard === undefined ||
    shard.logicalByteOffset !== 0n ||
    shard.logicalByteLength !== BigInt(plan.totalBytes) ||
    shard.allocatedByteLength < BigInt(plan.totalBytes)
  ) {
    try {
      allocation.destroy();
    } catch {
      throw diagnosticError(
        "uniform-arena-rollback-failed",
        "Qwen3.5 uniform arena rollback did not complete",
      );
    }
    throw diagnosticError(
      "uniform-arena-layout-invalid",
      "Qwen3.5 uniform arena allocation has an invalid layout",
    );
  }
  return new OwnedQwen35UniformArena({
    allocation,
    queue: options.queue,
    buffer: shard.buffer as object,
    slotCount: options.slotCount,
    slotWordCapacity: options.slotWordCapacity,
    stride: plan.stride,
  });
}
