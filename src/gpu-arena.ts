import { AllocationLedger } from "./allocation-ledger.js";

export type GpuAllocationCategory =
  | "model"
  | "activation"
  | "kv-cache"
  | "upload"
  | "scratch";

export interface GpuBufferLike {
  destroy(): void;
}

export interface GpuArenaDevice {
  readonly limits: {
    readonly maxBufferSize: number;
    readonly maxStorageBufferBindingSize: number;
  };
  createBuffer(descriptor: {
    readonly size: number;
    readonly usage: number;
    readonly label: string;
  }): GpuBufferLike;
}

export interface GpuAllocation {
  readonly buffers: readonly GpuBufferLike[];
  readonly logicalBytes: bigint;
  readonly allocatedBytes: bigint;
  destroy(): void;
}

export interface GpuAllocationRequest {
  readonly id: string;
  readonly category: GpuAllocationCategory;
  readonly byteLength: bigint;
  readonly usage: number;
  readonly alignment: number;
}

function requirePositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive safe integer`);
  }
}

function isPowerOfTwo(value: number): boolean {
  if (!Number.isSafeInteger(value) || value <= 0) {
    return false;
  }
  let remaining = BigInt(value);
  return (remaining & (remaining - 1n)) === 0n;
}

function destroyEveryBuffer(buffers: readonly GpuBufferLike[]): unknown {
  let firstError: unknown;
  for (const buffer of buffers) {
    try {
      buffer.destroy();
    } catch (error) {
      firstError ??= error;
    }
  }
  return firstError;
}

export class GpuArena {
  readonly #device: GpuArenaDevice;
  readonly #ledger: AllocationLedger;
  readonly #arenaCapBytes: bigint;

  constructor(
    device: GpuArenaDevice,
    ledger: AllocationLedger,
    options: { readonly arenaCapBytes: bigint },
  ) {
    if (options.arenaCapBytes <= 0n) {
      throw new Error("GPU arena cap must be greater than zero");
    }
    if (options.arenaCapBytes > ledger.snapshot().limitBytes) {
      throw new Error("GPU arena cap cannot exceed the allocation ledger limit");
    }
    this.#device = device;
    this.#ledger = ledger;
    this.#arenaCapBytes = options.arenaCapBytes;
  }

  allocate(request: GpuAllocationRequest): GpuAllocation {
    if (request.byteLength <= 0n) {
      throw new Error("GPU allocation byteLength must be greater than zero");
    }
    requirePositiveSafeInteger(request.usage, "GPU buffer usage");
    if (request.usage > 0xffff_ffff) {
      throw new Error("GPU buffer usage must fit an unsigned u32 bitmask");
    }
    if (!isPowerOfTwo(request.alignment)) {
      throw new Error("GPU buffer alignment must be a positive power of two");
    }
    requirePositiveSafeInteger(
      this.#device.limits.maxBufferSize,
      "maxBufferSize",
    );
    requirePositiveSafeInteger(
      this.#device.limits.maxStorageBufferBindingSize,
      "maxStorageBufferBindingSize",
    );

    const alignment = BigInt(request.alignment);
    const allocatedBytes =
      ((request.byteLength + alignment - 1n) / alignment) * alignment;
    if (allocatedBytes > this.#arenaCapBytes) {
      throw new Error("GPU allocation exceeds the selected arena cap");
    }
    if (
      this.#ledger.snapshot().currentBytes + allocatedBytes >
      this.#arenaCapBytes
    ) {
      throw new Error("GPU allocation would exceed the selected arena cap");
    }
    if (allocatedBytes > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error("GPU allocation cannot be converted to a safe integer");
    }

    const liveBufferLimit = Math.min(
      this.#device.limits.maxBufferSize,
      this.#device.limits.maxStorageBufferBindingSize,
    );
    const alignedChunkLimit =
      Math.floor(liveBufferLimit / request.alignment) * request.alignment;
    if (alignedChunkLimit < request.alignment) {
      throw new Error(
        "GPU device limits are smaller than the requested alignment",
      );
    }

    this.#ledger.reserve({
      id: request.id,
      category: request.category,
      bytes: allocatedBytes,
    });
    const buffers: GpuBufferLike[] = [];
    try {
      let remaining = Number(allocatedBytes);
      while (remaining > 0) {
        const size = Math.min(remaining, alignedChunkLimit);
        buffers.push(
          this.#device.createBuffer({
            size,
            usage: request.usage,
            // Tensor identifiers may contain model or user data; labels expose
            // only the public category and an allocation-local ordinal.
            label: `qwen-runtime:${request.category}:${buffers.length}`,
          }),
        );
        remaining -= size;
      }
    } catch (error) {
      // The arena owns each buffer as soon as createBuffer returns. On a later
      // failure it destroys that prefix before releasing the single ledger
      // reservation, so neither side can retain orphaned ownership.
      destroyEveryBuffer(buffers);
      this.#ledger.release(request.id);
      throw error;
    }

    let destroyed = false;
    return {
      buffers,
      logicalBytes: request.byteLength,
      allocatedBytes,
      destroy: () => {
        if (destroyed) {
          return;
        }
        destroyed = true;
        const destroyError = destroyEveryBuffer(buffers);
        this.#ledger.release(request.id);
        if (destroyError !== undefined) {
          throw destroyError;
        }
      },
    };
  }
}
