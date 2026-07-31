import {
  AllocationLedger,
  type AllocationHandle,
} from "./allocation-ledger.js";
import { diagnosticError } from "./diagnostics.js";

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
  pushErrorScope(filter: "validation" | "out-of-memory"): void;
  popErrorScope(): Promise<{ readonly message?: string } | null>;
  createBuffer(descriptor: {
    readonly size: number;
    readonly usage: number;
    readonly label: string;
  }): GpuBufferLike;
}

export interface GpuAllocation {
  readonly shards: readonly {
    readonly buffer: GpuBufferLike;
    readonly logicalByteOffset: bigint;
    readonly logicalByteLength: bigint;
    readonly allocatedByteLength: bigint;
  }[];
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
  readonly requiredShardQuantumBytes?: bigint;
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

const GPU_BUFFER_USAGE_ALL = 0x03ff;
const GPU_BUFFER_USAGE_MAP_READ = 0x0001;
const GPU_BUFFER_USAGE_MAP_WRITE = 0x0002;
const GPU_BUFFER_USAGE_COPY_SRC = 0x0004;
const GPU_BUFFER_USAGE_COPY_DST = 0x0008;

function requireLegalBufferUsage(usage: number): void {
  requirePositiveSafeInteger(usage, "GPU buffer usage");
  const hasUnknownBits =
    usage > GPU_BUFFER_USAGE_ALL ||
    (usage & ~GPU_BUFFER_USAGE_ALL) !== 0;
  const mapReadIsLegal =
    (usage & GPU_BUFFER_USAGE_MAP_READ) === 0 ||
    (usage & ~(GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST)) === 0;
  const mapWriteIsLegal =
    (usage & GPU_BUFFER_USAGE_MAP_WRITE) === 0 ||
    (usage & ~(GPU_BUFFER_USAGE_MAP_WRITE | GPU_BUFFER_USAGE_COPY_SRC)) === 0;
  if (hasUnknownBits || !mapReadIsLegal || !mapWriteIsLegal) {
    throw diagnosticError(
      "GPU_BUFFER_USAGE_INVALID",
      "GPU buffer usage is invalid",
    );
  }
}

export class GpuArena {
  readonly #device: GpuArenaDevice;
  readonly #ledger: AllocationLedger;
  readonly #bufferShardCapBytes: bigint;

  constructor(
    device: GpuArenaDevice,
    ledger: AllocationLedger,
    options: { readonly bufferShardCapBytes: bigint },
  ) {
    if (options.bufferShardCapBytes <= 0n) {
      throw new Error("GPU buffer shard cap must be greater than zero");
    }
    this.#device = device;
    this.#ledger = ledger;
    this.#bufferShardCapBytes = options.bufferShardCapBytes;
  }

  async allocate(request: GpuAllocationRequest): Promise<GpuAllocation> {
    if (request.byteLength <= 0n) {
      throw new Error("GPU allocation byteLength must be greater than zero");
    }
    requireLegalBufferUsage(request.usage);
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
    const shardQuantum =
      request.requiredShardQuantumBytes ?? alignment;
    if (
      shardQuantum <= 0n ||
      shardQuantum % alignment !== 0n ||
      (request.requiredShardQuantumBytes !== undefined &&
        request.byteLength % shardQuantum !== 0n)
    ) {
      throw new Error(
        "GPU shard quantum must align buffers and divide the logical allocation",
      );
    }
    const allocatedBytes =
      ((request.byteLength + alignment - 1n) / alignment) * alignment;
    if (allocatedBytes > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error("GPU allocation cannot be converted to a safe integer");
    }

    // The policy cap shapes each buffer, never the allocation or model total.
    // AllocationLedger is the independent cumulative budget and may be several
    // gigabytes when physical-device evidence supports that experiment.
    const liveBufferLimit = [
      BigInt(this.#device.limits.maxBufferSize),
      BigInt(this.#device.limits.maxStorageBufferBindingSize),
      this.#bufferShardCapBytes,
    ].reduce((smallest, value) => (value < smallest ? value : smallest));
    const alignedChunkLimit =
      (liveBufferLimit / shardQuantum) * shardQuantum;
    if (alignedChunkLimit < shardQuantum) {
      throw new Error(
        "GPU buffer shard cap or device limits cannot hold the requested alignment or shard quantum",
      );
    }
    const alignedChunkLimitNumber = Number(alignedChunkLimit);
    if (!Number.isSafeInteger(alignedChunkLimitNumber)) {
      throw new Error(
        "GPU buffer shard size cannot be converted to a safe integer",
      );
    }

    const ledgerHandle: AllocationHandle = this.#ledger.reserve({
      id: request.id,
      category: request.category,
      bytes: allocatedBytes,
    });
    const shards: Array<{
      buffer: GpuBufferLike;
      logicalByteOffset: bigint;
      logicalByteLength: bigint;
      allocatedByteLength: bigint;
    }> = [];
    let scopesPushed = 0;
    let creationFailed = false;
    let scopeFailed = false;
    let outOfMemoryError: { readonly message?: string } | null = null;
    let validationError: { readonly message?: string } | null = null;
    try {
      this.#device.pushErrorScope("validation");
      scopesPushed = 1;
      this.#device.pushErrorScope("out-of-memory");
      scopesPushed = 2;

      let remainingAllocated = allocatedBytes;
      let remainingLogical = request.byteLength;
      let logicalByteOffset = 0n;
      while (remainingAllocated > 0n) {
        const allocatedByteLength =
          remainingAllocated < alignedChunkLimit
            ? remainingAllocated
            : alignedChunkLimit;
        const logicalByteLength =
          remainingLogical < allocatedByteLength
            ? remainingLogical
            : allocatedByteLength;
        const buffer = this.#device.createBuffer({
          size: Number(allocatedByteLength),
          usage: request.usage,
          // Tensor identifiers may contain model or user data; labels expose
          // only the public category and an allocation-local ordinal.
          label: `qwen-runtime:${request.category}:${shards.length}`,
        });
        shards.push({
          buffer,
          logicalByteOffset,
          logicalByteLength,
          allocatedByteLength,
        });
        logicalByteOffset += logicalByteLength;
        remainingLogical -= logicalByteLength;
        remainingAllocated -= allocatedByteLength;
      }
    } catch {
      creationFailed = true;
    }

    // WebGPU reports createBuffer validation and OOM failures asynchronously.
    // Pop both scopes before transferring ownership to the returned allocation.
    if (scopesPushed === 2) {
      try {
        outOfMemoryError = await this.#device.popErrorScope();
      } catch {
        scopeFailed = true;
      }
    }
    if (scopesPushed >= 1) {
      try {
        validationError = await this.#device.popErrorScope();
      } catch {
        scopeFailed = true;
      }
    }

    if (
      creationFailed ||
      scopeFailed ||
      outOfMemoryError !== null ||
      validationError !== null
    ) {
      // The arena owns every returned prefix buffer until both async scopes
      // succeed. Any failure destroys the prefix and releases ledger ownership.
      destroyEveryBuffer(shards.map(({ buffer }) => buffer));
      this.#ledger.release(ledgerHandle);
      throw diagnosticError(
        "GPU_BUFFER_ALLOCATION_FAILED",
        "GPU buffer allocation failed",
      );
    }

    const ownedLedgerHandle = ledgerHandle;
    let destroyed = false;
    return {
      shards: Object.freeze(
        shards.map((shard) => Object.freeze({ ...shard })),
      ),
      logicalBytes: request.byteLength,
      allocatedBytes,
      destroy: () => {
        if (destroyed) {
          return;
        }
        destroyed = true;
        const destroyError = destroyEveryBuffer(
          shards.map(({ buffer }) => buffer),
        );
        this.#ledger.release(ownedLedgerHandle);
        if (destroyError !== undefined) {
          throw diagnosticError(
            "GPU_BUFFER_DESTRUCTION_FAILED",
            "GPU buffer destruction failed",
          );
        }
      },
    };
  }
}
