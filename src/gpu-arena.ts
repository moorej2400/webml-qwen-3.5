import {
  AllocationLedger,
  type AllocationHandle,
} from "./allocation-ledger.js";
import {
  allocationDiagnosticError,
  diagnosticError,
} from "./diagnostics.js";

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

function readErrorCode(error: unknown): unknown {
  if ((typeof error !== "object" || error === null) && typeof error !== "function") {
    return undefined;
  }
  try {
    return (error as { readonly code?: unknown }).code;
  } catch {
    return undefined;
  }
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

    let ledgerHandle: AllocationHandle;
    try {
      ledgerHandle = this.#ledger.reserve({
        id: request.id,
        category: request.category,
        bytes: allocatedBytes,
      });
    } catch (error) {
      const code = readErrorCode(error);
      if (code === "ALLOCATION_LIMIT_EXCEEDED") {
        throw diagnosticError(
          "allocation_conflict",
          "GPU allocation exceeds ledger limit",
        );
      }
      if (code === "ALLOCATION_DUPLICATE") {
        throw diagnosticError(
          "allocation_conflict",
          "GPU allocation ownership conflict",
        );
      }
      throw allocationDiagnosticError("unknown");
    }
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
    let outOfMemoryResult:
      | Promise<{ readonly message?: string } | null>
      | undefined;
    let validationResult:
      | Promise<{ readonly message?: string } | null>
      | undefined;
    try {
      this.#device.pushErrorScope("validation");
      scopesPushed = 1;
    } catch {
      scopeFailed = true;
    }
    if (!scopeFailed) {
      try {
        this.#device.pushErrorScope("out-of-memory");
        scopesPushed = 2;
      } catch {
        scopeFailed = true;
      }
    }

    if (!scopeFailed) {
      try {
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
    }

    // Error scopes share one device-wide stack. Capture both pops synchronously;
    // awaiting between them lets another allocation steal the outer scope.
    if (scopesPushed === 2) {
      try {
        outOfMemoryResult = this.#device.popErrorScope();
      } catch {
        scopeFailed = true;
      }
    }
    if (scopesPushed >= 1) {
      try {
        validationResult = this.#device.popErrorScope();
      } catch {
        scopeFailed = true;
      }
    }
    if (outOfMemoryResult !== undefined) {
      try {
        outOfMemoryError = await outOfMemoryResult;
      } catch {
        scopeFailed = true;
      }
    }
    if (validationResult !== undefined) {
      try {
        validationError = await validationResult;
      } catch {
        scopeFailed = true;
      }
    }

    const failureCode = scopeFailed
      ? "error_scope"
      : outOfMemoryError !== null && validationError !== null
        ? "gpu_ambiguous_scopes"
        : outOfMemoryError !== null
          ? "gpu_out_of_memory"
          : validationError !== null
            ? "gpu_validation"
            : creationFailed
              ? "buffer_creation"
              : undefined;
    if (failureCode !== undefined) {
      // The arena owns every returned prefix buffer until both async scopes
      // succeed. Any failure destroys the prefix and releases ledger ownership.
      destroyEveryBuffer(shards.map(({ buffer }) => buffer));
      this.#ledger.release(ledgerHandle);
      throw allocationDiagnosticError(failureCode);
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
