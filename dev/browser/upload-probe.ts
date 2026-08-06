import type { WebGpuProbeSurface } from "../../src/device-profile.js";

interface UploadProbeManifest {
  readonly shards: readonly {
    readonly offset: string;
    readonly length: string;
  }[];
  readonly tensorLayout: readonly {
    readonly shard: number;
    readonly shardOffset: string;
    readonly length: string;
  }[];
}

export interface LocalWeightUploadEvent {
  readonly stage: "before_write" | "after_write" | "after_retire";
  readonly ordinal: number;
  readonly shardIndex: number;
  readonly shardCount: number;
  readonly segmentIndex: number;
  readonly segmentCount: number;
  readonly globalOffset: number;
  readonly byteCount: number;
  readonly bufferShardBytes: number;
  readonly uploadLaneBytes: number;
  readonly retireAfterEachWrite: boolean;
}

interface UploadBoundary {
  readonly shardIndex: number;
  readonly shardCount: number;
  readonly segmentIndex: number;
  readonly segmentCount: number;
  readonly globalOffset: number;
  readonly byteCount: number;
}

interface UploadSegment {
  readonly shardIndex: number;
  readonly shardOffset: number;
  readonly byteCount: number;
  readonly segmentIndex: number;
  readonly segmentCount: number;
}

interface QueueLike {
  writeBuffer(...arguments_: unknown[]): unknown;
  onSubmittedWorkDone(...arguments_: unknown[]): Promise<unknown>;
}

const safeByteCount = (value: string): number => {
  if (!/^(?:0|[1-9][0-9]*)$/.test(value)) {
    throw new Error("Upload probe manifest byte count is invalid");
  }
  const parsed = BigInt(value);
  if (parsed > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error("Upload probe manifest byte count is unsafe");
  }
  return Number(parsed);
};

const requirePositiveU32Multiple = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 4 || value % 4 !== 0) {
    throw new Error(`${label} must be a positive u32 multiple`);
  }
  return value;
};

const buildUploadSegments = (
  manifest: UploadProbeManifest,
): readonly UploadSegment[] => {
  if (!Array.isArray(manifest.shards) || manifest.shards.length === 0) {
    throw new Error("Upload probe manifest has no shards");
  }
  if (!Array.isArray(manifest.tensorLayout) || manifest.tensorLayout.length === 0) {
    throw new Error("Upload probe manifest has no tensor layout");
  }
  const byShard: Array<Array<{
    readonly shardIndex: number;
    readonly shardOffset: number;
    readonly byteCount: number;
  }>> = manifest.shards.map(() => []);
  for (const entry of manifest.tensorLayout) {
    if (
      !Number.isSafeInteger(entry.shard) ||
      entry.shard < 0 ||
      entry.shard >= byShard.length
    ) {
      throw new Error("Upload probe tensor shard is invalid");
    }
    const shardOffset = safeByteCount(entry.shardOffset);
    const byteCount = safeByteCount(entry.length);
    requirePositiveU32Multiple(byteCount, "Upload probe segment length");
    if (shardOffset % 4 !== 0) {
      throw new Error("Upload probe segment offset is not u32 aligned");
    }
    byShard[entry.shard]!.push({
      shardIndex: entry.shard,
      shardOffset,
      byteCount,
    });
  }

  const result: UploadSegment[] = [];
  for (const segments of byShard) {
    segments.sort((left, right) => left.shardOffset - right.shardOffset);
    for (let index = 0; index < segments.length; index += 1) {
      const segment = segments[index]!;
      const previous = segments[index - 1];
      if (
        previous !== undefined &&
        segment.shardOffset < previous.shardOffset + previous.byteCount
      ) {
        throw new Error("Upload probe tensor segments overlap");
      }
      result.push(Object.freeze({
        ...segment,
        segmentIndex: index,
        segmentCount: segments.length,
      }));
    }
  }
  if (result.length === 0) {
    throw new Error("Upload probe manifest has no upload segments");
  }
  return Object.freeze(result);
};

const memberWithOriginalReceiver = (
  target: object,
  property: PropertyKey,
): unknown => {
  const value: unknown = Reflect.get(target, property, target);
  return typeof value === "function" ? value.bind(target) : value;
};

const writeByteCount = (arguments_: readonly unknown[]): number | null => {
  const explicitSize = arguments_[4];
  if (explicitSize !== undefined) {
    return Number.isSafeInteger(explicitSize) && (explicitSize as number) > 0
      ? explicitSize as number
      : null;
  }
  const dataOffset = arguments_[3] ?? 0;
  if (!Number.isSafeInteger(dataOffset) || (dataOffset as number) < 0) return null;
  const data = arguments_[2];
  let dataBytes: number | null = null;
  if (data instanceof ArrayBuffer) {
    dataBytes = data.byteLength;
  } else if (ArrayBuffer.isView(data)) {
    dataBytes = data.byteLength;
  }
  if (dataBytes === null || (dataOffset as number) >= dataBytes) return null;
  return dataBytes - (dataOffset as number);
};

export const createLocalWeightUploadProbe = (input: {
  readonly surface: WebGpuProbeSurface;
  readonly manifest: UploadProbeManifest;
  readonly bufferShardBytes: number;
  readonly uploadLaneBytes: number;
  readonly retireAfterEachWrite: boolean;
  readonly onEvent: (event: LocalWeightUploadEvent) => void;
}): Readonly<{
  readonly surface: WebGpuProbeSurface;
  beginWeightsUpload(): void;
  endWeightsUpload(): void;
}> => {
  const segments = buildUploadSegments(input.manifest);
  const bufferShardBytes = requirePositiveU32Multiple(
    input.bufferShardBytes,
    "Upload probe buffer shard size",
  );
  const uploadLaneBytes = requirePositiveU32Multiple(
    input.uploadLaneBytes,
    "Upload probe lane size",
  );
  if (typeof input.retireAfterEachWrite !== "boolean") {
    throw new Error("Upload probe retirement mode is invalid");
  }
  if (typeof input.onEvent !== "function") {
    throw new Error("Upload probe observer is invalid");
  }

  let active = false;
  let segmentCursor = 0;
  let segmentConsumed = 0;
  let globalOffset = 0;
  let ordinal = 0;
  let pendingRetirement: UploadBoundary[] = [];

  const emit = (
    stage: LocalWeightUploadEvent["stage"],
    boundary: UploadBoundary,
  ): void => {
    const event = Object.freeze({
      stage,
      ordinal: ordinal += 1,
      ...boundary,
      bufferShardBytes,
      uploadLaneBytes,
      retireAfterEachWrite: input.retireAfterEachWrite,
    });
    try {
      // Local telemetry is observational. It cannot change queue behavior.
      input.onEvent(event);
    } catch {
      // Ignore local observer failures so a diagnostic cannot stop inference.
    }
  };

  const planWrite = (byteCount: number | null): UploadBoundary | null => {
    const segment = segments[segmentCursor];
    if (
      !active ||
      segment === undefined ||
      byteCount === null ||
      byteCount < 4 ||
      byteCount % 4 !== 0 ||
      byteCount > uploadLaneBytes ||
      byteCount > segment.byteCount - segmentConsumed
    ) {
      // Once queue traffic differs from the static upload plan, later offsets
      // are ambiguous. Disable observation but never control the real queue.
      if (active) active = false;
      return null;
    }
    return Object.freeze({
      shardIndex: segment.shardIndex,
      shardCount: input.manifest.shards.length,
      segmentIndex: segment.segmentIndex,
      segmentCount: segment.segmentCount,
      globalOffset,
      byteCount,
    });
  };

  const acceptWrite = (boundary: UploadBoundary): void => {
    const segment = segments[segmentCursor]!;
    segmentConsumed += boundary.byteCount;
    globalOffset += boundary.byteCount;
    if (segmentConsumed === segment.byteCount) {
      segmentCursor += 1;
      segmentConsumed = 0;
    }
    pendingRetirement.push(boundary);
  };

  const wrapQueue = (queue: QueueLike): QueueLike => new Proxy(queue, {
    get(target, property) {
      if (property === "writeBuffer") {
        return (...arguments_: unknown[]): unknown => {
          const boundary = planWrite(writeByteCount(arguments_));
          if (boundary !== null) emit("before_write", boundary);
          let result: unknown;
          try {
            result = Reflect.apply(target.writeBuffer, target, arguments_);
          } catch (error) {
            if (boundary !== null) active = false;
            throw error;
          }
          if (boundary !== null) {
            acceptWrite(boundary);
            emit("after_write", boundary);
          }
          return result;
        };
      }
      if (property === "onSubmittedWorkDone") {
        return async (...arguments_: unknown[]): Promise<unknown> => {
          // A retirement call covers only writes accepted before that call.
          // Removing the batch now prevents concurrent waits from duplicating it.
          const retiring = pendingRetirement;
          pendingRetirement = [];
          const result = await Reflect.apply(
            target.onSubmittedWorkDone,
            target,
            arguments_,
          );
          for (const boundary of retiring) emit("after_retire", boundary);
          return result;
        };
      }
      return memberWithOriginalReceiver(target, property);
    },
  });

  const queueWrappers = new WeakMap<object, QueueLike>();
  const deviceWrappers = new WeakMap<object, object>();
  const adapterWrappers = new WeakMap<object, object>();

  const wrappedQueue = (queue: QueueLike): QueueLike => {
    const target = queue as object;
    const existing = queueWrappers.get(target);
    if (existing !== undefined) return existing;
    const wrapped = wrapQueue(queue);
    queueWrappers.set(target, wrapped);
    return wrapped;
  };

  const wrappedDevice = (device: object): object => {
    const existing = deviceWrappers.get(device);
    if (existing !== undefined) return existing;
    const wrapped = new Proxy(device, {
      get(target, property) {
        if (property === "queue") {
          const queue = Reflect.get(target, property, target) as QueueLike;
          return wrappedQueue(queue);
        }
        return memberWithOriginalReceiver(target, property);
      },
    });
    deviceWrappers.set(device, wrapped);
    return wrapped;
  };

  const wrappedAdapter = (adapter: object): object => {
    const existing = adapterWrappers.get(adapter);
    if (existing !== undefined) return existing;
    const wrapped = new Proxy(adapter, {
      get(target, property) {
        if (property === "requestDevice") {
          return async (...arguments_: unknown[]): Promise<object> => {
            const requestDevice = Reflect.get(target, property, target);
            const device = await Reflect.apply(
              requestDevice as (...values: unknown[]) => Promise<object>,
              target,
              arguments_,
            );
            return wrappedDevice(device);
          };
        }
        return memberWithOriginalReceiver(target, property);
      },
    });
    adapterWrappers.set(adapter, wrapped);
    return wrapped;
  };

  const gpu = input.surface.gpu;
  const wrappedGpu = gpu === undefined
    ? undefined
    : new Proxy(gpu as object, {
        get(target, property) {
          if (property === "requestAdapter") {
            return async (...arguments_: unknown[]): Promise<object | null> => {
              const requestAdapter = Reflect.get(target, property, target);
              const adapter = await Reflect.apply(
                requestAdapter as (...values: unknown[]) => Promise<object | null>,
                target,
                arguments_,
              );
              return adapter === null ? null : wrappedAdapter(adapter);
            };
          }
          return memberWithOriginalReceiver(target, property);
        },
      });
  const surface = new Proxy(input.surface as object, {
    get(target, property) {
      if (property === "gpu") return wrappedGpu;
      return memberWithOriginalReceiver(target, property);
    },
  }) as WebGpuProbeSurface;

  return Object.freeze({
    surface,
    beginWeightsUpload() {
      // A new upload must not relabel writes that are still waiting to retire.
      if (pendingRetirement.length > 0) {
        active = false;
        return;
      }
      segmentCursor = 0;
      segmentConsumed = 0;
      globalOffset = 0;
      ordinal = 0;
      active = true;
    },
    endWeightsUpload() {
      active = false;
    },
  });
};
