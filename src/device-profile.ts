export const DEFAULT_BUFFER_SHARD_CAP_BYTES = 256 * 1024 * 1024;
export const DEFAULT_UPLOAD_LANE_BYTES = 32 * 1024 * 1024;

export type BufferShardPolicy = "default" | "evidence-128" | "evidence-64";

export interface GpuLimitsLike {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly [name: string]: number;
}

export interface GpuDeviceProfileDevice {
  readonly features: Iterable<string>;
  readonly limits: GpuLimitsLike;
}

export interface GpuAdapterLike {
  readonly features: Iterable<string>;
  readonly limits: GpuLimitsLike;
  requestDevice(descriptor: {
    readonly requiredFeatures?: readonly string[];
    readonly requiredLimits?: Readonly<Record<string, number>>;
  }): Promise<GpuDeviceProfileDevice>;
}

export interface WebGpuProbeSurface {
  readonly gpu?: {
    requestAdapter(): Promise<GpuAdapterLike | null>;
  };
  readonly performance?: {
    readonly memory?: {
      readonly jsHeapSizeLimit?: number;
    };
  };
}

export interface DeviceProfileOptions {
  readonly requiredFeatures?: readonly string[];
  readonly requiredLimits?: Readonly<Record<string, number>>;
  readonly bufferShardPolicy?: BufferShardPolicy;
}

export interface DeviceProfile {
  readonly device: GpuDeviceProfileDevice;
  /**
   * Per-buffer shaping policy. This value never limits total resident model
   * bytes or claims a browser capability ceiling.
   */
  readonly bufferShardCapBytes: number;
  readonly uploadLaneBytes: number;
  readonly bufferShardCapIsCapabilityCeiling: false;
  readonly facts: {
    readonly adapter: {
      readonly features: readonly string[];
      readonly limits: Readonly<Record<string, number>>;
    };
    readonly device: {
      readonly features: readonly string[];
      readonly limits: Readonly<Record<string, number>>;
    };
    readonly jsHeapLimitBytes: number | null;
  };
}

const BUFFER_SHARD_POLICY_BYTES: Readonly<Record<BufferShardPolicy, number>> = {
  default: DEFAULT_BUFFER_SHARD_CAP_BYTES,
  "evidence-128": 128 * 1024 * 1024,
  "evidence-64": 64 * 1024 * 1024,
};

const MINIMUM_ALIGNMENT_LIMITS = new Set([
  "minUniformBufferOffsetAlignment",
  "minStorageBufferOffsetAlignment",
]);

function requireLimitValue(limits: GpuLimitsLike, name: string): number {
  const value = limits[name];
  if (value === undefined || !Number.isSafeInteger(value) || value <= 0) {
    throw new Error("WebGPU limit is missing or unsafe");
  }
  return value;
}

function supportsRequiredLimit(
  available: number,
  required: number,
  name: string,
): boolean {
  // These WebGPU limits are requirements on the smallest supported offset.
  // A lower available alignment is better, unlike max-capacity limits.
  return MINIMUM_ALIGNMENT_LIMITS.has(name)
    ? available <= required
    : available >= required;
}

function sanitizedLimitFacts(
  limits: GpuLimitsLike,
  requiredNames: readonly string[],
): Readonly<Record<string, number>> {
  const names = new Set([
    "maxBufferSize",
    "maxStorageBufferBindingSize",
    ...requiredNames,
  ]);
  const facts: Record<string, number> = {};
  for (const name of [...names].sort()) {
    const value = limits[name];
    if (
      /^[A-Za-z][A-Za-z0-9]{0,63}$/.test(name) &&
      value !== undefined &&
      Number.isSafeInteger(value) &&
      value > 0
    ) {
      facts[name] = value;
    }
  }
  return Object.freeze(facts);
}

function sanitizedHeapLimit(surface: WebGpuProbeSurface): number | null {
  const value = surface.performance?.memory?.jsHeapSizeLimit;
  return Number.isSafeInteger(value) && value !== undefined && value > 0
    ? value
    : null;
}

/**
 * Probes an injected browser surface so capability policy is testable without
 * a global navigator or a production WebGPU typing package.
 */
export async function probeDeviceProfile(
  surface: WebGpuProbeSurface,
  options: DeviceProfileOptions = {},
): Promise<DeviceProfile> {
  if (surface.gpu === undefined) {
    throw new Error("WebGPU is unavailable");
  }
  const adapter = await surface.gpu.requestAdapter();
  if (adapter === null) {
    throw new Error("WebGPU adapter is unavailable");
  }

  const availableFeatures = new Set(adapter.features);
  const requiredFeatures = [...(options.requiredFeatures ?? [])];
  for (const feature of requiredFeatures) {
    if (!availableFeatures.has(feature)) {
      throw new Error("Required WebGPU feature is unavailable");
    }
  }

  const requiredLimits = { ...(options.requiredLimits ?? {}) };
  for (const [name, required] of Object.entries(requiredLimits)) {
    if (!Number.isSafeInteger(required) || required <= 0) {
      throw new Error("Required WebGPU limit must be a positive safe integer");
    }
    const available = requireLimitValue(adapter.limits, name);
    if (!supportsRequiredLimit(available, required, name)) {
      throw new Error("Required WebGPU limit is unavailable");
    }
  }

  const descriptor: {
    requiredFeatures?: readonly string[];
    requiredLimits?: Readonly<Record<string, number>>;
  } = {};
  if (requiredFeatures.length > 0) {
    descriptor.requiredFeatures = requiredFeatures;
  }
  if (Object.keys(requiredLimits).length > 0) {
    descriptor.requiredLimits = requiredLimits;
  }
  const device = await adapter.requestDevice(descriptor);

  const enabledFeatures = new Set(device.features);
  for (const feature of requiredFeatures) {
    if (!enabledFeatures.has(feature)) {
      throw new Error("Returned WebGPU device is missing a required feature");
    }
  }
  for (const [name, required] of Object.entries(requiredLimits)) {
    const returned = requireLimitValue(device.limits, name);
    if (!supportsRequiredLimit(returned, required, name)) {
      throw new Error(
        "Returned WebGPU device does not satisfy required limits",
      );
    }
  }

  // Both limits are live allocation constraints. A high maxBufferSize does not
  // permit a storage binding that exceeds maxStorageBufferBindingSize.
  requireLimitValue(device.limits, "maxBufferSize");
  requireLimitValue(
    device.limits,
    "maxStorageBufferBindingSize",
  );
  const adapterFeatures = [...availableFeatures]
    .filter((feature) => /^[a-z0-9-]+$/.test(feature))
    .sort();
  // Optional adapter features are not usable until requestDevice enables them.
  const deviceFeatures = [...enabledFeatures]
    .filter((feature) => /^[a-z0-9-]+$/.test(feature))
    .sort();
  const requiredLimitNames = Object.keys(requiredLimits);

  return {
    device,
    bufferShardCapBytes:
      BUFFER_SHARD_POLICY_BYTES[options.bufferShardPolicy ?? "default"],
    uploadLaneBytes: DEFAULT_UPLOAD_LANE_BYTES,
    bufferShardCapIsCapabilityCeiling: false,
    facts: {
      adapter: {
        features: Object.freeze(adapterFeatures),
        limits: sanitizedLimitFacts(adapter.limits, requiredLimitNames),
      },
      device: {
        features: Object.freeze(deviceFeatures),
        limits: sanitizedLimitFacts(device.limits, requiredLimitNames),
      },
      // Safari does not expose performance.memory; absence must not look like a
      // measured zero-byte heap.
      jsHeapLimitBytes: sanitizedHeapLimit(surface),
    },
  };
}
