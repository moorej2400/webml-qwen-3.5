export const DEFAULT_INITIAL_ARENA_BYTES = 256 * 1024 * 1024;
export const DEFAULT_UPLOAD_LANE_BYTES = 32 * 1024 * 1024;

export type ArenaPolicy = "default" | "evidence-128" | "evidence-64";

export interface GpuLimitsLike {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly [name: string]: number;
}

export interface GpuDeviceProfileDevice {
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
  readonly arenaPolicy?: ArenaPolicy;
}

export interface DeviceProfile {
  readonly device: GpuDeviceProfileDevice;
  readonly arenaCapBytes: number;
  readonly uploadLaneBytes: number;
  /**
   * Policy caps are starting points for allocation experiments, not claims
   * about the maximum model or workload that a browser can execute.
   */
  readonly policyIsCapabilityCeiling: false;
  readonly facts: {
    readonly features: readonly string[];
    readonly limits: {
      readonly maxBufferSize: number;
      readonly maxStorageBufferBindingSize: number;
    };
    readonly jsHeapLimitBytes: number | null;
  };
}

const ARENA_POLICY_BYTES: Readonly<Record<ArenaPolicy, number>> = {
  default: DEFAULT_INITIAL_ARENA_BYTES,
  "evidence-128": 128 * 1024 * 1024,
  "evidence-64": 64 * 1024 * 1024,
};

function requireDeviceLimit(
  limits: GpuLimitsLike,
  name: string,
  minimum?: number,
): number {
  const value = limits[name];
  if (value === undefined || !Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`WebGPU limit ${name} is missing or unsafe`);
  }
  if (minimum !== undefined) {
    if (!Number.isSafeInteger(minimum) || minimum <= 0) {
      throw new Error(
        `Required WebGPU limit ${name} must be a positive safe integer`,
      );
    }
    if (value < minimum) {
      throw new Error(
        `WebGPU limit ${name} is ${value}, below required value ${minimum}`,
      );
    }
  }
  return value;
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
      throw new Error(`Required WebGPU feature is unavailable: ${feature}`);
    }
  }

  const requiredLimits = { ...(options.requiredLimits ?? {}) };
  for (const [name, minimum] of Object.entries(requiredLimits)) {
    requireDeviceLimit(adapter.limits, name, minimum);
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

  // Both limits are live allocation constraints. A high maxBufferSize does not
  // permit a storage binding that exceeds maxStorageBufferBindingSize.
  const maxBufferSize = requireDeviceLimit(device.limits, "maxBufferSize");
  const maxStorageBufferBindingSize = requireDeviceLimit(
    device.limits,
    "maxStorageBufferBindingSize",
  );
  const features = [...availableFeatures]
    .filter((feature) => /^[a-z0-9-]+$/.test(feature))
    .sort();

  return {
    device,
    arenaCapBytes: ARENA_POLICY_BYTES[options.arenaPolicy ?? "default"],
    uploadLaneBytes: DEFAULT_UPLOAD_LANE_BYTES,
    policyIsCapabilityCeiling: false,
    facts: {
      features,
      limits: {
        maxBufferSize,
        maxStorageBufferBindingSize,
      },
      // Safari does not expose performance.memory; absence must not look like a
      // measured zero-byte heap.
      jsHeapLimitBytes: sanitizedHeapLimit(surface),
    },
  };
}
