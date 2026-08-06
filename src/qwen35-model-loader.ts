import { AllocationLedger } from "./allocation-ledger.js";
import {
  probeDeviceProfile,
  type BufferShardPolicy,
  type DeviceProfile,
  type WebGpuProbeSurface,
} from "./device-profile.js";
import { diagnosticError } from "./diagnostics.js";
import {
  GpuArena,
  type GpuAllocation,
  type GpuArenaDevice,
} from "./gpu-arena.js";
import {
  HttpRangeReader,
  browserRangeFetch,
  type ImmutableRangeSource,
  type RangeFetch,
} from "./http-range-reader.js";
import {
  createQwen35HybridState,
  planQwen35HybridState,
  type Qwen35HybridState,
  type Qwen35HybridStateResource,
} from "./hybrid-state.js";
import {
  stringifyManifest,
  validateModelPackageManifest,
  type ImmutableArtifactIdentity,
  type ModelPackageManifest,
  type PackageShard,
} from "./manifest.js";
import {
  BrowserOpfsStorage,
  ImmutableOpfsModelCache,
  modelCacheKey,
  type ModelCacheStorage,
} from "./opfs-model-cache.js";
import {
  PINNED_QWEN35_COMPILED_TOKENIZER,
  loadPinnedQwen35Tokenizer,
} from "./qwen-tokenizer.js";
import { QWEN35_4B_CONFIG } from "./qwen35-config.js";
import { createQwen35GreedyExecutionDriverFactory } from "./qwen35-greedy-driver.js";
import {
  createQwen35DiskBackedTiedEmbeddingStore,
  createQwen35ModelCacheRangeReader,
  hasQwen35DiskBackedTiedEmbedding,
  type Qwen35DiskBackedTiedEmbeddingStore,
} from "./qwen35-disk-backed-tied-embedding.js";
import {
  qwen35PermanentWeightPackage,
  type Qwen35RollingLayerStore,
} from "./qwen35-rolling-layer-weights.js";
import {
  buildQwen35Program,
  type Qwen35Program,
  type Qwen35TensorDirectoryEntry,
} from "./qwen35-program.js";
import {
  qwen35TensorWeightBytes,
  type Qwen35PackageDirectory,
  type Qwen35PackageSegment,
  type Qwen35PackageTensor,
  type Qwen35WeightDirectory,
  type Qwen35WeightDirectoryView,
} from "./qwen35-weight-directory.js";
import {
  initializeQwen35WeightExecution,
  type Qwen35UploadRetirementPolicy,
  type Qwen35WeightResidencyPolicy,
  type Qwen35WeightWriteQueue,
} from "./qwen35-weight-upload.js";
import {
  createQwen35PerformanceCounters,
  createQwen35PerformanceWriteQueue,
  type Qwen35PerformanceCounters,
  type Qwen35PerformanceWriteQueue,
} from "./qwen35-performance.js";
import type { Qwen35WebGpuDevice } from "./qwen35-webgpu-executor.js";
import { createQwen35LazyVisionRuntime } from "./qwen35-vision-runtime.js";
import type { Qwen35VisionPackagePins } from "./qwen35-vision-package-loader.js";
import type {
  LoadOptions,
  Qwen35ExecutionDriver,
  Qwen35LoadedResources,
  RuntimeLoadEvent,
} from "./qwen35-session.js";

export const QWEN35_RUNTIME_ABI = "qwen35-webgpu-v1";
const TIED_INPUT_ROW_CAPACITY = 64;
const TIED_OUTPUT_TILE_ROWS = 1_024;
const QWEN35_DECODABLE_ROWS = 248_070;
const PINNED_LANGUAGE_SOURCE: Readonly<ImmutableArtifactIdentity> =
  Object.freeze({
    repository: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF",
    revision: "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
    file: "Qwen_Qwen3.5-4B-Q3_K_L.gguf",
    size: "2665441248",
    sha256: "41c3f1bf47e477693dab332e73347c7138d5e9fbfe74c6d2eaba590be1f3d20a",
  });
const PINNED_TOKENIZER_SOURCE: Readonly<ImmutableArtifactIdentity> =
  Object.freeze({
    repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
    revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    file: "tokenizer.json",
    size: "12807982",
    sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
  });

/**
 * One loader-owned device shared by allocation, upload, execution, and cleanup.
 * Factory contexts borrow this exact object and must never destroy it.
 */
export interface Qwen35ModelDevice extends Qwen35WebGpuDevice {
  readonly features: Iterable<string>;
  readonly limits: Qwen35WebGpuDevice["limits"] & {
    readonly maxBufferSize: number;
  };
  readonly queue: Qwen35WebGpuDevice["queue"] & Qwen35WeightWriteQueue;
  pushErrorScope(filter: "validation" | "out-of-memory"): void;
  readonly lost: Promise<unknown>;
  destroy(): void;
}

/** Non-owning device surface exposed to execution factories. */
export type Qwen35BorrowedModelDevice = Omit<Qwen35ModelDevice, "destroy">;

/** Profile view that cannot transfer device-destruction authority. */
export type Qwen35BorrowedDeviceProfile = Omit<DeviceProfile, "device"> & {
  readonly device: Qwen35BorrowedModelDevice;
};

export type {
  Qwen35PackageDirectory,
  Qwen35PackageSegment,
  Qwen35PackageTensor,
  Qwen35WeightDirectory,
  Qwen35WeightDirectoryView,
} from "./qwen35-weight-directory.js";

export interface Qwen35DriverFactoryContext {
  /** Borrowed from the loader; the factory may use but must not destroy it. */
  readonly device: Qwen35BorrowedModelDevice;
  readonly profile: Qwen35BorrowedDeviceProfile;
  readonly program: Qwen35Program;
  readonly arena: GpuArena;
  readonly hybridState: Qwen35HybridState;
  /** Permanently resident tensor-owned buffers; tied and rolling weights are excluded. */
  readonly weightDirectory: Qwen35WeightDirectoryView;
  /** Borrowed bounded cache; loader disposal remains the only owner. */
  readonly tiedEmbedding?: Qwen35DiskBackedTiedEmbeddingStore;
  /** Borrowed rolling layer owner; the loader disposes it after driver quiescence. */
  readonly rollingLayers?: Qwen35RollingLayerStore;
  /** Exact immutable mapping from logical tensors to uploaded shard ranges. */
  readonly packageDirectory: Qwen35PackageDirectory;
  /** Optional counters shared by production upload and execution paths. */
  readonly performanceCounters?: Qwen35PerformanceCounters;
}

export interface Qwen35StateAllocationClearContext {
  /** Borrowed from the loader; the factory may use but must not destroy it. */
  readonly device: Qwen35BorrowedModelDevice;
  readonly allocation: GpuAllocation;
  readonly resource: Qwen35HybridStateResource;
  /** Optional load-scoped counters for queue work performed by state clearing. */
  readonly performanceCounters?: Qwen35PerformanceCounters;
}

/** Installation point for the next model-scheduler milestone. */
export interface Qwen35ExecutionDriverFactory {
  clearStateAllocation(
    context: Qwen35StateAllocationClearContext,
  ): Promise<void>;
  /** A rejected create call must release any factory-private partial state. */
  create(context: Qwen35DriverFactoryContext): Promise<Qwen35ExecutionDriver>;
}

export type Qwen35DriverFactoryBoundaryContext = Omit<
  Qwen35DriverFactoryContext,
  "device" | "profile"
> & {
  readonly profile: DeviceProfile;
};

export interface Qwen35DriverFactoryBoundary {
  clearStateAllocation(
    allocation: GpuAllocation,
    resource: Qwen35HybridStateResource,
  ): Promise<void>;
  create(
    context: Qwen35DriverFactoryBoundaryContext,
  ): Promise<Qwen35ExecutionDriver>;
}

/** Freezes non-owning call contexts around one exact loader-owned device. */
export function createQwen35DriverFactoryBoundary(
  device: Qwen35ModelDevice,
  factory: Qwen35ExecutionDriverFactory,
  performanceCounters?: Qwen35PerformanceCounters,
): Qwen35DriverFactoryBoundary {
  return Object.freeze({
    clearStateAllocation(
      allocation: GpuAllocation,
      resource: Qwen35HybridStateResource,
    ) {
      return factory.clearStateAllocation(Object.freeze({
        device,
        allocation,
        resource,
        ...(performanceCounters === undefined ? {} : { performanceCounters }),
      }));
    },
    create(context: Qwen35DriverFactoryBoundaryContext) {
      const borrowedProfile: Qwen35BorrowedDeviceProfile = Object.freeze({
        ...context.profile,
        device,
      });
      return factory.create(Object.freeze({
        ...context,
        profile: borrowedProfile,
        device,
      }));
    },
  });
}

export interface Qwen35BrowserLoadOptions extends LoadOptions {
  readonly manifest: ModelPackageManifest;
  readonly packageBaseUrl: string;
  readonly expectedPackageBaseUrl: string;
  readonly expectedManifestSha256: string;
  readonly compiledTokenizerUrl: string;
  /** Optional test or experiment override; production uses the fixed greedy driver. */
  readonly executionDriverFactory?: Qwen35ExecutionDriverFactory;
  readonly fetchImplementation?: typeof fetch;
  readonly rangeFetch?: RangeFetch;
  readonly cacheStorage?: ModelCacheStorage;
  readonly webGpuSurface?: WebGpuProbeSurface;
  /** Local physical-device experiment selector; it shapes each GPU buffer only. */
  readonly bufferShardPolicy?: BufferShardPolicy;
  /** Local evidence run override; omitted builds keep the measured device default. */
  readonly uploadLaneBytes?: number;
  readonly uploadRetirementPolicy?: Qwen35UploadRetirementPolicy;
  /** Resident weights are preferred on desktop; rolling remains the mobile fallback. */
  readonly residencyPolicy?: Qwen35WeightResidencyPolicy;
  readonly gpuLedgerLimitBytes?: bigint;
  /** Explicit local Chrome smoke-test opt-in; public deployments keep HTTPS pins. */
  readonly allowInsecureLocalhost?: boolean;
  /** Optional local vision package pins for the Chrome development server. */
  readonly visionPackagePins?: Qwen35VisionPackagePins;
}

function browserLoadOptions(
  options: LoadOptions,
): Qwen35BrowserLoadOptions {
  return options as Qwen35BrowserLoadOptions;
}

function deepFreeze<T>(value: T): T {
  if (typeof value !== "object" || value === null || Object.isFrozen(value)) {
    return value;
  }
  for (const child of Object.values(value)) {
    deepFreeze(child);
  }
  return Object.freeze(value);
}

/** Copies caller-owned manifest data into an immutable canonical snapshot. */
export function snapshotQwen35Manifest(
  manifest: ModelPackageManifest,
): ModelPackageManifest {
  const canonical = stringifyManifest(manifest);
  const owned = JSON.parse(canonical) as ModelPackageManifest;
  return deepFreeze(validateModelPackageManifest(owned));
}

function sameIdentity(
  actual: ImmutableArtifactIdentity,
  expected: Readonly<ImmutableArtifactIdentity>,
): boolean {
  return (
    actual.repository === expected.repository &&
    actual.revision === expected.revision &&
    actual.file === expected.file &&
    actual.size === expected.size &&
    actual.sha256 === expected.sha256
  );
}

export function assertQwen35PackageIdentity(
  manifest: ModelPackageManifest,
): void {
  if (
    manifest.packageKind !== "language" ||
    manifest.runtime.abi !== QWEN35_RUNTIME_ABI ||
    !sameIdentity(manifest.source, PINNED_LANGUAGE_SOURCE) ||
    !sameIdentity(manifest.tokenizer, PINNED_TOKENIZER_SOURCE)
  ) {
    throw diagnosticError(
      "model-package-identity-mismatch",
      "Model package does not match the pinned Qwen3.5 sources",
    );
  }
}

export function assertQwen35ConvertedPackageTrust(
  manifest: ModelPackageManifest,
  pins: {
    readonly packageBaseUrl: string;
    readonly expectedPackageBaseUrl: string;
    readonly expectedManifestSha256: string;
    readonly allowInsecureLocalhost?: boolean;
  },
): void {
  const allowInsecureLocalhost = pins.allowInsecureLocalhost === true;
  const expectedBase = safeUrl(
    pins.expectedPackageBaseUrl,
    "Expected package base URL",
    undefined,
    allowInsecureLocalhost,
  );
  const localDevelopmentBase =
    allowInsecureLocalhost &&
    expectedBase.protocol === "http:" &&
    expectedBase.hostname === LOCAL_DEVELOPMENT_HOST &&
    expectedBase.pathname.endsWith("/");
  if (
    (!localDevelopmentBase && expectedBase.hostname !== "huggingface.co") ||
    expectedBase.search !== "" ||
    expectedBase.hash !== "" ||
    (!localDevelopmentBase &&
      !/^\/[^/]+\/[^/]+\/resolve\/[a-f0-9]{40}\/$/.test(expectedBase.pathname))
  ) {
    throw diagnosticError(
      "model-package-url-mutable",
      "Converted package pin must use an immutable Hugging Face revision URL",
    );
  }
  const actualBase = safeUrl(
    pins.packageBaseUrl,
    "Package base URL",
    undefined,
    allowInsecureLocalhost,
  );
  if (
    pins.packageBaseUrl !== pins.expectedPackageBaseUrl ||
    actualBase.href !== expectedBase.href
  ) {
    throw diagnosticError(
      "model-package-url-mismatch",
      "Package base URL does not match the application pin",
    );
  }
  if (
    !/^[a-f0-9]{64}$/.test(pins.expectedManifestSha256) ||
    modelCacheKey(manifest) !== pins.expectedManifestSha256
  ) {
    throw diagnosticError(
      "model-package-manifest-mismatch",
      "Model package manifest does not match the application pin",
    );
  }
}

function safeUrl(
  value: string,
  label: string,
  base?: URL,
  allowInsecureLocalhost = false,
): URL {
  let parsed: URL;
  try {
    parsed = base === undefined ? new URL(value) : new URL(value, base);
  } catch {
    throw diagnosticError("model-url-invalid", `${label} is invalid`);
  }
  const localDevelopmentUrl =
    allowInsecureLocalhost &&
    parsed.protocol === "http:" &&
    parsed.hostname === LOCAL_DEVELOPMENT_HOST;
  if (
    (!localDevelopmentUrl && parsed.protocol !== "https:") ||
    parsed.username.length > 0 ||
    parsed.password.length > 0
  ) {
    throw diagnosticError(
      "model-url-invalid",
      `${label} must use credential-free HTTPS`,
    );
  }
  return parsed;
}

function safeBytes(value: string, label: string): number {
  const parsed = BigInt(value);
  if (parsed > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw diagnosticError("model-size-unsafe", `${label} is too large`);
  }
  return Number(parsed);
}

export function buildQwen35TensorDirectory(
  manifest: ModelPackageManifest,
): readonly Qwen35TensorDirectoryEntry[] {
  const tensors = new Map<string, Qwen35TensorDirectoryEntry>();
  for (const segment of manifest.tensorLayout) {
    const shape = segment.shape.map((dimension) =>
      safeBytes(dimension, "Tensor dimension"),
    );
    const existing = tensors.get(segment.name);
    const entry = Object.freeze({
      name: segment.name,
      shape: Object.freeze(shape),
      ggmlType: segment.ggmlType,
      storageType: segment.storageType,
    }) as Qwen35TensorDirectoryEntry;
    if (existing === undefined) {
      tensors.set(segment.name, entry);
    } else if (
      existing.ggmlType !== entry.ggmlType ||
      existing.storageType !== entry.storageType ||
      existing.shape.some((value, index) => value !== entry.shape[index])
    ) {
      throw diagnosticError(
        "model-tensor-directory-invalid",
        "Tensor segments disagree at the program boundary",
      );
    }
  }
  return Object.freeze([...tensors.values()]);
}

export function buildQwen35PackageDirectory(
  sourceManifest: ModelPackageManifest,
): Qwen35PackageDirectory {
  const manifest = validateModelPackageManifest(sourceManifest);
  const tensors = new Map<string, {
    entry: Omit<Qwen35PackageTensor, "segments">;
    segments: Qwen35PackageSegment[];
  }>();
  for (const segment of manifest.tensorLayout) {
    const shape = Object.freeze(
      segment.shape.map((dimension) => safeBytes(dimension, "Tensor dimension")),
    );
    const existing = tensors.get(segment.name);
    const entry = Object.freeze({
      name: segment.name,
      shape,
      ggmlType: segment.ggmlType,
      storageType: segment.storageType,
    });
    if (
      existing !== undefined &&
      (existing.entry.ggmlType !== entry.ggmlType ||
        existing.entry.storageType !== entry.storageType ||
        existing.entry.shape.length !== entry.shape.length ||
        existing.entry.shape.some((value, index) => value !== entry.shape[index]))
    ) {
      throw diagnosticError(
        "model-tensor-directory-invalid",
        "Tensor segments disagree at the driver boundary",
      );
    }
    const grouped = existing ?? { entry, segments: [] };
    grouped.segments.push(Object.freeze({
      shardIndex: segment.shard,
      shardOffset: segment.shardOffset,
      tensorOffset: segment.tensorOffset,
      length: segment.length,
    }));
    tensors.set(segment.name, grouped);
  }
  return Object.freeze({
    manifestSha256: modelCacheKey(manifest),
    shards: Object.freeze(manifest.shards.map((shard, index) => Object.freeze({
      index,
      url: shard.url,
      offset: shard.offset,
      length: shard.length,
      sha256: shard.sha256,
    }))),
    tensors: Object.freeze([...tensors.values()].map(({ entry, segments }) =>
      Object.freeze({ ...entry, segments: Object.freeze(segments) }),
    )),
  });
}

async function* responseChunks(
  response: Response,
  signal: AbortSignal,
  onProgress?: (completedBytes: number) => void,
): AsyncIterable<Uint8Array> {
  const reader = response.body?.getReader();
  if (reader === undefined) {
    throw diagnosticError(
      "tokenizer-response-body-missing",
      "Tokenizer response body is unavailable",
    );
  }
  try {
    let completedBytes = 0;
    while (true) {
      signal.throwIfAborted();
      const item = await reader.read();
      if (item.done) return;
      completedBytes += item.value.byteLength;
      onProgress?.(completedBytes);
      yield item.value;
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

async function loadTokenizer(
  url: URL,
  fetchImplementation: typeof fetch,
  signal: AbortSignal,
  onProgress?: (completedBytes: number) => void,
) {
  const response = await fetchImplementation(url, {
    method: "GET",
    cache: "no-store",
    signal,
  });
  if (!response.ok) {
    await response.body?.cancel().catch(() => undefined);
    throw diagnosticError(
      "tokenizer-response-invalid",
      "Tokenizer artifact request failed",
    );
  }
  const contentLength = response.headers.get("content-length");
  if (
    contentLength !== null &&
    contentLength !== String(PINNED_QWEN35_COMPILED_TOKENIZER.byteLength)
  ) {
    await response.body?.cancel().catch(() => undefined);
    throw diagnosticError(
      "tokenizer-response-size-invalid",
      "Tokenizer artifact length does not match the pinned package",
    );
  }
  return loadPinnedQwen35Tokenizer(
    responseChunks(response, signal, onProgress),
    PINNED_QWEN35_COMPILED_TOKENIZER.byteLength,
  );
}

function shardSource(
  base: URL,
  shard: PackageShard,
  allowInsecureLocalhost = false,
): ImmutableRangeSource {
  return {
    locator: safeUrl(shard.url, "Model shard URL", base, allowInsecureLocalhost).href,
    byteLength: safeBytes(shard.length, "Model shard length"),
    // The cache authenticates every complete shard against its manifest hash.
    immutableUrl: true,
  };
}

function destroyReverse(allocations: readonly GpuAllocation[]): unknown {
  let firstError: unknown;
  for (let index = allocations.length - 1; index >= 0; index -= 1) {
    try {
      allocations[index]!.destroy();
    } catch (error) {
      firstError ??= error;
    }
  }
  return firstError;
}

export async function cleanupQwen35GpuResources(input: {
  readonly driver: Pick<Qwen35ExecutionDriver, "dispose"> | null;
  readonly tiedEmbedding?: Pick<Qwen35DiskBackedTiedEmbeddingStore, "dispose">;
  readonly rollingLayers?: Pick<Qwen35RollingLayerStore, "dispose">;
  readonly vision?: Pick<NonNullable<Qwen35LoadedResources["vision"]>, "dispose">;
  readonly device: {
    readonly queue: { onSubmittedWorkDone(): Promise<void> };
    destroy(): void;
  };
  readonly hybridState: Pick<Qwen35HybridState, "dispose"> | null;
  readonly weightAllocations: readonly GpuAllocation[];
  readonly ledger: Pick<AllocationLedger, "assertAllReleased">;
}): Promise<void> {
  let firstError: unknown;
  if (input.vision !== undefined) {
    try {
      await input.vision.dispose();
    } catch (error) {
      firstError ??= error;
    }
  }
  try {
    await input.driver?.dispose();
  } catch (error) {
    firstError ??= error;
  }
  if (input.rollingLayers !== undefined) {
    try {
      // Every staged layer is borrowed by the driver until its executor fence.
      await input.rollingLayers.dispose();
    } catch (error) {
      firstError ??= error;
    }
  }
  if (input.tiedEmbedding !== undefined) {
    try {
      // The driver must release every borrowed cache binding before its loader
      // owner fences and destroys the two tied-cache allocations.
      await input.tiedEmbedding.dispose();
    } catch (error) {
      firstError ??= error;
    }
  }
  try {
    // Submitted work may still reference weights or state after create/upload
    // fails. Destruction starts only after the queue reaches this boundary.
    await input.device.queue.onSubmittedWorkDone();
  } catch (error) {
    firstError ??= error;
  }
  try {
    input.hybridState?.dispose();
  } catch (error) {
    firstError ??= error;
  }
  const weightError = destroyReverse(input.weightAllocations);
  firstError ??= weightError;
  try {
    input.device.destroy();
  } catch (error) {
    firstError ??= error;
  }
  try {
    input.ledger.assertAllReleased();
  } catch (error) {
    firstError ??= error;
  }
  if (firstError !== undefined) {
    throw firstError;
  }
}

export function qwen35AllocatedWeightBytes(
  packageDirectory: Qwen35PackageDirectory,
  residencyPolicy: "rolling" | "resident" = "rolling",
): bigint {
  return qwen35TensorWeightBytes(
    residencyPolicy === "resident"
      ? packageDirectory
      : qwen35PermanentWeightPackage(packageDirectory),
  );
}

export function isQwen35AppleMobileBrowser(
  userAgent = globalThis.navigator?.userAgent ?? "",
  maxTouchPoints = globalThis.navigator?.maxTouchPoints ?? 0,
): boolean {
  return /\b(?:iPhone|iPad|iPod)\b/i.test(userAgent) ||
    (/\bMacintosh\b/i.test(userAgent) &&
      (/\bMobile\b/i.test(userAgent) || maxTouchPoints > 1));
}

function resolveQwen35ResidencyPolicy(
  options: Qwen35BrowserLoadOptions,
  packageDirectory: Qwen35PackageDirectory,
  stateBytes: bigint,
): Qwen35WeightResidencyPolicy {
  const requested = options.residencyPolicy ?? "auto";
  if (requested !== "auto") return requested;
  if (isQwen35AppleMobileBrowser()) return "rolling";
  const residentBytes = stateBytes + qwen35TensorWeightBytes(packageDirectory);
  if (
    options.gpuLedgerLimitBytes !== undefined &&
    options.gpuLedgerLimitBytes < residentBytes
  ) {
    return "rolling";
  }
  return "auto";
}

const GPU_LEDGER_REPRESENTATION_GUARD = BigInt(Number.MAX_SAFE_INTEGER);
const LOCAL_DEVELOPMENT_HOST = ["local", "host"].join("");

export function createQwen35GpuLedger(
  requiredBytes: bigint,
  experimentalBudgetBytes?: bigint,
): AllocationLedger {
  // The default is only the largest value metrics can represent exactly. It is
  // not a product limit, device capability claim, or inferred memory budget.
  const limitBytes =
    experimentalBudgetBytes ?? GPU_LEDGER_REPRESENTATION_GUARD;
  if (requiredBytes > GPU_LEDGER_REPRESENTATION_GUARD) {
    throw diagnosticError(
      "model-size-unsafe",
      "Tracked GPU ownership exceeds the safe metrics representation",
    );
  }
  if (
    experimentalBudgetBytes !== undefined &&
    experimentalBudgetBytes > GPU_LEDGER_REPRESENTATION_GUARD
  ) {
    throw diagnosticError(
      "gpu-ledger-limit-unsafe",
      "GPU ledger budget exceeds the safe metrics representation",
    );
  }
  if (limitBytes < requiredBytes) {
    throw diagnosticError(
      "gpu-ledger-limit-insufficient",
      "GPU ledger cannot own the complete model and context state",
    );
  }
  return new AllocationLedger(limitBytes);
}

const REQUIRED_QWEN35_DEVICE_LIMITS = Object.freeze([
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "minStorageBufferOffsetAlignment",
  "minUniformBufferOffsetAlignment",
  "maxUniformBufferBindingSize",
  "maxComputeWorkgroupsPerDimension",
] as const);

function isIterable(value: unknown): value is Iterable<unknown> {
  if (value === null || value === undefined) {
    return false;
  }
  try {
    return typeof (value as { [Symbol.iterator]?: unknown })[
      Symbol.iterator
    ] === "function";
  } catch {
    return false;
  }
}

/** Validates the complete model device surface before any GPU allocation. */
export function assertQwen35ModelDevice(
  value: unknown,
): asserts value is Qwen35ModelDevice {
  if (typeof value !== "object" || value === null) {
    throw diagnosticError(
      "webgpu-device-incomplete",
      "WebGPU device is missing required resource methods",
    );
  }
  const device = value as Record<string, unknown>;
  const queue = device.queue as Record<string, unknown> | null | undefined;
  const limits = device.limits as Record<string, unknown> | null | undefined;
  const lost = device.lost as Record<string, unknown> | null | undefined;
  const validLimits =
    limits !== null &&
    limits !== undefined &&
    REQUIRED_QWEN35_DEVICE_LIMITS.every((name) => {
      const limit = limits[name];
      return Number.isSafeInteger(limit) && (limit as number) > 0;
    });
  if (
    !validLimits ||
    !isIterable(device.features) ||
    queue === null ||
    queue === undefined ||
    typeof queue.writeBuffer !== "function" ||
    typeof queue.submit !== "function" ||
    typeof queue.onSubmittedWorkDone !== "function" ||
    typeof device.createBuffer !== "function" ||
    typeof device.pushErrorScope !== "function" ||
    typeof device.popErrorScope !== "function" ||
    typeof device.createShaderModule !== "function" ||
    typeof device.createComputePipelineAsync !== "function" ||
    typeof device.createBindGroup !== "function" ||
    typeof device.createCommandEncoder !== "function" ||
    typeof device.destroy !== "function" ||
    lost === null ||
    lost === undefined ||
    typeof lost.then !== "function"
  ) {
    throw diagnosticError(
      "webgpu-device-incomplete",
      "WebGPU device is missing required resource methods",
    );
  }
}

function destroyQwen35DeviceOrThrow(
  value: unknown,
  code: string,
  message: string,
): void {
  const destroy = (value as { destroy?: unknown } | null)?.destroy;
  if (typeof destroy !== "function") {
    throw diagnosticError(code, message);
  }
  try {
    destroy.call(value);
  } catch {
    throw diagnosticError(code, message);
  }
}

/** Validates an acquired device and preserves cleanup failure as public state. */
export function assertQwen35AcquiredModelDevice(
  value: unknown,
): asserts value is Qwen35ModelDevice {
  try {
    assertQwen35ModelDevice(value);
  } catch (validationError) {
    destroyQwen35DeviceOrThrow(
      value,
      "webgpu-device-rejection-cleanup-failed",
      "Rejected WebGPU device cleanup failed",
    );
    throw validationError;
  }
}

/** Builds the authenticated, GPU-owned resources for one lock-held session. */
export async function loadQwen35BrowserResources(
  signal: AbortSignal,
  rawOptions: LoadOptions,
): Promise<Qwen35LoadedResources> {
  const options = browserLoadOptions(rawOptions);
  const reportLoadEvent = (event: RuntimeLoadEvent): void => {
    try {
      options.onLoadEvent?.(Object.freeze(event));
    } catch {
      // Resource ownership and authentication never depend on telemetry.
    }
  };
  const factory =
    options.executionDriverFactory ??
    createQwen35GreedyExecutionDriverFactory();
  // No caller-owned manifest reference crosses the first asynchronous
  // boundary. Every later cache, URL, and driver read uses this snapshot.
  const manifest = snapshotQwen35Manifest(options.manifest);
  assertQwen35PackageIdentity(manifest);
  assertQwen35ConvertedPackageTrust(manifest, options);
  const packageBase = safeUrl(
    options.packageBaseUrl,
    "Package base URL",
    undefined,
    options.allowInsecureLocalhost === true,
  );
  const tokenizerUrl = safeUrl(
    options.compiledTokenizerUrl,
    "Compiled tokenizer URL",
    undefined,
    options.allowInsecureLocalhost === true,
  );
  const fetchImplementation = options.fetchImplementation ?? globalThis.fetch;
  if (fetchImplementation === undefined) {
    throw diagnosticError(
      "fetch-unavailable",
      "The browser Fetch API is unavailable",
    );
  }
  const rangeReader = new HttpRangeReader(
    options.rangeFetch ?? browserRangeFetch(fetchImplementation),
  );
  const packageBytes = manifest.shards.reduce(
    (total, shard) => total + safeBytes(shard.length, "Model shard length"),
    0,
  );
  reportLoadEvent({
    phase: "cache_scan",
    completedBytes: 0,
    totalBytes: packageBytes,
  });
  const browserStorageManager = globalThis.navigator?.storage;
  if (
    options.cacheStorage === undefined &&
    browserStorageManager === undefined
  ) {
    throw diagnosticError(
      "opfs-unavailable",
      "The browser origin-private file system is unavailable",
    );
  }
  const storage =
    options.cacheStorage ??
    (await BrowserOpfsStorage.open(browserStorageManager!));
  const cache = new ImmutableOpfsModelCache(storage, rangeReader, {
    onLoadEvent: reportLoadEvent,
  });
  const cached = await cache.ensure(
    manifest,
    (shard) => shardSource(packageBase, shard, options.allowInsecureLocalhost === true),
    signal,
  );
  signal.throwIfAborted();
  reportLoadEvent({
    phase: "tokenizer_load",
    completedBytes: 0,
    totalBytes: PINNED_QWEN35_COMPILED_TOKENIZER.byteLength,
  });
  const tokenizer = await loadTokenizer(
    tokenizerUrl,
    fetchImplementation,
    signal,
    (completedBytes) => reportLoadEvent({
      phase: "tokenizer_load",
      completedBytes,
      totalBytes: PINNED_QWEN35_COMPILED_TOKENIZER.byteLength,
    }),
  );
  const program = buildQwen35Program({
    config: QWEN35_4B_CONFIG,
    tensors: buildQwen35TensorDirectory(manifest),
  });
  const packageDirectory = buildQwen35PackageDirectory(manifest);
  const navigatorWithGpu = globalThis.navigator as
    | (Navigator & {
    readonly gpu?: WebGpuProbeSurface["gpu"];
      })
    | undefined;
  const browserSurface: WebGpuProbeSurface = {
    ...(navigatorWithGpu?.gpu === undefined
      ? {}
      : { gpu: navigatorWithGpu.gpu }),
  };
  const profileOptions = {
    requiredFeatures: ["shader-f16"],
    ...(options.bufferShardPolicy === undefined
      ? {}
      : {
          // This changes only physical buffer segmentation. It never changes
          // the selected model, its total residency, or the 16K context contract.
          bufferShardPolicy: options.bufferShardPolicy,
        }),
    rejectedDeviceCleanup: (rejectedDevice: unknown) => {
      destroyQwen35DeviceOrThrow(
        rejectedDevice,
        "webgpu-device-rejection-cleanup-failed",
        "Rejected WebGPU device cleanup failed",
      );
    },
  };
  reportLoadEvent({
    phase: "device_probe",
    completedBytes: 0,
    totalBytes: 0,
  });
  const profile = await probeDeviceProfile(
    options.webGpuSurface ?? browserSurface,
    profileOptions,
  );
  const device = profile.device as unknown;
  const stateBytesBigInt = planQwen35HybridState(
    QWEN35_4B_CONFIG.productContextLength,
  ).totalBytes;
  const residencyPolicy = resolveQwen35ResidencyPolicy(
    options,
    packageDirectory,
    stateBytesBigInt,
  );
  const weightBytesBigInt = qwen35AllocatedWeightBytes(
    packageDirectory,
    residencyPolicy === "rolling" ? "rolling" : "resident",
  );
  const stateBytes = Number(stateBytesBigInt);
  const weightBytes = Number(weightBytesBigInt);
  assertQwen35AcquiredModelDevice(device);
  let ledger: AllocationLedger;
  let arena: GpuArena;
  let driverBoundary: Qwen35DriverFactoryBoundary;
  const performanceCounters = createQwen35PerformanceCounters();
  const performanceQueue = createQwen35PerformanceWriteQueue({
    queue: device.queue as Qwen35PerformanceWriteQueue,
    counters: performanceCounters,
  });
  try {
    const minimumLedgerBytes = stateBytesBigInt + weightBytesBigInt;
    ledger = createQwen35GpuLedger(
      minimumLedgerBytes,
      options.gpuLedgerLimitBytes,
    );
    arena = new GpuArena(device, ledger, {
      bufferShardCapBytes: BigInt(profile.bufferShardCapBytes),
    });
    driverBoundary = createQwen35DriverFactoryBoundary(
      device,
      factory,
      performanceCounters,
    );
  } catch (error) {
    destroyQwen35DeviceOrThrow(
      device,
      "model-load-device-cleanup-failed",
      "Model device cleanup failed during load setup",
    );
    throw error;
  }
  let weightDirectory: Qwen35WeightDirectory | null = null;
  let hybridState: Qwen35HybridState | null = null;
  let driver: Qwen35ExecutionDriver | null = null;
  let tiedEmbedding: Qwen35DiskBackedTiedEmbeddingStore | null = null;
  let rollingLayers: Qwen35RollingLayerStore | null = null;
  try {
    reportLoadEvent({
      phase: "state_allocate",
      completedBytes: 0,
      totalBytes: stateBytes,
      currentGpuBytes: Number(ledger.snapshot().currentBytes),
      peakGpuBytes: Number(ledger.snapshot().peakBytes),
    });
    hybridState = await createQwen35HybridState({
      arena,
      capacity: QWEN35_4B_CONFIG.productContextLength,
      clearAllocation: (allocation, resource) =>
        driverBoundary.clearStateAllocation(allocation, resource),
      onProgress: (completedBytes) => reportLoadEvent({
        phase: "state_allocate",
        completedBytes,
        totalBytes: stateBytes,
        currentGpuBytes: Number(ledger.snapshot().currentBytes),
        peakGpuBytes: Number(ledger.snapshot().peakBytes),
      }),
    });
    // Safari rejected the second state buffer only when growth followed driver
    // creation. Prewarming one 256-token page keeps the 16K remainder lazy and
    // establishes persistent state before any driver-private GPU resources.
    await hybridState.ensureCapacity(1, signal);
    reportLoadEvent({
      phase: "weights_allocate",
      completedBytes: 0,
      totalBytes: weightBytes,
      currentGpuBytes: Number(ledger.snapshot().currentBytes),
      peakGpuBytes: Number(ledger.snapshot().peakBytes),
    });
    const initialized = await initializeQwen35WeightExecution({
      arena,
      packageDirectory,
      storage,
      cached,
      queue: performanceQueue,
      uploadLaneBytes: options.uploadLaneBytes ?? profile.uploadLaneBytes,
      uploadRetirementPolicy: options.uploadRetirementPolicy ?? "window",
      residencyPolicy,
      signal,
      onWeightsAllocated: (completedBytes) => reportLoadEvent({
        phase: "weights_allocate",
        completedBytes,
        totalBytes: weightBytes,
        currentGpuBytes: Number(ledger.snapshot().currentBytes),
        peakGpuBytes: Number(ledger.snapshot().peakBytes),
      }),
      onWeightsUploaded: (completedBytes) => reportLoadEvent({
        phase: "weights_upload",
        completedBytes,
        totalBytes: weightBytes,
        currentGpuBytes: Number(ledger.snapshot().currentBytes),
        peakGpuBytes: Number(ledger.snapshot().peakBytes),
      }),
      onDriverInitialize: () => reportLoadEvent({
        phase: "driver_initialize",
        completedBytes: 0,
        totalBytes: 0,
        currentGpuBytes: Number(ledger.snapshot().currentBytes),
        peakGpuBytes: Number(ledger.snapshot().peakBytes),
      }),
      createDriver: async (uploadedWeights, stagedRollingLayers) => {
        rollingLayers = stagedRollingLayers ?? null;
        if (
          tiedEmbedding === null &&
          stagedRollingLayers !== undefined &&
          hasQwen35DiskBackedTiedEmbedding(packageDirectory)
        ) {
          tiedEmbedding = await createQwen35DiskBackedTiedEmbeddingStore({
            arena,
            queue: performanceQueue,
            packageDirectory,
            cached,
            rangeReader: createQwen35ModelCacheRangeReader(storage),
            inputRowCapacity: TIED_INPUT_ROW_CAPACITY,
            outputTileRows: TIED_OUTPUT_TILE_ROWS,
            decodableRows: QWEN35_DECODABLE_ROWS,
          });
        }
        return driverBoundary.create({
          profile,
          program,
          arena,
          hybridState: hybridState!,
          weightDirectory: uploadedWeights,
          packageDirectory,
          performanceCounters,
          ...(tiedEmbedding === null ? {} : { tiedEmbedding }),
          ...(rollingLayers === null ? {} : { rollingLayers }),
        });
      },
    });
    weightDirectory = initialized.directory;
    driver = initialized.driver;
    signal.throwIfAborted();
  } catch (error) {
    try {
      await cleanupQwen35GpuResources({
        driver,
        ...(tiedEmbedding === null ? {} : { tiedEmbedding }),
        ...(rollingLayers === null ? {} : { rollingLayers }),
        device,
        hybridState,
        weightAllocations: weightDirectory?.allocations ?? [],
        ledger,
      });
    } catch {
      throw diagnosticError(
        "model-load-rollback-failed",
        "Model load rollback did not complete",
      );
    }
    throw error;
  }

  const ownedDriver = driver!;
  const ownedState = hybridState!;
  const ownedWeightDirectory = weightDirectory!;
  const ownedTiedEmbedding = ((): Qwen35DiskBackedTiedEmbeddingStore | null =>
    tiedEmbedding)();
  const ownedRollingLayers = ((): Qwen35RollingLayerStore | null =>
    rollingLayers)();
  const sharedGpuExecutor = ownedDriver.sharedGpuExecutor;
  const vision = sharedGpuExecutor === undefined
    ? undefined
    : createQwen35LazyVisionRuntime({
        device,
        profile,
        arena,
        ledger,
        executor: sharedGpuExecutor,
        performanceQueue,
        ...(options.fetchImplementation === undefined
          ? {}
          : { fetchImplementation: options.fetchImplementation }),
        ...(options.visionPackagePins === undefined
          ? {}
          : { visionPackagePins: options.visionPackagePins }),
      });
  let disposed = false;
  const performanceMetrics = () => {
    if (disposed) {
      performanceCounters.setGpuBreakdown({
        permanentGpuBytes: 0,
        transientGpuBytes: 0,
        stateGpuBytes: 0,
      });
      return performanceCounters.snapshot();
    }
    const tiedMetrics = ownedTiedEmbedding?.getMetrics();
    const rollingMetrics = ownedRollingLayers?.getMetrics();
    const permanentGpuBytes = Number(ownedWeightDirectory.allocatedBytes);
    const stateGpuBytes = Number(ownedState.allocatedBytes);
    const currentGpuBytes = Number(ledger.snapshot().currentBytes);
    performanceCounters.setGpuBreakdown({
      permanentGpuBytes,
      // The ledger includes activation, uniform, cache, and vision resources;
      // subtracting the two owned persistent categories reports all remaining
      // driver resources as transient without hiding scratch allocations.
      transientGpuBytes: Math.max(
        0,
        currentGpuBytes - permanentGpuBytes - stateGpuBytes,
      ),
      stateGpuBytes,
    });
    return Object.freeze({
      ...performanceCounters.snapshot(),
      diskReadBytes:
        (tiedMetrics?.diskReadBytes ?? 0) +
        (rollingMetrics?.diskReadBytes ?? 0),
    });
  };
  return {
    tokenizer,
    driver: ownedDriver,
    cacheHit: cached.cacheHit,
    trackedCpuBytes: PINNED_QWEN35_COMPILED_TOKENIZER.byteLength,
    trackedGpuBytes: Number(ledger.snapshot().currentBytes),
    gpuByteMetrics() {
      const snapshot = ledger.snapshot();
      return Object.freeze({
        currentBytes: Number(snapshot.currentBytes),
        peakBytes: Number(snapshot.peakBytes),
      });
    },
    performanceMetrics,
    deviceLost: device.lost,
    ...(vision === undefined ? {} : { vision }),
    async dispose() {
      if (disposed) return;
      disposed = true;
      try {
        await cleanupQwen35GpuResources({
          driver: ownedDriver,
          ...(ownedTiedEmbedding === null ? {} : { tiedEmbedding: ownedTiedEmbedding }),
          ...(ownedRollingLayers === null ? {} : { rollingLayers: ownedRollingLayers }),
          ...(vision === undefined ? {} : { vision }),
          device,
          hybridState: ownedState,
          weightAllocations: ownedWeightDirectory.allocations,
          ledger,
        });
      } catch {
        throw diagnosticError(
          "model-resource-cleanup-failed",
          "Model resource cleanup did not complete",
        );
      }
    },
  };
}
