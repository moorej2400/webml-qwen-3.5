import { AllocationLedger } from "./allocation-ledger.js";
import {
  probeDeviceProfile,
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
  type CachedModelPackage,
  type ModelCacheStorage,
} from "./opfs-model-cache.js";
import {
  PINNED_QWEN35_COMPILED_TOKENIZER,
  loadPinnedQwen35Tokenizer,
} from "./qwen-tokenizer.js";
import { QWEN35_4B_CONFIG } from "./qwen35-config.js";
import {
  buildQwen35Program,
  type Qwen35Program,
  type Qwen35TensorDirectoryEntry,
} from "./qwen35-program.js";
import type {
  LoadOptions,
  Qwen35ExecutionDriver,
  Qwen35LoadedResources,
} from "./qwen35-session.js";

export const QWEN35_RUNTIME_ABI = "qwen35-webgpu-v1";
const GPU_STORAGE_AND_COPY_DST = 0x0080 | 0x0008;
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

interface Qwen35OwnedDevice extends GpuArenaDevice {
  readonly lost: Promise<unknown>;
  readonly queue: {
    onSubmittedWorkDone(): Promise<void>;
  };
  destroy(): void;
}

export interface Qwen35PackageSegment {
  readonly shardIndex: number;
  readonly shardOffset: string;
  readonly tensorOffset: string;
  readonly length: string;
}

export interface Qwen35PackageTensor {
  readonly name: string;
  readonly shape: readonly number[];
  readonly ggmlType: number;
  readonly storageType: string;
  readonly segments: readonly Qwen35PackageSegment[];
}

export interface Qwen35PackageDirectory {
  readonly manifestSha256: string;
  readonly shards: readonly {
    readonly index: number;
    readonly url: string;
    readonly offset: string;
    readonly length: string;
    readonly sha256: string;
  }[];
  readonly tensors: readonly Qwen35PackageTensor[];
}

export interface Qwen35WeightUploadInput {
  readonly shardIndex: number;
  readonly allocation: GpuAllocation;
  readonly chunk: Uint8Array;
  readonly byteOffset: number;
  readonly signal: AbortSignal;
}

export interface Qwen35DriverFactoryContext {
  readonly profile: DeviceProfile;
  readonly program: Qwen35Program;
  readonly arena: GpuArena;
  readonly hybridState: Qwen35HybridState;
  readonly weightAllocations: readonly GpuAllocation[];
  /** Exact immutable mapping from logical tensors to uploaded shard ranges. */
  readonly packageDirectory: Qwen35PackageDirectory;
}

/** Installation point for the next model-scheduler milestone. */
export interface Qwen35ExecutionDriverFactory {
  clearStateAllocation(
    allocation: GpuAllocation,
    resource: Qwen35HybridStateResource,
  ): Promise<void>;
  /** A rejected create call must release any factory-private partial state. */
  create(context: Qwen35DriverFactoryContext): Promise<Qwen35ExecutionDriver>;
  uploadWeightChunk(
    driver: Qwen35ExecutionDriver,
    input: Qwen35WeightUploadInput,
  ): Promise<void>;
}

export interface Qwen35BrowserLoadOptions extends LoadOptions {
  readonly manifest: ModelPackageManifest;
  readonly packageBaseUrl: string;
  readonly expectedPackageBaseUrl: string;
  readonly expectedManifestSha256: string;
  readonly compiledTokenizerUrl: string;
  readonly executionDriverFactory?: Qwen35ExecutionDriverFactory;
  readonly fetchImplementation?: typeof fetch;
  readonly rangeFetch?: RangeFetch;
  readonly cacheStorage?: ModelCacheStorage;
  readonly webGpuSurface?: WebGpuProbeSurface;
  readonly gpuLedgerLimitBytes?: bigint;
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
  },
): void {
  const expectedBase = safeUrl(
    pins.expectedPackageBaseUrl,
    "Expected package base URL",
  );
  if (
    expectedBase.hostname !== "huggingface.co" ||
    expectedBase.search !== "" ||
    expectedBase.hash !== "" ||
    !/^\/[^/]+\/[^/]+\/resolve\/[a-f0-9]{40}\/$/.test(
      expectedBase.pathname,
    )
  ) {
    throw diagnosticError(
      "model-package-url-mutable",
      "Converted package pin must use an immutable Hugging Face revision URL",
    );
  }
  const actualBase = safeUrl(pins.packageBaseUrl, "Package base URL");
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

function safeUrl(value: string, label: string, base?: URL): URL {
  let parsed: URL;
  try {
    parsed = base === undefined ? new URL(value) : new URL(value, base);
  } catch {
    throw diagnosticError("model-url-invalid", `${label} is invalid`);
  }
  if (
    parsed.protocol !== "https:" ||
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
): AsyncIterable<Uint8Array> {
  const reader = response.body?.getReader();
  if (reader === undefined) {
    throw diagnosticError(
      "tokenizer-response-body-missing",
      "Tokenizer response body is unavailable",
    );
  }
  try {
    while (true) {
      signal.throwIfAborted();
      const item = await reader.read();
      if (item.done) return;
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
    responseChunks(response, signal),
    PINNED_QWEN35_COMPILED_TOKENIZER.byteLength,
  );
}

function shardSource(
  base: URL,
  shard: PackageShard,
): ImmutableRangeSource {
  return {
    locator: safeUrl(shard.url, "Model shard URL", base).href,
    byteLength: safeBytes(shard.length, "Model shard length"),
    // The cache authenticates every complete shard against its manifest hash.
    immutableUrl: true,
  };
}

export async function streamQwen35CachedWeights(input: {
  storage: ModelCacheStorage;
  cached: CachedModelPackage;
  allocations: readonly GpuAllocation[];
  driver: Qwen35ExecutionDriver;
  factory: Qwen35ExecutionDriverFactory;
  uploadLaneBytes: number;
  signal: AbortSignal;
}): Promise<void> {
  for (const [shardIndex, shard] of input.cached.shards.entries()) {
    const stream = await input.storage.openRead(shard.storagePath);
    if (stream === null) {
      throw diagnosticError(
        "model-cache-shard-missing",
        "Authenticated model cache shard is unavailable",
      );
    }
    let byteOffset = 0;
    for await (const sourceChunk of stream) {
      for (
        let chunkOffset = 0;
        chunkOffset < sourceChunk.byteLength;
        chunkOffset += input.uploadLaneBytes
      ) {
        input.signal.throwIfAborted();
        const chunk = sourceChunk.subarray(
          chunkOffset,
          Math.min(sourceChunk.byteLength, chunkOffset + input.uploadLaneBytes),
        );
        await input.factory.uploadWeightChunk(input.driver, {
          shardIndex,
          allocation: input.allocations[shardIndex]!,
          chunk,
          byteOffset,
          signal: input.signal,
        });
        byteOffset += chunk.byteLength;
      }
    }
    if (byteOffset !== shard.byteLength) {
      throw diagnosticError(
        "model-cache-shard-size-invalid",
        "Authenticated model cache shard has an invalid length",
      );
    }
  }
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
  readonly device: {
    readonly queue: { onSubmittedWorkDone(): Promise<void> };
    destroy(): void;
  };
  readonly hybridState: Pick<Qwen35HybridState, "dispose"> | null;
  readonly weightAllocations: readonly GpuAllocation[];
  readonly ledger: Pick<AllocationLedger, "assertAllReleased">;
}): Promise<void> {
  let firstError: unknown;
  try {
    await input.driver?.dispose();
  } catch (error) {
    firstError ??= error;
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
  shards: readonly { readonly byteLength: number }[],
): bigint {
  return shards.reduce(
    (sum, shard) =>
      sum + ((BigInt(shard.byteLength) + 3n) / 4n) * 4n,
    0n,
  );
}

const GPU_LEDGER_REPRESENTATION_GUARD = BigInt(Number.MAX_SAFE_INTEGER);

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

/** Builds the authenticated, GPU-owned resources for one lock-held session. */
export async function loadQwen35BrowserResources(
  signal: AbortSignal,
  rawOptions: LoadOptions,
): Promise<Qwen35LoadedResources> {
  const options = browserLoadOptions(rawOptions);
  const factory = options.executionDriverFactory;
  if (factory === undefined) {
    throw diagnosticError(
      "qwen-execution-driver-not-installed",
      "The Qwen3.5 execution driver is not installed",
    );
  }
  // No caller-owned manifest reference crosses the first asynchronous
  // boundary. Every later cache, URL, and driver read uses this snapshot.
  const manifest = snapshotQwen35Manifest(options.manifest);
  assertQwen35PackageIdentity(manifest);
  assertQwen35ConvertedPackageTrust(manifest, options);
  const packageBase = safeUrl(options.packageBaseUrl, "Package base URL");
  const tokenizerUrl = safeUrl(
    options.compiledTokenizerUrl,
    "Compiled tokenizer URL",
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
  const cache = new ImmutableOpfsModelCache(storage, rangeReader);
  const cached = await cache.ensure(
    manifest,
    (shard) => shardSource(packageBase, shard),
    signal,
  );
  signal.throwIfAborted();
  const tokenizer = await loadTokenizer(
    tokenizerUrl,
    fetchImplementation,
    signal,
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
  const profile = await probeDeviceProfile(
    options.webGpuSurface ?? browserSurface,
    { requiredFeatures: ["shader-f16"] },
  );
  const device = profile.device as unknown as Qwen35OwnedDevice;
  if (
    typeof device.createBuffer !== "function" ||
    typeof device.pushErrorScope !== "function" ||
    typeof device.popErrorScope !== "function" ||
    typeof device.queue?.onSubmittedWorkDone !== "function" ||
    typeof device.destroy !== "function" ||
    typeof device.lost?.then !== "function"
  ) {
    device.destroy?.();
    throw diagnosticError(
      "webgpu-device-incomplete",
      "WebGPU device is missing required resource methods",
    );
  }
  const stateBytes = planQwen35HybridState(
    QWEN35_4B_CONFIG.productContextLength,
  ).totalBytes;
  const weightBytes = qwen35AllocatedWeightBytes(cached.shards);
  const minimumLedgerBytes = stateBytes + weightBytes;
  let ledger: AllocationLedger;
  try {
    ledger = createQwen35GpuLedger(
      minimumLedgerBytes,
      options.gpuLedgerLimitBytes,
    );
  } catch (error) {
    device.destroy();
    throw error;
  }
  const arena = new GpuArena(device, ledger, {
    bufferShardCapBytes: BigInt(profile.bufferShardCapBytes),
  });
  const weightAllocations: GpuAllocation[] = [];
  let hybridState: Qwen35HybridState | null = null;
  let driver: Qwen35ExecutionDriver | null = null;
  try {
    for (const [index, shard] of cached.shards.entries()) {
      weightAllocations.push(await arena.allocate({
        id: `model-shard-${index}`,
        category: "model",
        byteLength: BigInt(shard.byteLength),
        usage: GPU_STORAGE_AND_COPY_DST,
        alignment: 4,
        requiredShardQuantumBytes: 4n,
      }));
    }
    hybridState = await createQwen35HybridState({
      arena,
      capacity: QWEN35_4B_CONFIG.productContextLength,
      clearAllocation: (allocation, resource) =>
        factory.clearStateAllocation(allocation, resource),
    });
    driver = await factory.create({
      profile,
      program,
      arena,
      hybridState,
      weightAllocations: Object.freeze([...weightAllocations]),
      packageDirectory,
    });
    await streamQwen35CachedWeights({
      storage,
      cached,
      allocations: weightAllocations,
      driver,
      factory,
      uploadLaneBytes: profile.uploadLaneBytes,
      signal,
    });
    signal.throwIfAborted();
  } catch (error) {
    try {
      await cleanupQwen35GpuResources({
        driver,
        device,
        hybridState,
        weightAllocations,
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
  let disposed = false;
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
    deviceLost: device.lost,
    async dispose() {
      if (disposed) return;
      disposed = true;
      try {
        await cleanupQwen35GpuResources({
          driver: ownedDriver,
          device,
          hybridState: ownedState,
          weightAllocations,
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
