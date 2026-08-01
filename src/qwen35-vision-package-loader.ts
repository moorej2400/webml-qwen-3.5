import { diagnosticError } from "./diagnostics.js";
import {
  HttpRangeReader,
  browserRangeFetch,
  type ImmutableRangeSource,
  type RangeChunkMetadata,
  type RangeFetch,
} from "./http-range-reader.js";
import { IncrementalSha256 } from "./incremental-sha256.js";
import {
  validateModelPackageManifest,
  type ImmutableArtifactIdentity,
  type ModelPackageManifest,
  type VisionProcessorSettings,
} from "./manifest.js";

const MAX_METADATA_BYTES = 1024 * 1024;
const VISION_LAYER_COUNT = 24;
const VISION_SHARD_COUNT = VISION_LAYER_COUNT + 2;
const SHA256 = /^[a-f0-9]{64}$/u;
const VISION_SHARD_NAME = /^vision-([0-9]{5})\.bin$/u;
const IMMUTABLE_HUGGING_FACE_BASE = /^\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+\/resolve\/([a-f0-9]{40})\/$/u;

const PINNED_VISION_SOURCE: Readonly<ImmutableArtifactIdentity> = Object.freeze({
  repository: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF",
  revision: "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
  file: "mmproj-Qwen_Qwen3.5-4B-bf16.gguf",
  size: "675569216",
  sha256: "463f39bd1c291c1186c319a8c90ff8640aafa678b14cbee2232d695113dfbb66",
});
const PINNED_TOKENIZER_SOURCE: Readonly<ImmutableArtifactIdentity> = Object.freeze({
  repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
  revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
  file: "tokenizer.json",
  size: "12807982",
  sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
});
const PINNED_PROCESSOR_SOURCE: Readonly<ImmutableArtifactIdentity> = Object.freeze({
  repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
  revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
  file: "preprocessor_config.json",
  size: "390",
  sha256: "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
});
const PINNED_PROCESSOR_SETTINGS: Readonly<VisionProcessorSettings> = Object.freeze({
  processorClass: "Qwen3VLProcessor",
  imageProcessorType: "Qwen2VLImageProcessorFast",
  patchSize: 16,
  temporalPatchSize: 2,
  mergeSize: 2,
  shortestEdge: 65_536,
  longestEdge: 16_777_216,
  imageMean: Object.freeze([0.5, 0.5, 0.5]),
  imageStd: Object.freeze([0.5, 0.5, 0.5]),
});

export interface Qwen35VisionPackagePins {
  readonly packageBaseUrl: string;
  readonly expectedPackageBaseUrl: string;
  readonly expectedManifestSha256: string;
  readonly expectedLayerIndexSha256: string;
}

export interface Qwen35VisionPackageLoadOptions {
  readonly manifestBytes: Uint8Array;
  readonly layerIndexBytes: Uint8Array;
  readonly pins: Qwen35VisionPackagePins;
  readonly rangeFetch?: RangeFetch;
  /** Tests may reduce this; production keeps the existing conservative ranges. */
  readonly rangeStrategies?: readonly number[];
}

export interface Qwen35VisionLayerShard {
  readonly index: number;
  readonly byteLength: number;
  readonly sha256: string;
}

export interface Qwen35VisionLayer {
  readonly index: number | "bootstrap";
  readonly shards: readonly Qwen35VisionLayerShard[];
}

/**
 * Stages one layer without publishing it. `abort` must destroy or roll back
 * every staged resource; only `commit` may make verified weights observable.
 */
export interface Qwen35VisionLayerSink {
  write(chunk: Uint8Array, metadata: RangeChunkMetadata): Promise<void>;
  commit(): Promise<void>;
  abort(): Promise<void>;
}

export interface Qwen35VisionPackage {
  readonly manifest: ModelPackageManifest;
  readonly manifestSha256: string;
  readonly layerIndexSha256: string;
  readonly bootstrap: Qwen35VisionLayer;
  readonly layers: readonly Qwen35VisionLayer[];
  /** Streams one layer only; it never constructs a model-sized byte array. */
  streamLayer(
    layer: number | "bootstrap",
    sink: Qwen35VisionLayerSink,
    signal?: AbortSignal,
  ): Promise<void>;
}

/**
 * A package whose caller-supplied immutable metadata pins and layer hashes
 * passed validation. This is not a claim that it is the fixed public release.
 */
export type Qwen35IntegrityValidatedVisionPackage = Qwen35VisionPackage;

/** The exact immutable metadata identities accepted as the public release. */
export const QWEN35_PRODUCTION_VISION_PACKAGE_PINS = Object.freeze({
  packageBaseUrl: "https://huggingface.co/moorejared97/Qwen3.5-4B-Q3-K-L-WebGPU/resolve/c03d9b750d8ff5ba7e48effcdc2ff16db2ad7153/",
  expectedManifestSha256: "7323d564ea10a1305db61a4af9c226eef7288202e8a97b5b33ec9be136dbf9a5",
  expectedLayerIndexSha256: "e71fa47629be790db5a9afb247901cea411a95d27e8dbad4029270a7cd61ded2",
});

/** An integrity-validated package that also matches the fixed public release pins. */
declare const PRODUCTION_RELEASE_PACKAGE: unique symbol;
export type Qwen35ProductionVisionPackage = Qwen35IntegrityValidatedVisionPackage & Readonly<{
  [PRODUCTION_RELEASE_PACKAGE]: true;
}>;

// These brands are intentionally module-private. A matching public shape, or
// caller-selected metadata pins, must not claim fixed-release production trust.
const INTEGRITY_VALIDATED_PACKAGES = new WeakSet<Qwen35IntegrityValidatedVisionPackage>();
const PRODUCTION_TRUSTED_PACKAGES = new WeakSet<Qwen35IntegrityValidatedVisionPackage>();

interface InternalLayer extends Qwen35VisionLayer {
  readonly sources: readonly ImmutableRangeSource[];
}

function fail(code: string, message: string): never {
  throw diagnosticError(code, message);
}

/** Rejects structural lookalikes before low-level vision execution reads package metadata. */
export function assertIntegrityValidatedQwen35VisionPackage(
  value: unknown,
): Qwen35IntegrityValidatedVisionPackage {
  if (
    typeof value !== "object" ||
    value === null ||
    !INTEGRITY_VALIDATED_PACKAGES.has(value as Qwen35IntegrityValidatedVisionPackage)
  ) {
    fail("vision-package-integrity-unvalidated", "Vision package was not integrity-validated by the loader");
  }
  return value as Qwen35IntegrityValidatedVisionPackage;
}

/** Requires the exact release pins at the production bootstrap or session boundary. */
export function assertProductionTrustedQwen35VisionPackage(
  value: unknown,
): Qwen35ProductionVisionPackage {
  const package_ = assertIntegrityValidatedQwen35VisionPackage(value);
  if (!PRODUCTION_TRUSTED_PACKAGES.has(package_)) {
    fail("vision-package-production-untrusted", "Vision package is not the fixed production release");
  }
  return package_ as Qwen35ProductionVisionPackage;
}

function sha256(bytes: Uint8Array): string {
  return new IncrementalSha256().update(bytes).digestHex();
}

function sameIdentity(
  actual: ImmutableArtifactIdentity | undefined,
  expected: Readonly<ImmutableArtifactIdentity>,
): boolean {
  return actual !== undefined &&
    actual.repository === expected.repository &&
    actual.revision === expected.revision &&
    actual.file === expected.file &&
    actual.size === expected.size &&
    actual.sha256 === expected.sha256;
}

function sameSettings(actual: VisionProcessorSettings | undefined): boolean {
  return actual !== undefined &&
    actual.processorClass === PINNED_PROCESSOR_SETTINGS.processorClass &&
    actual.imageProcessorType === PINNED_PROCESSOR_SETTINGS.imageProcessorType &&
    actual.patchSize === PINNED_PROCESSOR_SETTINGS.patchSize &&
    actual.temporalPatchSize === PINNED_PROCESSOR_SETTINGS.temporalPatchSize &&
    actual.mergeSize === PINNED_PROCESSOR_SETTINGS.mergeSize &&
    actual.shortestEdge === PINNED_PROCESSOR_SETTINGS.shortestEdge &&
    actual.longestEdge === PINNED_PROCESSOR_SETTINGS.longestEdge &&
    actual.imageMean.length === 3 && actual.imageStd.length === 3 &&
    actual.imageMean.every((value, index) => value === PINNED_PROCESSOR_SETTINGS.imageMean[index]) &&
    actual.imageStd.every((value, index) => value === PINNED_PROCESSOR_SETTINGS.imageStd[index]);
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

function validateMetadataBytes(bytes: Uint8Array, code: string): void {
  const shared = typeof SharedArrayBuffer !== "undefined" &&
    bytes instanceof Uint8Array &&
    bytes.buffer instanceof SharedArrayBuffer;
  if (
    !(bytes instanceof Uint8Array) ||
    shared ||
    bytes.byteLength < 1 ||
    bytes.byteLength > MAX_METADATA_BYTES
  ) {
    fail(code, "Vision package metadata is invalid");
  }
}

function snapshotMetadata(bytes: Uint8Array, code: string): Uint8Array {
  validateMetadataBytes(bytes, code);
  // A private ArrayBuffer prevents later caller mutation from changing the
  // bytes between SHA verification and manifest or layer-index parsing.
  return new Uint8Array(bytes);
}

function parseMetadata(bytes: Uint8Array, code: string): unknown {
  validateMetadataBytes(bytes, code);
  try {
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    fail(code, "Vision package metadata is invalid");
  }
}

function requirePinnedHash(actual: string, expected: unknown, code: string): void {
  if (typeof expected !== "string" || !SHA256.test(expected) || actual !== expected) {
    fail(code, "Vision package metadata does not match the application pin");
  }
}

function packageBase(pins: Qwen35VisionPackagePins): URL {
  if (pins.packageBaseUrl !== pins.expectedPackageBaseUrl) {
    fail("vision-package-base-mismatch", "Vision package base URL does not match the application pin");
  }
  let base: URL;
  try {
    base = new URL(pins.expectedPackageBaseUrl);
  } catch {
    fail("vision-package-base-mismatch", "Vision package base URL does not match the application pin");
  }
  if (
    base.protocol !== "https:" ||
    base.hostname !== "huggingface.co" ||
    base.username !== "" ||
    base.password !== "" ||
    base.search !== "" ||
    base.hash !== "" ||
    !IMMUTABLE_HUGGING_FACE_BASE.test(base.pathname) ||
    base.href !== pins.expectedPackageBaseUrl
  ) {
    fail("vision-package-base-mismatch", "Vision package base URL does not match the application pin");
  }
  return base;
}

function byteLength(value: string): number {
  const parsed = BigInt(value);
  if (parsed > BigInt(Number.MAX_SAFE_INTEGER)) {
    fail("vision-package-shard-invalid", "Vision package shard metadata is invalid");
  }
  return Number(parsed);
}

function validateManifest(bytes: Uint8Array): ModelPackageManifest {
  const parsed = parseMetadata(bytes, "vision-package-manifest-invalid") as ModelPackageManifest;
  let manifest: ModelPackageManifest;
  try {
    manifest = validateModelPackageManifest(parsed);
  } catch {
    fail("vision-package-manifest-invalid", "Vision package manifest is invalid");
  }
  if (
    manifest.packageKind !== "vision" ||
    manifest.runtime.abi !== "qwen35-webgpu-vision-v1" ||
    manifest.shards.length !== VISION_SHARD_COUNT ||
    !sameIdentity(manifest.source, PINNED_VISION_SOURCE) ||
    !sameIdentity(manifest.tokenizer, PINNED_TOKENIZER_SOURCE) ||
    !sameIdentity(manifest.processor, PINNED_PROCESSOR_SOURCE) ||
    !sameSettings(manifest.processorSettings)
  ) {
    fail("vision-package-identity-mismatch", "Vision package does not match pinned Qwen3.5 sources");
  }
  return deepFreeze(manifest);
}

function validateLayerIndex(
  bytes: Uint8Array,
  manifest: ModelPackageManifest,
  base: URL,
): readonly InternalLayer[] {
  const parsed = parseMetadata(bytes, "vision-layer-index-invalid");
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    fail("vision-layer-index-invalid", "Vision layer index is invalid");
  }
  const value = parsed as { format?: unknown; version?: unknown; groups?: unknown };
  if (
    value.format !== "webml-qwen-vision-layer-index" ||
    value.version !== 1 ||
    !Array.isArray(value.groups) ||
    value.groups.length !== VISION_LAYER_COUNT + 1
  ) {
    fail("vision-layer-index-invalid", "Vision layer index is invalid");
  }
  const expectedGroups: Array<{ layer: string; shards: readonly number[] }> = [
    { layer: "bootstrap", shards: [0, 1] },
    ...Array.from({ length: VISION_LAYER_COUNT }, (_, index) => ({
      layer: String(index),
      shards: [index + 2],
    })),
  ];
  return Object.freeze(value.groups.map((candidate, groupIndex) => {
    const expected = expectedGroups[groupIndex]!;
    if (
      typeof candidate !== "object" || candidate === null || Array.isArray(candidate) ||
      Object.keys(candidate).length !== 2 ||
      !Object.hasOwn(candidate, "layer") || !Object.hasOwn(candidate, "shards")
    ) {
      fail("vision-layer-index-invalid", "Vision layer index is invalid");
    }
    const group = candidate as { layer: unknown; shards: unknown };
    if (
      group.layer !== expected.layer ||
      !Array.isArray(group.shards) ||
      group.shards.length !== expected.shards.length ||
      group.shards.some((shard, index) => shard !== expected.shards[index])
    ) {
      fail("vision-layer-index-invalid", "Vision layer index is invalid");
    }
    const shards = Object.freeze(group.shards.map((index) => {
      const shard = manifest.shards[index as number];
      if (shard === undefined || shard.url !== `vision-${String(index).padStart(5, "0")}.bin`) {
        fail("vision-package-shard-invalid", "Vision package shard metadata is invalid");
      }
      const match = VISION_SHARD_NAME.exec(shard.url);
      const length = byteLength(shard.length);
      if (match === null || Number(match[1]) !== index || length < 1) {
        fail("vision-package-shard-invalid", "Vision package shard metadata is invalid");
      }
      const locator = new URL(shard.url, base);
      if (locator.origin !== base.origin || locator.pathname !== `${base.pathname}${shard.url}` || locator.search !== "" || locator.hash !== "") {
        fail("vision-package-shard-invalid", "Vision package shard metadata is invalid");
      }
      return Object.freeze({
        index,
        byteLength: length,
        sha256: shard.sha256,
        source: Object.freeze({ locator: locator.href, byteLength: length, immutableUrl: true }),
      });
    }));
    return Object.freeze({
      index: expected.layer === "bootstrap" ? "bootstrap" : Number(expected.layer),
      shards: Object.freeze(shards.map(({ index, byteLength: length, sha256: hash }) =>
        Object.freeze({ index, byteLength: length, sha256: hash }),
      )),
      sources: Object.freeze(shards.map((shard) => shard.source)),
    }) as InternalLayer;
  }));
}

/**
 * Authenticates metadata first, then exposes one bounded, hash-checked layer
 * stream at a time. This is the weight boundary for the future VisionEncoder.
 */
export function createIntegrityValidatedQwen35VisionPackage(
  options: Qwen35VisionPackageLoadOptions,
): Qwen35IntegrityValidatedVisionPackage {
  const manifestBytes = snapshotMetadata(
    options.manifestBytes,
    "vision-package-manifest-invalid",
  );
  const layerIndexBytes = snapshotMetadata(
    options.layerIndexBytes,
    "vision-layer-index-invalid",
  );
  const base = packageBase(options.pins);
  const manifestSha256 = sha256(manifestBytes);
  requirePinnedHash(manifestSha256, options.pins.expectedManifestSha256, "vision-package-manifest-hash-mismatch");
  const layerIndexSha256 = sha256(layerIndexBytes);
  requirePinnedHash(layerIndexSha256, options.pins.expectedLayerIndexSha256, "vision-layer-index-hash-mismatch");
  const manifest = validateManifest(manifestBytes);
  const groups = validateLayerIndex(layerIndexBytes, manifest, base);
  const bootstrap = groups[0]!;
  const layers = Object.freeze(groups.slice(1));
  const rangeReader = new HttpRangeReader(
    options.rangeFetch ?? browserRangeFetch(),
    options.rangeStrategies === undefined ? {} : { rangeStrategies: options.rangeStrategies },
  );
  let streaming = false;

  const authenticatedPackage = Object.freeze({
    manifest,
    manifestSha256,
    layerIndexSha256,
    bootstrap,
    layers,
    async streamLayer(
      layer: number | "bootstrap",
      sink: Qwen35VisionLayerSink,
      signal: AbortSignal = new AbortController().signal,
    ): Promise<void> {
      if (streaming) {
        fail("vision-layer-stream-busy", "Vision package already has an active layer stream");
      }
      const selected = layer === "bootstrap" ? bootstrap : layers[layer];
      if (
        (!Number.isSafeInteger(layer) && layer !== "bootstrap") ||
        selected === undefined ||
        typeof sink !== "object" || sink === null ||
        typeof sink.write !== "function" ||
        typeof sink.commit !== "function" ||
        typeof sink.abort !== "function"
      ) {
        fail("vision-layer-stream-invalid", "Vision layer stream request is invalid");
      }
      streaming = true;
      try {
        for (let shardIndex = 0; shardIndex < selected.shards.length; shardIndex += 1) {
          const shard = selected.shards[shardIndex]!;
          const source = selected.sources[shardIndex]!;
          const hash = new IncrementalSha256();
          await rangeReader.stream(source, async (chunk, metadata) => {
            hash.update(chunk);
            await sink.write(chunk, metadata);
          }, signal);
          if (hash.digestHex() !== shard.sha256) {
            fail("vision-layer-shard-hash-mismatch", "Vision layer shard failed integrity verification");
          }
        }
        // Bootstrap spans two shards, so publication waits until both hashes
        // pass rather than committing a partially authenticated layer.
        // The final range can be the final source chunk, so cancellation must
        // be checked again after its hash before commit publishes staged bytes.
        signal.throwIfAborted();
        await sink.commit();
      } catch (error) {
        try {
          await sink.abort();
        } catch {
          // The hash, transport, cancellation, or staging failure remains the
          // primary diagnostic even when best-effort rollback also fails.
        }
        throw error;
      } finally {
        streaming = false;
      }
    },
  });
  INTEGRITY_VALIDATED_PACKAGES.add(authenticatedPackage);
  return authenticatedPackage;
}

/**
 * Creates the only package type the production bootstrap may return.
 * Callers cannot supply alternate release pins through this entrypoint.
 */
export function createProductionQwen35VisionPackage(
  options: Omit<Qwen35VisionPackageLoadOptions, "pins">,
): Qwen35ProductionVisionPackage {
  const package_ = createIntegrityValidatedQwen35VisionPackage({
    ...options,
    pins: {
      packageBaseUrl: QWEN35_PRODUCTION_VISION_PACKAGE_PINS.packageBaseUrl,
      expectedPackageBaseUrl: QWEN35_PRODUCTION_VISION_PACKAGE_PINS.packageBaseUrl,
      expectedManifestSha256: QWEN35_PRODUCTION_VISION_PACKAGE_PINS.expectedManifestSha256,
      expectedLayerIndexSha256: QWEN35_PRODUCTION_VISION_PACKAGE_PINS.expectedLayerIndexSha256,
    },
  });
  PRODUCTION_TRUSTED_PACKAGES.add(package_);
  return package_ as Qwen35ProductionVisionPackage;
}
