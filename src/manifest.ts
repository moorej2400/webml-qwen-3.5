export interface ImmutableArtifactIdentity {
  repository: string;
  revision: string;
  file: string;
  /** Canonical unsigned decimal bytes; strings preserve values above 2^53. */
  size: string;
  sha256: string;
}

export interface RuntimeIdentity {
  abi: string;
}

export type PackageKind = "language" | "vision";
export type TensorStorageType = "raw" | "q3-k-112";

export interface TensorLayoutEntry {
  name: string;
  shape: string[];
  ggmlType: number;
  storageType: TensorStorageType;
  shard: number;
  shardOffset: string;
  tensorOffset: string;
  length: string;
  quantization?: {
    blockElements: number;
    blockBytes: number;
  };
}

export interface PackageShard {
  url: string;
  /** Logical offset in the complete package byte stream. */
  offset: string;
  length: string;
  sha256: string;
}

export interface ExcludedTensor {
  name: string;
  reason: string;
}

export interface ModelPackageManifest {
  format: "webml-qwen-package";
  version: 1;
  packageKind: PackageKind;
  source: ImmutableArtifactIdentity;
  runtime: RuntimeIdentity;
  tokenizer: ImmutableArtifactIdentity;
  processor?: ImmutableArtifactIdentity;
  tensorLayout: TensorLayoutEntry[];
  shards: PackageShard[];
  excludedTensors: ExcludedTensor[];
}

const SHA256 = /^[a-f0-9]{64}$/;
const COMMIT_REVISION = /^[a-f0-9]{40}$/;
const UNSIGNED_DECIMAL = /^(?:0|[1-9][0-9]*)$/;
const SAFE_NAME = /^[^\u0000-\u001f\u007f]+$/;

function requireString(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string" || value.length === 0 || !SAFE_NAME.test(value)) {
    throw new Error(`${label} must be a non-empty printable string`);
  }
}

function decimal(value: unknown, label: string, allowZero = true): bigint {
  if (typeof value !== "string" || !UNSIGNED_DECIMAL.test(value)) {
    throw new Error(`${label} must be a canonical unsigned decimal byte count`);
  }
  const parsed = BigInt(value);
  if (!allowZero && parsed === 0n) {
    throw new Error(`${label} must be greater than zero`);
  }
  return parsed;
}

function validateArtifact(
  artifact: ImmutableArtifactIdentity,
  label: string,
): void {
  if (typeof artifact !== "object" || artifact === null) {
    throw new Error(`${label} identity is required`);
  }
  requireString(artifact.repository, `${label} repository`);
  requireString(artifact.file, `${label} file`);
  if (!COMMIT_REVISION.test(artifact.revision)) {
    throw new Error(`${label} must use an immutable 40-character commit revision`);
  }
  decimal(artifact.size, `${label} size`, false);
  if (!SHA256.test(artifact.sha256)) {
    throw new Error(`${label} SHA-256 must contain 64 lowercase hex characters`);
  }
}

function validateShardUrl(value: string): void {
  requireString(value, "shard URL");
  if (
    value.startsWith("/") ||
    value.includes("\\") ||
    value.split("/").includes("..")
  ) {
    throw new Error("shard URL must be a safe relative URL or HTTPS URL");
  }
  if (value.includes(":") && !value.startsWith("https://")) {
    throw new Error("shard URL must use HTTPS");
  }
}

/**
 * Validates the browser trust boundary before a manifest can select remote
 * bytes, tensor ranges, or runtime ABI behavior.
 */
export function validateModelPackageManifest(
  manifest: ModelPackageManifest,
): ModelPackageManifest {
  if (typeof manifest !== "object" || manifest === null) {
    throw new Error("Model package manifest must be an object");
  }
  if (manifest.format !== "webml-qwen-package" || manifest.version !== 1) {
    throw new Error("Unsupported model package manifest format or version");
  }
  if (
    manifest.packageKind !== "language" &&
    manifest.packageKind !== "vision"
  ) {
    throw new Error("Model package kind must be language or vision");
  }

  validateArtifact(manifest.source, "source");
  validateArtifact(manifest.tokenizer, "tokenizer");
  if (manifest.processor !== undefined) {
    validateArtifact(manifest.processor, "processor");
  }
  if (manifest.packageKind === "vision" && manifest.processor === undefined) {
    throw new Error("processor identity is required for a vision package");
  }
  requireString(manifest.runtime?.abi, "runtime ABI");

  if (!Array.isArray(manifest.shards) || manifest.shards.length === 0) {
    throw new Error("Model package requires at least one shard");
  }
  let expectedShardOffset = 0n;
  for (const [index, shard] of manifest.shards.entries()) {
    validateShardUrl(shard.url);
    const offset = decimal(shard.offset, `shard ${index} offset`);
    const length = decimal(shard.length, `shard ${index} length`, false);
    if (offset !== expectedShardOffset) {
      throw new Error(`shard ${index} offset must be contiguous`);
    }
    if (!SHA256.test(shard.sha256)) {
      throw new Error(`shard ${index} SHA-256 is malformed`);
    }
    expectedShardOffset += length;
  }

  if (!Array.isArray(manifest.tensorLayout)) {
    throw new Error("tensorLayout must be an array");
  }
  const segments = new Set<string>();
  for (const tensor of manifest.tensorLayout) {
    requireString(tensor.name, "tensor name");
    if (
      !Array.isArray(tensor.shape) ||
      tensor.shape.length === 0 ||
      tensor.shape.some((dimension) => decimal(dimension, "tensor dimension", false) < 1n)
    ) {
      throw new Error(`tensor ${tensor.name} has an invalid shape`);
    }
    if (!Number.isInteger(tensor.ggmlType) || tensor.ggmlType < 0) {
      throw new Error(`tensor ${tensor.name} has an invalid GGML type`);
    }
    if (tensor.storageType !== "raw" && tensor.storageType !== "q3-k-112") {
      throw new Error(`tensor ${tensor.name} has an unsupported storage type`);
    }
    if (
      !Number.isSafeInteger(tensor.shard) ||
      tensor.shard < 0 ||
      tensor.shard >= manifest.shards.length
    ) {
      throw new Error(`tensor ${tensor.name} refers to an invalid shard`);
    }
    const shardOffset = decimal(
      tensor.shardOffset,
      `tensor ${tensor.name} shardOffset`,
    );
    const tensorOffset = decimal(
      tensor.tensorOffset,
      `tensor ${tensor.name} tensorOffset`,
    );
    const length = decimal(tensor.length, `tensor ${tensor.name} length`, false);
    const shardLength = decimal(
      manifest.shards[tensor.shard]!.length,
      `shard ${tensor.shard} length`,
      false,
    );
    if (shardOffset + length > shardLength) {
      throw new Error(`tensor ${tensor.name} exceeds shard bounds`);
    }

    const segmentKey = `${tensor.name}\u0000${tensorOffset}`;
    if (segments.has(segmentKey)) {
      throw new Error(`duplicate tensor segment: ${tensor.name}@${tensorOffset}`);
    }
    segments.add(segmentKey);

    if (tensor.storageType === "q3-k-112") {
      if (
        tensor.quantization?.blockElements !== 256 ||
        tensor.quantization.blockBytes !== 112 ||
        tensorOffset % 112n !== 0n ||
        length % 112n !== 0n
      ) {
        throw new Error(`tensor ${tensor.name} violates Q3_K block alignment`);
      }
    }
  }

  if (!Array.isArray(manifest.excludedTensors)) {
    throw new Error("excludedTensors must be an array");
  }
  const excludedNames = new Set<string>();
  for (const excluded of manifest.excludedTensors) {
    requireString(excluded.name, "excluded tensor name");
    requireString(excluded.reason, "excluded tensor reason");
    if (excludedNames.has(excluded.name)) {
      throw new Error(`duplicate excluded tensor: ${excluded.name}`);
    }
    excludedNames.add(excluded.name);
  }

  return manifest;
}

function sortJson(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortJson);
  }
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
        .map(([key, child]) => [key, sortJson(child)]),
    );
  }
  return value;
}

export function stringifyManifest(manifest: ModelPackageManifest): string {
  validateModelPackageManifest(manifest);
  return `${JSON.stringify(sortJson(manifest), null, 2)}\n`;
}
