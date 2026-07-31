import {
  GgmlType,
  ggmlTensorByteLength,
  ggmlTypeLayout,
} from "./gguf.js";
import {
  MTP_EXCLUSION_REASON,
  WEBGPU_LANGUAGE_TENSOR_LAYOUTS,
  isMtpTensorName,
  webGpuLanguageTensorLayout,
  type WebGpuTensorStorageType,
} from "./tensor-policy.js";

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
export type TensorStorageType = "raw" | WebGpuTensorStorageType;

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
const MAX_UINT64 = (1n << 64n) - 1n;
const MAX_DECIMAL_DIGITS = 20;
const MAX_MANIFEST_STRING_BYTES = 65_535;
export const MAX_SHARDS = 4_096;
export const MAX_TENSOR_SEGMENTS = 100_000;
export const MAX_EXCLUDED_TENSORS = 100_000;
const textEncoder = new TextEncoder();
const GGML_TYPE_NAMES = new Map<GgmlType, string>([
  [GgmlType.F32, "F32"],
  [GgmlType.Q8_0, "Q8_0"],
  [GgmlType.Q3_K, "Q3_K"],
  [GgmlType.Q4_K, "Q4_K"],
  [GgmlType.Q5_K, "Q5_K"],
  [GgmlType.Q6_K, "Q6_K"],
]);
const TENSOR_STORAGE_TYPES: ReadonlySet<TensorStorageType> = new Set([
  "raw",
  "f32",
  "q8-0-36",
  "q3-k-112",
  "q4-k-144",
  "q5-k-176",
  "q6-k-212",
]);

function requireString(
  value: unknown,
  label: string,
  maxBytes = MAX_MANIFEST_STRING_BYTES,
): asserts value is string {
  if (typeof value !== "string" || value.length === 0 || !SAFE_NAME.test(value)) {
    throw new Error(`${label} must be a non-empty printable string`);
  }
  if (
    value.length > maxBytes ||
    textEncoder.encode(value).byteLength > maxBytes
  ) {
    throw new Error(`${label} exceeds its byte length bound`);
  }
}

function decimal(value: unknown, label: string, allowZero = true): bigint {
  if (typeof value !== "string") {
    throw new Error(`${label} must be a canonical unsigned decimal byte count`);
  }
  // Reject attacker-controlled digit runs before BigInt allocates their value.
  if (value.length > MAX_DECIMAL_DIGITS) {
    throw new Error(`${label} exceeds the decimal digit bound`);
  }
  if (!UNSIGNED_DECIMAL.test(value)) {
    throw new Error(`${label} must be a canonical unsigned decimal byte count`);
  }
  const parsed = BigInt(value);
  if (parsed > MAX_UINT64) {
    throw new Error(`${label} exceeds the unsigned 64-bit bound`);
  }
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
  requireString(artifact.repository, `${label} repository`, 1_024);
  requireString(artifact.file, `${label} file`, 1_024);
  if (!COMMIT_REVISION.test(artifact.revision)) {
    throw new Error(`${label} must use an immutable 40-character commit revision`);
  }
  decimal(artifact.size, `${label} size`, false);
  if (!SHA256.test(artifact.sha256)) {
    throw new Error(`${label} SHA-256 must contain 64 lowercase hex characters`);
  }
}

function validateShardUrl(value: string): void {
  requireString(value, "shard URL", 2_048);
  if (/%(?:2e|2f|5c)/i.test(value)) {
    throw new Error("shard URL contains encoded traversal syntax");
  }
  if (/%(?![a-f0-9]{2})/i.test(value)) {
    throw new Error("shard URL contains malformed percent encoding");
  }
  const rawPath = value.split(/[?#]/, 1)[0]!;
  if (
    rawPath.includes("\\") ||
    rawPath.split("/").some((segment) => segment === "." || segment === "..")
  ) {
    throw new Error("shard URL contains path traversal syntax");
  }

  const hasScheme = /^[a-z][a-z0-9+.-]*:/i.test(value);
  let parsed: URL;
  try {
    // The synthetic base lets WHATWG normalization prove that a relative path
    // stays inside its package directory.
    parsed = hasScheme
      ? new URL(value)
      : new URL(value, "https://package.invalid/package/");
  } catch {
    throw new Error("shard URL must be a valid WHATWG URL");
  }

  if (hasScheme) {
    if (
      parsed.protocol !== "https:" ||
      parsed.username !== "" ||
      parsed.password !== ""
    ) {
      throw new Error("absolute shard URL must use credential-free HTTPS");
    }
  } else if (
    value.startsWith("//") ||
    parsed.origin !== "https://package.invalid" ||
    !parsed.pathname.startsWith("/package/")
  ) {
    throw new Error("relative shard URL must remain inside the package directory");
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
  requireString(manifest.runtime?.abi, "runtime ABI", 256);

  if (!Array.isArray(manifest.shards) || manifest.shards.length === 0) {
    throw new Error("Model package requires at least one shard");
  }
  if (manifest.shards.length > MAX_SHARDS) {
    throw new Error("Model package shard count exceeds the configured bound");
  }
  let expectedShardOffset = 0n;
  for (const [index, shard] of manifest.shards.entries()) {
    validateShardUrl(shard.url);
    const offset = decimal(shard.offset, `shard ${index} offset`);
    const length = decimal(shard.length, `shard ${index} length`, false);
    if (offset < expectedShardOffset) {
      throw new Error(`shard ${index} has an overlapping shard mapping`);
    }
    if (offset > expectedShardOffset) {
      throw new Error(`shard ${index} offset must be contiguous`);
    }
    if (!SHA256.test(shard.sha256)) {
      throw new Error(`shard ${index} SHA-256 is malformed`);
    }
    expectedShardOffset += length;
  }

  if (
    !Array.isArray(manifest.tensorLayout) ||
    manifest.tensorLayout.length === 0
  ) {
    throw new Error("tensorLayout must be a non-empty array");
  }
  if (manifest.tensorLayout.length > MAX_TENSOR_SEGMENTS) {
    throw new Error("tensorLayout count exceeds the configured bound");
  }
  const segments = new Set<string>();
  const tensorGroups = new Map<
    string,
    {
      attributes: string;
      expectedLength: bigint;
      segments: Array<{ offset: bigint; length: bigint }>;
    }
  >();
  const shardMappings: Array<Array<{ offset: bigint; end: bigint }>> =
    manifest.shards.map(() => []);
  for (const tensor of manifest.tensorLayout) {
    requireString(tensor.name, "tensor name", 64);
    if (
      !Array.isArray(tensor.shape) ||
      tensor.shape.length < 1 ||
      tensor.shape.length > 4
    ) {
      throw new Error(`tensor rank for ${tensor.name} must be from 1 through 4`);
    }
    const shape = tensor.shape.map((dimension) =>
      decimal(dimension, "tensor dimension", false),
    );
    if (!Number.isInteger(tensor.ggmlType) || tensor.ggmlType < 0) {
      throw new Error(`tensor ${tensor.name} has an invalid GGML type`);
    }
    if (!TENSOR_STORAGE_TYPES.has(tensor.storageType)) {
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

    let expectedLength: bigint;
    if (tensor.storageType !== "raw") {
      if (shardOffset % 4n !== 0n) {
        throw new Error(
          `tensor ${tensor.name} shardOffset must be u32 aligned`,
        );
      }
      const policy = webGpuLanguageTensorLayout(
        tensor.ggmlType as GgmlType,
      );
      if (policy?.storageType !== tensor.storageType) {
        const expectedType = WEBGPU_LANGUAGE_TENSOR_LAYOUTS.find(
          (candidate) => candidate.storageType === tensor.storageType,
        )?.ggmlType;
        throw new Error(
          `tensor ${tensor.name} uses ${tensor.storageType} storage without GGML ${
            expectedType === undefined
              ? "supported"
              : GGML_TYPE_NAMES.get(expectedType)
          } type`,
        );
      }
      const outputBlockBytes = BigInt(policy.outputBlockBytes);
      if (tensorOffset % outputBlockBytes !== 0n || length % outputBlockBytes !== 0n) {
        throw new Error(
          `tensor ${tensor.name} violates ${tensor.storageType} block alignment`,
        );
      }
      if (policy.blockElements === 1) {
        if (tensor.quantization !== undefined) {
          throw new Error(`F32 tensor ${tensor.name} cannot declare quantization`);
        }
      } else if (
        tensor.quantization?.blockElements !== policy.blockElements ||
        tensor.quantization.blockBytes !== policy.outputBlockBytes
      ) {
        throw new Error(
          `tensor ${tensor.name} has invalid ${tensor.storageType} quantization metadata`,
        );
      }
      const sourceBytes = ggmlTensorByteLength(
        tensor.ggmlType as GgmlType,
        shape,
      );
      expectedLength =
        (sourceBytes / BigInt(policy.sourceBlockBytes)) * outputBlockBytes;
    } else {
      if (
        webGpuLanguageTensorLayout(tensor.ggmlType as GgmlType) !== undefined
      ) {
        throw new Error(
          `tensor ${tensor.name} must use its explicit WebGPU storage layout`,
        );
      }
      if (tensor.quantization !== undefined) {
        throw new Error(`raw tensor ${tensor.name} cannot declare quantization`);
      }
      const nativeLayout = ggmlTypeLayout(tensor.ggmlType as GgmlType);
      if (
        tensorOffset % nativeLayout.blockBytes !== 0n ||
        length % nativeLayout.blockBytes !== 0n
      ) {
        throw new Error(
          `tensor ${tensor.name} violates native storage block alignment`,
        );
      }
      expectedLength = ggmlTensorByteLength(
        tensor.ggmlType as GgmlType,
        shape,
      );
    }

    const attributes = JSON.stringify({
      shape: tensor.shape,
      ggmlType: tensor.ggmlType,
      storageType: tensor.storageType,
      quantization:
        tensor.quantization === undefined
          ? null
          : [
              tensor.quantization.blockElements,
              tensor.quantization.blockBytes,
            ],
    });
    const group = tensorGroups.get(tensor.name);
    if (group === undefined) {
      tensorGroups.set(tensor.name, {
        attributes,
        expectedLength,
        segments: [{ offset: tensorOffset, length }],
      });
    } else {
      if (
        group.attributes !== attributes ||
        group.expectedLength !== expectedLength
      ) {
        throw new Error(
          `tensor ${tensor.name} must use consistent attributes across segments`,
        );
      }
      group.segments.push({ offset: tensorOffset, length });
    }
    shardMappings[tensor.shard]!.push({
      offset: shardOffset,
      end: shardOffset + length,
    });
  }

  // Logical tensor coverage and physical shard mappings are independent:
  // validate both so a complete tensor cannot alias another tensor's bytes.
  for (const [name, group] of tensorGroups) {
    group.segments.sort((left, right) =>
      left.offset < right.offset ? -1 : left.offset > right.offset ? 1 : 0,
    );
    let covered = 0n;
    for (const segment of group.segments) {
      if (segment.offset !== covered) {
        throw new Error(`tensor ${name} segments must be contiguous`);
      }
      covered += segment.length;
    }
    if (covered !== group.expectedLength) {
      throw new Error(
        `tensor ${name} segments must cover the exact shape-derived length`,
      );
    }
  }

  for (const mappings of shardMappings) {
    mappings.sort((left, right) =>
      left.offset < right.offset ? -1 : left.offset > right.offset ? 1 : 0,
    );
    for (let index = 1; index < mappings.length; index += 1) {
      if (mappings[index]!.offset < mappings[index - 1]!.end) {
        throw new Error("overlapping tensor shard mapping");
      }
    }
  }

  if (!Array.isArray(manifest.excludedTensors)) {
    throw new Error("excludedTensors must be an array");
  }
  if (manifest.excludedTensors.length > MAX_EXCLUDED_TENSORS) {
    throw new Error("excludedTensors count exceeds the configured bound");
  }
  const excludedNames = new Set<string>();
  for (const excluded of manifest.excludedTensors) {
    requireString(excluded.name, "excluded tensor name", 64);
    requireString(excluded.reason, "excluded tensor reason", 128);
    if (!isMtpTensorName(excluded.name)) {
      throw new Error(
        `excluded tensor ${excluded.name} does not match the MTP version-1 name policy`,
      );
    }
    if (excluded.reason !== MTP_EXCLUSION_REASON) {
      throw new Error(
        `excluded tensor ${excluded.name} reason is invalid for manifest version 1`,
      );
    }
    if (tensorGroups.has(excluded.name)) {
      throw new Error(
        "included and excluded tensor names must remain disjoint",
      );
    }
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
