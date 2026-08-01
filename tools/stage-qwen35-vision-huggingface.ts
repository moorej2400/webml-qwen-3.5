import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  lstat,
  mkdir,
  open,
  readdir,
  realpath,
  writeFile,
  type FileHandle,
} from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  stringifyManifest,
  validateModelPackageManifest,
  type ImmutableArtifactIdentity,
  type ModelPackageManifest,
  type VisionProcessorSettings,
} from "../src/manifest.js";
import { PINNED_QWEN35_VISION_CONVERTER_SOURCE } from "./convert-qwen35-vision.js";

const COPY_CHUNK_BYTES = 8 * 1024 * 1024;
const MAX_METADATA_BYTES = COPY_CHUNK_BYTES;
const VISION_RELEASE_LAYER_COUNT = 24;
const VISION_RELEASE_SHARD_COUNT = VISION_RELEASE_LAYER_COUNT + 2;
const VISION_SHARD_NAME = /^vision-([0-9]{5})\.bin$/u;

export interface Qwen35VisionHuggingFaceStagingArguments {
  readonly packageDirectory: string;
  readonly outputDirectory: string;
}

export interface Qwen35VisionHuggingFaceStageContract {
  readonly source: ImmutableArtifactIdentity;
  readonly tokenizer: ImmutableArtifactIdentity;
  readonly processor: ImmutableArtifactIdentity;
  readonly processorSettings: VisionProcessorSettings;
  readonly shardCount: number;
}

export interface Qwen35VisionHuggingFaceStageReport {
  readonly format: "webml-qwen-vision-huggingface-stage-report";
  readonly version: 1;
  readonly files: readonly string[];
  readonly copiedShardCount: number;
}

/** Test-only seam for deterministic metadata mutation checks; it never receives a filesystem path. */
export interface Qwen35VisionHuggingFaceStagingTestHooks {
  readonly afterMetadataStat?: (metadata: Readonly<{ label: string; size: number }>) => Promise<void> | void;
}

export class Qwen35VisionHuggingFaceStageError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "Qwen35VisionHuggingFaceStageError";
  }
}

function fail(code: string, message: string): never {
  throw new Qwen35VisionHuggingFaceStageError(code, message);
}

export const PINNED_QWEN35_VISION_HUGGINGFACE_STAGE_CONTRACT:
Readonly<Qwen35VisionHuggingFaceStageContract> = Object.freeze({
  source: PINNED_QWEN35_VISION_CONVERTER_SOURCE.source,
  tokenizer: PINNED_QWEN35_VISION_CONVERTER_SOURCE.tokenizer,
  processor: PINNED_QWEN35_VISION_CONVERTER_SOURCE.processor,
  processorSettings: PINNED_QWEN35_VISION_CONVERTER_SOURCE.processorSettings,
  shardCount: 26,
});

export function parseQwen35VisionHuggingFaceStagingArguments(
  arguments_: readonly string[],
): Qwen35VisionHuggingFaceStagingArguments {
  const values = new Map<string, string>();
  for (let index = 0; index < arguments_.length; index += 2) {
    const name = arguments_[index];
    const value = arguments_[index + 1];
    if (
      (name !== "--package" && name !== "--output") ||
      value === undefined || value.length === 0 || value.startsWith("--") ||
      values.has(name)
    ) {
      fail("vision-hf-stage-arguments-invalid", "Vision Hugging Face staging received an unknown argument");
    }
    values.set(name, value);
  }
  const packageDirectory = values.get("--package");
  const outputDirectory = values.get("--output");
  if (packageDirectory === undefined || outputDirectory === undefined) {
    fail("vision-hf-stage-arguments-invalid", "Vision Hugging Face staging requires --package and --output");
  }
  return Object.freeze({ packageDirectory, outputDirectory });
}

async function exists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    fail("vision-hf-stage-path-check-failed", "Vision Hugging Face staging path could not be checked");
  }
}

async function regularFile(path: string, label: string): Promise<string> {
  try {
    const facts = await lstat(path);
    if (!facts.isFile() || facts.isSymbolicLink()) {
      fail("vision-hf-stage-input-invalid", `${label} must be a regular file`);
    }
    return await realpath(path);
  } catch (error) {
    if (error instanceof Qwen35VisionHuggingFaceStageError) throw error;
    fail("vision-hf-stage-input-invalid", `${label} must be a regular file`);
  }
}

async function outsideGitOutput(output: string): Promise<string> {
  const requested = resolve(output);
  if (basename(requested) === ".git" || basename(requested).length === 0) {
    fail("vision-hf-stage-output-invalid", "Vision Hugging Face staging output is invalid");
  }
  if (await exists(requested)) {
    fail("vision-hf-stage-output-exists", "Vision Hugging Face staging output already exists");
  }
  let parent: string;
  try {
    parent = await realpath(dirname(requested));
  } catch {
    fail("vision-hf-stage-output-parent-invalid", "Vision Hugging Face staging output parent must exist");
  }
  for (let current = parent; ; current = dirname(current)) {
    if (await exists(join(current, ".git"))) {
      fail("vision-hf-stage-output-inside-git", "Vision Hugging Face staging output must be outside Git");
    }
    if (dirname(current) === current) break;
  }
  return join(parent, basename(requested));
}

async function boundedMetadataFile(path: string, label: string): Promise<string> {
  const resolved = await regularFile(path, label);
  const facts = await lstat(resolved);
  if (facts.size < 1 || facts.size > MAX_METADATA_BYTES) {
    fail("vision-hf-stage-input-invalid", `${label} size is invalid`);
  }
  return resolved;
}

async function boundedRead(
  path: string,
  label: string,
  testHooks: Qwen35VisionHuggingFaceStagingTestHooks,
): Promise<Uint8Array> {
  let handle: FileHandle | undefined;
  try {
    // O_NOFOLLOW keeps the opened descriptor, rather than a pre-open path check,
    // as the file authority for public package metadata.
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const facts = await handle.stat();
    if (!facts.isFile() || facts.size < 1 || facts.size > MAX_METADATA_BYTES) {
      fail("vision-hf-stage-input-invalid", `${label} size is invalid`);
    }
    await testHooks.afterMetadataStat?.({ label, size: facts.size });
    const bytes = Buffer.allocUnsafe(facts.size);
    for (let position = 0; position < bytes.byteLength;) {
      const read = await handle.read(bytes, position, bytes.byteLength - position, position);
      if (read.bytesRead === 0) {
        fail("vision-hf-stage-input-mutated", "Converted vision metadata changed while it was read");
      }
      position += read.bytesRead;
    }
    const eofProbe = Buffer.allocUnsafe(1);
    if ((await handle.read(eofProbe, 0, eofProbe.byteLength, bytes.byteLength)).bytesRead !== 0) {
      fail("vision-hf-stage-input-mutated", "Converted vision metadata changed while it was read");
    }
    return bytes;
  } catch (error) {
    if (error instanceof Qwen35VisionHuggingFaceStageError) throw error;
    return fail("vision-hf-stage-input-invalid", `${label} could not be read`);
  } finally {
    await handle?.close().catch(() => undefined);
  }
}

async function sha256File(path: string): Promise<string> {
  const handle = await open(path, "r");
  const buffer = Buffer.allocUnsafe(COPY_CHUNK_BYTES);
  const hash = createHash("sha256");
  try {
    for (let position = 0; ; ) {
      const result = await handle.read(buffer, 0, buffer.byteLength, position);
      if (result.bytesRead === 0) return hash.digest("hex");
      hash.update(buffer.subarray(0, result.bytesRead));
      position += result.bytesRead;
    }
  } finally {
    await handle.close();
  }
}

async function copyAndHash(
  source: string,
  destination: string,
): Promise<{ readonly size: bigint; readonly sha256: string }> {
  const input = await open(source, "r");
  let output: FileHandle | undefined;
  const buffer = Buffer.allocUnsafe(COPY_CHUNK_BYTES);
  const hash = createHash("sha256");
  let copied = 0n;
  try {
    output = await open(destination, "wx");
    for (let position = 0; ; ) {
      const read = await input.read(buffer, 0, buffer.byteLength, position);
      if (read.bytesRead === 0) break;
      const chunk = buffer.subarray(0, read.bytesRead);
      hash.update(chunk);
      let written = 0;
      while (written < chunk.byteLength) {
        const result = await output.write(chunk, written, chunk.byteLength - written, Number(copied) + written);
        if (result.bytesWritten === 0) {
          fail("vision-hf-stage-copy-failed", "Vision Hugging Face shard copy failed");
        }
        written += result.bytesWritten;
      }
      copied += BigInt(chunk.byteLength);
      position += read.bytesRead;
    }
    return { size: copied, sha256: hash.digest("hex") };
  } catch (error) {
    if (error instanceof Qwen35VisionHuggingFaceStageError) throw error;
    return fail("vision-hf-stage-copy-failed", "Vision Hugging Face shard copy failed");
  } finally {
    await output?.close().catch(() => undefined);
    await input.close().catch(() => undefined);
  }
}

function parseChecksums(input: string): ReadonlyMap<string, string> {
  const result = new Map<string, string>();
  for (const line of input.trimEnd().split("\n")) {
    const match = /^([a-f0-9]{64})  ([A-Za-z0-9._/-]+)$/u.exec(line);
    if (
      match === null || match[2]!.startsWith("/") ||
      match[2]!.split("/").some((part) => part === "" || part === "." || part === "..") ||
      result.has(match[2]!)
    ) {
      fail("vision-hf-stage-checksums-invalid", "Converted vision package checksums are invalid");
    }
    result.set(match[2]!, match[1]!);
  }
  return result;
}

function sameIdentity(actual: ImmutableArtifactIdentity, expected: ImmutableArtifactIdentity): boolean {
  return actual.repository === expected.repository && actual.revision === expected.revision &&
    actual.file === expected.file && actual.size === expected.size && actual.sha256 === expected.sha256;
}

function sameSettings(actual: VisionProcessorSettings, expected: VisionProcessorSettings): boolean {
  return actual.processorClass === expected.processorClass &&
    actual.imageProcessorType === expected.imageProcessorType &&
    actual.patchSize === expected.patchSize &&
    actual.temporalPatchSize === expected.temporalPatchSize &&
    actual.mergeSize === expected.mergeSize &&
    actual.shortestEdge === expected.shortestEdge &&
    actual.longestEdge === expected.longestEdge &&
    actual.imageMean.length === expected.imageMean.length &&
    actual.imageStd.length === expected.imageStd.length &&
    actual.imageMean.every((value, index) => value === expected.imageMean[index]) &&
    actual.imageStd.every((value, index) => value === expected.imageStd[index]);
}

type AuthenticatedVisionPackage = Readonly<{
  root: string;
  manifest: ModelPackageManifest;
  checksums: ReadonlyMap<string, string>;
}>;

function validateLayerIndex(bytes: Uint8Array, shardCount: number): void {
  let parsed: unknown;
  try { parsed = JSON.parse(new TextDecoder().decode(bytes)); } catch {
    fail("vision-hf-stage-layer-index-invalid", "Converted vision layer index is invalid");
  }
  const value = parsed as { format?: unknown; version?: unknown; groups?: unknown };
  if (value.format !== "webml-qwen-vision-layer-index" || value.version !== 1 || !Array.isArray(value.groups)) {
    fail("vision-hf-stage-layer-index-invalid", "Converted vision layer index is invalid");
  }
  if (shardCount !== VISION_RELEASE_SHARD_COUNT || value.groups.length !== VISION_RELEASE_LAYER_COUNT + 1) {
    fail("vision-hf-stage-layer-index-invalid", "Converted vision layer index is invalid");
  }

  // The streamed executor consumes this release index positionally; coverage alone
  // cannot detect reordered groups or a shard assigned to the wrong vision layer.
  const matchesGroup = (group: unknown, layer: string, shards: readonly number[]): boolean => {
    if (typeof group !== "object" || group === null || Array.isArray(group)) return false;
    const candidate = group as { layer?: unknown; shards?: unknown };
    const keys = Object.keys(candidate).sort();
    return keys.length === 2 && keys[0] === "layer" && keys[1] === "shards" &&
      candidate.layer === layer && Array.isArray(candidate.shards) &&
      candidate.shards.length === shards.length &&
      candidate.shards.every((shard, index) => shard === shards[index]);
  };
  if (!matchesGroup(value.groups[0], "bootstrap", [0, 1])) {
    fail("vision-hf-stage-layer-index-invalid", "Converted vision layer index is invalid");
  }
  for (let layer = 0; layer < VISION_RELEASE_LAYER_COUNT; layer += 1) {
    if (!matchesGroup(value.groups[layer + 1], String(layer), [layer + 2])) {
      fail("vision-hf-stage-layer-index-invalid", "Converted vision layer index is invalid");
    }
  }
}

async function authenticatePackage(
  packageDirectory: string,
  contract: Qwen35VisionHuggingFaceStageContract,
  testHooks: Qwen35VisionHuggingFaceStagingTestHooks,
): Promise<AuthenticatedVisionPackage> {
  const facts = await lstat(packageDirectory).catch(() => undefined);
  if (facts === undefined || !facts.isDirectory() || facts.isSymbolicLink()) {
    fail("vision-hf-stage-package-invalid", "Converted vision package must be a regular directory");
  }
  const root = await realpath(packageDirectory);
  const checksums = parseChecksums(new TextDecoder().decode(
    await boundedRead(join(root, "SHA256SUMS"), "Vision package checksums", testHooks),
  ));
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(
      await boundedRead(join(root, "manifest.json"), "Vision package manifest", testHooks),
    ));
  } catch {
    fail("vision-hf-stage-manifest-invalid", "Converted vision package manifest is invalid");
  }
  const manifest = validateModelPackageManifest(parsed as ModelPackageManifest);
  if (
    manifest.packageKind !== "vision" ||
    manifest.runtime.abi !== "qwen35-webgpu-vision-v1" ||
    !sameIdentity(manifest.source, contract.source) ||
    !sameIdentity(manifest.tokenizer, contract.tokenizer) ||
    manifest.processor === undefined || !sameIdentity(manifest.processor, contract.processor) ||
    manifest.processorSettings === undefined || !sameSettings(manifest.processorSettings, contract.processorSettings) ||
    manifest.shards.length !== contract.shardCount
  ) {
    fail("vision-hf-stage-manifest-invalid", "Converted vision package identity does not match the release contract");
  }
  const expectedFiles = new Set(["LICENSES.json", "layer-index.json", "manifest.json", "source-provenance.json"]);
  for (const [index, shard] of manifest.shards.entries()) {
    const name = `vision-${String(index).padStart(5, "0")}.bin`;
    if (shard.url !== `shards/${name}` || checksums.get(shard.url) !== shard.sha256) {
      fail("vision-hf-stage-manifest-invalid", "Converted vision shard mapping is invalid");
    }
    const file = join(root, "shards", name);
    const shardFacts = await lstat(file);
    if (!shardFacts.isFile() || shardFacts.isSymbolicLink() || BigInt(shardFacts.size) !== BigInt(shard.length)) {
      fail("vision-hf-stage-manifest-invalid", "Converted vision shard length is invalid");
    }
    expectedFiles.add(shard.url);
  }
  if (checksums.size !== expectedFiles.size || [...expectedFiles].some((file) => !checksums.has(file))) {
    fail("vision-hf-stage-checksums-invalid", "Converted vision package checksum set is not exact");
  }
  const topLevel = (await readdir(root)).sort();
  if (JSON.stringify(topLevel) !== JSON.stringify(["LICENSES.json", "SHA256SUMS", "layer-index.json", "manifest.json", "shards", "source-provenance.json"])) {
    fail("vision-hf-stage-package-invalid", "Converted vision package file set is not exact");
  }
  const sourceShards = (await readdir(join(root, "shards"))).sort();
  if (JSON.stringify(sourceShards) !== JSON.stringify(manifest.shards.map((_, index) => `vision-${String(index).padStart(5, "0")}.bin`))) {
    fail("vision-hf-stage-package-invalid", "Converted vision shard file set is not exact");
  }
  for (const [file, expected] of checksums) {
    const source = join(root, ...file.split("/"));
    if (await sha256File(await regularFile(source, "Converted vision package file")) !== expected) {
      fail("vision-hf-stage-source-checksum-mismatch", "Converted vision package file checksum does not match");
    }
  }
  // Authenticate metadata size before reserving output so an oversized source
  // cannot leave an output directory after a pre-publication validation failure.
  await boundedMetadataFile(join(root, "LICENSES.json"), "Vision package licenses");
  await boundedMetadataFile(join(root, "source-provenance.json"), "Vision package provenance");
  validateLayerIndex(
    await boundedRead(join(root, "layer-index.json"), "Vision layer index", testHooks),
    manifest.shards.length,
  );
  return Object.freeze({ root, manifest, checksums });
}

async function readAuthenticatedMetadata(
  source: AuthenticatedVisionPackage,
  file: string,
  testHooks: Qwen35VisionHuggingFaceStagingTestHooks,
): Promise<Uint8Array> {
  const bytes = await boundedRead(join(source.root, file), "Converted vision metadata", testHooks);
  if (createHash("sha256").update(bytes).digest("hex") !== source.checksums.get(file)) {
    fail("vision-hf-stage-source-checksum-mismatch", "Converted vision package file checksum does not match");
  }
  return bytes;
}

function publicationProvenance(source: ModelPackageManifest, staged: ModelPackageManifest): string {
  return `${JSON.stringify({
    format: "webml-qwen-vision-huggingface-publication-provenance",
    version: 1,
    layout: "huggingface-flat-root-v1",
    source: source.source,
    tokenizer: source.tokenizer,
    processor: source.processor,
    runtimeAbi: source.runtime.abi,
    sourceManifestSha256: createHash("sha256").update(stringifyManifest(source)).digest("hex"),
    stagedManifestSha256: createHash("sha256").update(stringifyManifest(staged)).digest("hex"),
  }, null, 2)}\n`;
}

/** Stages only the vision package using names that cannot collide with language files. */
export async function stageQwen35VisionHuggingFacePackage(
  arguments_: Qwen35VisionHuggingFaceStagingArguments,
  contract: Qwen35VisionHuggingFaceStageContract = PINNED_QWEN35_VISION_HUGGINGFACE_STAGE_CONTRACT,
  testHooks: Qwen35VisionHuggingFaceStagingTestHooks = Object.freeze({}),
): Promise<Qwen35VisionHuggingFaceStageReport> {
  const outputDirectory = await outsideGitOutput(arguments_.outputDirectory);
  const source = await authenticatePackage(resolve(arguments_.packageDirectory), contract, testHooks);
  const stagedManifest = validateModelPackageManifest({
    ...source.manifest,
    shards: source.manifest.shards.map((shard) => ({ ...shard, url: basename(shard.url) })),
  });
  try {
    await mkdir(outputDirectory);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EEXIST") {
      fail("vision-hf-stage-output-exists", "Vision Hugging Face staging output already exists");
    }
    fail("vision-hf-stage-publish-failed", "Vision Hugging Face staging output could not be reserved");
  }
  let copiedShardCount = 0;
  for (const shard of source.manifest.shards) {
    const name = basename(shard.url);
    if (!VISION_SHARD_NAME.test(name)) {
      fail("vision-hf-stage-manifest-invalid", "Converted vision shard name is invalid");
    }
    const copied = await copyAndHash(join(source.root, "shards", name), join(outputDirectory, name));
    if (copied.size !== BigInt(shard.length) || copied.sha256 !== shard.sha256) {
      fail("vision-hf-stage-staged-shard-mismatch", "Staged vision shard does not match its manifest");
    }
    copiedShardCount += 1;
  }
  const metadata = new Map<string, Uint8Array | string>([
    ["vision-LICENSES.json", await readAuthenticatedMetadata(source, "LICENSES.json", testHooks)],
    ["vision-layer-index.json", await readAuthenticatedMetadata(source, "layer-index.json", testHooks)],
    ["vision-manifest.json", stringifyManifest(stagedManifest)],
    ["vision-publication-provenance.json", publicationProvenance(source.manifest, stagedManifest)],
    ["vision-source-provenance.json", await readAuthenticatedMetadata(source, "source-provenance.json", testHooks)],
  ]);
  for (const [name, bytes] of metadata) {
    await writeFile(join(outputDirectory, name), bytes, { flag: "wx" });
  }
  const checksumNames = [...source.manifest.shards.map((shard) => basename(shard.url)), ...metadata.keys()].sort();
  const checksumLines: string[] = [];
  for (const name of checksumNames) {
    checksumLines.push(`${await sha256File(join(outputDirectory, name))}  ${name}`);
  }
  await writeFile(join(outputDirectory, "vision-SHA256SUMS"), `${checksumLines.join("\n")}\n`, { encoding: "utf8", flag: "wx" });
  return Object.freeze({
    format: "webml-qwen-vision-huggingface-stage-report" as const,
    version: 1 as const,
    files: Object.freeze([
      "vision-LICENSES.json",
      "vision-SHA256SUMS",
      "vision-layer-index.json",
      "vision-manifest.json",
      "vision-publication-provenance.json",
      "vision-source-provenance.json",
      ...source.manifest.shards.map((shard) => basename(shard.url)),
    ].sort()),
    copiedShardCount,
  });
}

const invokedPath = process.argv[1];
if (invokedPath !== undefined && import.meta.url === pathToFileURL(resolve(invokedPath)).href) {
  try {
    process.stdout.write(`${JSON.stringify(await stageQwen35VisionHuggingFacePackage(
      parseQwen35VisionHuggingFaceStagingArguments(process.argv.slice(2)),
    ), null, 2)}\n`);
  } catch (error) {
    const safe = error instanceof Qwen35VisionHuggingFaceStageError
      ? error
      : new Qwen35VisionHuggingFaceStageError("vision-hf-stage-failed", "Vision Hugging Face staging failed");
    process.stderr.write(`${safe.code}: ${safe.message}\n`);
    process.exitCode = 1;
  }
}
