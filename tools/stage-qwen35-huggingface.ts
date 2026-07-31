import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  copyFile,
  lstat,
  mkdir,
  open,
  readFile,
  realpath,
  writeFile,
} from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  stringifyManifest,
  validateModelPackageManifest,
  type ImmutableArtifactIdentity,
  type ModelPackageManifest,
} from "../src/manifest.js";
import { PINNED_QWEN35_COMPILED_TOKENIZER } from "../src/qwen-tokenizer.js";
import {
  PINNED_QWEN35_CHAT_TEMPLATE_SHA256,
  PINNED_QWEN35_TOKENIZER_CONFIG,
} from "../src/tokenizer-compiler.js";
import {
  deserializeAuthenticatedTokenizerArtifact,
} from "../src/tokenizer-binary.js";
import { PINNED_QWEN35_LANGUAGE_CONVERTER_SOURCE } from "./convert-qwen35-language.js";

const MAX_METADATA_BYTES = 16 * 1024 * 1024;
const DEFAULT_COPY_CHUNK_BYTES = 8 * 1024 * 1024;
const SHARD_NAME = /^model-[0-9]{5}\.bin$/u;

export interface Qwen35HuggingFaceStagingArguments {
  readonly packageDirectory: string;
  readonly tokenizerBinPath: string;
  readonly tokenizerManifestPath: string;
  readonly outputDirectory: string;
}

export interface Qwen35HuggingFaceStageContract {
  readonly source: ImmutableArtifactIdentity;
  readonly tokenizer: ImmutableArtifactIdentity;
  readonly tokenizerConfig: ImmutableArtifactIdentity;
  readonly chatTemplateSha256: string;
  readonly tokenizerArtifact: {
    readonly byteLength: number;
    readonly sha256: string;
    readonly baseVocabSize: number;
    readonly mergeCount: number;
    readonly addedTokenCount: number;
    readonly decodableTokenCount: number;
    readonly modelLogitRows: number;
  };
}

export interface Qwen35HuggingFaceStageOptions {
  readonly cloneFile?: (source: string, destination: string) => Promise<void>;
}

export interface Qwen35HuggingFaceStageReport {
  readonly format: "webml-qwen-huggingface-stage-report";
  readonly version: 1;
  readonly files: readonly string[];
  readonly clonedShardCount: number;
}

export class Qwen35HuggingFaceStageError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "Qwen35HuggingFaceStageError";
    this.code = code;
  }
}

function fail(code: string, message: string): never {
  throw new Qwen35HuggingFaceStageError(code, message);
}

export const PINNED_QWEN35_HUGGINGFACE_STAGE_CONTRACT:
Readonly<Qwen35HuggingFaceStageContract> = Object.freeze({
  source: PINNED_QWEN35_LANGUAGE_CONVERTER_SOURCE.source,
  tokenizer: PINNED_QWEN35_LANGUAGE_CONVERTER_SOURCE.tokenizer,
  tokenizerConfig: Object.freeze({
    ...PINNED_QWEN35_TOKENIZER_CONFIG,
    size: String(PINNED_QWEN35_TOKENIZER_CONFIG.size),
  }),
  chatTemplateSha256: PINNED_QWEN35_CHAT_TEMPLATE_SHA256,
  tokenizerArtifact: Object.freeze({
    byteLength: PINNED_QWEN35_COMPILED_TOKENIZER.byteLength,
    sha256: PINNED_QWEN35_COMPILED_TOKENIZER.sha256,
    baseVocabSize: PINNED_QWEN35_COMPILED_TOKENIZER.baseVocabSize,
    mergeCount: PINNED_QWEN35_COMPILED_TOKENIZER.mergeCount,
    addedTokenCount: PINNED_QWEN35_COMPILED_TOKENIZER.addedTokenCount,
    decodableTokenCount: PINNED_QWEN35_COMPILED_TOKENIZER.decodableTokenCount,
    modelLogitRows: PINNED_QWEN35_COMPILED_TOKENIZER.modelLogitRows,
  }),
});

export function parseQwen35HuggingFaceStagingArguments(
  arguments_: readonly string[],
): Qwen35HuggingFaceStagingArguments {
  const allowed = new Set([
    "--package",
    "--tokenizer-bin",
    "--tokenizer-manifest",
    "--output",
  ]);
  const values = new Map<string, string>();
  for (let index = 0; index < arguments_.length; index += 2) {
    const name = arguments_[index];
    const value = arguments_[index + 1];
    if (
      name === undefined ||
      value === undefined ||
      !allowed.has(name) ||
      value.length === 0 ||
      value.startsWith("--") ||
      values.has(name)
    ) {
      fail("hf-stage-arguments-invalid", "Hugging Face staging received an unknown argument");
    }
    values.set(name, value);
  }
  const packageDirectory = values.get("--package");
  const tokenizerBinPath = values.get("--tokenizer-bin");
  const tokenizerManifestPath = values.get("--tokenizer-manifest");
  const outputDirectory = values.get("--output");
  if (
    packageDirectory === undefined ||
    tokenizerBinPath === undefined ||
    tokenizerManifestPath === undefined ||
    outputDirectory === undefined
  ) {
    fail(
      "hf-stage-arguments-invalid",
      "Hugging Face staging requires --package, --tokenizer-bin, --tokenizer-manifest, and --output",
    );
  }
  return Object.freeze({
    packageDirectory,
    tokenizerBinPath,
    tokenizerManifestPath,
    outputDirectory,
  });
}

async function exists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    fail("hf-stage-path-check-failed", "Hugging Face staging path could not be checked");
  }
}

async function regularFile(path: string, label: string): Promise<string> {
  try {
    const facts = await lstat(path);
    if (!facts.isFile() || facts.isSymbolicLink()) {
      fail("hf-stage-input-invalid", `${label} must be a regular file`);
    }
    return await realpath(path);
  } catch (error) {
    if (error instanceof Qwen35HuggingFaceStageError) throw error;
    fail("hf-stage-input-invalid", `${label} must be a regular file`);
  }
}

async function outsideGitOutput(output: string): Promise<string> {
  const requested = resolve(output);
  if (basename(requested).length === 0 || basename(requested) === ".git") {
    fail("hf-stage-output-invalid", "Hugging Face staging output is invalid");
  }
  if (await exists(requested)) {
    fail("hf-stage-output-exists", "Hugging Face staging output already exists");
  }
  let parent: string;
  try {
    parent = await realpath(dirname(requested));
  } catch {
    fail("hf-stage-output-parent-invalid", "Hugging Face staging output parent must exist");
  }
  let current = parent;
  while (true) {
    if (await exists(join(current, ".git"))) {
      fail("hf-stage-output-inside-git", "Hugging Face staging output must be outside Git");
    }
    const next = dirname(current);
    if (next === current) break;
    current = next;
  }
  return join(parent, basename(requested));
}

async function boundedRead(path: string, label: string): Promise<Uint8Array> {
  const resolved = await regularFile(path, label);
  const facts = await lstat(resolved);
  if (facts.size < 1 || facts.size > MAX_METADATA_BYTES) {
    fail("hf-stage-input-invalid", `${label} size is invalid`);
  }
  return readFile(resolved);
}

async function sha256File(path: string): Promise<string> {
  const handle = await open(path, "r");
  const hash = createHash("sha256");
  const buffer = Buffer.allocUnsafe(DEFAULT_COPY_CHUNK_BYTES);
  try {
    let position = 0;
    while (true) {
      const { bytesRead } = await handle.read(buffer, 0, buffer.byteLength, position);
      if (bytesRead === 0) break;
      hash.update(buffer.subarray(0, bytesRead));
      position += bytesRead;
    }
    return hash.digest("hex");
  } finally {
    await handle.close();
  }
}

function parseChecksums(input: string): ReadonlyMap<string, string> {
  const result = new Map<string, string>();
  for (const line of input.trimEnd().split("\n")) {
    const match = /^([a-f0-9]{64})  ([A-Za-z0-9._/-]+)$/u.exec(line);
    if (
      match === null ||
      match[2]!.startsWith("/") ||
      match[2]!.split("/").some((part) => part === "" || part === "." || part === "..") ||
      result.has(match[2]!)
    ) {
      fail("hf-stage-checksums-invalid", "Converted package checksums are invalid");
    }
    result.set(match[2]!, match[1]!);
  }
  return result;
}

function sameIdentity(
  actual: ImmutableArtifactIdentity,
  expected: ImmutableArtifactIdentity,
): boolean {
  return actual.repository === expected.repository &&
    actual.revision === expected.revision &&
    actual.file === expected.file &&
    actual.size === expected.size &&
    actual.sha256 === expected.sha256;
}

function unknownIdentityMatches(
  actual: unknown,
  expected: ImmutableArtifactIdentity,
): boolean {
  if (typeof actual !== "object" || actual === null || Array.isArray(actual)) {
    return false;
  }
  const value = actual as Record<string, unknown>;
  return value.repository === expected.repository &&
    value.revision === expected.revision &&
    value.file === expected.file &&
    String(value.size) === expected.size &&
    value.sha256 === expected.sha256;
}

async function authenticateConvertedPackage(
  packageDirectory: string,
  contract: Qwen35HuggingFaceStageContract,
): Promise<{ manifest: ModelPackageManifest; files: ReadonlyMap<string, string> }> {
  const rootFacts = await lstat(packageDirectory).catch(() => undefined);
  if (rootFacts === undefined || !rootFacts.isDirectory() || rootFacts.isSymbolicLink()) {
    fail("hf-stage-package-invalid", "Converted package must be a regular directory");
  }
  const root = await realpath(packageDirectory);
  const checksums = parseChecksums(
    new TextDecoder().decode(await boundedRead(join(root, "SHA256SUMS"), "Package checksums")),
  );
  const requiredChecksums = new Set([
    "LICENSES.json",
    "manifest.json",
    "source-provenance.json",
  ]);
  for (const [relative, expected] of checksums) {
    const candidate = join(root, ...relative.split("/"));
    const resolved = await regularFile(candidate, "Converted package file");
    if (!resolved.startsWith(`${root}/`) || await sha256File(resolved) !== expected) {
      fail("hf-stage-source-checksum-mismatch", "Converted package file checksum does not match");
    }
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(
      new TextDecoder().decode(await boundedRead(join(root, "manifest.json"), "Package manifest")),
    );
  } catch {
    fail("hf-stage-manifest-invalid", "Converted package manifest is invalid");
  }
  const manifest = validateModelPackageManifest(parsed as ModelPackageManifest);
  if (
    manifest.packageKind !== "language" ||
    manifest.runtime.abi !== "qwen35-webgpu-v1" ||
    !sameIdentity(manifest.source, contract.source) ||
    !sameIdentity(manifest.tokenizer, contract.tokenizer)
  ) {
    fail("hf-stage-manifest-invalid", "Converted package identity does not match the release contract");
  }
  const shardNames = new Set<string>();
  for (const shard of manifest.shards) {
    const match = /^shards\/(model-[0-9]{5}\.bin)$/u.exec(shard.url);
    if (match === null || shardNames.has(match[1]!) || checksums.get(shard.url) !== shard.sha256) {
      fail("hf-stage-manifest-invalid", "Converted package shard mapping is invalid");
    }
    requiredChecksums.add(shard.url);
    const shardFacts = await lstat(join(root, ...shard.url.split("/")));
    if (BigInt(shardFacts.size) !== BigInt(shard.length)) {
      fail("hf-stage-manifest-invalid", "Converted package shard length is invalid");
    }
    shardNames.add(match[1]!);
  }
  if (manifest.shards.length !== shardNames.size) {
    fail("hf-stage-manifest-invalid", "Converted package shard mapping is incomplete");
  }
  if (
    checksums.size !== requiredChecksums.size ||
    [...requiredChecksums].some((name) => !checksums.has(name))
  ) {
    fail("hf-stage-checksums-invalid", "Converted package checksum set is not exact");
  }
  return { manifest, files: checksums };
}

async function authenticateTokenizer(
  binPath: string,
  manifestPath: string,
  contract: Qwen35HuggingFaceStageContract,
): Promise<{ binary: Uint8Array; manifestBytes: Uint8Array }> {
  const binary = await boundedRead(binPath, "Compiled tokenizer");
  const manifestBytes = await boundedRead(manifestPath, "Compiled tokenizer manifest");
  let manifest: Record<string, unknown>;
  try {
    manifest = JSON.parse(new TextDecoder().decode(manifestBytes)) as Record<string, unknown>;
  } catch {
    fail("hf-stage-tokenizer-manifest-invalid", "Compiled tokenizer manifest is invalid");
  }
  const exactKeys = [
    "addedTokenCount",
    "artifactByteLength",
    "artifactSha256",
    "baseVocabSize",
    "chatTemplateSha256",
    "chatTemplateSource",
    "decodableTokenCount",
    "format",
    "mergeCount",
    "modelLogitRows",
    "normalization",
    "preTokenizer",
    "runtimeAbi",
    "source",
    "tokenCount",
    "undecodableLogitRows",
    "version",
  ].sort();
  if (JSON.stringify(Object.keys(manifest).sort()) !== JSON.stringify(exactKeys)) {
    fail("hf-stage-tokenizer-manifest-invalid", "Compiled tokenizer manifest schema is invalid");
  }
  if (
    manifest.format !== "webml-qwen35-tokenizer" ||
    manifest.version !== 1 ||
    manifest.runtimeAbi !== "qwen35-tokenizer-v1" ||
    !unknownIdentityMatches(manifest.source, contract.tokenizer) ||
    !unknownIdentityMatches(manifest.chatTemplateSource, contract.tokenizerConfig) ||
    manifest.chatTemplateSha256 !== contract.chatTemplateSha256 ||
    manifest.normalization !== "NFC" ||
    manifest.preTokenizer !== "qwen35-bytelevel-v1" ||
    manifest.baseVocabSize !== contract.tokenizerArtifact.baseVocabSize ||
    manifest.tokenCount !== contract.tokenizerArtifact.decodableTokenCount ||
    manifest.decodableTokenCount !== contract.tokenizerArtifact.decodableTokenCount ||
    manifest.modelLogitRows !== contract.tokenizerArtifact.modelLogitRows ||
    manifest.undecodableLogitRows !==
      contract.tokenizerArtifact.modelLogitRows - contract.tokenizerArtifact.decodableTokenCount ||
    manifest.mergeCount !== contract.tokenizerArtifact.mergeCount ||
    manifest.addedTokenCount !== contract.tokenizerArtifact.addedTokenCount
  ) {
    fail("hf-stage-tokenizer-manifest-invalid", "Compiled tokenizer manifest identity is invalid");
  }
  if (
    manifest.artifactByteLength !== binary.byteLength ||
    manifest.artifactSha256 !== createHash("sha256").update(binary).digest("hex")
  ) {
    fail("hf-stage-tokenizer-checksum-mismatch", "Compiled tokenizer SHA-256 does not match its manifest");
  }
  try {
    const tables = await deserializeAuthenticatedTokenizerArtifact(
      (async function* (): AsyncIterable<Uint8Array> { yield binary; })(),
      contract.tokenizerArtifact,
      binary.byteLength,
    );
    if (
      tables.baseVocabSize !== contract.tokenizerArtifact.baseVocabSize ||
      tables.tokenCount !== contract.tokenizerArtifact.decodableTokenCount ||
      tables.merges.length / 3 !== contract.tokenizerArtifact.mergeCount ||
      tables.addedTokenIds.length !== contract.tokenizerArtifact.addedTokenCount ||
      manifest.modelLogitRows !== contract.tokenizerArtifact.modelLogitRows
    ) {
      fail("hf-stage-tokenizer-invalid", "Compiled tokenizer does not match the release contract");
    }
  } catch {
    fail("hf-stage-tokenizer-invalid", "Compiled tokenizer does not match the release contract");
  }
  const canonicalManifest = `${JSON.stringify({
    format: "webml-qwen35-tokenizer",
    version: 1,
    runtimeAbi: "qwen35-tokenizer-v1",
    source: manifest.source,
    chatTemplateSource: manifest.chatTemplateSource,
    chatTemplateSha256: manifest.chatTemplateSha256,
    normalization: "NFC",
    preTokenizer: "qwen35-bytelevel-v1",
    baseVocabSize: contract.tokenizerArtifact.baseVocabSize,
    tokenCount: contract.tokenizerArtifact.decodableTokenCount,
    decodableTokenCount: contract.tokenizerArtifact.decodableTokenCount,
    modelLogitRows: contract.tokenizerArtifact.modelLogitRows,
    undecodableLogitRows:
      contract.tokenizerArtifact.modelLogitRows - contract.tokenizerArtifact.decodableTokenCount,
    mergeCount: contract.tokenizerArtifact.mergeCount,
    addedTokenCount: contract.tokenizerArtifact.addedTokenCount,
    artifactByteLength: binary.byteLength,
    artifactSha256: createHash("sha256").update(binary).digest("hex"),
  }, null, 2)}\n`;
  return { binary, manifestBytes: new TextEncoder().encode(canonicalManifest) };
}

function publicationProvenance(
  sourceManifest: ModelPackageManifest,
  stagedManifest: ModelPackageManifest,
  tokenizerHash: string,
): string {
  return `${JSON.stringify({
    format: "webml-qwen-huggingface-publication-provenance",
    version: 1,
    layout: "huggingface-flat-root-v1",
    source: sourceManifest.source,
    tokenizer: sourceManifest.tokenizer,
    sourceManifestSha256: createHash("sha256").update(stringifyManifest(sourceManifest)).digest("hex"),
    stagedManifestSha256: createHash("sha256").update(stringifyManifest(stagedManifest)).digest("hex"),
    compiledTokenizerSha256: tokenizerHash,
  }, null, 2)}\n`;
}

/**
 * Authenticates a converted package and publishes a flat, web-uploader-ready
 * directory with independently owned clone/copy-on-write shard files.
 */
export async function stageQwen35HuggingFacePackage(
  arguments_: Qwen35HuggingFaceStagingArguments,
  contract: Qwen35HuggingFaceStageContract =
    PINNED_QWEN35_HUGGINGFACE_STAGE_CONTRACT,
  options: Qwen35HuggingFaceStageOptions = {},
): Promise<Qwen35HuggingFaceStageReport> {
  const outputDirectory = await outsideGitOutput(arguments_.outputDirectory);
  const authenticated = await authenticateConvertedPackage(
    resolve(arguments_.packageDirectory),
    contract,
  );
  const tokenizer = await authenticateTokenizer(
    resolve(arguments_.tokenizerBinPath),
    resolve(arguments_.tokenizerManifestPath),
    contract,
  );
  const stagedManifest = validateModelPackageManifest({
    ...authenticated.manifest,
    shards: authenticated.manifest.shards.map((shard) => ({
      ...shard,
      url: basename(shard.url),
    })),
  });
  try {
    await mkdir(outputDirectory);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EEXIST") {
      fail("hf-stage-output-exists", "Hugging Face staging output already exists");
    }
    fail("hf-stage-publish-failed", "Hugging Face staging output could not be reserved");
  }
  // The exclusive directory reservation removes the final-rename overwrite
  // race. A failed output remains outside Git for inspection and is never
  // silently reused as a complete release.
  const staging = outputDirectory;
  let clonedShardCount = 0;
  const cloneFile = options.cloneFile ?? ((source: string, destination: string) =>
    copyFile(source, destination, constants.COPYFILE_FICLONE));
  for (const shard of authenticated.manifest.shards) {
    const name = basename(shard.url);
    if (!SHARD_NAME.test(name)) fail("hf-stage-manifest-invalid", "Shard name is invalid");
    const source = join(resolve(arguments_.packageDirectory), "shards", name);
    const destination = join(staging, name);
    await cloneFile(source, destination);
    const facts = await lstat(destination);
    if (
      !facts.isFile() ||
      facts.isSymbolicLink() ||
      BigInt(facts.size) !== BigInt(shard.length) ||
      await sha256File(destination) !== shard.sha256
    ) {
      fail("hf-stage-staged-shard-mismatch", "Staged shard does not match its manifest");
    }
    clonedShardCount += 1;
  }
  const sourceRoot = resolve(arguments_.packageDirectory);
  const manifestText = stringifyManifest(stagedManifest);
  const tokenizerHash = createHash("sha256").update(tokenizer.binary).digest("hex");
  const metadata = new Map<string, Uint8Array | string>([
    ["LICENSES.json", await boundedRead(join(sourceRoot, "LICENSES.json"), "License metadata")],
    ["manifest.json", manifestText],
    ["publication-provenance.json", publicationProvenance(authenticated.manifest, stagedManifest, tokenizerHash)],
    ["source-provenance.json", await boundedRead(join(sourceRoot, "source-provenance.json"), "Source provenance")],
    ["tokenizer.bin", tokenizer.binary],
    ["tokenizer.manifest.json", tokenizer.manifestBytes],
  ]);
  for (const [name, bytes] of metadata) {
    await writeFile(join(staging, name), bytes, { flag: "wx" });
  }
  const checksumNames = [
    ...authenticated.manifest.shards.map((shard) => basename(shard.url)),
    ...metadata.keys(),
  ].sort();
  const checksumLines: string[] = [];
  for (const name of checksumNames) {
    checksumLines.push(`${await sha256File(join(staging, name))}  ${name}`);
  }
  await writeFile(join(staging, "SHA256SUMS"), `${checksumLines.join("\n")}\n`, {
    encoding: "utf8",
    flag: "wx",
  });
  const files = Object.freeze([
    "LICENSES.json",
    "SHA256SUMS",
    "manifest.json",
    ...authenticated.manifest.shards.map((shard) => basename(shard.url)),
    "publication-provenance.json",
    "source-provenance.json",
    "tokenizer.bin",
    "tokenizer.manifest.json",
  ].sort());
  return Object.freeze({
    format: "webml-qwen-huggingface-stage-report" as const,
    version: 1 as const,
    files,
    clonedShardCount,
  });
}

const invokedPath = process.argv[1];
if (invokedPath !== undefined && import.meta.url === pathToFileURL(resolve(invokedPath)).href) {
  try {
    const report = await stageQwen35HuggingFacePackage(
      parseQwen35HuggingFaceStagingArguments(process.argv.slice(2)),
    );
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  } catch (error) {
    const safe = error instanceof Qwen35HuggingFaceStageError
      ? error
      : new Qwen35HuggingFaceStageError("hf-stage-failed", "Hugging Face staging failed");
    process.stderr.write(`${safe.code}: ${safe.message}\n`);
    process.exitCode = 1;
  }
}
