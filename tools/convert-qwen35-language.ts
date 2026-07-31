import { createHash, randomUUID } from "node:crypto";
import {
  lstat,
  mkdir,
  open,
  realpath,
  rename,
  writeFile,
  type FileHandle,
} from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  MAX_STREAM_READ_BYTES,
  createManifestFromPlan,
  executeConversionPlan,
  planConversion,
  type ConversionPlan,
  type RandomAccessWriter,
} from "../src/converter.js";
import {
  GgmlType,
  parseGguf,
  type ParsedGguf,
  type RandomAccessReader,
} from "../src/gguf.js";
import {
  stringifyManifest,
  type ImmutableArtifactIdentity,
} from "../src/manifest.js";

const RUNTIME_ABI = "qwen35-webgpu-v1";
const DEFAULT_MAX_SHARD_BYTES = 128n * 1024n * 1024n;
const DEFAULT_TENSOR_ALIGNMENT = 256;

const GGML_TYPE_NAMES = new Map<GgmlType, string>([
  [GgmlType.F32, "F32"],
  [GgmlType.F16, "F16"],
  [GgmlType.Q4_0, "Q4_0"],
  [GgmlType.Q4_1, "Q4_1"],
  [GgmlType.Q5_0, "Q5_0"],
  [GgmlType.Q5_1, "Q5_1"],
  [GgmlType.Q8_0, "Q8_0"],
  [GgmlType.Q8_1, "Q8_1"],
  [GgmlType.Q2_K, "Q2_K"],
  [GgmlType.Q3_K, "Q3_K"],
  [GgmlType.Q4_K, "Q4_K"],
  [GgmlType.Q5_K, "Q5_K"],
  [GgmlType.Q6_K, "Q6_K"],
  [GgmlType.Q8_K, "Q8_K"],
  [GgmlType.I8, "I8"],
  [GgmlType.I16, "I16"],
  [GgmlType.I32, "I32"],
  [GgmlType.I64, "I64"],
  [GgmlType.F64, "F64"],
  [GgmlType.BF16, "BF16"],
]);

export interface Qwen35LanguageSourceContract {
  readonly source: ImmutableArtifactIdentity;
  readonly tokenizer: ImmutableArtifactIdentity;
  readonly dataOffset: bigint;
  readonly alignment: number;
  readonly blockCount: number;
  readonly tensorTypeCounts: Readonly<Record<string, number>>;
}

export const PINNED_QWEN35_LANGUAGE_CONVERTER_SOURCE:
Readonly<Qwen35LanguageSourceContract> = Object.freeze({
  source: Object.freeze({
    repository: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF",
    revision: "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
    file: "Qwen_Qwen3.5-4B-Q3_K_L.gguf",
    size: "2665441248",
    sha256: "41c3f1bf47e477693dab332e73347c7138d5e9fbfe74c6d2eaba590be1f3d20a",
  }),
  tokenizer: Object.freeze({
    repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
    revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    file: "tokenizer.json",
    size: "12807982",
    sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
  }),
  dataOffset: 10_969_056n,
  alignment: 32,
  blockCount: 33,
  tensorTypeCounts: Object.freeze({
    F32: 232,
    Q8_0: 20,
    Q3_K: 112,
    Q4_K: 8,
    Q5_K: 64,
    Q6_K: 5,
  }),
});

export type Qwen35LanguageConverterMode = "convert" | "dry-run" | "inventory";

export interface Qwen35LanguageConverterArguments {
  readonly sourcePath: string;
  readonly outputDirectory: string;
  readonly mode: Qwen35LanguageConverterMode;
}

export interface Qwen35LanguageConverterOptions {
  readonly maxShardBytes?: bigint;
  readonly tensorAlignment?: number;
  readonly maxBlocksPerRead?: number;
}

export interface Qwen35LanguageConverterReport {
  readonly format: "webml-qwen-converter-report";
  readonly version: 1;
  readonly mode: Qwen35LanguageConverterMode;
  readonly source: ImmutableArtifactIdentity;
  readonly inventory: readonly {
    readonly ggmlType: string;
    readonly ggmlTypeId: number;
    readonly tensorCount: number;
    readonly sourceBytes: string;
  }[];
  readonly excludedTensors: ConversionPlan["excludedTensors"];
  readonly plan?: {
    readonly shardCount: number;
    readonly segmentCount: number;
    readonly outputBytes: string;
  };
  readonly published?: {
    readonly files: readonly string[];
  };
}

export class Qwen35LanguageConverterError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "Qwen35LanguageConverterError";
    this.code = code;
  }
}

function fail(code: string, message: string): never {
  throw new Qwen35LanguageConverterError(code, message);
}

export function parseQwen35LanguageConverterArguments(
  arguments_: readonly string[],
): Qwen35LanguageConverterArguments {
  const paths = new Map<string, string>();
  let mode: Qwen35LanguageConverterMode = "convert";
  let modeWasSet = false;
  for (let index = 0; index < arguments_.length; index += 1) {
    const name = arguments_[index];
    if (name === "--dry-run" || name === "--inventory") {
      if (modeWasSet) {
        fail("converter-arguments-invalid", "Converter accepts only one mode");
      }
      mode = name === "--dry-run" ? "dry-run" : "inventory";
      modeWasSet = true;
      continue;
    }
    if (name !== "--source" && name !== "--output") {
      fail("converter-arguments-invalid", "Converter received an unknown argument");
    }
    const value = arguments_[index + 1];
    if (
      value === undefined ||
      value.length === 0 ||
      value.startsWith("--") ||
      paths.has(name)
    ) {
      fail("converter-arguments-invalid", "Converter received an unknown argument");
    }
    paths.set(name, value);
    index += 1;
  }
  const sourcePath = paths.get("--source");
  const outputDirectory = paths.get("--output");
  if (sourcePath === undefined || outputDirectory === undefined) {
    fail(
      "converter-arguments-invalid",
      "Converter requires --source and --output",
    );
  }
  return Object.freeze({ sourcePath, outputDirectory, mode });
}

class FileReader implements RandomAccessReader {
  constructor(
    readonly handle: FileHandle,
    readonly size: bigint,
  ) {}

  async read(offset: bigint, length: number): Promise<Uint8Array> {
    if (
      !Number.isSafeInteger(length) ||
      length < 0 ||
      length > MAX_STREAM_READ_BYTES ||
      offset < 0n ||
      offset > BigInt(Number.MAX_SAFE_INTEGER) ||
      offset + BigInt(length) > this.size
    ) {
      fail("converter-source-read-failed", "Converter source read failed");
    }
    const bytes = Buffer.allocUnsafe(length);
    let completed = 0;
    try {
      while (completed < length) {
        const result = await this.handle.read(
          bytes,
          completed,
          length - completed,
          Number(offset) + completed,
        );
        if (result.bytesRead === 0) {
          fail("converter-source-read-failed", "Converter source read failed");
        }
        completed += result.bytesRead;
      }
    } catch (error) {
      if (error instanceof Qwen35LanguageConverterError) throw error;
      fail("converter-source-read-failed", "Converter source read failed");
    }
    return bytes;
  }
}

class ShardWriter implements RandomAccessWriter {
  constructor(readonly handle: FileHandle) {}

  async write(offset: bigint, bytes: Uint8Array): Promise<void> {
    if (
      offset < 0n ||
      offset > BigInt(Number.MAX_SAFE_INTEGER) ||
      bytes.byteLength > MAX_STREAM_READ_BYTES
    ) {
      fail("converter-shard-write-failed", "Converter shard write failed");
    }
    let completed = 0;
    try {
      while (completed < bytes.byteLength) {
        const result = await this.handle.write(
          bytes,
          completed,
          bytes.byteLength - completed,
          Number(offset) + completed,
        );
        if (result.bytesWritten === 0) {
          fail("converter-shard-write-failed", "Converter shard write failed");
        }
        completed += result.bytesWritten;
      }
    } catch (error) {
      if (error instanceof Qwen35LanguageConverterError) throw error;
      fail("converter-shard-write-failed", "Converter shard write failed");
    }
  }
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    fail("converter-output-check-failed", "Converter output could not be checked");
  }
}

async function resolveOutsideGitOutput(outputDirectory: string): Promise<string> {
  const requested = resolve(outputDirectory);
  const name = basename(requested);
  if (name === ".git" || name.length === 0) {
    fail("converter-output-invalid", "Converter output directory is invalid");
  }
  if (await pathExists(requested)) {
    fail("converter-output-exists", "Converter output directory already exists");
  }
  let parent: string;
  try {
    parent = await realpath(dirname(requested));
  } catch {
    fail(
      "converter-output-parent-invalid",
      "Converter output parent directory must already exist",
    );
  }
  let current = parent;
  while (true) {
    if (await pathExists(join(current, ".git"))) {
      fail(
        "converter-output-inside-git",
        "Converter output must be outside a Git worktree",
      );
    }
    const next = dirname(current);
    if (next === current) break;
    current = next;
  }
  return join(parent, name);
}

async function sourceHash(reader: FileReader): Promise<string> {
  const hash = createHash("sha256");
  let offset = 0n;
  while (offset < reader.size) {
    const remaining = reader.size - offset;
    const length = Number(
      remaining < BigInt(MAX_STREAM_READ_BYTES)
        ? remaining
        : BigInt(MAX_STREAM_READ_BYTES),
    );
    hash.update(await reader.read(offset, length));
    offset += BigInt(length);
  }
  return hash.digest("hex");
}

type StableFileFacts = Readonly<{
  dev: bigint;
  ino: bigint;
  size: bigint;
  mtimeNs: bigint;
  ctimeNs: bigint;
}>;

async function fileFacts(handle: FileHandle): Promise<StableFileFacts> {
  try {
    const facts = await handle.stat({ bigint: true });
    if (!facts.isFile()) {
      fail("converter-source-invalid", "Converter source must be a regular file");
    }
    return {
      dev: facts.dev,
      ino: facts.ino,
      size: facts.size,
      mtimeNs: facts.mtimeNs,
      ctimeNs: facts.ctimeNs,
    };
  } catch (error) {
    if (error instanceof Qwen35LanguageConverterError) throw error;
    fail("converter-source-stat-failed", "Converter source could not be inspected");
  }
}

function sameFileFacts(left: StableFileFacts, right: StableFileFacts): boolean {
  return left.dev === right.dev &&
    left.ino === right.ino &&
    left.size === right.size &&
    left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs;
}

function typeName(type: GgmlType): string {
  return GGML_TYPE_NAMES.get(type) ?? `GGML_${type}`;
}

function validateParsedSource(
  parsed: ParsedGguf,
  plan: ConversionPlan,
  contract: Qwen35LanguageSourceContract,
): void {
  if (
    parsed.dataOffset !== contract.dataOffset ||
    parsed.alignment !== contract.alignment ||
    parsed.metadata["general.architecture"] !== "qwen35" ||
    parsed.metadata["qwen35.block_count"] !== contract.blockCount
  ) {
    fail(
      "converter-source-contract-mismatch",
      "Converter source does not match the pinned Qwen3.5 directory contract",
    );
  }
  const actual = Object.fromEntries(
    plan.inventory.map((entry) => [typeName(entry.ggmlType), entry.tensorCount]),
  );
  const expectedEntries = Object.entries(contract.tensorTypeCounts)
    .sort(([left], [right]) => left.localeCompare(right));
  const actualEntries = Object.entries(actual)
    .sort(([left], [right]) => left.localeCompare(right));
  if (JSON.stringify(actualEntries) !== JSON.stringify(expectedEntries)) {
    fail(
      "converter-source-inventory-mismatch",
      "Converter source GGML inventory does not match the pinned contract",
    );
  }
}

function reportFor(
  mode: Qwen35LanguageConverterMode,
  contract: Qwen35LanguageSourceContract,
  plan: ConversionPlan,
  publishedFiles?: readonly string[],
): Qwen35LanguageConverterReport {
  const inventory = plan.inventory.map((entry) => Object.freeze({
    ggmlType: typeName(entry.ggmlType),
    ggmlTypeId: entry.ggmlType,
    tensorCount: entry.tensorCount,
    sourceBytes: entry.sourceBytes.toString(),
  }));
  const planSummary = mode === "inventory"
    ? undefined
    : Object.freeze({
        shardCount: plan.shards.length,
        segmentCount: plan.segments.length,
        outputBytes: plan.shards
          .reduce((total, shard) => total + shard.length, 0n)
          .toString(),
      });
  return Object.freeze({
    format: "webml-qwen-converter-report" as const,
    version: 1 as const,
    mode,
    source: contract.source,
    inventory: Object.freeze(inventory),
    excludedTensors: plan.excludedTensors,
    ...(planSummary === undefined ? {} : { plan: planSummary }),
    ...(publishedFiles === undefined
      ? {}
      : { published: Object.freeze({ files: Object.freeze([...publishedFiles]) }) }),
  });
}

function sourceProvenance(
  parsed: ParsedGguf,
  plan: ConversionPlan,
  contract: Qwen35LanguageSourceContract,
): string {
  const document = {
    format: "webml-qwen-source-provenance",
    version: 1,
    source: contract.source,
    tokenizer: contract.tokenizer,
    runtimeAbi: RUNTIME_ABI,
    observed: {
      alignment: parsed.alignment,
      dataOffset: parsed.dataOffset.toString(),
      tensorCount: parsed.tensors.length,
      inventory: plan.inventory.map((entry) => ({
        ggmlType: typeName(entry.ggmlType),
        ggmlTypeId: entry.ggmlType,
        tensorCount: entry.tensorCount,
        sourceBytes: entry.sourceBytes.toString(),
      })),
      excludedTensors: plan.excludedTensors,
    },
  };
  return `${JSON.stringify(document, null, 2)}\n`;
}

function licenseMetadata(contract: Qwen35LanguageSourceContract): string {
  const document = {
    format: "webml-qwen-license-metadata",
    version: 1,
    materials: [{
      name: "Qwen3.5 4B model materials and converted weights",
      spdx: "Apache-2.0",
      licenseUrl:
        `${contract.tokenizer.repository}/blob/${contract.tokenizer.revision}/LICENSE`,
      notice: "Distribution remains subject to the upstream model license.",
    }],
  };
  return `${JSON.stringify(document, null, 2)}\n`;
}

async function hashOutputFile(path: string): Promise<string> {
  let handle: FileHandle;
  try {
    handle = await open(path, "r");
  } catch {
    fail("converter-checksum-failed", "Converter checksum generation failed");
  }
  try {
    const facts = await fileFacts(handle);
    return await sourceHash(new FileReader(handle, facts.size));
  } catch (error) {
    if (error instanceof Qwen35LanguageConverterError) throw error;
    fail("converter-checksum-failed", "Converter checksum generation failed");
  } finally {
    await handle.close().catch(() => undefined);
  }
}

async function publishPackage(input: {
  readonly outputDirectory: string;
  readonly reader: FileReader;
  readonly parsed: ParsedGguf;
  readonly plan: ConversionPlan;
  readonly contract: Qwen35LanguageSourceContract;
  readonly sourceFacts: StableFileFacts;
  readonly maxBlocksPerRead?: number;
}): Promise<readonly string[]> {
  const parent = dirname(input.outputDirectory);
  const name = basename(input.outputDirectory);
  // Failed staging stays outside Git for inspection; only this final rename
  // makes a complete package visible at the requested output path.
  const staging = join(parent, `.${name}.staging-${randomUUID()}`);
  const shardDirectory = join(staging, "shards");
  try {
    await mkdir(shardDirectory, { recursive: true });
  } catch {
    fail("converter-staging-failed", "Converter staging directory could not be created");
  }

  const shardFiles = input.plan.shards.map(
    (shard) => `shards/model-${String(shard.index).padStart(5, "0")}.bin`,
  );
  const handles: FileHandle[] = [];
  try {
    for (const file of shardFiles) {
      handles.push(await open(join(staging, file), "wx"));
    }
    await executeConversionPlan(
      input.plan,
      input.reader,
      handles.map((handle) => new ShardWriter(handle)),
      input.maxBlocksPerRead === undefined
        ? {}
        : { maxBlocksPerRead: input.maxBlocksPerRead },
    );
  } catch (error) {
    if (error instanceof Qwen35LanguageConverterError) throw error;
    fail("converter-conversion-failed", "Converter shard generation failed");
  } finally {
    const closed = await Promise.allSettled(handles.map((handle) => handle.close()));
    if (closed.some((result) => result.status === "rejected")) {
      fail("converter-shard-close-failed", "Converter shard close failed");
    }
  }

  const sourceAfterConversion = await fileFacts(input.reader.handle);
  if (!sameFileFacts(input.sourceFacts, sourceAfterConversion)) {
    fail("converter-source-changed", "Converter source changed during conversion");
  }

  const shardArtifacts = await Promise.all(
    shardFiles.map(async (file) => ({
      url: file,
      sha256: await hashOutputFile(join(staging, file)),
    })),
  );
  const manifest = createManifestFromPlan(input.plan, {
    packageKind: "language",
    source: input.contract.source,
    runtimeAbi: RUNTIME_ABI,
    tokenizer: input.contract.tokenizer,
    shards: shardArtifacts,
  });
  try {
    await writeFile(join(staging, "manifest.json"), stringifyManifest(manifest), {
      encoding: "utf8",
      flag: "wx",
    });
    await writeFile(
      join(staging, "source-provenance.json"),
      sourceProvenance(input.parsed, input.plan, input.contract),
      { encoding: "utf8", flag: "wx" },
    );
    await writeFile(
      join(staging, "LICENSES.json"),
      licenseMetadata(input.contract),
      { encoding: "utf8", flag: "wx" },
    );
  } catch {
    fail("converter-metadata-write-failed", "Converter metadata write failed");
  }

  const checksummedFiles = [
    "LICENSES.json",
    "manifest.json",
    ...shardFiles,
    "source-provenance.json",
  ].sort();
  const checksumLines: string[] = [];
  for (const file of checksummedFiles) {
    checksumLines.push(`${await hashOutputFile(join(staging, file))}  ${file}`);
  }
  try {
    await writeFile(
      join(staging, "SHA256SUMS"),
      `${checksumLines.join("\n")}\n`,
      { encoding: "utf8", flag: "wx" },
    );
    await rename(staging, input.outputDirectory);
  } catch {
    fail("converter-publish-failed", "Converter package publication failed");
  }
  return Object.freeze([
    "LICENSES.json",
    "SHA256SUMS",
    "manifest.json",
    ...shardFiles,
    "source-provenance.json",
  ]);
}

/**
 * Authenticates one immutable GGUF through a single open handle, then plans or
 * publishes bounded complete-row shards without retaining model-sized buffers.
 */
export async function convertQwen35LanguagePackage(
  arguments_: Qwen35LanguageConverterArguments,
  contract: Qwen35LanguageSourceContract =
    PINNED_QWEN35_LANGUAGE_CONVERTER_SOURCE,
  options: Qwen35LanguageConverterOptions = {},
): Promise<Qwen35LanguageConverterReport> {
  const outputDirectory = await resolveOutsideGitOutput(arguments_.outputDirectory);
  let handle: FileHandle;
  try {
    handle = await open(resolve(arguments_.sourcePath), "r");
  } catch {
    fail("converter-source-open-failed", "Converter source could not be opened");
  }
  try {
    const initialFacts = await fileFacts(handle);
    if (initialFacts.size.toString() !== contract.source.size) {
      fail(
        "converter-source-size-mismatch",
        "Converter source size does not match the pinned identity",
      );
    }
    const reader = new FileReader(handle, initialFacts.size);
    if (await sourceHash(reader) !== contract.source.sha256) {
      fail(
        "converter-source-hash-mismatch",
        "Converter source SHA-256 does not match the pinned identity",
      );
    }
    if (!sameFileFacts(initialFacts, await fileFacts(handle))) {
      fail("converter-source-changed", "Converter source changed during authentication");
    }

    let parsed: ParsedGguf;
    let plan: ConversionPlan;
    try {
      parsed = await parseGguf(reader);
      plan = planConversion(parsed, {
        maxShardBytes: options.maxShardBytes ?? DEFAULT_MAX_SHARD_BYTES,
        tensorAlignment: options.tensorAlignment ?? DEFAULT_TENSOR_ALIGNMENT,
      });
    } catch (error) {
      if (error instanceof Qwen35LanguageConverterError) throw error;
      fail("converter-source-parse-failed", "Converter source directory is invalid");
    }
    validateParsedSource(parsed, plan, contract);
    if (arguments_.mode !== "convert") {
      return reportFor(arguments_.mode, contract, plan);
    }
    const publishedFiles = await publishPackage({
      outputDirectory,
      reader,
      parsed,
      plan,
      contract,
      sourceFacts: initialFacts,
      ...(options.maxBlocksPerRead === undefined
        ? {}
        : { maxBlocksPerRead: options.maxBlocksPerRead }),
    });
    return reportFor(arguments_.mode, contract, plan, publishedFiles);
  } finally {
    await handle.close().catch(() => undefined);
  }
}

function publicFailure(error: unknown): Qwen35LanguageConverterError {
  return error instanceof Qwen35LanguageConverterError
    ? error
    : new Qwen35LanguageConverterError(
        "converter-failed",
        "Qwen3.5 language package conversion failed",
      );
}

const invokedPath = process.argv[1];
if (
  invokedPath !== undefined &&
  import.meta.url === pathToFileURL(resolve(invokedPath)).href
) {
  try {
    const report = await convertQwen35LanguagePackage(
      parseQwen35LanguageConverterArguments(process.argv.slice(2)),
    );
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  } catch (error) {
    const safe = publicFailure(error);
    process.stderr.write(`${safe.code}: ${safe.message}\n`);
    process.exitCode = 1;
  }
}
