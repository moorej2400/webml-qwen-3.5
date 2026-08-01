import { createHash } from "node:crypto";
import {
  lstat,
  mkdir,
  open,
  realpath,
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
  type PlannedSegment,
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
  type VisionProcessorSettings,
} from "../src/manifest.js";

const RUNTIME_ABI = "qwen35-webgpu-vision-v1";
const DEFAULT_MAX_SHARD_BYTES = 64n * 1024n * 1024n;
const DEFAULT_TENSOR_ALIGNMENT = 256;

const GGML_TYPE_NAMES = new Map<GgmlType, string>([
  [GgmlType.F32, "F32"],
  [GgmlType.BF16, "BF16"],
  [GgmlType.F16, "F16"],
  [GgmlType.Q8_0, "Q8_0"],
  [GgmlType.Q3_K, "Q3_K"],
  [GgmlType.Q4_K, "Q4_K"],
  [GgmlType.Q5_K, "Q5_K"],
  [GgmlType.Q6_K, "Q6_K"],
]);

export interface Qwen35VisionSourceContract {
  readonly source: ImmutableArtifactIdentity;
  readonly tokenizer: ImmutableArtifactIdentity;
  readonly processor: ImmutableArtifactIdentity;
  readonly processorSettings: VisionProcessorSettings;
  readonly dataOffset: bigint;
  readonly alignment: number;
  readonly tensorCount: number;
  readonly layerCount: number;
  readonly tensorTypeCounts: Readonly<Record<string, number>>;
  readonly ggufSettings: Readonly<{
    embeddingLength: number;
    feedForwardLength: number;
    headCount: number;
    projectionDim: number;
    imageSize: number;
    patchSize: number;
    spatialMergeSize: number;
    projectorType: string;
    useGelu: boolean;
    attentionLayerNormEpsilon: number;
  }>;
}

export const PINNED_QWEN35_VISION_CONVERTER_SOURCE:
Readonly<Qwen35VisionSourceContract> = Object.freeze({
  source: Object.freeze({
    repository: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF",
    revision: "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
    file: "mmproj-Qwen_Qwen3.5-4B-bf16.gguf",
    size: "675569216",
    sha256: "463f39bd1c291c1186c319a8c90ff8640aafa678b14cbee2232d695113dfbb66",
  }),
  tokenizer: Object.freeze({
    repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
    revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    file: "tokenizer.json",
    size: "12807982",
    sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
  }),
  processor: Object.freeze({
    repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
    revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    file: "preprocessor_config.json",
    size: "390",
    sha256: "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
  }),
  processorSettings: Object.freeze({
    processorClass: "Qwen3VLProcessor",
    imageProcessorType: "Qwen2VLImageProcessorFast",
    patchSize: 16,
    temporalPatchSize: 2,
    mergeSize: 2,
    shortestEdge: 65_536,
    longestEdge: 16_777_216,
    imageMean: Object.freeze([0.5, 0.5, 0.5]),
    imageStd: Object.freeze([0.5, 0.5, 0.5]),
  }),
  dataOffset: 17_984n,
  alignment: 32,
  tensorCount: 298,
  layerCount: 24,
  tensorTypeCounts: Object.freeze({ F32: 200, BF16: 98 }),
  ggufSettings: Object.freeze({
    embeddingLength: 1024,
    feedForwardLength: 4096,
    headCount: 16,
    projectionDim: 2560,
    imageSize: 768,
    patchSize: 16,
    spatialMergeSize: 2,
    projectorType: "qwen3vl_merger",
    useGelu: true,
    attentionLayerNormEpsilon: 0.0000009999999974752427,
  }),
});

export type Qwen35VisionConverterMode = "convert" | "dry-run" | "inventory";

export interface Qwen35VisionConverterArguments {
  readonly sourcePath: string;
  readonly outputDirectory: string;
  readonly mode: Qwen35VisionConverterMode;
}

export interface Qwen35VisionConverterOptions {
  readonly maxShardBytes?: bigint;
  readonly tensorAlignment?: number;
  readonly maxBlocksPerRead?: number;
  /** Test hook for source-mutation checks between planning and publication. */
  readonly afterPlanning?: () => Promise<void>;
  /** Test hook for proving output ownership before any package bytes are written. */
  readonly afterOutputReservation?: (outputDirectory: string) => Promise<void>;
}

export interface Qwen35VisionConverterReport {
  readonly format: "webml-qwen-vision-converter-report";
  readonly version: 1;
  readonly mode: Qwen35VisionConverterMode;
  readonly source: ImmutableArtifactIdentity;
  readonly inventory: readonly {
    readonly ggmlType: string;
    readonly ggmlTypeId: number;
    readonly tensorCount: number;
    readonly sourceBytes: string;
  }[];
  readonly layerShards: readonly { readonly layer: string; readonly shards: readonly number[] }[];
  readonly plan?: { readonly shardCount: number; readonly segmentCount: number; readonly outputBytes: string };
  readonly published?: { readonly files: readonly string[] };
}

export class Qwen35VisionConverterError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "Qwen35VisionConverterError";
  }
}

function fail(code: string, message: string): never {
  throw new Qwen35VisionConverterError(code, message);
}

export function parseQwen35VisionConverterArguments(
  arguments_: readonly string[],
): Qwen35VisionConverterArguments {
  const paths = new Map<string, string>();
  let mode: Qwen35VisionConverterMode = "convert";
  let modeWasSet = false;
  for (let index = 0; index < arguments_.length; index += 1) {
    const name = arguments_[index];
    if (name === "--dry-run" || name === "--inventory") {
      if (modeWasSet) fail("vision-converter-arguments-invalid", "Vision converter accepts only one mode");
      mode = name === "--dry-run" ? "dry-run" : "inventory";
      modeWasSet = true;
      continue;
    }
    if (name !== "--source" && name !== "--output") {
      fail("vision-converter-arguments-invalid", "Vision converter received an unknown argument");
    }
    const value = arguments_[index + 1];
    if (value === undefined || value.length === 0 || value.startsWith("--") || paths.has(name)) {
      fail("vision-converter-arguments-invalid", "Vision converter received an invalid path argument");
    }
    paths.set(name, value);
    index += 1;
  }
  const sourcePath = paths.get("--source");
  const outputDirectory = paths.get("--output");
  if (sourcePath === undefined || outputDirectory === undefined) {
    fail("vision-converter-arguments-invalid", "Vision converter requires --source and --output");
  }
  return Object.freeze({ sourcePath, outputDirectory, mode });
}

class FileReader implements RandomAccessReader {
  constructor(readonly handle: FileHandle, readonly size: bigint) {}

  async read(offset: bigint, length: number): Promise<Uint8Array> {
    if (!Number.isSafeInteger(length) || length < 0 || length > MAX_STREAM_READ_BYTES ||
      offset < 0n || offset > BigInt(Number.MAX_SAFE_INTEGER) || offset + BigInt(length) > this.size) {
      fail("vision-converter-source-read-failed", "Vision converter source read failed");
    }
    const bytes = Buffer.allocUnsafe(length);
    let completed = 0;
    try {
      while (completed < length) {
        const result = await this.handle.read(bytes, completed, length - completed, Number(offset) + completed);
        if (result.bytesRead === 0) fail("vision-converter-source-read-failed", "Vision converter source read failed");
        completed += result.bytesRead;
      }
    } catch (error) {
      if (error instanceof Qwen35VisionConverterError) throw error;
      fail("vision-converter-source-read-failed", "Vision converter source read failed");
    }
    return bytes;
  }
}

class ShardWriter implements RandomAccessWriter {
  constructor(readonly handle: FileHandle) {}

  async write(offset: bigint, bytes: Uint8Array): Promise<void> {
    if (offset < 0n || offset > BigInt(Number.MAX_SAFE_INTEGER) || bytes.byteLength > MAX_STREAM_READ_BYTES) {
      fail("vision-converter-shard-write-failed", "Vision converter shard write failed");
    }
    let completed = 0;
    try {
      while (completed < bytes.byteLength) {
        const result = await this.handle.write(bytes, completed, bytes.byteLength - completed, Number(offset) + completed);
        if (result.bytesWritten === 0) fail("vision-converter-shard-write-failed", "Vision converter shard write failed");
        completed += result.bytesWritten;
      }
    } catch (error) {
      if (error instanceof Qwen35VisionConverterError) throw error;
      fail("vision-converter-shard-write-failed", "Vision converter shard write failed");
    }
  }
}

async function pathExists(path: string): Promise<boolean> {
  try { await lstat(path); return true; } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    fail("vision-converter-output-check-failed", "Vision converter output could not be checked");
  }
}

async function resolveOutsideGitOutput(outputDirectory: string): Promise<string> {
  const requested = resolve(outputDirectory);
  if (basename(requested) === ".git") {
    fail("vision-converter-output-invalid", "Vision converter output directory is invalid");
  }
  if (await pathExists(requested)) {
    fail("vision-converter-output-exists", "Vision converter output directory already exists");
  }
  let parent: string;
  try { parent = await realpath(dirname(requested)); } catch {
    fail("vision-converter-output-parent-invalid", "Vision converter output parent directory must already exist");
  }
  for (let current = parent; ; current = dirname(current)) {
    if (await pathExists(join(current, ".git"))) {
      fail("vision-converter-output-inside-git", "Vision converter output must be outside a Git worktree");
    }
    if (dirname(current) === current) break;
  }
  return join(parent, basename(requested));
}

async function reserveOutputDirectory(outputDirectory: string): Promise<void> {
  // A final rename can replace a directory created after validation; mkdir gives
  // this conversion exclusive ownership before any package bytes are written.
  try {
    await mkdir(outputDirectory);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EEXIST") {
      fail("vision-converter-output-exists", "Vision converter output directory already exists");
    }
    fail("vision-converter-output-reserve-failed", "Vision converter output directory could not be reserved");
  }
}

async function fileFacts(handle: FileHandle): Promise<Readonly<{ dev: bigint; ino: bigint; size: bigint; mtimeNs: bigint; ctimeNs: bigint }>> {
  try {
    const facts = await handle.stat({ bigint: true });
    if (!facts.isFile()) fail("vision-converter-source-invalid", "Vision converter source must be a regular file");
    return { dev: facts.dev, ino: facts.ino, size: facts.size, mtimeNs: facts.mtimeNs, ctimeNs: facts.ctimeNs };
  } catch (error) {
    if (error instanceof Qwen35VisionConverterError) throw error;
    fail("vision-converter-source-stat-failed", "Vision converter source could not be inspected");
  }
}

function sameFacts(left: Awaited<ReturnType<typeof fileFacts>>, right: Awaited<ReturnType<typeof fileFacts>>): boolean {
  return left.dev === right.dev && left.ino === right.ino && left.size === right.size &&
    left.mtimeNs === right.mtimeNs && left.ctimeNs === right.ctimeNs;
}

async function hashReader(reader: FileReader): Promise<string> {
  const hash = createHash("sha256");
  for (let offset = 0n; offset < reader.size;) {
    const length = Number((reader.size - offset) < BigInt(MAX_STREAM_READ_BYTES)
      ? reader.size - offset : BigInt(MAX_STREAM_READ_BYTES));
    hash.update(await reader.read(offset, length));
    offset += BigInt(length);
  }
  return hash.digest("hex");
}

async function hashFile(path: string): Promise<string> {
  let handle: FileHandle;
  try { handle = await open(path, "r"); } catch { fail("vision-converter-checksum-failed", "Vision converter checksum generation failed"); }
  try {
    const facts = await fileFacts(handle);
    return await hashReader(new FileReader(handle, facts.size));
  } finally { await handle.close().catch(() => undefined); }
}

function typeName(type: GgmlType): string { return GGML_TYPE_NAMES.get(type) ?? `GGML_${type}`; }

function layerName(name: string): string {
  const match = /(?:^|\.)blk\.(\d+)(?:\.|$)/u.exec(name);
  return match === null ? "bootstrap" : match[1]!;
}

/** Orders bootstrap work first and numbered vision layers by numeric index. */
export function orderQwen35VisionLayerGroups(
  tensorNames: readonly string[],
): readonly string[] {
  return [...new Set(tensorNames.map(layerName))].sort((left, right) => {
    if (left === "bootstrap") return -1;
    if (right === "bootstrap") return 1;
    return Number(left) - Number(right);
  });
}

function groupPlans(parsed: ParsedGguf, maxShardBytes: bigint, tensorAlignment: number): {
  readonly plan: ConversionPlan;
  readonly layerShards: readonly { readonly layer: string; readonly shards: readonly number[] }[];
} {
  const groups = new Map<string, ParsedGguf["tensors"]>();
  for (const tensor of parsed.tensors) {
    const group = layerName(tensor.name);
    groups.set(group, [...(groups.get(group) ?? []), tensor]);
  }
  const orderedLayers = orderQwen35VisionLayerGroups(
    parsed.tensors.map((tensor) => tensor.name),
  );
  const plans: Array<{ layer: string; plan: ConversionPlan }> = [];
  // Layer names are numbers: lexical order would put layer 10 before layer 2
  // and make the streamed package order disagree with the layer index.
  for (const layer of orderedLayers) {
    const plan = planConversion(
      { ...parsed, tensors: groups.get(layer)! },
      { maxShardBytes, tensorAlignment },
    );
    if (plan.excludedTensors.length !== 0) {
      fail(
        "vision-converter-source-contract-mismatch",
        "Vision converter source must not apply the language MTP exclusion policy",
      );
    }
    plans.push({ layer, plan });
  }
  const first = plans[0]?.plan;
  if (first === undefined) fail("vision-converter-source-contract-mismatch", "Vision converter source has no tensor groups");
  const shards: Array<{ index: number; length: bigint }> = [];
  const segments: PlannedSegment[] = [];
  const inventory = new Map<GgmlType, { tensorCount: number; sourceBytes: bigint }>();
  const layerShards: Array<{ layer: string; shards: number[] }> = [];
  for (const item of plans) {
    const shardBase = shards.length;
    for (const shard of item.plan.shards) shards.push({ index: shard.index + shardBase, length: shard.length });
    for (const segment of item.plan.segments) segments.push({ ...segment, shard: segment.shard + shardBase });
    for (const entry of item.plan.inventory) {
      const previous = inventory.get(entry.ggmlType) ?? { tensorCount: 0, sourceBytes: 0n };
      inventory.set(entry.ggmlType, { tensorCount: previous.tensorCount + entry.tensorCount, sourceBytes: previous.sourceBytes + entry.sourceBytes });
    }
    layerShards.push({ layer: item.layer, shards: item.plan.shards.map((shard) => shard.index + shardBase) });
  }
  return {
    plan: { inventory: [...inventory.entries()].sort(([a], [b]) => a - b).map(([ggmlType, value]) => ({ ggmlType, ...value })), excludedTensors: [], shards, segments },
    layerShards: Object.freeze(layerShards.map((entry) => Object.freeze({ layer: entry.layer, shards: Object.freeze(entry.shards) }))),
  };
}

function validateParsedSource(parsed: ParsedGguf, plan: ConversionPlan, layerShards: readonly { readonly layer: string; readonly shards: readonly number[] }[], contract: Qwen35VisionSourceContract): void {
  const inventory = Object.fromEntries(plan.inventory.map((entry) => [typeName(entry.ggmlType), entry.tensorCount]));
  const expectedLayers = Array.from({ length: contract.layerCount }, (_, index) => String(index));
  const layers = layerShards.filter((entry) => entry.layer !== "bootstrap").map((entry) => entry.layer);
  const settings = contract.ggufSettings;
  const epsilon = parsed.metadata["clip.vision.attention.layer_norm_epsilon"];
  if (parsed.dataOffset !== contract.dataOffset || parsed.alignment !== contract.alignment ||
    parsed.tensors.length !== contract.tensorCount || parsed.metadata["general.architecture"] !== "clip" ||
    parsed.metadata["general.type"] !== "mmproj" || parsed.metadata["clip.vision.block_count"] !== contract.layerCount ||
    parsed.metadata["clip.vision.embedding_length"] !== settings.embeddingLength ||
    parsed.metadata["clip.vision.feed_forward_length"] !== settings.feedForwardLength ||
    parsed.metadata["clip.vision.attention.head_count"] !== settings.headCount ||
    parsed.metadata["clip.vision.projection_dim"] !== settings.projectionDim ||
    parsed.metadata["clip.vision.image_size"] !== settings.imageSize ||
    parsed.metadata["clip.vision.patch_size"] !== settings.patchSize ||
    parsed.metadata["clip.vision.spatial_merge_size"] !== settings.spatialMergeSize ||
    parsed.metadata["clip.projector_type"] !== settings.projectorType ||
    parsed.metadata["clip.use_gelu"] !== settings.useGelu ||
    typeof epsilon !== "number" || Math.abs(epsilon - settings.attentionLayerNormEpsilon) > 1e-12 ||
    JSON.stringify(Object.entries(inventory).sort()) !== JSON.stringify(Object.entries(contract.tensorTypeCounts).sort()) ||
    JSON.stringify(layers) !== JSON.stringify(expectedLayers) || layerShards[0]?.layer !== "bootstrap") {
    fail("vision-converter-source-contract-mismatch", "Vision converter source does not match the pinned Qwen3.5 directory contract");
  }
  const owners = new Set<number>();
  for (const group of layerShards) for (const shard of group.shards) {
    if (owners.has(shard)) fail("vision-converter-plan-invalid", "Vision converter created a shard with multiple layer owners");
    owners.add(shard);
  }
}

function reportFor(mode: Qwen35VisionConverterMode, contract: Qwen35VisionSourceContract, plan: ConversionPlan, layerShards: readonly { readonly layer: string; readonly shards: readonly number[] }[], publishedFiles?: readonly string[]): Qwen35VisionConverterReport {
  const inventory = plan.inventory.map((entry) => Object.freeze({ ggmlType: typeName(entry.ggmlType), ggmlTypeId: entry.ggmlType, tensorCount: entry.tensorCount, sourceBytes: entry.sourceBytes.toString() }));
  const planSummary = mode === "inventory" ? undefined : Object.freeze({ shardCount: plan.shards.length, segmentCount: plan.segments.length, outputBytes: plan.shards.reduce((total, shard) => total + shard.length, 0n).toString() });
  return Object.freeze({ format: "webml-qwen-vision-converter-report" as const, version: 1 as const, mode, source: contract.source, inventory: Object.freeze(inventory), layerShards, ...(planSummary === undefined ? {} : { plan: planSummary }), ...(publishedFiles === undefined ? {} : { published: Object.freeze({ files: Object.freeze([...publishedFiles]) }) }) });
}

function metadataDocument(parsed: ParsedGguf, plan: ConversionPlan, layerShards: readonly { readonly layer: string; readonly shards: readonly number[] }[], contract: Qwen35VisionSourceContract): string {
  return `${JSON.stringify({ format: "webml-qwen-vision-source-provenance", version: 1, source: contract.source, tokenizer: contract.tokenizer, processor: contract.processor, processorSettings: contract.processorSettings, ggufSettings: contract.ggufSettings, runtimeAbi: RUNTIME_ABI, observed: { alignment: parsed.alignment, dataOffset: parsed.dataOffset.toString(), tensorCount: parsed.tensors.length, inventory: plan.inventory.map((entry) => ({ ggmlType: typeName(entry.ggmlType), ggmlTypeId: entry.ggmlType, tensorCount: entry.tensorCount, sourceBytes: entry.sourceBytes.toString() })), layerShards } }, null, 2)}\n`;
}

function layerIndexDocument(layerShards: readonly { readonly layer: string; readonly shards: readonly number[] }[]): string {
  return `${JSON.stringify({ format: "webml-qwen-vision-layer-index", version: 1, groups: layerShards }, null, 2)}\n`;
}

function licenseDocument(contract: Qwen35VisionSourceContract): string {
  return `${JSON.stringify({ format: "webml-qwen-license-metadata", version: 1, materials: [{ name: "Qwen3.5 4B vision model materials and converted weights", spdx: "Apache-2.0", licenseUrl: `${contract.processor.repository}/blob/${contract.processor.revision}/LICENSE`, notice: "Distribution remains subject to the upstream model license." }] }, null, 2)}\n`;
}

async function publishPackage(input: { outputDirectory: string; reader: FileReader; parsed: ParsedGguf; plan: ConversionPlan; layerShards: readonly { readonly layer: string; readonly shards: readonly number[] }[]; contract: Qwen35VisionSourceContract; sourceFacts: Awaited<ReturnType<typeof fileFacts>>; maxBlocksPerRead?: number }): Promise<readonly string[]> {
  const shardDirectory = join(input.outputDirectory, "shards");
  try { await mkdir(shardDirectory); } catch { fail("vision-converter-staging-failed", "Vision converter shard directory could not be created"); }
  const shardFiles = input.plan.shards.map((shard) => `shards/vision-${String(shard.index).padStart(5, "0")}.bin`);
  const handles: FileHandle[] = [];
  try {
    for (const file of shardFiles) handles.push(await open(join(input.outputDirectory, file), "wx"));
    await executeConversionPlan(input.plan, input.reader, handles.map((handle) => new ShardWriter(handle)), input.maxBlocksPerRead === undefined ? {} : { maxBlocksPerRead: input.maxBlocksPerRead });
  } catch (error) {
    if (error instanceof Qwen35VisionConverterError) throw error;
    fail("vision-converter-conversion-failed", "Vision converter shard generation failed");
  } finally {
    const closed = await Promise.allSettled(handles.map((handle) => handle.close()));
    if (closed.some((result) => result.status === "rejected")) fail("vision-converter-shard-close-failed", "Vision converter shard close failed");
  }
  if (!sameFacts(input.sourceFacts, await fileFacts(input.reader.handle))) {
    fail("vision-converter-source-changed", "Vision converter source changed during conversion");
  }
  const artifacts: Array<{ url: string; sha256: string }> = [];
  // Hash one shard at a time so package verification never creates a
  // model-sized set of simultaneous streaming read buffers.
  for (const file of shardFiles) {
    artifacts.push({
      url: file,
      sha256: await hashFile(join(input.outputDirectory, file)),
    });
  }
  const manifest = createManifestFromPlan(input.plan, { packageKind: "vision", source: input.contract.source, runtimeAbi: RUNTIME_ABI, tokenizer: input.contract.tokenizer, processor: input.contract.processor, processorSettings: input.contract.processorSettings, shards: artifacts });
  try {
    await writeFile(join(input.outputDirectory, "manifest.json"), stringifyManifest(manifest), { encoding: "utf8", flag: "wx" });
    await writeFile(join(input.outputDirectory, "layer-index.json"), layerIndexDocument(input.layerShards), { encoding: "utf8", flag: "wx" });
    await writeFile(join(input.outputDirectory, "source-provenance.json"), metadataDocument(input.parsed, input.plan, input.layerShards, input.contract), { encoding: "utf8", flag: "wx" });
    await writeFile(join(input.outputDirectory, "LICENSES.json"), licenseDocument(input.contract), { encoding: "utf8", flag: "wx" });
  } catch { fail("vision-converter-metadata-write-failed", "Vision converter metadata write failed"); }
  const checksummed = ["LICENSES.json", "layer-index.json", "manifest.json", ...shardFiles, "source-provenance.json"].sort();
  try {
    const lines: string[] = [];
    for (const file of checksummed) {
      lines.push(`${await hashFile(join(input.outputDirectory, file))}  ${file}`);
    }
    await writeFile(
      join(input.outputDirectory, "SHA256SUMS"),
      `${lines.join("\n")}\n`,
      { encoding: "utf8", flag: "wx" },
    );
  } catch { fail("vision-converter-publish-failed", "Vision converter package publication failed"); }
  return Object.freeze(["LICENSES.json", "SHA256SUMS", "layer-index.json", "manifest.json", ...shardFiles, "source-provenance.json"]);
}

/** Converts authenticated vision tensors in bounded ranges, keeping layer groups isolated for OPFS streaming. */
export async function convertQwen35VisionPackage(arguments_: Qwen35VisionConverterArguments, contract: Qwen35VisionSourceContract = PINNED_QWEN35_VISION_CONVERTER_SOURCE, options: Qwen35VisionConverterOptions = {}): Promise<Qwen35VisionConverterReport> {
  const outputDirectory = await resolveOutsideGitOutput(arguments_.outputDirectory);
  let handle: FileHandle;
  try { handle = await open(resolve(arguments_.sourcePath), "r"); } catch { fail("vision-converter-source-open-failed", "Vision converter source could not be opened"); }
  try {
    const initialFacts = await fileFacts(handle);
    if (initialFacts.size.toString() !== contract.source.size) fail("vision-converter-source-size-mismatch", "Vision converter source size does not match the pinned identity");
    const reader = new FileReader(handle, initialFacts.size);
    if (await hashReader(reader) !== contract.source.sha256) fail("vision-converter-source-hash-mismatch", "Vision converter source SHA-256 does not match the pinned identity");
    if (!sameFacts(initialFacts, await fileFacts(handle))) fail("vision-converter-source-changed", "Vision converter source changed during authentication");
    let parsed: ParsedGguf;
    let grouped: ReturnType<typeof groupPlans>;
    try {
      parsed = await parseGguf(reader);
      grouped = groupPlans(parsed, options.maxShardBytes ?? DEFAULT_MAX_SHARD_BYTES, options.tensorAlignment ?? DEFAULT_TENSOR_ALIGNMENT);
    } catch (error) {
      if (error instanceof Qwen35VisionConverterError) throw error;
      fail("vision-converter-source-parse-failed", "Vision converter source directory is invalid");
    }
    validateParsedSource(parsed, grouped.plan, grouped.layerShards, contract);
    if (options.afterPlanning !== undefined) {
      await options.afterPlanning();
    }
    // Inventory and dry-run also report source-derived facts, so they must not
    // return a plan from a file that changed after its authenticated hash.
    if (!sameFacts(initialFacts, await fileFacts(handle))) {
      fail("vision-converter-source-changed", "Vision converter source changed during planning");
    }
    if (arguments_.mode !== "convert") return reportFor(arguments_.mode, contract, grouped.plan, grouped.layerShards);
    await reserveOutputDirectory(outputDirectory);
    if (options.afterOutputReservation !== undefined) {
      await options.afterOutputReservation(outputDirectory);
    }
    const published = await publishPackage({ outputDirectory, reader, parsed, plan: grouped.plan, layerShards: grouped.layerShards, contract, sourceFacts: initialFacts, ...(options.maxBlocksPerRead === undefined ? {} : { maxBlocksPerRead: options.maxBlocksPerRead }) });
    return reportFor(arguments_.mode, contract, grouped.plan, grouped.layerShards, published);
  } finally { await handle.close().catch(() => undefined); }
}

function publicFailure(error: unknown): Qwen35VisionConverterError {
  return error instanceof Qwen35VisionConverterError ? error : new Qwen35VisionConverterError("vision-converter-failed", "Qwen3.5 vision package conversion failed");
}

const invokedPath = process.argv[1];
if (invokedPath !== undefined && import.meta.url === pathToFileURL(resolve(invokedPath)).href) {
  try { process.stdout.write(`${JSON.stringify(await convertQwen35VisionPackage(parseQwen35VisionConverterArguments(process.argv.slice(2))), null, 2)}\n`); }
  catch (error) { const safe = publicFailure(error); process.stderr.write(`${safe.code}: ${safe.message}\n`); process.exitCode = 1; }
}
