import type {
  ImmutableRangeSource,
  RangeChunkConsumer,
} from "./http-range-reader.js";
import { IncrementalSha256 } from "./incremental-sha256.js";
import {
  stringifyManifest,
  validateModelPackageManifest,
  type ModelPackageManifest,
  type PackageShard,
} from "./manifest.js";

const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: true });
const SAFE_PATH_SEGMENT = /^[a-z0-9][a-z0-9.-]{0,127}$/;
const SHA256 = /^[a-f0-9]{64}$/;
const REVISION = /^[a-f0-9]{40}$/;
const MAX_READY_RECORD_BYTES = 1024 * 1024;
const MAX_READY_GENERATIONS = 1_024;
const CACHE_ENUMERATION_LIMITS: CacheEnumerationLimits = Object.freeze({
  maxDepth: 8,
  maxEntries: 4_096,
});

export interface CacheAtomicWriter {
  write(chunk: Uint8Array): Promise<void>;
  commit(): Promise<void>;
  /** Makes an interrupted temp recoverable instead of deleting it. */
  preserveIncomplete(): Promise<void>;
}

/**
 * Small OPFS boundary used by the cache and in-memory crash tests.
 *
 * `commit()` must publish all writes as one file version. OPFS
 * FileSystemWritableFileStream provides this behavior when close() settles.
 */
export interface ModelCacheStorage {
  openAtomicWriter(path: string): Promise<CacheAtomicWriter>;
  openRead(path: string): Promise<AsyncIterable<Uint8Array> | null>;
  move(source: string, destination: string): Promise<boolean>;
  list(
    prefix: string,
    limits: CacheEnumerationLimits,
  ): Promise<readonly string[]>;
}

export interface CacheEnumerationLimits {
  maxDepth: number;
  maxEntries: number;
}

export interface ShardByteStreamer {
  stream(
    source: ImmutableRangeSource,
    consume: RangeChunkConsumer,
    signal?: AbortSignal,
  ): Promise<void>;
}

export type CacheSourceResolver = (
  shard: PackageShard,
  index: number,
) => ImmutableRangeSource;

export interface CachedShard {
  storagePath: string;
  byteLength: number;
  sha256: string;
}

export interface CachedModelPackage {
  cacheKey: string;
  manifestSha256: string;
  cacheHit: boolean;
  shards: readonly CachedShard[];
}

export interface ModelCacheMetrics {
  cacheHits: number;
  cacheMisses: number;
  bytesWritten: number;
  verifiedParts: number;
  hashMilliseconds: number;
  recoveredArtifacts: number;
}

export interface ImmutableOpfsModelCacheOptions {
  attemptId?: () => string;
  now?: () => number;
}

export class ModelCacheError extends Error {
  constructor(readonly code: string) {
    super(code);
    this.name = "ModelCacheError";
  }
}

interface ReadyRecord {
  format: "webml-qwen-cache-ready";
  version: 1;
  cacheKey: string;
  manifestSha256: string;
  sourceRevision: string;
  sourceSha256: string;
  sourceSize: string;
  runtimeAbi: string;
  shards: CachedShard[];
}

interface VerifiedFile {
  byteLength: number;
  sha256: string;
}

export function modelCacheKey(manifest: ModelPackageManifest): string {
  const canonical = stringifyManifest(
    validateModelPackageManifest(manifest),
  );
  return new IncrementalSha256()
    .update(encoder.encode(canonical))
    .digestHex();
}

/**
 * Publishes immutable OPFS generations by writing the ready record last.
 *
 * Interrupted weights remain as unreferenced attempt-unique blobs; staging
 * metadata remains under tmp/ or stale/. Cache discovery reads only fully
 * parsed ready records whose identities and hashes match, so an orphan can
 * never become a cache hit or require a quota-doubling copy.
 */
export class ImmutableOpfsModelCache {
  private readonly makeAttemptId: () => string;
  private readonly now: () => number;
  private cacheHits = 0;
  private cacheMisses = 0;
  private bytesWritten = 0;
  private verifiedParts = 0;
  private hashMilliseconds = 0;
  private recoveredArtifacts = 0;

  constructor(
    private readonly storage: ModelCacheStorage,
    private readonly streamer: ShardByteStreamer,
    options: ImmutableOpfsModelCacheOptions = {},
  ) {
    this.makeAttemptId =
      options.attemptId ??
      (() => crypto.randomUUID().replaceAll("-", "").toLowerCase());
    this.now = options.now ?? (() => performance.now());
  }

  get metrics(): ModelCacheMetrics {
    return {
      cacheHits: this.cacheHits,
      cacheMisses: this.cacheMisses,
      bytesWritten: this.bytesWritten,
      verifiedParts: this.verifiedParts,
      hashMilliseconds: this.hashMilliseconds,
      recoveredArtifacts: this.recoveredArtifacts,
    };
  }

  async ensure(
    manifest: ModelPackageManifest,
    resolveSource: CacheSourceResolver,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<CachedModelPackage> {
    validateModelPackageManifest(manifest);
    const key = modelCacheKey(manifest);
    const existing = await this.findReady(manifest, key, signal);
    if (existing !== null) {
      this.cacheHits += 1;
      return { ...existing, cacheHit: true };
    }
    this.cacheMisses += 1;

    const attempt = this.makeAttemptId();
    if (!SAFE_PATH_SEGMENT.test(attempt)) {
      throw new ModelCacheError("attempt-id-invalid");
    }
    const attemptPrefix = `tmp/${key}/${attempt}`;
    await this.recoverTemps(key, attempt);
    const planned = manifest.shards.map((shard, index) => ({
      storagePath: `blobs/${shard.sha256}/${attempt}-${index
        .toString()
        .padStart(5, "0")}.bin`,
      byteLength: safeByteLength(shard.length),
      sha256: shard.sha256,
    }));
    await this.requireUnusedAttempt(attemptPrefix, planned);
    await this.writeStagingManifest(
      manifest,
      key,
      attemptPrefix,
      planned,
    );

    try {
      for (const [index, shard] of manifest.shards.entries()) {
        signal.throwIfAborted();
        const expectedLength = safeByteLength(shard.length);
        const source = resolveSource(shard, index);
        if (source.byteLength !== expectedLength) {
          throw new ModelCacheError("source-length-mismatch");
        }
        await this.downloadImmutablePart(
          planned[index]!.storagePath,
          source,
          expectedLength,
          shard.sha256,
          signal,
        );
      }

      const promoted: CachedShard[] = [];
      for (const part of planned) {
        signal.throwIfAborted();
        const verified = await this.verifyFile(part.storagePath, signal);
        if (
          verified.byteLength !== part.byteLength ||
          verified.sha256 !== part.sha256
        ) {
          throw new ModelCacheError("published-part-invalid");
        }
        promoted.push({
          storagePath: part.storagePath,
          byteLength: part.byteLength,
          sha256: part.sha256,
        });
      }

      const manifestSha256 = key;
      const record: ReadyRecord = {
        format: "webml-qwen-cache-ready",
        version: 1,
        cacheKey: key,
        manifestSha256,
        sourceRevision: manifest.source.revision,
        sourceSha256: manifest.source.sha256,
        sourceSize: manifest.source.size,
        runtimeAbi: manifest.runtime.abi,
        shards: promoted,
      };
      const recordBytes = encoder.encode(JSON.stringify(record));
      const recordHash = new IncrementalSha256()
        .update(recordBytes)
        .digestHex();
      const readyPath = `ready/${key}/${recordHash}.json`;
      await this.writeAtomic(readyPath, [recordBytes]);

      const publishedRecord = await readSmall(
        this.storage,
        readyPath,
        MAX_READY_RECORD_BYTES,
      );
      if (
        publishedRecord === null ||
        !equalBytes(publishedRecord, recordBytes)
      ) {
        throw new ModelCacheError("ready-publication-invalid");
      }
      return {
        cacheKey: key,
        manifestSha256,
        cacheHit: false,
        shards: promoted,
      };
    } catch (error) {
      await this.quarantineAttempt(attemptPrefix, key, attempt);
      throw error;
    }
  }

  private async findReady(
    manifest: ModelPackageManifest,
    key: string,
    signal: AbortSignal,
  ): Promise<Omit<CachedModelPackage, "cacheHit"> | null> {
    const paths = (
      await this.storage.list(
        `ready/${key}/`,
        CACHE_ENUMERATION_LIMITS,
      )
    )
      .filter((path) => path.endsWith(".json"))
      .sort()
      .slice(0, MAX_READY_GENERATIONS);
    for (const path of paths) {
      signal.throwIfAborted();
      try {
        const bytes = await readSmall(
          this.storage,
          path,
          MAX_READY_RECORD_BYTES,
        );
        if (bytes === null) {
          continue;
        }
        const record = parseReadyRecord(bytes, manifest, key);
        let valid = true;
        for (const shard of record.shards) {
          const verified = await this.verifyFile(shard.storagePath, signal);
          if (
            verified.byteLength !== shard.byteLength ||
            verified.sha256 !== shard.sha256
          ) {
            valid = false;
            break;
          }
        }
        if (valid) {
          return {
            cacheKey: key,
            manifestSha256: key,
            shards: record.shards,
          };
        }
      } catch (error) {
        if (signal.aborted) {
          throw error;
        }
        // Invalid generations are immutable evidence. Ignore them and publish
        // a separate generation instead of overwriting or deleting user data.
      }
    }
    return null;
  }

  private async writeStagingManifest(
    manifest: ModelPackageManifest,
    key: string,
    attemptPrefix: string,
    planned: readonly CachedShard[],
  ): Promise<void> {
    const staging = encoder.encode(
      JSON.stringify({
        format: "webml-qwen-cache-staging",
        version: 1,
        cacheKey: key,
        sourceRevision: manifest.source.revision,
        sourceSha256: manifest.source.sha256,
        sourceSize: manifest.source.size,
        partCount: manifest.shards.length,
        parts: planned,
      }),
    );
    await this.writeAtomic(`${attemptPrefix}/manifest.json`, [staging]);
  }

  private async requireUnusedAttempt(
    attemptPrefix: string,
    planned: readonly CachedShard[],
  ): Promise<void> {
    const existingTemp = await this.storage.list(
      `${attemptPrefix}/`,
      CACHE_ENUMERATION_LIMITS,
    );
    if (existingTemp.length > 0) {
      throw new ModelCacheError("cache-attempt-collision");
    }
    for (const part of planned) {
      if ((await this.storage.openRead(part.storagePath)) !== null) {
        throw new ModelCacheError("cache-attempt-collision");
      }
    }
  }

  private async downloadImmutablePart(
    path: string,
    source: ImmutableRangeSource,
    expectedLength: number,
    expectedSha256: string,
    signal: AbortSignal,
  ): Promise<void> {
    const writer = await this.storage.openAtomicWriter(path);
    const hasher = new IncrementalSha256();
    let byteLength = 0;
    try {
      await this.streamer.stream(
        source,
        async (chunk) => {
          signal.throwIfAborted();
          this.updateHash(hasher, chunk);
          await writer.write(chunk);
          byteLength += chunk.byteLength;
          this.bytesWritten += chunk.byteLength;
        },
        signal,
      );
      await writer.commit();
    } catch (error) {
      await writer.preserveIncomplete();
      throw error;
    }
    const actualSha256 = hasher.digestHex();
    this.verifiedParts += 1;
    if (
      byteLength !== expectedLength ||
      actualSha256 !== expectedSha256
    ) {
      throw new ModelCacheError("shard-hash-mismatch");
    }
  }

  private async verifyFile(
    path: string,
    signal: AbortSignal,
  ): Promise<VerifiedFile> {
    const source = await this.storage.openRead(path);
    if (source === null) {
      return { byteLength: 0, sha256: "" };
    }
    const hasher = new IncrementalSha256();
    let byteLength = 0;
    for await (const chunk of source) {
      signal.throwIfAborted();
      this.updateHash(hasher, chunk);
      byteLength += chunk.byteLength;
    }
    const sha256 = hasher.digestHex();
    this.verifiedParts += 1;
    return { byteLength, sha256 };
  }

  private updateHash(hasher: IncrementalSha256, chunk: Uint8Array): void {
    const started = this.now();
    hasher.update(chunk);
    this.hashMilliseconds += Math.max(0, this.now() - started);
  }

  private async writeAtomic(
    path: string,
    chunks: readonly Uint8Array[],
  ): Promise<void> {
    const writer = await this.storage.openAtomicWriter(path);
    try {
      for (const chunk of chunks) {
        await writer.write(chunk);
      }
      await writer.commit();
    } catch (error) {
      await writer.preserveIncomplete();
      throw error;
    }
  }

  private async recoverTemps(key: string, currentAttempt: string): Promise<void> {
    const prefix = `tmp/${key}/`;
    for (const source of await this.storage.list(
      prefix,
      CACHE_ENUMERATION_LIMITS,
    )) {
      const relative = source.slice(prefix.length);
      if (relative.startsWith(`${currentAttempt}/`)) {
        continue;
      }
      const destination = `stale/${key}/${relative}`;
      if (await this.storage.move(source, destination)) {
        this.recoveredArtifacts += 1;
      }
    }
  }

  private async quarantineAttempt(
    attemptPrefix: string,
    key: string,
    attempt: string,
  ): Promise<void> {
    for (const source of await this.storage.list(
      `${attemptPrefix}/`,
      CACHE_ENUMERATION_LIMITS,
    )) {
      const relative = source.slice(attemptPrefix.length + 1);
      const destination = `stale/${key}/${attempt}/${relative}`;
      if (await this.storage.move(source, destination)) {
        this.recoveredArtifacts += 1;
      }
    }
  }
}

function safeByteLength(value: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new ModelCacheError("browser-byte-length-unsupported");
  }
  return parsed;
}

function parseReadyRecord(
  bytes: Uint8Array,
  manifest: ModelPackageManifest,
  key: string,
): ReadyRecord {
  let value: unknown;
  try {
    value = JSON.parse(decoder.decode(bytes));
  } catch {
    throw new ModelCacheError("ready-record-invalid");
  }
  if (typeof value !== "object" || value === null) {
    throw new ModelCacheError("ready-record-invalid");
  }
  const record = value as Partial<ReadyRecord>;
  if (
    record.format !== "webml-qwen-cache-ready" ||
    record.version !== 1 ||
    record.cacheKey !== key ||
    record.manifestSha256 !== key ||
    record.sourceRevision !== manifest.source.revision ||
    record.sourceSha256 !== manifest.source.sha256 ||
    record.sourceSize !== manifest.source.size ||
    record.runtimeAbi !== manifest.runtime.abi ||
    !REVISION.test(record.sourceRevision) ||
    !SHA256.test(record.sourceSha256) ||
    !Array.isArray(record.shards) ||
    record.shards.length !== manifest.shards.length
  ) {
    throw new ModelCacheError("ready-identity-mismatch");
  }

  const shards: CachedShard[] = [];
  for (const [index, entry] of record.shards.entries()) {
    const expected = manifest.shards[index]!;
    if (
      typeof entry !== "object" ||
      entry === null ||
      !SHA256.test(entry.sha256) ||
      entry.sha256 !== expected.sha256 ||
      entry.byteLength !== safeByteLength(expected.length) ||
      typeof entry.storagePath !== "string" ||
      !entry.storagePath.startsWith(`blobs/${entry.sha256}/`) ||
      !isSafeCachePath(entry.storagePath)
    ) {
      throw new ModelCacheError("ready-part-invalid");
    }
    shards.push({
      storagePath: entry.storagePath,
      byteLength: entry.byteLength,
      sha256: entry.sha256,
    });
  }
  return { ...(record as ReadyRecord), shards };
}

async function readSmall(
  storage: ModelCacheStorage,
  path: string,
  limit: number,
): Promise<Uint8Array | null> {
  const source = await storage.openRead(path);
  if (source === null) {
    return null;
  }
  const chunks: Uint8Array[] = [];
  let byteLength = 0;
  for await (const chunk of source) {
    byteLength += chunk.byteLength;
    if (byteLength > limit) {
      throw new ModelCacheError("small-file-limit-exceeded");
    }
    chunks.push(chunk);
  }
  const output = new Uint8Array(byteLength);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) {
    return false;
  }
  for (let index = 0; index < left.byteLength; index += 1) {
    if (left[index] !== right[index]) {
      return false;
    }
  }
  return true;
}

function isSafeCachePath(path: string): boolean {
  return path.split("/").every((segment) => SAFE_PATH_SEGMENT.test(segment));
}

/**
 * OPFS adapter. FileSystemWritableFileStream.close() commits through the
 * browser's temporary backing file, while move() is used only when the
 * implementation exposes the current OPFS rename extension.
 */
export class BrowserOpfsStorage implements ModelCacheStorage {
  private constructor(private readonly root: FileSystemDirectoryHandle) {}

  static async open(
    storageManager: StorageManager = navigator.storage,
  ): Promise<BrowserOpfsStorage> {
    return new BrowserOpfsStorage(await storageManager.getDirectory());
  }

  static fromRoot(
    root: FileSystemDirectoryHandle,
  ): BrowserOpfsStorage {
    return new BrowserOpfsStorage(root);
  }

  async openAtomicWriter(path: string): Promise<CacheAtomicWriter> {
    const { directory, name } = await this.parent(path, true);
    const handle = await directory.getFileHandle(name, { create: true });
    const stream = await handle.createWritable({ keepExistingData: false });
    let settled = false;
    const close = async (): Promise<void> => {
      if (!settled) {
        settled = true;
        await stream.close();
      }
    };
    return {
      write: async (chunk) => {
        if (settled) {
          throw new ModelCacheError("writer-already-settled");
        }
        await stream.write(fileSystemWriteChunk(chunk));
      },
      commit: close,
      // Closing an interrupted temp preserves it for a later stale move.
      preserveIncomplete: close,
    };
  }

  async openRead(path: string): Promise<AsyncIterable<Uint8Array> | null> {
    let handle: FileSystemFileHandle;
    try {
      const { directory, name } = await this.parent(path, false);
      handle = await directory.getFileHandle(name);
    } catch (error) {
      if (isNotFound(error)) {
        return null;
      }
      throw error;
    }
    const file = await handle.getFile();
    return (async function* streamFile(): AsyncIterable<Uint8Array> {
      const reader = file.stream().getReader();
      try {
        while (true) {
          const result = await reader.read();
          if (result.done) {
            return;
          }
          yield result.value;
        }
      } finally {
        reader.releaseLock();
      }
    })();
  }

  async move(source: string, destination: string): Promise<boolean> {
    const { directory: sourceDirectory, name: sourceName } =
      await this.parent(source, false);
    const sourceHandle = await sourceDirectory.getFileHandle(sourceName);
    const movable = sourceHandle as FileSystemFileHandle & {
      move?: (
        destinationDirectory: FileSystemDirectoryHandle,
        name: string,
      ) => Promise<void>;
    };
    if (typeof movable.move !== "function") {
      return false;
    }
    const { directory, name } = await this.parent(destination, true);
    await movable.move(directory, name);
    return true;
  }

  async list(
    prefix: string,
    limits: CacheEnumerationLimits,
  ): Promise<readonly string[]> {
    validateEnumerationLimits(limits);
    const segments = safeSegments(prefix);
    if (segments.length > limits.maxDepth) {
      throw new ModelCacheError("cache-enumeration-depth");
    }
    let directory = this.root;
    try {
      for (const segment of segments) {
        directory = await directory.getDirectoryHandle(segment);
      }
    } catch (error) {
      if (isNotFound(error)) {
        return [];
      }
      throw error;
    }
    const normalizedPrefix =
      segments.length === 0 ? "" : `${segments.join("/")}/`;
    const output: string[] = [];
    const counter = { entries: 0 };
    await walkDirectory(
      directory,
      normalizedPrefix,
      output,
      limits,
      counter,
      segments.length,
    );
    return output;
  }

  private async parent(
    path: string,
    create: boolean,
  ): Promise<{ directory: FileSystemDirectoryHandle; name: string }> {
    const segments = safeSegments(path);
    const name = segments.pop();
    if (name === undefined) {
      throw new ModelCacheError("cache-path-invalid");
    }
    let directory = this.root;
    for (const segment of segments) {
      directory = await directory.getDirectoryHandle(segment, { create });
    }
    return { directory, name };
  }
}

function safeSegments(path: string): string[] {
  const normalized = path.replace(/\/+$/, "");
  if (normalized.length === 0) {
    return [];
  }
  const segments = normalized.split("/");
  if (
    segments.length === 0 ||
    segments.some((segment) => !SAFE_PATH_SEGMENT.test(segment))
  ) {
    throw new ModelCacheError("cache-path-invalid");
  }
  return segments;
}

async function walkDirectory(
  directory: FileSystemDirectoryHandle,
  prefix: string,
  output: string[],
  limits: CacheEnumerationLimits,
  counter: { entries: number },
  depth: number,
): Promise<void> {
  const iterableDirectory = directory as FileSystemDirectoryHandle & {
    entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
  };
  for await (const [name, handle] of iterableDirectory.entries()) {
    counter.entries += 1;
    if (counter.entries > limits.maxEntries) {
      throw new ModelCacheError("cache-enumeration-limit");
    }
    if (!SAFE_PATH_SEGMENT.test(name)) {
      continue;
    }
    const path = `${prefix}${name}`;
    if (handle.kind === "file") {
      output.push(path);
    } else {
      if (depth >= limits.maxDepth) {
        throw new ModelCacheError("cache-enumeration-depth");
      }
      await walkDirectory(
        handle as FileSystemDirectoryHandle,
        `${path}/`,
        output,
        limits,
        counter,
        depth + 1,
      );
    }
  }
}

function validateEnumerationLimits(limits: CacheEnumerationLimits): void {
  if (
    !Number.isSafeInteger(limits.maxDepth) ||
    limits.maxDepth < 0 ||
    limits.maxDepth > CACHE_ENUMERATION_LIMITS.maxDepth ||
    !Number.isSafeInteger(limits.maxEntries) ||
    limits.maxEntries < 1 ||
    limits.maxEntries > CACHE_ENUMERATION_LIMITS.maxEntries
  ) {
    throw new ModelCacheError("cache-enumeration-bounds-invalid");
  }
}

function fileSystemWriteChunk(
  chunk: Uint8Array,
): Uint8Array<ArrayBuffer> {
  if (chunk.buffer instanceof ArrayBuffer) {
    return new Uint8Array(
      chunk.buffer,
      chunk.byteOffset,
      chunk.byteLength,
    );
  }
  // OPFS rejects SharedArrayBuffer-backed views. Network range chunks use
  // ArrayBuffer, so this defensive copy is outside the weight hot path.
  return new Uint8Array(chunk);
}

function isNotFound(error: unknown): boolean {
  return (
    error instanceof DOMException
      ? error.name === "NotFoundError"
      : error instanceof Error && error.name === "NotFoundError"
  );
}
