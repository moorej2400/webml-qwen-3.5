import assert from "node:assert/strict";
import test from "node:test";

import {
  HttpRangeReader,
  type ImmutableRangeSource,
  type RangeFetch,
} from "../src/http-range-reader.js";
import { IncrementalSha256 } from "../src/incremental-sha256.js";
import type { ModelPackageManifest, PackageShard } from "../src/manifest.js";
import {
  BrowserOpfsStorage,
  ImmutableOpfsModelCache,
  ModelCacheError,
  modelCacheKey,
  type CacheAtomicWriter,
  type CacheEnumerationLimits,
  type ModelCacheStorage,
} from "../src/opfs-model-cache.js";
import type { Qwen35PackedRangeReader } from "../src/qwen35-disk-backed-tied-embedding.js";

interface ModelCacheRangeSubject {
  createQwen35ModelCacheRangeReader(
    storage: ModelCacheStorage,
  ): Qwen35PackedRangeReader;
}

async function modelCacheRangeSubject(): Promise<ModelCacheRangeSubject> {
  return await import("../src/qwen35-disk-backed-tied-embedding.js") as unknown as ModelCacheRangeSubject;
}

function hash(bytes: Uint8Array): string {
  return new IncrementalSha256().update(bytes).digestHex();
}

function concat(chunks: readonly Uint8Array[]): Uint8Array {
  const output = new Uint8Array(
    chunks.reduce((total, chunk) => total + chunk.byteLength, 0),
  );
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

class MemoryCacheStorage implements ModelCacheStorage {
  readonly files = new Map<string, Uint8Array>();
  readonly committed: string[] = [];

  constructor(readonly moveSupported = true) {}

  async openAtomicWriter(path: string): Promise<CacheAtomicWriter> {
    const chunks: Uint8Array[] = [];
    let settled = false;
    const publish = (): void => {
      if (!settled) {
        this.files.set(path, concat(chunks));
        settled = true;
      }
    };
    return {
      write: async (chunk) => {
        assert.equal(settled, false);
        chunks.push(chunk.slice());
      },
      commit: async () => {
        publish();
        this.committed.push(path);
      },
      preserveIncomplete: async () => publish(),
    };
  }

  async openRead(path: string): Promise<AsyncIterable<Uint8Array> | null> {
    const bytes = this.files.get(path);
    if (bytes === undefined) {
      return null;
    }
    return (async function* stream(): AsyncIterable<Uint8Array> {
      for (let offset = 0; offset < bytes.byteLength; offset += 7) {
        yield bytes.subarray(offset, Math.min(bytes.byteLength, offset + 7));
      }
    })();
  }

  async move(source: string, destination: string): Promise<boolean> {
    if (!this.moveSupported) {
      return false;
    }
    const bytes = this.files.get(source);
    if (bytes === undefined) {
      throw new Error("missing move source");
    }
    this.files.set(destination, bytes);
    this.files.delete(source);
    return true;
  }

  async list(
    prefix: string,
    limits: CacheEnumerationLimits,
  ): Promise<readonly string[]> {
    const matches = [...this.files.keys()].filter((path) =>
      path.startsWith(prefix),
    );
    if (matches.length > limits.maxEntries) {
      throw new ModelCacheError("cache-enumeration-limit");
    }
    return matches;
  }
}

const REVISION = "1".repeat(40);
const TOKENIZER_REVISION = "2".repeat(40);

function manifestFor(bytes: Uint8Array, expectedHash = hash(bytes)): ModelPackageManifest {
  return {
    format: "webml-qwen-package",
    version: 1,
    packageKind: "language",
    source: {
      repository: "public/model",
      revision: REVISION,
      file: "model.gguf",
      size: String(bytes.byteLength),
      sha256: expectedHash,
    },
    runtime: { abi: "qwen35-webgpu-v2" },
    tokenizer: {
      repository: "public/tokenizer",
      revision: TOKENIZER_REVISION,
      file: "tokenizer.json",
      size: "1",
      sha256: "b".repeat(64),
    },
    tensorLayout: [
      {
        name: "blk.0.attn_q.weight",
        shape: ["256"],
        ggmlType: 11,
        storageType: "q3-k-112",
        shard: 0,
        shardOffset: "0",
        tensorOffset: "0",
        length: "112",
        quantization: { blockElements: 256, blockBytes: 112 },
      },
    ],
    shards: [
      {
        url: "shards/model-00000.bin",
        offset: "0",
        length: String(bytes.byteLength),
        sha256: expectedHash,
      },
    ],
    excludedTensors: [],
  };
}

function readerFor(
  bytes: Uint8Array,
  beforeResponse?: () => Promise<void>,
): HttpRangeReader {
  const fetcher: RangeFetch = async (_locator, init) => {
    await beforeResponse?.();
    const match = /^bytes=(\d+)-(\d+)$/.exec(init.range);
    assert.ok(match);
    const start = Number(match[1]);
    const end = Number(match[2]);
    const body = bytes.subarray(start, end + 1);
    return new Response(body, {
      status: 206,
      headers: {
        "content-length": String(body.byteLength),
        "content-range": `bytes ${start}-${end}/${bytes.byteLength}`,
      },
    });
  };
  return new HttpRangeReader(fetcher, { rangeStrategies: [16] });
}

function resolver(
  shard: PackageShard,
): ImmutableRangeSource {
  return {
    locator: `https://public.example/immutable/${shard.url}`,
    byteLength: Number(shard.length),
    immutableUrl: true,
  };
}

function fixtureBytes(): Uint8Array {
  return Uint8Array.from({ length: 112 }, (_, index) => (index * 17) & 0xff);
}

test("publishes no ready manifest until every shard is hashed and committed", async () => {
  const bytes = fixtureBytes();
  const storage = new MemoryCacheStorage();
  let release!: () => void;
  const responseGate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const cache = new ImmutableOpfsModelCache(
    storage,
    readerFor(bytes, async () => responseGate),
    { attemptId: () => "attempt-a" },
  );

  const pending = cache.ensure(manifestFor(bytes), resolver);
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(
    [...storage.files.keys()].some((path) => path.startsWith("ready/")),
    false,
  );

  release();
  const result = await pending;
  assert.equal(result.cacheHit, false);
  assert.equal(result.shards.length, 1);
  const readyPaths = [...storage.files.keys()].filter((path) =>
    path.startsWith("ready/"),
  );
  assert.equal(readyPaths.length, 1);
  assert.equal(storage.committed.at(-1), readyPaths[0]);
});

test("reconnect verifies an existing immutable package without network access", async () => {
  const bytes = fixtureBytes();
  const storage = new MemoryCacheStorage();
  const manifest = manifestFor(bytes);
  const first = new ImmutableOpfsModelCache(storage, readerFor(bytes), {
    attemptId: () => "attempt-a",
  });
  const populated = await first.ensure(manifest, resolver);
  let networkCalls = 0;
  const offlineReader = new HttpRangeReader(async () => {
    networkCalls += 1;
    throw new TypeError("offline");
  });
  const reconnect = new ImmutableOpfsModelCache(storage, offlineReader, {
    attemptId: () => "attempt-b",
  });

  const cached = await reconnect.ensure(manifest, resolver);

  assert.equal(populated.cacheHit, false);
  assert.equal(cached.cacheHit, true);
  assert.equal(networkCalls, 0);
  assert.equal(cached.manifestSha256, populated.manifestSha256);
});

test("hash mismatch preserves an ignored recovery artifact and never appears ready", async () => {
  const bytes = fixtureBytes();
  const storage = new MemoryCacheStorage();
  const wrongHash = "f".repeat(64);
  const cache = new ImmutableOpfsModelCache(storage, readerFor(bytes), {
    attemptId: () => "attempt-a",
  });

  await assert.rejects(
    cache.ensure(manifestFor(bytes, wrongHash), resolver),
    (error: unknown) =>
      error instanceof ModelCacheError && error.code === "shard-hash-mismatch",
  );

  assert.equal(
    [...storage.files.keys()].some((path) => path.startsWith("ready/")),
    false,
  );
  assert.equal(
    [...storage.files.keys()].some(
      (path) => path.startsWith("stale/") || path.startsWith("tmp/"),
    ),
    true,
  );
});

test("a later run quarantines an interrupted temp and creates a new valid generation", async () => {
  const bytes = fixtureBytes();
  const manifest = manifestFor(bytes);
  const storage = new MemoryCacheStorage();
  const key = modelCacheKey(manifest);
  storage.files.set(
    `tmp/${key}/crashed-attempt/shard-00000.part`,
    new Uint8Array([1, 2, 3]),
  );
  const cache = new ImmutableOpfsModelCache(storage, readerFor(bytes), {
    attemptId: () => "attempt-b",
  });

  const result = await cache.ensure(manifest, resolver);

  assert.equal(result.cacheHit, false);
  assert.equal(
    [...storage.files.keys()].some((path) =>
      path.startsWith(`stale/${key}/`),
    ),
    true,
  );
  assert.equal(
    [...storage.files.keys()].some((path) =>
      path.startsWith(`tmp/${key}/crashed-attempt/`),
    ),
    false,
  );
});

test("no-move storage never retains duplicate multi-gigabyte part data", async () => {
  const bytes = fixtureBytes();
  const storage = new MemoryCacheStorage(false);
  const cache = new ImmutableOpfsModelCache(storage, readerFor(bytes), {
    attemptId: () => "attempt-a",
  });

  const result = await cache.ensure(manifestFor(bytes), resolver);

  assert.equal(result.cacheHit, false);
  assert.ok(storage.files.has(result.shards[0]!.storagePath));
  assert.equal(hash(storage.files.get(result.shards[0]!.storagePath)!), hash(bytes));
  const fullWeightFiles = [...storage.files.values()].filter(
    (value) => value.byteLength === bytes.byteLength,
  );
  assert.equal(fullWeightFiles.length, 1);
  assert.ok(
    storage.committed.indexOf(result.shards[0]!.storagePath) <
      storage.committed.findIndex((path) => path.startsWith("ready/")),
  );
});

test("an attempt-id collision cannot overwrite an existing immutable blob", async () => {
  const bytes = fixtureBytes();
  const storage = new MemoryCacheStorage(false);
  const sha256 = hash(bytes);
  const existingPath = `blobs/${sha256}/attempt-a-00000.bin`;
  const existing = new Uint8Array([9, 8, 7]);
  storage.files.set(existingPath, existing);
  const cache = new ImmutableOpfsModelCache(storage, readerFor(bytes), {
    attemptId: () => "attempt-a",
  });

  await assert.rejects(
    cache.ensure(manifestFor(bytes), resolver),
    (error: unknown) =>
      error instanceof ModelCacheError &&
      error.code === "cache-attempt-collision",
  );
  assert.deepEqual(storage.files.get(existingPath), existing);
});

test("corrupt ready data is ignored and replaced by a separate valid generation", async () => {
  const bytes = fixtureBytes();
  const manifest = manifestFor(bytes);
  const storage = new MemoryCacheStorage();
  const first = new ImmutableOpfsModelCache(storage, readerFor(bytes), {
    attemptId: () => "attempt-a",
  });
  const original = await first.ensure(manifest, resolver);
  storage.files.set(original.shards[0]!.storagePath, new Uint8Array([9]));
  const second = new ImmutableOpfsModelCache(storage, readerFor(bytes), {
    attemptId: () => "attempt-b",
  });

  const recovered = await second.ensure(manifest, resolver);

  assert.equal(recovered.cacheHit, false);
  assert.notEqual(
    recovered.shards[0]!.storagePath,
    original.shards[0]!.storagePath,
  );
  assert.ok(storage.files.has(original.shards[0]!.storagePath));
});

test("cache metrics contain counts and timing but no source or storage identifiers", async () => {
  const bytes = fixtureBytes();
  const storage = new MemoryCacheStorage();
  const cache = new ImmutableOpfsModelCache(storage, readerFor(bytes), {
    attemptId: () => "attempt-a",
  });
  await cache.ensure(manifestFor(bytes), resolver);

  const serialized = JSON.stringify(cache.metrics);
  assert.doesNotMatch(serialized, /public|model|tokenizer|shard|https|tmp|ready|stack/i);
  assert.match(serialized, /hashMilliseconds/);
  assert.equal(cache.metrics.verifiedParts, 2);
});

test("hash timing measures incremental update work instead of network or storage waits", async () => {
  const bytes = fixtureBytes();
  const storage = new MemoryCacheStorage();
  let tick = 0;
  const cache = new ImmutableOpfsModelCache(storage, readerFor(bytes), {
    attemptId: () => "attempt-a",
    now: () => tick++,
  });

  await cache.ensure(manifestFor(bytes), resolver);

  // Seven 16-byte HTTP ranges are hashed during download. The in-memory OPFS
  // reader then verifies the promoted file in sixteen 7-byte pieces.
  assert.equal(cache.metrics.hashMilliseconds, 23);
});

interface FakeDirectoryHandle {
  kind: "directory";
  entries(): AsyncIterableIterator<[string, FakeDirectoryHandle | FakeFileHandle]>;
  getDirectoryHandle(name: string): Promise<FakeDirectoryHandle>;
}

interface FakeFileHandle {
  kind: "file";
}

function fakeTree(
  entries: Readonly<Record<string, FakeDirectoryHandle | FakeFileHandle>>,
  onYield?: () => void,
): FakeDirectoryHandle {
  return {
    kind: "directory",
    async *entries() {
      for (const entry of Object.entries(entries)) {
        onYield?.();
        yield entry;
      }
    },
    async getDirectoryHandle(name: string) {
      const child = entries[name];
      if (child?.kind !== "directory") {
        throw new DOMException("not found", "NotFoundError");
      }
      return child;
    },
  };
}

test("OPFS enumeration stops as soon as the total-entry limit is exceeded", async () => {
  let yielded = 0;
  const entries = Object.fromEntries(
    Array.from({ length: 100 }, (_, index) => [
      `part-${index}`,
      { kind: "file" as const },
    ]),
  );
  const root = fakeTree(entries, () => {
    yielded += 1;
  });
  const storage = BrowserOpfsStorage.fromRoot(
    root as unknown as FileSystemDirectoryHandle,
  );

  await assert.rejects(
    storage.list("", { maxDepth: 4, maxEntries: 3 }),
    (error: unknown) =>
      error instanceof ModelCacheError &&
      error.code === "cache-enumeration-limit",
  );
  assert.equal(yielded, 4);
});

test("OPFS enumeration rejects excessive tree depth before entering it", async () => {
  let deepestDirectoryRead = false;
  const tooDeep: FakeDirectoryHandle = {
    kind: "directory",
    async *entries() {
      deepestDirectoryRead = true;
    },
    async getDirectoryHandle() {
      throw new DOMException("not found", "NotFoundError");
    },
  };
  const levelTwo = fakeTree({ three: tooDeep });
  const levelOne = fakeTree({ two: levelTwo });
  const root = fakeTree({ one: levelOne });
  const storage = BrowserOpfsStorage.fromRoot(
    root as unknown as FileSystemDirectoryHandle,
  );

  await assert.rejects(
    storage.list("", { maxDepth: 2, maxEntries: 100 }),
    (error: unknown) =>
      error instanceof ModelCacheError &&
      error.code === "cache-enumeration-depth",
  );
  assert.equal(deepestDirectoryRead, false);
});

test("OPFS enumeration counts the requested prefix against the depth limit", async () => {
  let deepestDirectoryRead = false;
  const tooDeep: FakeDirectoryHandle = {
    kind: "directory",
    async *entries() {
      deepestDirectoryRead = true;
    },
    async getDirectoryHandle() {
      throw new DOMException("not found", "NotFoundError");
    },
  };
  const levelTwo = fakeTree({ three: tooDeep });
  const levelOne = fakeTree({ two: levelTwo });
  const root = fakeTree({ one: levelOne });
  const storage = BrowserOpfsStorage.fromRoot(
    root as unknown as FileSystemDirectoryHandle,
  );

  await assert.rejects(
    storage.list("one/two/three", { maxDepth: 2, maxEntries: 100 }),
    (error: unknown) =>
      error instanceof ModelCacheError &&
      error.code === "cache-enumeration-depth",
  );
  assert.equal(deepestDirectoryRead, false);
});

test("OPFS random reads slice only the requested immutable byte range", async () => {
  const bytes = Uint8Array.from({ length: 32 }, (_, index) => index);
  const slices: Array<{ readonly start: number; readonly end: number }> = [];
  let directoryLookups = 0;
  let fileHandleLookups = 0;
  let fileSnapshots = 0;
  const fileHandle = {
    kind: "file" as const,
    async getFile() {
      fileSnapshots += 1;
      return {
        size: bytes.byteLength,
        slice(start = 0, end = bytes.byteLength) {
          slices.push({ start, end });
          const selected = bytes.slice(start, end);
          return new Blob([selected]);
        },
        stream() {
          assert.fail("a bounded range read must not stream the complete file");
        },
      };
    },
  };
  const blobs = {
    kind: "directory" as const,
    async getFileHandle(name: string) {
      fileHandleLookups += 1;
      if (name !== "model.bin") {
        throw new DOMException("not found", "NotFoundError");
      }
      return fileHandle;
    },
  };
  const root = {
    kind: "directory" as const,
    async getDirectoryHandle(name: string) {
      directoryLookups += 1;
      if (name !== "blobs") {
        throw new DOMException("not found", "NotFoundError");
      }
      return blobs;
    },
  };
  const storage = BrowserOpfsStorage.fromRoot(
    root as unknown as FileSystemDirectoryHandle,
  ) as BrowserOpfsStorage & {
    readRange(
      path: string,
      offset: number,
      byteLength: number,
      signal: AbortSignal,
    ): Promise<Uint8Array | null>;
  };
  assert.equal(
    typeof storage.readRange,
    "function",
    "the disk-backed tied tensor requires bounded OPFS random reads",
  );

  const result = await storage.readRange(
    "blobs/model.bin",
    7,
    9,
    new AbortController().signal,
  );
  const second = await storage.readRange(
    "blobs/model.bin",
    16,
    4,
    new AbortController().signal,
  );

  assert.deepEqual(result, bytes.subarray(7, 16));
  assert.deepEqual(second, bytes.subarray(16, 20));
  assert.deepEqual(slices, [{ start: 7, end: 16 }, { start: 16, end: 20 }]);
  assert.equal(directoryLookups, 1);
  assert.equal(fileHandleLookups, 1);
  assert.equal(fileSnapshots, 1);
});

test("model-cache range adapter always prefers bounded random access", async () => {
  const subject = await modelCacheRangeSubject();
  assert.equal(
    typeof subject.createQwen35ModelCacheRangeReader,
    "function",
    "the loader requires a model-cache range-reader adapter",
  );
  const bytes = Uint8Array.of(7, 8, 9);
  const calls: unknown[] = [];
  const storage: ModelCacheStorage = {
    async openAtomicWriter() { throw new Error("not used"); },
    async openRead() {
      assert.fail("production random access must not use the stream fallback");
    },
    async readRange(path, offset, byteLength, signal) {
      calls.push({ path, offset, byteLength, signal });
      return bytes;
    },
    async move() { return false; },
    async list() { return []; },
  };
  const signal = new AbortController().signal;
  const reader = subject.createQwen35ModelCacheRangeReader(storage);

  const result = await reader.read({
    storagePath: "blobs/model.bin",
    offset: 11,
    byteLength: 3,
    signal,
  });

  assert.equal(result, bytes);
  assert.deepEqual(calls, [{
    path: "blobs/model.bin",
    offset: 11,
    byteLength: 3,
    signal,
  }]);
});
