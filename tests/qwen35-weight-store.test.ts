import assert from "node:assert/strict";
import test from "node:test";

import type {
  GpuAllocation,
  GpuAllocationRequest,
  GpuBufferLike,
} from "../src/gpu-arena.js";
import type { CachedModelPackage, ModelCacheStorage } from "../src/opfs-model-cache.js";
import {
  allocateQwen35WeightDirectory,
  type Qwen35PackageDirectory,
  type Qwen35WeightArena,
  type Qwen35WeightDirectory,
} from "../src/qwen35-weight-directory.js";
import {
  initializeQwen35WeightExecution,
  uploadQwen35CachedWeights,
  type Qwen35WeightWriteQueue,
} from "../src/qwen35-weight-upload.js";

class MemoryBuffer implements GpuBufferLike {
  readonly bytes: Uint8Array;
  destroyed = false;

  constructor(byteLength: number) {
    this.bytes = new Uint8Array(byteLength).fill(0xcc);
  }

  destroy(): void {
    this.destroyed = true;
  }
}

function allocation(
  logicalBytes: number,
  shardBytes: readonly number[] = [logicalBytes],
): GpuAllocation {
  let logicalByteOffset = 0n;
  const buffers = shardBytes.map((byteLength) => {
    const buffer = new MemoryBuffer(byteLength);
    const shard = {
      buffer,
      logicalByteOffset,
      logicalByteLength: BigInt(byteLength),
      allocatedByteLength: BigInt(byteLength),
    };
    logicalByteOffset += BigInt(byteLength);
    return shard;
  });
  let destroyed = false;
  return {
    shards: buffers,
    logicalBytes: BigInt(logicalBytes),
    allocatedBytes: BigInt(logicalBytes),
    destroy() {
      if (destroyed) return;
      destroyed = true;
      for (const shard of buffers) shard.buffer.destroy();
    },
  };
}

function arenaWithSplits(
  splitByAllocation: readonly (readonly number[])[] = [],
): {
  arena: Qwen35WeightArena;
  requests: GpuAllocationRequest[];
  allocations: GpuAllocation[];
} {
  const requests: GpuAllocationRequest[] = [];
  const allocations: GpuAllocation[] = [];
  const arena: Qwen35WeightArena = {
    async allocate(request) {
      requests.push(request);
      const created = allocation(
        Number(request.byteLength),
        splitByAllocation[allocations.length] ?? [Number(request.byteLength)],
      );
      allocations.push(created);
      return created;
    },
  };
  return { arena, requests, allocations };
}

function packageDirectory(): Qwen35PackageDirectory {
  return {
    manifestSha256: "a".repeat(64),
    shards: [
      { index: 0, url: "part-0.bin", offset: "0", length: "32", sha256: "b".repeat(64) },
      { index: 1, url: "part-1.bin", offset: "32", length: "32", sha256: "c".repeat(64) },
    ],
    tensors: [
      {
        name: "token_embd.weight",
        shape: [2, 2],
        ggmlType: 0,
        storageType: "f32",
        segments: [
          { shardIndex: 0, shardOffset: "4", tensorOffset: "0", length: "8" },
          { shardIndex: 1, shardOffset: "8", tensorOffset: "8", length: "8" },
        ],
      },
      {
        name: "output_norm.weight",
        shape: [1, 2],
        ggmlType: 0,
        storageType: "f32",
        segments: [
          { shardIndex: 0, shardOffset: "20", tensorOffset: "0", length: "8" },
        ],
      },
    ],
  };
}

function cachedPackage(): CachedModelPackage {
  return {
    cacheKey: "cache",
    manifestSha256: "a".repeat(64),
    cacheHit: true,
    shards: [
      { storagePath: "shard-0", byteLength: 32, sha256: "b".repeat(64) },
      { storagePath: "shard-1", byteLength: 32, sha256: "c".repeat(64) },
    ],
  };
}

function chunkedStorage(
  bytesByPath: Readonly<Record<string, Uint8Array>>,
  chunkWidths: readonly number[],
): ModelCacheStorage {
  return {
    async openAtomicWriter() {
      throw new Error("not used");
    },
    async openRead(path) {
      const bytes = bytesByPath[path];
      if (bytes === undefined) return null;
      return (async function* () {
        let offset = 0;
        let widthIndex = 0;
        while (offset < bytes.byteLength) {
          const width = chunkWidths[widthIndex % chunkWidths.length]!;
          yield bytes.subarray(offset, Math.min(offset + width, bytes.byteLength));
          offset += width;
          widthIndex += 1;
        }
      })();
    },
    async move() {
      return false;
    },
    async list() {
      return [];
    },
  };
}

function memoryQueue(events: string[] = []): Qwen35WeightWriteQueue {
  return {
    writeBuffer(buffer, bufferOffset, data, dataOffset = 0, size) {
      const target = buffer as MemoryBuffer;
      const byteLength = size ?? data.byteLength - dataOffset;
      assert.equal(bufferOffset % 4, 0);
      assert.equal(dataOffset % 4, 0);
      assert.equal(byteLength % 4, 0);
      target.bytes.set(
        new Uint8Array(data.buffer, data.byteOffset + dataOffset, byteLength),
        bufferOffset,
      );
      events.push(`write:${byteLength}`);
    },
    async onSubmittedWorkDone() {
      events.push("upload-complete");
    },
  };
}

function tensorBytes(
  directory: Qwen35WeightDirectory,
  name: string,
): Uint8Array {
  const tensor = directory.get(name)!;
  const bytes = new Uint8Array(Number(tensor.logicalBytes));
  for (const view of tensor.physicalRows) {
    const buffer = view.buffer as MemoryBuffer;
    bytes.set(
      buffer.bytes.subarray(
        view.bufferByteOffset,
        view.bufferByteOffset + view.byteLength,
      ),
      view.tensorByteOffset,
    );
  }
  return bytes;
}

test("allocates one row-quantized logical allocation per tensor", async () => {
  const fixture = arenaWithSplits([[8, 8], [8]]);
  const directory = await allocateQwen35WeightDirectory(
    fixture.arena,
    packageDirectory(),
  );

  assert.equal(directory.size, 2);
  assert.equal(
    directory.tensors.filter((tensor) => tensor.name === "token_embd.weight").length,
    1,
  );
  assert.deepEqual(
    fixture.requests.map((request) => ({
      byteLength: request.byteLength,
      quantum: request.requiredShardQuantumBytes,
    })),
    [
      { byteLength: 16n, quantum: 8n },
      { byteLength: 8n, quantum: 4n },
    ],
  );
  assert.deepEqual(directory.get("token_embd.weight")?.physicalRows, [
    {
      buffer: fixture.allocations[0]!.shards[0]!.buffer,
      firstRow: 0,
      rowCount: 1,
      tensorByteOffset: 0,
      bufferByteOffset: 0,
      byteLength: 8,
    },
    {
      buffer: fixture.allocations[0]!.shards[1]!.buffer,
      firstRow: 1,
      rowCount: 1,
      tensorByteOffset: 8,
      bufferByteOffset: 0,
      byteLength: 8,
    },
  ]);
  assert.equal(directory.allocatedBytes, 24n);
  assert.equal(Object.isFrozen(directory), true);
  assert.equal(Object.isFrozen(directory.get("token_embd.weight")?.physicalRows), true);
});

test("derives the exact packed row quantum for every language layout", async () => {
  const layouts = [
    ["f32", 1, 4],
    ["q8-0-36", 32, 36],
    ["q3-k-112", 256, 112],
    ["q4-k-144", 256, 144],
    ["q5-k-176", 256, 176],
    ["q6-k-212", 256, 212],
  ] as const;
  let shardOffset = 0;
  const tensors = layouts.map(([storageType, width, rowBytes], index) => {
    const tensor = {
      name: `tensor-${index}`,
      shape: [width, 2],
      ggmlType: index,
      storageType,
      segments: [{
        shardIndex: 0,
        shardOffset: String(shardOffset),
        tensorOffset: "0",
        length: String(rowBytes * 2),
      }],
    };
    shardOffset += rowBytes * 2;
    return tensor;
  });
  const fixture = arenaWithSplits();

  await allocateQwen35WeightDirectory(fixture.arena, {
    manifestSha256: "a".repeat(64),
    shards: [{
      index: 0,
      url: "part.bin",
      offset: "0",
      length: String(shardOffset),
      sha256: "b".repeat(64),
    }],
    tensors,
  });

  assert.deepEqual(
    fixture.requests.map((request) => request.requiredShardQuantumBytes),
    layouts.map(([, , rowBytes]) => BigInt(rowBytes)),
  );
});

test("rejects a physical buffer split through a packed row and rolls back", async () => {
  const fixture = arenaWithSplits([[12, 4]]);

  await assert.rejects(
    allocateQwen35WeightDirectory(fixture.arena, packageDirectory()),
    { code: "model-weight-row-split-invalid" },
  );
  assert.equal((fixture.allocations[0]!.shards[0]!.buffer as MemoryBuffer).destroyed, true);
});

test("rejects package segments that split a packed tensor row", async () => {
  const fixture = arenaWithSplits();
  const invalid = structuredClone(packageDirectory());
  (invalid.tensors[0] as { segments: Qwen35PackageDirectory["tensors"][number]["segments"] }).segments = [
    { shardIndex: 0, shardOffset: "4", tensorOffset: "0", length: "4" },
    { shardIndex: 1, shardOffset: "8", tensorOffset: "4", length: "12" },
  ];

  await assert.rejects(
    allocateQwen35WeightDirectory(fixture.arena, invalid),
    { code: "model-weight-segments-invalid" },
  );
  assert.equal(fixture.allocations.length, 0);
});

test("scatters tensors across package padding, package shards, and physical row buffers", async () => {
  const fixture = arenaWithSplits([[8, 8], [8]]);
  const directory = await allocateQwen35WeightDirectory(fixture.arena, packageDirectory());
  const first = new Uint8Array(32).fill(0xee);
  first.set([1, 2, 3, 4, 5, 6, 7, 8], 4);
  first.set([21, 22, 23, 24, 25, 26, 27, 28], 20);
  const second = new Uint8Array(32).fill(0xdd);
  second.set([9, 10, 11, 12, 13, 14, 15, 16], 8);
  const writes: string[] = [];

  await uploadQwen35CachedWeights({
    storage: chunkedStorage({ "shard-0": first, "shard-1": second }, [3, 7, 2, 9]),
    cached: cachedPackage(),
    directory,
    queue: memoryQueue(writes),
    uploadLaneBytes: 12,
    signal: new AbortController().signal,
  });

  assert.deepEqual([...tensorBytes(directory, "token_embd.weight")], [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  ]);
  assert.deepEqual([...tensorBytes(directory, "output_norm.weight")], [
    21, 22, 23, 24, 25, 26, 27, 28,
  ]);
  assert.equal(writes.at(-1), "upload-complete");
  assert.equal(writes.every((event) => event === "upload-complete" || Number(event.slice(6)) <= 12), true);
});

test("per-write retirement settles every accepted queue copy independently", async () => {
  const fixture = arenaWithSplits([[8, 8], [8]]);
  const directory = await allocateQwen35WeightDirectory(fixture.arena, packageDirectory());
  const queueEvents: string[] = [];
  const options = {
    storage: chunkedStorage(
      { "shard-0": new Uint8Array(32), "shard-1": new Uint8Array(32) },
      [32],
    ),
    cached: cachedPackage(),
    directory,
    queue: memoryQueue(queueEvents),
    uploadLaneBytes: 16,
    uploadRetirementPolicy: "per-write" as const,
    signal: new AbortController().signal,
  };

  await assert.doesNotReject(uploadQwen35CachedWeights(options));
  assert.deepEqual([...tensorBytes(directory, "token_embd.weight")], new Array(16).fill(0));
  assert.deepEqual(queueEvents, [
    "write:8",
    "upload-complete",
    "write:8",
    "upload-complete",
    "write:8",
    "upload-complete",
  ]);
});

test("retires queue-owned upload copies within the configured staging window", async () => {
  const fixture = arenaWithSplits([[8, 8], [8]]);
  const directory = await allocateQwen35WeightDirectory(fixture.arena, packageDirectory());
  const target = memoryQueue();
  let outstandingBytes = 0;
  let peakOutstandingBytes = 0;
  let retirements = 0;
  const queue: Qwen35WeightWriteQueue = {
    writeBuffer(...args) {
      const size = args[4] ?? args[2].byteLength - (args[3] ?? 0);
      outstandingBytes += size;
      peakOutstandingBytes = Math.max(peakOutstandingBytes, outstandingBytes);
      if (outstandingBytes > 8) {
        throw new Error("queue-owned upload staging exceeded its bound");
      }
      target.writeBuffer(...args);
    },
    async onSubmittedWorkDone() {
      retirements += 1;
      outstandingBytes = 0;
    },
  };

  await uploadQwen35CachedWeights({
    storage: chunkedStorage(
      { "shard-0": new Uint8Array(32), "shard-1": new Uint8Array(32) },
      [32],
    ),
    cached: cachedPackage(),
    directory,
    queue,
    uploadLaneBytes: 8,
    signal: new AbortController().signal,
  });

  assert.equal(peakOutstandingBytes, 8);
  assert.ok(retirements >= 3);
  assert.equal(outstandingBytes, 0);
});

for (let remainder = 0; remainder <= 3; remainder += 1) {
  test(`coalesces an input chunk with a ${remainder}-byte u32 remainder`, async () => {
    const fixture = arenaWithSplits([[8, 8], [8]]);
    const directory = await allocateQwen35WeightDirectory(fixture.arena, packageDirectory());
    const first = Uint8Array.from({ length: 32 }, (_, index) => index);
    const second = Uint8Array.from({ length: 32 }, (_, index) => 32 + index);

    await uploadQwen35CachedWeights({
      storage: chunkedStorage(
        { "shard-0": first, "shard-1": second },
        [4 + remainder, 1, 11],
      ),
      cached: cachedPackage(),
      directory,
      queue: memoryQueue(),
      uploadLaneBytes: 8,
      signal: new AbortController().signal,
    });

    assert.deepEqual([...tensorBytes(directory, "token_embd.weight")], [
      ...first.subarray(4, 12),
      ...second.subarray(8, 16),
    ]);
  });
}

test("cancellation rolls back every tensor allocation", async () => {
  const fixture = arenaWithSplits([[8, 8], [8]]);
  const controller = new AbortController();
  const first = new Uint8Array(32);
  const second = new Uint8Array(32);
  const events: string[] = [];
  const queue = memoryQueue(events);
  const originalWrite = queue.writeBuffer.bind(queue);
  let writes = 0;
  queue.writeBuffer = (...args) => {
    originalWrite(...args);
    writes += 1;
    controller.abort();
  };

  await assert.rejects(
    initializeQwen35WeightExecution({
      arena: fixture.arena,
      packageDirectory: packageDirectory(),
      storage: chunkedStorage({ "shard-0": first, "shard-1": second }, [32]),
      cached: cachedPackage(),
      queue,
      uploadLaneBytes: 8,
      signal: controller.signal,
      async createDriver() {
        throw new Error("driver must not be created");
      },
    }),
    { name: "AbortError" },
  );
  assert.equal(writes, 1);
  assert.equal(events.at(-1), "upload-complete");
  assert.equal(
    fixture.allocations.every((item) =>
      item.shards.every((shard) => (shard.buffer as MemoryBuffer).destroyed),
    ),
    true,
  );
});

test("driver creation failure settles uploads and rolls back every tensor", async () => {
  const events: string[] = [];
  const fixture = arenaWithSplits([[8, 8], [8]]);

  await assert.rejects(
    initializeQwen35WeightExecution({
      arena: fixture.arena,
      packageDirectory: packageDirectory(),
      storage: chunkedStorage(
        { "shard-0": new Uint8Array(32), "shard-1": new Uint8Array(32) },
        [32],
      ),
      cached: cachedPackage(),
      queue: memoryQueue(events),
      uploadLaneBytes: 8,
      signal: new AbortController().signal,
      async createDriver() {
        events.push("create");
        throw new Error("driver construction failed");
      },
    }),
    /driver construction failed/,
  );
  assert.deepEqual(events.slice(-3), [
    "upload-complete",
    "create",
    "upload-complete",
  ]);
  assert.equal(
    fixture.allocations.every((item) =>
      item.shards.every((shard) => (shard.buffer as MemoryBuffer).destroyed),
    ),
    true,
  );
});

test("creates the execution driver only after every weight upload completes", async () => {
  const events: string[] = [];
  const fixture = arenaWithSplits([[8, 8], [8]]);
  const initialized = await initializeQwen35WeightExecution({
    arena: fixture.arena,
    packageDirectory: packageDirectory(),
    storage: chunkedStorage(
      { "shard-0": new Uint8Array(32), "shard-1": new Uint8Array(32) },
      [5, 7],
    ),
    cached: cachedPackage(),
    queue: memoryQueue(events),
    uploadLaneBytes: 8,
    signal: new AbortController().signal,
    async createDriver(directory) {
      assert.equal("destroy" in directory, false);
      assert.equal("allocations" in directory, false);
      events.push(`create:${directory.allocatedBytes}`);
      return { marker: "driver" };
    },
  });

  assert.equal(initialized.driver.marker, "driver");
  assert.equal(events.at(-2), "upload-complete");
  assert.equal(events.at(-1), "create:24");
});
