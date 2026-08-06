import assert from "node:assert/strict";
import test from "node:test";

import { AllocationLedger } from "../src/allocation-ledger.js";
import { diagnosticError } from "../src/diagnostics.js";
import { GpuArena, type GpuAllocationRequest } from "../src/gpu-arena.js";

const LOGICAL_CONTEXT_CAPACITY = 16_384;
const KV_PAGE_TOKENS = 256;
const FIRST_RESIDENT_BYTES = 61_865_984n;
const SECOND_RESIDENT_BYTES = 70_254_592n;
const FULL_RESIDENT_BYTES = 590_348_288n;

interface TestShard {
  readonly buffer: object;
  readonly logicalByteOffset: bigint;
  readonly logicalByteLength: bigint;
  readonly allocatedByteLength: bigint;
}

interface TestResourceView {
  readonly byteLength: bigint;
  readonly shards: readonly TestShard[];
}

interface TestLazyHybridState {
  readonly capacity: number;
  readonly residentCapacity: number;
  readonly position: number;
  readonly byteLength: bigint;
  readonly resourceCount: number;
  ensureCapacity(requiredEnd: number, signal?: AbortSignal): Promise<void>;
  advance(tokens: number): { readonly start: number; readonly end: number };
  getResource(
    layer: number,
    kind: "key" | "value" | "conv" | "recurrent",
  ): TestResourceView;
  reset(): Promise<void>;
  dispose(): void;
}

interface AllocationRecord {
  readonly request: GpuAllocationRequest;
  readonly destroy: () => void;
  readonly destroyed: () => boolean;
}

function trackingArena(options: {
  readonly failAtRequest?: number;
  readonly failure?: unknown;
  readonly afterAllocate?: (requestNumber: number) => void;
} = {}): {
  readonly arena: {
    allocate(request: GpuAllocationRequest): Promise<{
      readonly shards: readonly TestShard[];
      readonly logicalBytes: bigint;
      readonly allocatedBytes: bigint;
      destroy(): void;
    }>;
  };
  readonly records: AllocationRecord[];
  readonly requests: GpuAllocationRequest[];
  readonly requestCount: () => number;
} {
  const records: AllocationRecord[] = [];
  const requests: GpuAllocationRequest[] = [];
  let requestCount = 0;
  return {
    arena: {
      async allocate(request) {
        requestCount += 1;
        requests.push(request);
        if (requestCount === options.failAtRequest) {
          throw options.failure ?? new Error("synthetic page allocation failure");
        }
        let destroyed = false;
        const record = {
          request,
          destroy() {
            destroyed = true;
          },
          destroyed: () => destroyed,
        };
        records.push(record);
        options.afterAllocate?.(requestCount);
        return {
          shards: Object.freeze([Object.freeze({
            buffer: {},
            logicalByteOffset: 0n,
            logicalByteLength: request.byteLength,
            allocatedByteLength: request.byteLength,
          })]),
          logicalBytes: request.byteLength,
          allocatedBytes: request.byteLength,
          destroy: record.destroy,
        };
      },
    },
    records,
    requests,
    requestCount: () => requestCount,
  };
}

async function createState(
  arena: ReturnType<typeof trackingArena>["arena"] | GpuArena,
): Promise<TestLazyHybridState> {
  const module = await import("../src/hybrid-state.js") as unknown as {
    createQwen35HybridState(options: {
      readonly arena: typeof arena;
      readonly capacity: number;
      readonly clearAllocation: () => Promise<void>;
    }): Promise<TestLazyHybridState>;
  };
  return module.createQwen35HybridState({
    arena,
    capacity: LOGICAL_CONTEXT_CAPACITY,
    clearAllocation: async () => {},
  });
}

test("keeps the complete 16K state non-resident during model load", async () => {
  const tracked = trackingArena();
  const state = await createState(tracked.arena);

  assert.equal(state.capacity, LOGICAL_CONTEXT_CAPACITY);
  assert.equal(state.residentCapacity, 0);
  assert.equal(state.byteLength, 0n);
  assert.equal(tracked.requestCount(), 0);
  assert.throws(() => state.advance(1), /resident|ensure|allocate/i);

  state.dispose();
  assert.equal(tracked.records.length, 0);
});

test("second DeltaNet allocation failure preserves its diagnostic code and rolls back conv", async () => {
  const tracked = trackingArena({
    failAtRequest: 2,
    failure: diagnosticError(
      "gpu_out_of_memory",
      "GPU buffer allocation failed",
    ),
  });
  const state = await createState(tracked.arena);

  await assert.rejects(
    state.ensureCapacity(1),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.equal(
        (error as Error & { readonly code?: unknown }).code,
        "gpu_out_of_memory",
      );
      assert.equal(error.message, "GPU buffer allocation failed");
      return true;
    },
  );

  assert.deepEqual(
    tracked.requests.map(({ byteLength }) => byteLength),
    [131_072n, 2_097_152n],
  );
  assert.equal(tracked.records.length, 1);
  assert.equal(tracked.records[0]?.destroyed(), true);
  assert.equal(state.residentCapacity, 0);
  assert.equal(state.byteLength, 0n);
  assert.equal(state.resourceCount, 0);
  state.dispose();
});

test("HybridState maps an unapproved arena diagnostic to unknown without raw text", async () => {
  const failure = Object.assign(
    diagnosticError(
      "private_dynamic_gpu_error",
      "private allocation message at local-path:<path>/<model-id>.gguf",
    ),
    {
      path: "local-path:<path>/<model-id>.gguf",
      stack: "private stack local-path:<path>/<model-id>.gguf",
    },
  );
  const tracked = trackingArena({
    failAtRequest: 2,
    failure,
  });
  const state = await createState(tracked.arena);

  await assert.rejects(
    state.ensureCapacity(1),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.equal(
        (error as Error & { readonly code?: unknown }).code,
        "unknown",
      );
      assert.doesNotMatch(
        error.message,
        /private|dynamic|local|model\.gguf|allocation message/i,
      );
      return true;
    },
  );
  assert.deepEqual(
    tracked.requests.map(({ byteLength }) => byteLength),
    [131_072n, 2_097_152n],
  );
  assert.equal(tracked.records[0]?.destroyed(), true);
  assert.equal(state.residentCapacity, 0);
  assert.equal(state.byteLength, 0n);
  state.dispose();
});

test("grows state in bounded pages and preserves the 16K logical target", async () => {
  const tracked = trackingArena();
  const state = await createState(tracked.arena);

  await state.ensureCapacity(1);
  assert.equal(state.residentCapacity, KV_PAGE_TOKENS);
  assert.equal(state.byteLength, FIRST_RESIDENT_BYTES);
  assert.ok(state.byteLength < 64n * 1_024n * 1_024n);
  assert.equal(tracked.requestCount(), 64);
  assert.equal(new Set(tracked.records.map(({ request }) => request.id)).size, 64);

  const firstKey = state.getResource(3, "key");
  assert.equal(firstKey.byteLength, 524_288n);
  assert.deepEqual(
    firstKey.shards.map(({ logicalByteOffset, logicalByteLength }) => [
      logicalByteOffset,
      logicalByteLength,
    ]),
    [[0n, 524_288n]],
  );

  await state.ensureCapacity(KV_PAGE_TOKENS);
  assert.equal(tracked.requestCount(), 64);

  await state.ensureCapacity(KV_PAGE_TOKENS + 1);
  assert.equal(state.residentCapacity, KV_PAGE_TOKENS * 2);
  assert.equal(state.byteLength, SECOND_RESIDENT_BYTES);
  assert.equal(tracked.requestCount(), 80);
  assert.deepEqual(
    state.getResource(3, "key").shards.map(
      ({ logicalByteOffset, logicalByteLength }) => [
        logicalByteOffset,
        logicalByteLength,
      ],
    ),
    [[0n, 524_288n], [524_288n, 524_288n]],
  );

  assert.deepEqual(state.advance(KV_PAGE_TOKENS * 2), {
    start: 0,
    end: KV_PAGE_TOKENS * 2,
  });
  assert.throws(() => state.advance(1), /resident|ensure|allocate/i);

  await state.ensureCapacity(LOGICAL_CONTEXT_CAPACITY);
  assert.equal(state.residentCapacity, LOGICAL_CONTEXT_CAPACITY);
  assert.equal(state.byteLength, FULL_RESIDENT_BYTES);
  assert.equal(state.getResource(3, "key").shards.length, 64);
  assert.equal(
    state.getResource(3, "key").shards.at(-1)?.logicalByteOffset,
    63n * 524_288n,
  );
  assert.deepEqual(state.advance(LOGICAL_CONTEXT_CAPACITY - 512), {
    start: 512,
    end: LOGICAL_CONTEXT_CAPACITY,
  });
  await assert.rejects(
    state.ensureCapacity(LOGICAL_CONTEXT_CAPACITY + 1),
    /capacity|16384/i,
  );

  state.dispose();
  assert.equal(tracked.records.every(({ destroyed }) => destroyed()), true);
});

test("rolls back a failed page growth without destroying resident state", async () => {
  const tracked = trackingArena({ failAtRequest: 68 });
  const state = await createState(tracked.arena);

  await state.ensureCapacity(1);
  const stable = tracked.records.slice();
  await assert.rejects(
    state.ensureCapacity(KV_PAGE_TOKENS + 1),
    /allocation|growth|state/i,
  );

  assert.equal(state.residentCapacity, KV_PAGE_TOKENS);
  assert.equal(state.byteLength, FIRST_RESIDENT_BYTES);
  assert.equal(state.getResource(3, "key").shards.length, 1);
  assert.equal(stable.every(({ destroyed }) => !destroyed()), true);
  assert.equal(
    tracked.records.slice(stable.length).every(({ destroyed }) => destroyed()),
    true,
  );

  state.dispose();
  assert.equal(stable.every(({ destroyed }) => destroyed()), true);
});

test("snapshots validated page metadata before the transactional commit", async () => {
  const records: Array<{ destroyed: boolean }> = [];
  let requestCount = 0;
  let firstShard:
    | {
        logicalByteOffset: bigint;
        logicalByteLength: bigint;
        allocatedByteLength: bigint;
        buffer: object;
      }
    | undefined;
  const arena = {
    async allocate(request: GpuAllocationRequest) {
      requestCount += 1;
      const shard = {
        buffer: {},
        logicalByteOffset: 0n,
        logicalByteLength: request.byteLength,
        allocatedByteLength: request.byteLength,
      };
      if (firstShard === undefined) firstShard = shard;
      if (requestCount === 64) firstShard.logicalByteOffset = 4n;
      const record = { destroyed: false };
      records.push(record);
      return {
        shards: [shard],
        logicalBytes: request.byteLength,
        allocatedBytes: request.byteLength,
        destroy() { record.destroyed = true; },
      };
    },
  };
  const state = await createState(arena);

  await assert.rejects(state.ensureCapacity(1), /allocation|growth|state/i);
  assert.equal(state.residentCapacity, 0);
  assert.equal(state.byteLength, 0n);
  assert.equal(state.resourceCount, 0);
  assert.equal(records.every(({ destroyed }) => destroyed), true);

  state.dispose();
});

test("cancellation rolls back an in-flight page and leaves the prior capacity usable", async () => {
  const controller = new AbortController();
  const tracked = trackingArena({
    afterAllocate(requestNumber) {
      if (requestNumber === 65) controller.abort();
    },
  });
  const state = await createState(tracked.arena);

  await state.ensureCapacity(1);
  const stable = tracked.records.slice();
  await assert.rejects(
    state.ensureCapacity(KV_PAGE_TOKENS + 1, controller.signal),
    (error: unknown) => error instanceof DOMException && error.name === "AbortError",
  );

  assert.equal(state.residentCapacity, KV_PAGE_TOKENS);
  assert.equal(state.byteLength, FIRST_RESIDENT_BYTES);
  assert.equal(stable.every(({ destroyed }) => !destroyed()), true);
  assert.equal(tracked.records.length, stable.length + 1);
  assert.equal(tracked.records.at(-1)?.destroyed(), true);
  assert.deepEqual(state.advance(1), { start: 0, end: 1 });

  state.dispose();
  assert.equal(stable.every(({ destroyed }) => destroyed()), true);
});

test("dispose during growth releases the allocation that resolves after disposal", async () => {
  let resolveAllocation!: () => void;
  const allocationGate = new Promise<void>((resolve) => {
    resolveAllocation = resolve;
  });
  const tracked = trackingArena();
  const gatedArena = {
    async allocate(request: GpuAllocationRequest) {
      await allocationGate;
      return tracked.arena.allocate(request);
    },
  };
  const state = await createState(gatedArena);

  const growth = state.ensureCapacity(1);
  await Promise.resolve();
  state.dispose();
  resolveAllocation();

  await assert.rejects(growth, /disposed|growth|allocation/i);
  assert.equal(tracked.records.length, 1);
  assert.equal(tracked.records[0]?.destroyed(), true);
  assert.throws(() => state.advance(1), /disposed/i);
});

test("the production allocation ledger reports real current and peak page bytes", async () => {
  const ledger = new AllocationLedger(1_024n * 1_024n * 1_024n);
  const device = {
    limits: {
      maxBufferSize: 256 * 1_024 * 1_024,
      maxStorageBufferBindingSize: 256 * 1_024 * 1_024,
    },
    pushErrorScope() {},
    async popErrorScope() { return null; },
    createBuffer() { return { destroy() {} }; },
  };
  const arena = new GpuArena(device, ledger, {
    bufferShardCapBytes: 256n * 1_024n * 1_024n,
  });
  const state = await createState(arena);

  assert.deepEqual(
    { current: ledger.snapshot().currentBytes, peak: ledger.snapshot().peakBytes },
    { current: 0n, peak: 0n },
  );
  await state.ensureCapacity(1);
  assert.deepEqual(
    { current: ledger.snapshot().currentBytes, peak: ledger.snapshot().peakBytes },
    { current: FIRST_RESIDENT_BYTES, peak: FIRST_RESIDENT_BYTES },
  );
  await state.ensureCapacity(KV_PAGE_TOKENS + 1);
  assert.deepEqual(
    { current: ledger.snapshot().currentBytes, peak: ledger.snapshot().peakBytes },
    { current: SECOND_RESIDENT_BYTES, peak: SECOND_RESIDENT_BYTES },
  );

  state.dispose();
  assert.deepEqual(
    { current: ledger.snapshot().currentBytes, peak: ledger.snapshot().peakBytes },
    { current: 0n, peak: SECOND_RESIDENT_BYTES },
  );
  assert.doesNotThrow(() => ledger.assertAllReleased());
});

test("HybridState distinguishes a progress callback failure from GPU allocation", async () => {
  const tracked = trackingArena();
  const { createQwen35HybridState } = await import("../src/hybrid-state.js");
  const state = await createQwen35HybridState({
    arena: tracked.arena,
    capacity: LOGICAL_CONTEXT_CAPACITY,
    clearAllocation: async () => {},
    onProgress() { throw new Error("private callback text"); },
  });

  await assert.rejects(
    state.ensureCapacity(1),
    (error: unknown) => {
      assert.equal((error as { readonly code?: unknown }).code, "state_progress");
      assert.doesNotMatch((error as Error).message, /private|callback/i);
      return true;
    },
  );
  state.dispose();
});
