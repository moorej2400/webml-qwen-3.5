import assert from "node:assert/strict";
import test from "node:test";

async function loadHybridStateModule(): Promise<Record<string, unknown>> {
  return import("../src/hybrid-state.js").catch(() => ({}));
}

test("plans exact independently bindable Qwen3.5 hybrid resources at 16K", async () => {
  const module = await loadHybridStateModule();
  assert.equal(typeof module.planQwen35HybridState, "function");

  const plan = (
    module.planQwen35HybridState as (capacity: number) => {
      readonly resources: readonly {
        readonly layer: number;
        readonly kind: string;
        readonly bytes: bigint;
      }[];
      readonly totalBytes: bigint;
    }
  )(16_384);

  assert.equal(plan.resources.length, 64);
  assert.equal(plan.totalBytes, 590_348_288n);
  assert.deepEqual(
    plan.resources
      .filter(({ kind }) => kind === "key")
      .map(({ layer }) => layer),
    [3, 7, 11, 15, 19, 23, 27, 31],
  );
  assert.deepEqual(
    plan.resources
      .filter(({ kind }) => kind === "value")
      .map(({ layer }) => layer),
    [3, 7, 11, 15, 19, 23, 27, 31],
  );
  assert.equal(
    plan.resources.filter(({ kind }) => kind === "conv").length,
    24,
  );
  assert.equal(
    plan.resources.filter(({ kind }) => kind === "recurrent").length,
    24,
  );
  assert.equal(
    plan.resources
      .filter(({ kind }) => kind === "key" || kind === "value")
      .every(({ bytes }) => bytes === 33_554_432n),
    true,
  );
  assert.equal(
    plan.resources
      .filter(({ kind }) => kind === "conv")
      .every(({ bytes }) => bytes === 131_072n),
    true,
  );
  assert.equal(
    plan.resources
      .filter(({ kind }) => kind === "recurrent")
      .every(({ bytes }) => bytes === 2_097_152n),
    true,
  );
});

test("allocates, resets, advances, and disposes every logical state exactly once", async () => {
  const module = await loadHybridStateModule();
  assert.equal(typeof module.createQwen35HybridState, "function");
  const destroyed: string[] = [];
  const cleared: string[] = [];
  const allocated: Array<{
    readonly id: string;
    readonly byteLength: bigint;
    readonly category: string;
  }> = [];
  const arena = {
    async allocate(request: {
      readonly id: string;
      readonly byteLength: bigint;
      readonly category: string;
    }) {
      allocated.push(request);
      let didDestroy = false;
      return {
        shards: [{
          buffer: { destroy() {} },
          logicalByteOffset: 0n,
          logicalByteLength: request.byteLength,
          allocatedByteLength: request.byteLength,
        }],
        logicalBytes: request.byteLength,
        allocatedBytes: request.byteLength,
        destroy() {
          if (!didDestroy) {
            didDestroy = true;
            destroyed.push(request.id);
          }
        },
      };
    },
  };
  const state = await (
    module.createQwen35HybridState as (options: {
      readonly arena: typeof arena;
      readonly capacity: number;
      readonly clearAllocation: (
        allocation: unknown,
        resource: { readonly id: string },
      ) => Promise<void>;
    }) => Promise<{
      readonly capacity: number;
      readonly position: number;
      readonly byteLength: bigint;
      readonly resourceCount: number;
      advance(tokens: number): { readonly start: number; readonly end: number };
      reset(): Promise<void>;
      dispose(): void;
    }>
  )({
    arena,
    capacity: 16_384,
    clearAllocation: async (_allocation, resource) => {
      cleared.push(resource.id);
    },
  });

  assert.equal(state.capacity, 16_384);
  assert.equal(state.position, 0);
  assert.equal(state.byteLength, 590_348_288n);
  assert.equal(state.resourceCount, 64);
  assert.equal(allocated.length, 64);
  assert.deepEqual(state.advance(2), { start: 0, end: 2 });
  assert.equal(state.position, 2);
  await state.reset();
  assert.equal(state.position, 0);
  assert.equal(cleared.length, 64);
  assert.throws(() => state.advance(16_385), /capacity/i);
  assert.throws(() => state.advance(0), /positive/i);

  state.dispose();
  state.dispose();
  assert.equal(destroyed.length, 64);
  assert.throws(() => state.advance(1), /disposed/i);
  await assert.rejects(state.reset(), /disposed/i);
});

test("rolls back all owned allocations when state creation fails", async () => {
  const module = await loadHybridStateModule();
  const destroyed: string[] = [];
  let allocationCount = 0;
  const arena = {
    async allocate(request: { readonly id: string; readonly byteLength: bigint }) {
      allocationCount += 1;
      if (allocationCount === 7) {
        throw new Error("synthetic allocation failure");
      }
      return {
        shards: [{
          buffer: { destroy() {} },
          logicalByteOffset: 0n,
          logicalByteLength: request.byteLength,
          allocatedByteLength: request.byteLength,
        }],
        logicalBytes: request.byteLength,
        allocatedBytes: request.byteLength,
        destroy() {
          destroyed.push(request.id);
        },
      };
    },
  };

  await assert.rejects(
    (
      module.createQwen35HybridState as (options: {
        readonly arena: typeof arena;
        readonly capacity: number;
        readonly clearAllocation: () => Promise<void>;
      }) => Promise<unknown>
    )({
      arena,
      capacity: 16_384,
      clearAllocation: async () => {},
    }),
    /state allocation failed/i,
  );
  assert.equal(allocationCount, 7);
  assert.equal(destroyed.length, 6);
  assert.equal(new Set(destroyed).size, 6);
});

test("poisons partially cleared state so only disposal remains legal", async () => {
  const module = await loadHybridStateModule();
  let clearCount = 0;
  let destroyCount = 0;
  const arena = {
    async allocate(request: { readonly byteLength: bigint }) {
      return {
        shards: [{
          buffer: { destroy() {} },
          logicalByteOffset: 0n,
          logicalByteLength: request.byteLength,
          allocatedByteLength: request.byteLength,
        }],
        logicalBytes: request.byteLength,
        allocatedBytes: request.byteLength,
        destroy() {
          destroyCount += 1;
        },
      };
    },
  };
  const state = await (
    module.createQwen35HybridState as (options: {
      readonly arena: typeof arena;
      readonly capacity: number;
      readonly clearAllocation: () => Promise<void>;
    }) => Promise<{
      advance(tokens: number): unknown;
      reset(): Promise<void>;
      dispose(): void;
    }>
  )({
    arena,
    capacity: 4,
    clearAllocation: async () => {
      clearCount += 1;
      if (clearCount === 2) {
        throw new Error("synthetic clear failure");
      }
    },
  });
  state.advance(1);

  await assert.rejects(state.reset(), /reset failed/i);
  assert.throws(() => state.advance(1), /poisoned/i);
  await assert.rejects(state.reset(), /poisoned/i);
  assert.doesNotThrow(() => state.dispose());
  assert.equal(destroyCount, 64);
});

test("validates all supported state capacities and positions", async () => {
  const module = await loadHybridStateModule();
  const plan = module.planQwen35HybridState as (capacity: number) => unknown;
  for (const invalid of [0, -1, 16_385, 1.5, Number.NaN]) {
    assert.throws(() => plan(invalid), /capacity/i);
  }
  for (const boundary of [1, 2, 4, 63, 64, 65, 16_384]) {
    assert.doesNotThrow(() => plan(boundary));
  }
});

interface TestShard {
  readonly buffer: object;
  readonly logicalByteOffset: bigint;
  readonly logicalByteLength: bigint;
  readonly allocatedByteLength: bigint;
}

interface TestResourceView {
  readonly id: string;
  readonly layer: number;
  readonly kind: "key" | "value" | "conv" | "recurrent";
  readonly byteLength: bigint;
  readonly shards: readonly TestShard[];
}

interface TestHybridState {
  readonly byteLength: bigint;
  getResource(
    layer: number,
    kind: TestResourceView["kind"],
  ): TestResourceView;
  getLayerResources(layer: number):
    | {
        readonly layer: number;
        readonly kind: "full-attention";
        readonly key: TestResourceView;
        readonly value: TestResourceView;
      }
    | {
        readonly layer: number;
        readonly kind: "gated-deltanet";
        readonly conv: TestResourceView;
        readonly recurrent: TestResourceView;
      };
  reset(): Promise<void>;
  dispose(): void;
}

test("exposes immutable resource views for all 32 model layers", async () => {
  const module = await loadHybridStateModule();
  const requests: Array<{
    readonly id: string;
    readonly byteLength: bigint;
    readonly requiredShardQuantumBytes: bigint;
  }> = [];
  const allocations = new Map<string, { readonly shards: readonly TestShard[] }>();
  const arena = {
    async allocate(request: {
      readonly id: string;
      readonly byteLength: bigint;
      readonly requiredShardQuantumBytes: bigint;
    }) {
      requests.push(request);
      const buffer = { destroy() {} };
      const shards = [{
        buffer,
        logicalByteOffset: 0n,
        logicalByteLength: request.byteLength,
        allocatedByteLength: request.byteLength,
      }];
      const allocation = {
        shards,
        logicalBytes: request.byteLength,
        allocatedBytes: request.byteLength,
        destroy() {},
      };
      allocations.set(request.id, allocation);
      return allocation;
    },
  };
  const state = await (
    module.createQwen35HybridState as (options: {
      readonly arena: typeof arena;
      readonly capacity: number;
      readonly clearAllocation: () => Promise<void>;
    }) => Promise<TestHybridState>
  )({ arena, capacity: 16_384, clearAllocation: async () => {} });

  for (let layer = 0; layer < 32; layer += 1) {
    const resources = state.getLayerResources(layer);
    assert.equal(Object.isFrozen(resources), true);
    if ([3, 7, 11, 15, 19, 23, 27, 31].includes(layer)) {
      assert.equal(resources.kind, "full-attention");
      if (resources.kind !== "full-attention") throw new Error("wrong layer kind");
      assert.equal(resources.key, state.getResource(layer, "key"));
      assert.equal(resources.value, state.getResource(layer, "value"));
      assert.notEqual(resources.key, resources.value);
    } else {
      assert.equal(resources.kind, "gated-deltanet");
      if (resources.kind !== "gated-deltanet") throw new Error("wrong layer kind");
      assert.equal(resources.conv, state.getResource(layer, "conv"));
      assert.equal(resources.recurrent, state.getResource(layer, "recurrent"));
    }
  }

  const key = state.getResource(3, "key");
  const ownedKey = allocations.get(key.id)!;
  assert.equal(Object.isFrozen(key), true);
  assert.equal(Object.isFrozen(key.shards), true);
  assert.equal(Object.isFrozen(key.shards[0]), true);
  assert.equal(key.shards[0]!.buffer, ownedKey.shards[0]!.buffer);
  assert.equal("destroy" in key, false);
  assert.equal("release" in key, false);
  assert.equal("destroy" in key.shards[0]!, false);
  assert.equal("release" in key.shards[0]!, false);
  assert.throws(() => state.getResource(3, "conv"), /resource/i);
  assert.throws(() => state.getLayerResources(32), /layer/i);
  assert.equal(requests.length, 64);
  assert.equal(
    requests
      .filter(({ id }) => id.endsWith("-key") || id.endsWith("-value"))
      .every(({ requiredShardQuantumBytes }) => requiredShardQuantumBytes === 2_048n),
    true,
  );
  assert.equal(
    requests
      .filter(({ id }) => id.endsWith("-conv"))
      .every(({ requiredShardQuantumBytes }) => requiredShardQuantumBytes === 16n),
    true,
  );
  assert.equal(
    requests
      .filter(({ id }) => id.endsWith("-recurrent"))
      .every(({ requiredShardQuantumBytes }) => requiredShardQuantumBytes === 512n),
    true,
  );

  const keyBeforeReset = state.getResource(3, "key");
  await state.reset();
  assert.equal(state.getResource(3, "key"), keyBeforeReset);

  state.dispose();
  assert.throws(() => state.getResource(3, "key"), /disposed/i);
});

test("keeps K and V as independent row-aligned pages under small buffer caps", async () => {
  const module = await loadHybridStateModule();
  const allocations = new Map<string, { readonly shards: readonly TestShard[] }>();
  const arena = {
    async allocate(request: {
      readonly id: string;
      readonly byteLength: bigint;
      readonly requiredShardQuantumBytes: bigint;
    }) {
      const cap = request.id.endsWith("-key") || request.id.endsWith("-value")
        ? 4_096n
        : request.byteLength;
      const shards: Array<{
        readonly buffer: { destroy(): void };
        readonly logicalByteOffset: bigint;
        readonly logicalByteLength: bigint;
        readonly allocatedByteLength: bigint;
      }> = [];
      let offset = 0n;
      while (offset < request.byteLength) {
        const length = request.byteLength - offset < cap
          ? request.byteLength - offset
          : cap;
        assert.equal(length % request.requiredShardQuantumBytes, 0n);
        shards.push({
          buffer: { destroy() {} },
          logicalByteOffset: offset,
          logicalByteLength: length,
          allocatedByteLength: length,
        });
        offset += length;
      }
      const allocation = {
        shards,
        logicalBytes: request.byteLength,
        allocatedBytes: request.byteLength,
        destroy() {},
      };
      allocations.set(request.id, allocation);
      return allocation;
    },
  };
  const state = await (
    module.createQwen35HybridState as (options: {
      readonly arena: typeof arena;
      readonly capacity: number;
      readonly clearAllocation: () => Promise<void>;
    }) => Promise<TestHybridState>
  )({ arena, capacity: 5, clearAllocation: async () => {} });

  const key = state.getResource(3, "key");
  const value = state.getResource(3, "value");
  assert.equal(key.byteLength, 10_240n);
  assert.equal(value.byteLength, 10_240n);
  assert.deepEqual(
    key.shards.map(({ logicalByteOffset, logicalByteLength }) => [
      logicalByteOffset,
      logicalByteLength,
    ]),
    [[0n, 4_096n], [4_096n, 4_096n], [8_192n, 2_048n]],
  );
  assert.deepEqual(
    value.shards.map(({ logicalByteOffset, logicalByteLength }) => [
      logicalByteOffset,
      logicalByteLength,
    ]),
    [[0n, 4_096n], [4_096n, 4_096n], [8_192n, 2_048n]],
  );
  assert.equal(
    key.shards[1]!.buffer,
    allocations.get(key.id)!.shards[1]!.buffer,
  );
  assert.equal(
    value.shards[2]!.buffer,
    allocations.get(value.id)!.shards[2]!.buffer,
  );
  assert.equal(state.byteLength, 53_641_216n);

  state.dispose();
});

test("destroys a malformed returned allocation once before rollback", async () => {
  const module = await loadHybridStateModule();
  let destroyCount = 0;
  const arena = {
    async allocate(request: { readonly byteLength: bigint }) {
      return {
        shards: [{
          buffer: { destroy() {} },
          logicalByteOffset: 2_048n,
          logicalByteLength: request.byteLength,
          allocatedByteLength: request.byteLength,
        }],
        logicalBytes: request.byteLength,
        allocatedBytes: request.byteLength,
        destroy() {
          destroyCount += 1;
        },
      };
    },
  };

  await assert.rejects(
    (
      module.createQwen35HybridState as (options: {
        readonly arena: typeof arena;
        readonly capacity: number;
        readonly clearAllocation: () => Promise<void>;
      }) => Promise<unknown>
    )({ arena, capacity: 16_384, clearAllocation: async () => {} }),
    /state allocation failed/i,
  );
  assert.equal(destroyCount, 1);
});
