import assert from "node:assert/strict";
import test from "node:test";

async function loadHybridStateModule(): Promise<Record<string, unknown>> {
  return import("../src/hybrid-state.js").catch(() => ({}));
}

test("plans exactly the Qwen3.5 4B hybrid state at 16K", async () => {
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

  assert.equal(plan.resources.length, 56);
  assert.equal(plan.totalBytes, 590_348_288n);
  assert.deepEqual(
    plan.resources
      .filter(({ kind }) => kind === "kv-pair")
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
      .filter(({ kind }) => kind === "kv-pair")
      .every(({ bytes }) => bytes === 67_108_864n),
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
        shards: [],
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
  assert.equal(state.resourceCount, 56);
  assert.equal(allocated.length, 56);
  assert.deepEqual(state.advance(2), { start: 0, end: 2 });
  assert.equal(state.position, 2);
  await state.reset();
  assert.equal(state.position, 0);
  assert.equal(cleared.length, 56);
  assert.throws(() => state.advance(16_385), /capacity/i);
  assert.throws(() => state.advance(0), /positive/i);

  state.dispose();
  state.dispose();
  assert.equal(destroyed.length, 56);
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
        shards: [],
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
});

test("poisons partially cleared state so only disposal remains legal", async () => {
  const module = await loadHybridStateModule();
  let clearCount = 0;
  let destroyCount = 0;
  const arena = {
    async allocate(request: { readonly byteLength: bigint }) {
      return {
        shards: [],
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
  assert.equal(destroyCount, 56);
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
