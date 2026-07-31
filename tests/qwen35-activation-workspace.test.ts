import assert from "node:assert/strict";
import test from "node:test";

import type {
  GpuAllocation,
  GpuAllocationRequest,
} from "../src/gpu-arena.js";
import {
  createQwen35ActivationWorkspace,
  planQwen35ActivationWorkspace,
  type Qwen35ActivationResourceView,
} from "../src/qwen35-activation-workspace.js";

interface FakeBuffer {
  readonly ordinal: number;
  destroyCount: number;
  destroy(): void;
}

class RecordingArena {
  readonly requests: GpuAllocationRequest[] = [];
  readonly buffers: FakeBuffer[] = [];
  failAt: number | undefined;
  malformedAt: number | undefined;

  async allocate(request: GpuAllocationRequest): Promise<GpuAllocation> {
    const ordinal = this.requests.length;
    this.requests.push(request);
    if (ordinal === this.failAt) {
      throw new Error("synthetic device detail");
    }
    const buffer: FakeBuffer = {
      ordinal,
      destroyCount: 0,
      destroy() {
        this.destroyCount += 1;
      },
    };
    this.buffers.push(buffer);
    const malformed = ordinal === this.malformedAt;
    return {
      shards: Object.freeze([
        Object.freeze({
          buffer,
          logicalByteOffset: malformed ? 4n : 0n,
          logicalByteLength: request.byteLength,
          allocatedByteLength: request.byteLength,
        }),
      ]),
      logicalBytes: request.byteLength,
      allocatedBytes: request.byteLength,
      destroy: () => buffer.destroy(),
    };
  }
}

function deferred(): {
  readonly promise: Promise<void>;
  readonly resolve: () => void;
} {
  let resolve!: () => void;
  const promise = new Promise<void>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

test("plans the exact one-token Qwen3.5 activation widths without a vocabulary matrix", () => {
  const plan = planQwen35ActivationWorkspace();
  const elements = Object.fromEntries(
    plan.resources.map((resource) => [resource.kind, resource.elementCount]),
  );

  assert.equal(plan.resourceCount, 18);
  assert.equal(plan.vocabularySize, 248_320);
  assert.equal(plan.logitsTileRows, 1_024);
  assert.deepEqual(elements, {
    "packed-embedding-output": 2_560,
    "hidden-secondary": 2_560,
    "normalized-hidden": 2_560,
    "attention-projection-primary": 8_192,
    "attention-projection-secondary": 8_192,
    "attention-inner-primary": 4_096,
    "attention-inner-secondary": 4_096,
    "full-attention-key": 1_024,
    "full-attention-value": 1_024,
    "deltanet-alpha": 32,
    "deltanet-beta": 32,
    "ffn-gate": 9_216,
    "ffn-up": 9_216,
    "ffn-product": 9_216,
    "logits-tile": 1_024,
    "top-k-scores": 256,
    "top-k-indices": 256,
    "selected-token": 1,
  });
  assert.equal(plan.totalBytes, 254_212n);
  assert.ok(
    plan.resources.every(
      (resource) =>
        resource.elementCount !== plan.vocabularySize &&
        resource.bytes !== BigInt(plan.vocabularySize * 2) &&
        resource.bytes !== BigInt(plan.vocabularySize * 4),
    ),
  );
  assert.equal(Object.isFrozen(plan), true);
  assert.equal(Object.isFrozen(plan.resources), true);
});

test("keeps mathematical logits tiles separate from physical dispatch planning", () => {
  const plan = planQwen35ActivationWorkspace();

  assert.equal(plan.mathematicalVocabularyTileCount, 243);
  assert.equal("logitsTileCount" in plan, false);
  assert.equal("logitsDispatchCount" in plan, false);
  assert.equal(plan.topKCandidateCapacity, 256);
  assert.equal(plan.selectedTokenInvalidSentinel, 0xffff_ffff);
  assert.ok(
    plan.topKCandidateCapacity >= plan.mathematicalVocabularyTileCount,
  );
  assert.equal(
    plan.resources.map(({ kind }) => String(kind)).includes("top-k-valid-count"),
    false,
  );
  const readbackResources = plan.resources
    .filter(({ usage }) => (usage & 0x0004) !== 0)
    .map(({ kind }) => kind);
  assert.deepEqual(readbackResources, ["selected-token"]);
});

test("maps four non-overlapping DeltaNet parameter ranges only for DeltaNet layers", () => {
  const { deltanetParameterLiveness } = planQwen35ActivationWorkspace();

  assert.equal(deltanetParameterLiveness.activeLayerKind, "gated-deltanet");
  assert.deepEqual(
    deltanetParameterLiveness.borrowedFullAttentionResources,
    ["full-attention-key", "full-attention-value"],
  );
  assert.deepEqual(deltanetParameterLiveness.ranges, {
    rawAlpha: {
      resource: "deltanet-alpha",
      byteOffset: 0,
      byteLength: 128,
      access: "read",
    },
    rawBeta: {
      resource: "deltanet-beta",
      byteOffset: 0,
      byteLength: 128,
      access: "read",
    },
    transformedBeta: {
      resource: "full-attention-key",
      byteOffset: 0,
      byteLength: 128,
      access: "write",
    },
    decay: {
      resource: "full-attention-value",
      byteOffset: 0,
      byteLength: 128,
      access: "write",
    },
  });
  const ranges = Object.values(deltanetParameterLiveness.ranges);
  assert.equal(new Set(ranges.map(({ resource }) => resource)).size, 4);
  for (const [index, left] of ranges.entries()) {
    for (const right of ranges.slice(index + 1)) {
      const overlaps =
        left.resource === right.resource &&
        left.byteOffset < right.byteOffset + right.byteLength &&
        right.byteOffset < left.byteOffset + left.byteLength;
      assert.equal(overlaps, false);
    }
  }
  assert.equal(Object.isFrozen(deltanetParameterLiveness), true);
  assert.equal(Object.isFrozen(deltanetParameterLiveness.ranges), true);
  assert.ok(ranges.every((range) => Object.isFrozen(range)));
});

test("allocates bounded complete vectors and exposes stable non-owning binding views", async () => {
  const arena = new RecordingArena();
  const workspace = await createQwen35ActivationWorkspace({
    arena,
    async clearAllocation() {},
  });
  const first = workspace.get("ffn-gate");
  const second = workspace.get("ffn-gate");

  assert.equal(workspace.resourceCount, 18);
  assert.equal(arena.requests.length, 18);
  assert.equal(first, second);
  assert.equal(first.elementCount, 9_216);
  assert.equal(first.byteLength, 36_864);
  assert.equal(first.binding.offset, 0);
  assert.equal(first.binding.size, 36_864);
  assert.equal("destroy" in first.binding, false);
  assert.equal(Object.isFrozen(first), true);
  assert.equal(Object.isFrozen(first.binding), true);
  assert.equal(Object.isFrozen(workspace.resources), true);
  assert.ok(
    arena.requests.every(
      (request) =>
        request.category === "activation" &&
        request.alignment === 4 &&
        request.requiredShardQuantumBytes === request.byteLength,
    ),
  );

  await workspace.dispose();
  assert.deepEqual(
    arena.buffers.map((buffer) => buffer.destroyCount),
    new Array(18).fill(1),
  );
});

test("clears every allocation before a workspace is reused", async () => {
  const arena = new RecordingArena();
  const cleared: string[] = [];
  const workspace = await createQwen35ActivationWorkspace({
    arena,
    async clearAllocation(_allocation, resource) {
      cleared.push(resource.kind);
    },
  });

  await workspace.reset();

  assert.deepEqual(
    cleared,
    planQwen35ActivationWorkspace().resources.map((resource) => resource.kind),
  );
  assert.equal(workspace.get("normalized-hidden").elementCount, 2_560);
  await workspace.dispose();
});

test("poisons a partially cleared workspace while keeping disposal available", async () => {
  const arena = new RecordingArena();
  let clearCount = 0;
  const workspace = await createQwen35ActivationWorkspace({
    arena,
    async clearAllocation() {
      clearCount += 1;
      if (clearCount === 3) throw new Error("private clear detail");
    },
  });

  await assert.rejects(workspace.reset(), {
    message: "Qwen3.5 activation workspace reset failed",
  });
  assert.throws(
    () => workspace.get("packed-embedding-output"),
    /workspace is poisoned/i,
  );
  await assert.rejects(workspace.reset(), /workspace is poisoned/i);
  await assert.doesNotReject(workspace.dispose());
  assert.deepEqual(
    arena.buffers.map((buffer) => buffer.destroyCount),
    new Array(18).fill(1),
  );
});

test("rolls back a partial layout allocation in reverse ownership order", async () => {
  const arena = new RecordingArena();
  arena.failAt = 4;

  await assert.rejects(
    createQwen35ActivationWorkspace({
      arena,
      async clearAllocation() {},
    }),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.equal(
        error.message,
        "Qwen3.5 activation workspace layout allocation failed",
      );
      assert.doesNotMatch(error.message, /browser|impossible|synthetic/i);
      return true;
    },
  );
  assert.deepEqual(
    arena.buffers.map((buffer) => [buffer.ordinal, buffer.destroyCount]),
    [[0, 1], [1, 1], [2, 1], [3, 1]],
  );
});

test("rejects malformed arena metadata without exposing allocation ownership", async () => {
  const arena = new RecordingArena();
  arena.malformedAt = 2;

  await assert.rejects(
    createQwen35ActivationWorkspace({
      arena,
      async clearAllocation() {},
    }),
    {
      message: "Qwen3.5 activation workspace layout allocation failed",
    },
  );
  assert.deepEqual(
    arena.buffers.map((buffer) => buffer.destroyCount),
    [1, 1, 1],
  );
});

test("disposal waits for an in-flight reset and remains idempotent", async () => {
  const arena = new RecordingArena();
  const gate = deferred();
  let firstResource: Qwen35ActivationResourceView | undefined;
  const workspace = await createQwen35ActivationWorkspace({
    arena,
    async clearAllocation(_allocation, resource) {
      firstResource ??= resource;
      if (resource === firstResource) await gate.promise;
    },
  });
  const reset = workspace.reset();
  const disposal = workspace.dispose();
  await Promise.resolve();

  assert.deepEqual(
    arena.buffers.map((buffer) => buffer.destroyCount),
    new Array(18).fill(0),
  );
  gate.resolve();
  await reset;
  await disposal;
  await workspace.dispose();
  assert.deepEqual(
    arena.buffers.map((buffer) => buffer.destroyCount),
    new Array(18).fill(1),
  );
  assert.throws(() => workspace.get("ffn-product"), /workspace is disposed/i);
});
