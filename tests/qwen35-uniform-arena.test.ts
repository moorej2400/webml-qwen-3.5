import assert from "node:assert/strict";
import test from "node:test";

import type { GpuAllocation, GpuAllocationRequest } from "../src/gpu-arena.js";
import {
  createQwen35UniformArena,
  type Qwen35UniformWriteQueue,
} from "../src/qwen35-uniform-arena.js";
import * as uniformModule from "../src/qwen35-uniform-arena.js";

interface FakeBuffer {
  readonly bytes: ArrayBuffer;
  destroyCount: number;
}

function harness(options: {
  readonly allocationFails?: boolean;
  readonly destroyFails?: boolean;
  readonly malformedAllocation?: boolean;
  readonly queueFails?: boolean;
  readonly writeFails?: boolean;
} = {}) {
  const requests: GpuAllocationRequest[] = [];
  const buffer: FakeBuffer = { bytes: new ArrayBuffer(2_048), destroyCount: 0 };
  const writes: { offset: number; bytes: number[] }[] = [];
  const events: string[] = [];
  let releaseQueue: (() => void) | undefined;
  let queueGate: Promise<void> | undefined;
  const arena = {
    async allocate(request: GpuAllocationRequest): Promise<GpuAllocation> {
      requests.push(request);
      if (options.allocationFails === true) {
        throw new Error("private allocation detail");
      }
      return {
        logicalBytes: request.byteLength,
        allocatedBytes: request.byteLength,
        shards: Object.freeze([Object.freeze({
          buffer: { ...buffer, destroy() { buffer.destroyCount += 1; } },
          logicalByteOffset: options.malformedAllocation === true ? 4n : 0n,
          logicalByteLength: request.byteLength,
          allocatedByteLength: request.byteLength,
        })]),
        destroy() {
          buffer.destroyCount += 1;
          if (options.destroyFails === true) {
            throw new Error("private destroy detail");
          }
        },
      };
    },
  };
  const queue: Qwen35UniformWriteQueue = {
    writeBuffer(_buffer, offset, data, dataOffset, size) {
      if (options.writeFails === true) throw new Error("private write detail");
      writes.push({
        offset,
        bytes: Array.from(new Uint8Array(data, dataOffset, size)),
      });
    },
    async onSubmittedWorkDone() {
      events.push("queue-start");
      await queueGate;
      if (options.queueFails === true) {
        throw new Error("private queue detail");
      }
      events.push("queue-end");
    },
  };
  return {
    arena,
    queue,
    requests,
    buffer,
    writes,
    events,
    blockQueue() {
      queueGate = new Promise<void>((resolve) => { releaseQueue = resolve; });
    },
    releaseQueue() { releaseQueue?.(); },
  };
}

test("allocates one aligned bounded arena and exposes distinct uniform slots", async () => {
  assert.equal("Qwen35UniformArena" in uniformModule, false);
  const fake = harness();
  const uniforms = await createQwen35UniformArena({
    arena: fake.arena,
    queue: fake.queue,
    slotCount: 8,
    slotWordCapacity: 8,
    minUniformBufferOffsetAlignment: 256,
    maxUniformBufferBindingSize: 65_536,
  });

  assert.deepEqual(fake.requests, [{
    id: "qwen35-uniform-arena",
    category: "scratch",
    byteLength: 2_048n,
    usage: 0x0048,
    alignment: 256,
    requiredShardQuantumBytes: 2_048n,
  }]);
  const first = uniforms.slot(0, 5);
  const second = uniforms.slot(1, 4);
  assert.deepEqual(first.binding, {
    buffer: first.binding.buffer,
    offset: 0,
    byteLength: 20,
  });
  assert.equal(second.binding.buffer, first.binding.buffer);
  assert.deepEqual(second.binding, {
    buffer: first.binding.buffer,
    offset: 256,
    byteLength: 16,
  });
  assert.equal(first, uniforms.slot(0, 5));
  assert.equal(Object.isFrozen(first), true);
  assert.equal(Object.isFrozen(first.binding), true);
  assert.throws(() => uniforms.slot(0, 4), /word count/i);

  first.update(Uint32Array.of(1, 2, 3, 4, 5));
  second.update(Uint32Array.of(9, 8, 7, 6));
  assert.deepEqual(fake.writes.map(({ offset }) => offset), [0, 256]);
  assert.deepEqual(fake.writes[0]?.bytes.slice(0, 8), [1, 0, 0, 0, 2, 0, 0, 0]);
  await uniforms.dispose();
  assert.equal(fake.buffer.destroyCount, 1);
});

test("accepts a caller-owned allocation id for a second live arena", async () => {
  const fake = harness();
  const uniforms = await createQwen35UniformArena({
    arena: fake.arena,
    queue: fake.queue,
    allocationId: "qwen35-vision-uniform-arena",
    slotCount: 1,
    slotWordCapacity: 4,
    minUniformBufferOffsetAlignment: 256,
    maxUniformBufferBindingSize: 16,
  });

  assert.equal(fake.requests[0]?.id, "qwen35-vision-uniform-arena");
  await uniforms.dispose();
});

test("does not upload a uniform slot when all words are unchanged", async () => {
  const fake = harness();
  const uniforms = await createQwen35UniformArena({
    arena: fake.arena,
    queue: fake.queue,
    slotCount: 1,
    slotWordCapacity: 4,
    minUniformBufferOffsetAlignment: 256,
    maxUniformBufferBindingSize: 16,
  });
  const slot = uniforms.slot(0, 4);
  slot.update(Uint32Array.of(1, 2, 3, 4));
  slot.update(Uint32Array.of(1, 2, 3, 4));
  assert.equal(fake.writes.length, 1);
  slot.update(Uint32Array.of(1, 2, 3, 5));
  assert.equal(fake.writes.length, 2);
  await uniforms.dispose();
});

test("rejects invalid plans and sanitizes allocation failure", async () => {
  const fake = harness();
  const base = {
    arena: fake.arena,
    queue: fake.queue,
    slotCount: 1,
    slotWordCapacity: 4,
    minUniformBufferOffsetAlignment: 256,
    maxUniformBufferBindingSize: 16,
  } as const;
  for (const overrides of [
    { slotCount: 0 },
    { slotWordCapacity: 0 },
    { minUniformBufferOffsetAlignment: 3 },
    { maxUniformBufferBindingSize: 15 },
  ]) {
    await assert.rejects(
      createQwen35UniformArena({ ...base, ...overrides }),
      { code: "uniform-arena-plan-invalid" },
    );
  }

  const failed = harness({ allocationFails: true });
  await assert.rejects(
    createQwen35UniformArena({
      ...base,
      arena: failed.arena,
      queue: failed.queue,
    }),
    (error: unknown) => {
      assert.equal(
        (error as { readonly code?: unknown }).code,
        "uniform-arena-allocation-failed",
      );
      assert.equal((error as Error).message.includes("private"), false);
      return true;
    },
  );
});

test("rejects capacity drift and fails uniform uploads closed", async () => {
  const fake = harness({ writeFails: true });
  const uniforms = await createQwen35UniformArena({
    arena: fake.arena,
    queue: fake.queue,
    slotCount: 2,
    slotWordCapacity: 8,
    minUniformBufferOffsetAlignment: 256,
    maxUniformBufferBindingSize: 32,
  });

  assert.throws(() => uniforms.slot(2, 4), /slot/i);
  assert.throws(() => uniforms.slot(0, 9), /word/i);
  const slot = uniforms.slot(0, 4);
  assert.throws(
    () => slot.update(Uint32Array.of(1, 2, 3)),
    { code: "uniform-arena-update-size-invalid" },
  );
  assert.throws(
    () => slot.update(Uint32Array.of(1, 2, 3, 4)),
    { code: "uniform-arena-upload-failed" },
  );
  assert.throws(() => uniforms.slot(1, 4), /poisoned/i);
  await uniforms.dispose();
});

test("keeps exact-capacity updates usable and disposal idempotent", async () => {
  const fake = harness();
  const uniforms = await createQwen35UniformArena({
    arena: fake.arena,
    queue: fake.queue,
    slotCount: 1,
    slotWordCapacity: 8,
    minUniformBufferOffsetAlignment: 256,
    maxUniformBufferBindingSize: 32,
  });
  const slot = uniforms.slot(0, 8);
  assert.throws(
    () => slot.update(Uint32Array.of(1, 2, 3, 4, 5, 6, 7)),
    { code: "uniform-arena-update-size-invalid" },
  );
  slot.update(Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 8));
  assert.equal(fake.writes.length, 1);

  const first = uniforms.dispose();
  assert.equal(uniforms.dispose(), first);
  await first;
  assert.throws(() => uniforms.slot(0, 8), { code: "uniform-arena-disposed" });
  assert.throws(
    () => slot.update(Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 8)),
    { code: "uniform-arena-disposed" },
  );
  assert.equal(fake.buffer.destroyCount, 1);
});

test("reports cleanup failure after attempting queue retirement and destruction", async () => {
  for (const options of [
    { queueFails: true },
    { destroyFails: true },
    { queueFails: true, destroyFails: true },
  ]) {
    const fake = harness(options);
    const uniforms = await createQwen35UniformArena({
      arena: fake.arena,
      queue: fake.queue,
      slotCount: 1,
      slotWordCapacity: 4,
      minUniformBufferOffsetAlignment: 256,
      maxUniformBufferBindingSize: 16,
    });
    await assert.rejects(
      uniforms.dispose(),
      (error: unknown) => {
        assert.equal(
          (error as { readonly code?: unknown }).code,
          "uniform-arena-cleanup-failed",
        );
        assert.equal((error as Error).message.includes("private"), false);
        return true;
      },
    );
    assert.deepEqual(
      fake.events,
      options.queueFails === true
        ? ["queue-start"]
        : ["queue-start", "queue-end"],
    );
    assert.equal(fake.buffer.destroyCount, 1);
  }
});

test("disposal fences queued writes before releasing allocation ownership", async () => {
  const fake = harness();
  fake.blockQueue();
  const uniforms = await createQwen35UniformArena({
    arena: fake.arena,
    queue: fake.queue,
    slotCount: 1,
    slotWordCapacity: 4,
    minUniformBufferOffsetAlignment: 256,
    maxUniformBufferBindingSize: 16,
  });
  uniforms.slot(0, 4).update(Uint32Array.of(1, 2, 3, 4));

  let disposed = false;
  const disposal = uniforms.dispose().then(() => { disposed = true; });
  await Promise.resolve();
  assert.equal(disposed, false);
  assert.equal(fake.buffer.destroyCount, 0);
  fake.releaseQueue();
  await disposal;
  assert.equal(disposed, true);
  assert.deepEqual(fake.events, ["queue-start", "queue-end"]);
  assert.equal(fake.buffer.destroyCount, 1);
});

test("rejects malformed allocation metadata and reports failed rollback", async () => {
  const malformed = harness({ malformedAllocation: true });
  await assert.rejects(
    createQwen35UniformArena({
      arena: malformed.arena,
      queue: malformed.queue,
      slotCount: 1,
      slotWordCapacity: 4,
      minUniformBufferOffsetAlignment: 256,
      maxUniformBufferBindingSize: 16,
    }),
    { code: "uniform-arena-layout-invalid" },
  );
  assert.equal(malformed.buffer.destroyCount, 1);

  const failedRollback = harness({
    malformedAllocation: true,
    destroyFails: true,
  });
  await assert.rejects(
    createQwen35UniformArena({
      arena: failedRollback.arena,
      queue: failedRollback.queue,
      slotCount: 1,
      slotWordCapacity: 4,
      minUniformBufferOffsetAlignment: 256,
      maxUniformBufferBindingSize: 16,
    }),
    { code: "uniform-arena-rollback-failed" },
  );
  assert.equal(failedRollback.buffer.destroyCount, 1);
});
