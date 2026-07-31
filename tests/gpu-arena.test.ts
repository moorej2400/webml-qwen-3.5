import assert from "node:assert/strict";
import test from "node:test";

import { AllocationLedger } from "../src/allocation-ledger.js";
import { GpuArena } from "../src/gpu-arena.js";

function fakeDevice(options?: {
  readonly maxBufferSize?: number;
  readonly maxStorageBufferBindingSize?: number;
  readonly failAt?: number;
}) {
  const descriptors: Array<{ size: number; usage: number; label?: string }> = [];
  const buffers: Array<{ destroyCount: number; destroy(): void }> = [];
  return {
    descriptors,
    buffers,
    device: {
      limits: {
        maxBufferSize: options?.maxBufferSize ?? 64,
        maxStorageBufferBindingSize:
          options?.maxStorageBufferBindingSize ?? 48,
      },
      createBuffer(descriptor: {
        size: number;
        usage: number;
        label?: string;
      }) {
        if (descriptors.length === options?.failAt) {
          throw new Error("synthetic allocation failure");
        }
        descriptors.push(descriptor);
        const buffer = {
          destroyCount: 0,
          destroy() {
            this.destroyCount += 1;
          },
        };
        buffers.push(buffer);
        return buffer;
      },
    },
  };
}

test("splits allocations by both live device limits and explicit alignment", () => {
  const fake = fakeDevice();
  const ledger = new AllocationLedger(128n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 128n,
  });

  const allocation = arena.allocate({
    id: "private-tensor-name",
    category: "model",
    byteLength: 80n,
    usage: 128,
    alignment: 16,
  });

  assert.deepEqual(
    fake.descriptors.map(({ size, usage, label }) => ({ size, usage, label })),
    [
      { size: 48, usage: 128, label: "qwen-runtime:model:0" },
      { size: 32, usage: 128, label: "qwen-runtime:model:1" },
    ],
  );
  assert.equal(allocation.logicalBytes, 80n);
  assert.equal(allocation.allocatedBytes, 80n);
  allocation.destroy();
  allocation.destroy();
  assert.deepEqual(fake.buffers.map((buffer) => buffer.destroyCount), [1, 1]);
  ledger.assertAllReleased();
});

test("uses maxBufferSize when it is lower than the storage binding limit", () => {
  const fake = fakeDevice({
    maxBufferSize: 32,
    maxStorageBufferBindingSize: 64,
  });
  const ledger = new AllocationLedger(96n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 96n,
  });

  const allocation = arena.allocate({
    id: "buffer-limited",
    category: "activation",
    byteLength: 64n,
    usage: 128,
    alignment: 16,
  });

  assert.deepEqual(
    fake.descriptors.map(({ size }) => size),
    [32, 32],
  );
  allocation.destroy();
});

test("rolls back created buffers and ledger ownership after partial failure", () => {
  const fake = fakeDevice({ failAt: 1 });
  const ledger = new AllocationLedger(128n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 128n,
  });

  assert.throws(
    () =>
      arena.allocate({
        id: "partial",
        category: "scratch",
        byteLength: 80n,
        usage: 128,
        alignment: 16,
      }),
    /synthetic allocation failure/i,
  );

  assert.equal(fake.buffers[0]?.destroyCount, 1);
  ledger.assertAllReleased();
});

test("rejects unsafe numeric conversion before reserving the ledger", () => {
  const fake = fakeDevice({
    maxBufferSize: Number.MAX_SAFE_INTEGER,
    maxStorageBufferBindingSize: Number.MAX_SAFE_INTEGER,
  });
  const ledger = new AllocationLedger(BigInt(Number.MAX_SAFE_INTEGER) + 16n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: BigInt(Number.MAX_SAFE_INTEGER),
  });

  assert.throws(
    () =>
      arena.allocate({
        id: "unsafe",
        category: "model",
        byteLength: BigInt(Number.MAX_SAFE_INTEGER) + 1n,
        usage: 128,
        alignment: 4,
      }),
    /safe integer/i,
  );
  ledger.assertAllReleased();
});

test("splits one logical allocation that exceeds the buffer shard cap", () => {
  const fake = fakeDevice({
    maxBufferSize: 64,
    maxStorageBufferBindingSize: 64,
  });
  const ledger = new AllocationLedger(160n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 32n,
  });

  const allocation = arena.allocate({
    id: "large-logical-tensor",
    category: "model",
    byteLength: 80n,
    usage: 128,
    alignment: 16,
  });

  assert.deepEqual(
    fake.descriptors.map(({ size }) => size),
    [32, 32, 16],
  );
  assert.equal(ledger.snapshot().currentBytes, 80n);
  allocation.destroy();
});

test("permits multiple allocations whose total exceeds the buffer shard cap", () => {
  const fake = fakeDevice();
  const ledger = new AllocationLedger(256n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 64n,
  });
  const first = arena.allocate({
    id: "first",
    category: "model",
    byteLength: 64n,
    usage: 128,
    alignment: 16,
  });

  const second = arena.allocate({
    id: "second",
    category: "activation",
    byteLength: 48n,
    usage: 128,
    alignment: 16,
  });

  assert.equal(ledger.snapshot().currentBytes, 112n);
  assert.ok(fake.descriptors.every(({ size }) => size <= 64));
  second.destroy();
  first.destroy();
  ledger.assertAllReleased();
});

test("uses only the ledger as the cumulative allocation budget", () => {
  const fake = fakeDevice();
  const ledger = new AllocationLedger(96n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 32n,
  });
  const first = arena.allocate({
    id: "first",
    category: "model",
    byteLength: 64n,
    usage: 128,
    alignment: 16,
  });

  assert.throws(
    () =>
      arena.allocate({
        id: "over-ledger",
        category: "activation",
        byteLength: 48n,
        usage: 128,
        alignment: 16,
      }),
    /ledger limit/i,
  );
  assert.equal(ledger.snapshot().currentBytes, 64n);
  first.destroy();
  ledger.assertAllReleased();
});

test("rejects invalid usage, alignment, and device limit combinations", () => {
  const bad = fakeDevice({
    maxBufferSize: 64,
    maxStorageBufferBindingSize: 2,
  });
  const ledger = new AllocationLedger(64n);
  const arena = new GpuArena(bad.device, ledger, {
    bufferShardCapBytes: 64n,
  });

  assert.throws(
    () =>
      arena.allocate({
        id: "usage",
        category: "upload",
        byteLength: 4n,
        usage: 0,
        alignment: 4,
      }),
    /usage/i,
  );
  assert.throws(
    () =>
      arena.allocate({
        id: "usage-overflow",
        category: "upload",
        byteLength: 4n,
        usage: 2 ** 32,
        alignment: 4,
      }),
    /usage.*u32/i,
  );
  assert.throws(
    () =>
      arena.allocate({
        id: "alignment",
        category: "upload",
        byteLength: 4n,
        usage: 8,
        alignment: 3,
      }),
    /power of two/i,
  );
  assert.throws(
    () =>
      arena.allocate({
        id: "limits",
        category: "upload",
        byteLength: 4n,
        usage: 8,
        alignment: 4,
      }),
    /device limits.*alignment/i,
  );
});
