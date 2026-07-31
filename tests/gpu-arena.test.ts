import assert from "node:assert/strict";
import test from "node:test";

import { AllocationLedger } from "../src/allocation-ledger.js";
import { GpuArena } from "../src/gpu-arena.js";
import { planQ3KMatrixShardDispatch } from "../src/q3k-gemv.js";

function fakeDevice(options?: {
  readonly maxBufferSize?: number;
  readonly maxStorageBufferBindingSize?: number;
  readonly failAt?: number;
  readonly destroyErrorMessage?: string;
  readonly popResults?: readonly (
    | null
    | { readonly message: string }
  )[];
}) {
  const descriptors: Array<{ size: number; usage: number; label?: string }> = [];
  const buffers: Array<{ destroyCount: number; destroy(): void }> = [];
  const pushedScopes: string[] = [];
  const poppedScopes: string[] = [];
  const scopeStack: string[] = [];
  const popResults = [...(options?.popResults ?? [])];
  return {
    descriptors,
    buffers,
    pushedScopes,
    poppedScopes,
    device: {
      limits: {
        maxBufferSize: options?.maxBufferSize ?? 64,
        maxStorageBufferBindingSize:
          options?.maxStorageBufferBindingSize ?? 48,
      },
      pushErrorScope(filter: string) {
        pushedScopes.push(filter);
        scopeStack.push(filter);
      },
      async popErrorScope() {
        const filter = scopeStack.pop();
        if (filter !== undefined) {
          poppedScopes.push(filter);
        }
        return popResults.shift() ?? null;
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
            if (options?.destroyErrorMessage !== undefined) {
              throw new Error(options.destroyErrorMessage);
            }
          },
        };
        buffers.push(buffer);
        return buffer;
      },
    },
  };
}

test("splits allocations by both live device limits and explicit alignment", async () => {
  const fake = fakeDevice();
  const ledger = new AllocationLedger(128n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 128n,
  });

  const allocation = await arena.allocate({
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

test("uses maxBufferSize when it is lower than the storage binding limit", async () => {
  const fake = fakeDevice({
    maxBufferSize: 32,
    maxStorageBufferBindingSize: 64,
  });
  const ledger = new AllocationLedger(96n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 96n,
  });

  const allocation = await arena.allocate({
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

test("rolls back created buffers and ledger ownership after partial failure", async () => {
  const fake = fakeDevice({ failAt: 1 });
  const ledger = new AllocationLedger(128n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 128n,
  });

  await assert.rejects(
    async () =>
      await arena.allocate({
        id: "partial",
        category: "scratch",
        byteLength: 80n,
        usage: 128,
        alignment: 16,
      }),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, /GPU buffer allocation failed/i);
      assert.doesNotMatch(error.message, /synthetic|partial/i);
      return true;
    },
  );

  assert.equal(fake.buffers[0]?.destroyCount, 1);
  assert.deepEqual(fake.pushedScopes, ["validation", "out-of-memory"]);
  assert.deepEqual(fake.poppedScopes, ["out-of-memory", "validation"]);
  ledger.assertAllReleased();
});

test("rejects unsafe numeric conversion before reserving the ledger", async () => {
  const fake = fakeDevice({
    maxBufferSize: Number.MAX_SAFE_INTEGER,
    maxStorageBufferBindingSize: Number.MAX_SAFE_INTEGER,
  });
  const ledger = new AllocationLedger(BigInt(Number.MAX_SAFE_INTEGER) + 16n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: BigInt(Number.MAX_SAFE_INTEGER),
  });

  await assert.rejects(
    async () =>
      await arena.allocate({
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

test("splits one logical allocation that exceeds the buffer shard cap", async () => {
  const fake = fakeDevice({
    maxBufferSize: 64,
    maxStorageBufferBindingSize: 64,
  });
  const ledger = new AllocationLedger(160n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 32n,
  });

  const allocation = await arena.allocate({
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

test("permits multiple allocations whose total exceeds the buffer shard cap", async () => {
  const fake = fakeDevice();
  const ledger = new AllocationLedger(256n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 64n,
  });
  const first = await arena.allocate({
    id: "first",
    category: "model",
    byteLength: 64n,
    usage: 128,
    alignment: 16,
  });

  const second = await arena.allocate({
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

test("uses only the ledger as the cumulative allocation budget", async () => {
  const fake = fakeDevice();
  const ledger = new AllocationLedger(96n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 32n,
  });
  const first = await arena.allocate({
    id: "first",
    category: "model",
    byteLength: 64n,
    usage: 128,
    alignment: 16,
  });

  await assert.rejects(
    async () =>
      await arena.allocate({
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

test("rejects invalid usage, alignment, and device limit combinations", async () => {
  const bad = fakeDevice({
    maxBufferSize: 64,
    maxStorageBufferBindingSize: 2,
  });
  const ledger = new AllocationLedger(64n);
  const arena = new GpuArena(bad.device, ledger, {
    bufferShardCapBytes: 64n,
  });

  await assert.rejects(
    async () =>
      await arena.allocate({
        id: "usage",
        category: "upload",
        byteLength: 4n,
        usage: 0,
        alignment: 4,
      }),
    /usage/i,
  );
  await assert.rejects(
    async () =>
      await arena.allocate({
        id: "usage-overflow",
        category: "upload",
        byteLength: 4n,
        usage: 2 ** 32,
        alignment: 4,
      }),
    /usage.*invalid/i,
  );
  await assert.rejects(
    async () =>
      await arena.allocate({
        id: "alignment",
        category: "upload",
        byteLength: 4n,
        usage: 8,
        alignment: 3,
      }),
    /power of two/i,
  );
  await assert.rejects(
    async () =>
      await arena.allocate({
        id: "limits",
        category: "upload",
        byteLength: 4n,
        usage: 8,
        alignment: 4,
      }),
    /device limits.*alignment/i,
  );
  assert.deepEqual(bad.pushedScopes, []);
});

test("rejects illegal GPUBufferUsage combinations before opening error scopes", async () => {
  const fake = fakeDevice();
  const ledger = new AllocationLedger(64n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 64n,
  });

  await assert.rejects(
    async () =>
      await arena.allocate({
        id: "illegal-map-read",
        category: "upload",
        byteLength: 16n,
        usage: 1 | 128,
        alignment: 4,
      }),
    /GPU buffer usage is invalid/i,
  );
  await assert.rejects(
    async () =>
      await arena.allocate({
        id: "unknown-bit",
        category: "upload",
        byteLength: 16n,
        usage: 1 << 20,
        alignment: 4,
      }),
    /GPU buffer usage is invalid/i,
  );
  assert.deepEqual(fake.pushedScopes, []);
  ledger.assertAllReleased();
});

test("rolls back every buffer after asynchronous validation or OOM errors", async () => {
  for (const popResults of [
    [
      null,
      { message: "validation at local-path:<path>/<tensor-id>.gguf" },
    ],
    [
      { message: "OOM for https://example.invalid/<model-id>.gguf" },
      null,
    ],
  ] as const) {
    const fake = fakeDevice({
      maxBufferSize: 32,
      maxStorageBufferBindingSize: 32,
      popResults,
    });
    const ledger = new AllocationLedger(128n);
    const arena = new GpuArena(fake.device, ledger, {
      bufferShardCapBytes: 32n,
    });

    await assert.rejects(
      arena.allocate({
        id: "local-path:<path>/<tensor-id>.gguf",
        category: "model",
        byteLength: 64n,
        usage: 128,
        alignment: 16,
      }),
      (error: unknown) => {
        assert.ok(error instanceof Error);
        assert.match(error.message, /GPU buffer allocation failed/i);
        assert.doesNotMatch(
          error.message,
          /local-path|tensor-id|example\.invalid|model-id|gguf/i,
        );
        return true;
      },
    );

    assert.deepEqual(fake.buffers.map((buffer) => buffer.destroyCount), [1, 1]);
    assert.deepEqual(fake.pushedScopes, ["validation", "out-of-memory"]);
    assert.deepEqual(fake.poppedScopes, ["out-of-memory", "validation"]);
    ledger.assertAllReleased();
  }
});

test("releases ledger ownership without exposing buffer destruction errors", async () => {
  const fake = fakeDevice({
    destroyErrorMessage:
      "destroy failed at local-path:<path>/<tensor-id>.gguf",
  });
  const ledger = new AllocationLedger(64n);
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 64n,
  });
  const allocation = await arena.allocate({
    id: "temporary-output",
    category: "activation",
    byteLength: 16n,
    usage: 128,
    alignment: 4,
  });

  assert.throws(
    () => allocation.destroy(),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, /GPU buffer destruction failed/i);
      assert.doesNotMatch(error.message, /local-path|tensor-id|gguf/i);
      return true;
    },
  );
  ledger.assertAllReleased();
  assert.doesNotThrow(() => allocation.destroy());
});

test("keeps complete Q3 rows in 128 MiB shards and covers every output row", async () => {
  const MIB = 1024 * 1024;
  const rows = 248_320;
  const columns = 2_560;
  const rowBytes = 10 * 112;
  const matrixBytes = BigInt(rows * rowBytes);
  const fake = fakeDevice({
    maxBufferSize: 512 * MIB,
    maxStorageBufferBindingSize: 512 * MIB,
  });
  const ledger = new AllocationLedger(512n * BigInt(MIB));
  const arena = new GpuArena(fake.device, ledger, {
    bufferShardCapBytes: 128n * BigInt(MIB),
  });

  const allocation = await arena.allocate({
    id: "output-projection",
    category: "model",
    byteLength: matrixBytes,
    usage: 128,
    alignment: 16,
    requiredShardQuantumBytes: BigInt(rowBytes),
  });
  const dispatches = planQ3KMatrixShardDispatch({
    rows,
    columns,
    shards: allocation.shards,
  });

  assert.ok(allocation.shards.length > 1);
  for (const shard of allocation.shards) {
    assert.ok(shard.allocatedByteLength <= 128n * BigInt(MIB));
    assert.equal(shard.logicalByteOffset % 112n, 0n);
    assert.equal(shard.logicalByteLength % 112n, 0n);
    assert.equal(shard.logicalByteOffset % BigInt(rowBytes), 0n);
    assert.equal(shard.logicalByteLength % BigInt(rowBytes), 0n);
  }
  let nextOutputRow = 0;
  for (const dispatch of dispatches) {
    assert.equal(dispatch.uniforms.outputRowOffset, nextOutputRow);
    assert.equal(dispatch.uniforms.weightWordOffset, 0);
    nextOutputRow += dispatch.uniforms.localRows;
  }
  assert.equal(nextOutputRow, rows);
  allocation.destroy();
  ledger.assertAllReleased();
});
