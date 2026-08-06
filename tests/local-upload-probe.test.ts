import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import test from "node:test";

const MIB = 1024 * 1024;
const moduleUrl = new URL("../dev/browser/upload-probe.ts", import.meta.url);

interface ProbeEvent {
  readonly stage: "before_write" | "after_write" | "after_retire";
  readonly ordinal: number;
  readonly shardIndex: number;
  readonly shardCount: number;
  readonly segmentIndex: number;
  readonly segmentCount: number;
  readonly globalOffset: number;
  readonly byteCount: number;
  readonly bufferShardBytes: number;
  readonly uploadLaneBytes: number;
  readonly retireAfterEachWrite: boolean;
}

const manifest = {
  shards: [
    { offset: "0", length: "32" },
    { offset: "32", length: "32" },
  ],
  tensorLayout: [
    {
      name: "token_embd.weight",
      shard: 0,
      shardOffset: "4",
      tensorOffset: "0",
      length: "8",
    },
    {
      name: "token_embd.weight",
      shard: 1,
      shardOffset: "8",
      tensorOffset: "8",
      length: "8",
    },
    {
      name: "output_norm.weight",
      shard: 0,
      shardOffset: "20",
      tensorOffset: "0",
      length: "8",
    },
  ],
};

const loadProbeModule = async (): Promise<Record<string, unknown>> => {
  assert.equal(
    existsSync(moduleUrl),
    true,
    "the local upload probe must exist only in the development module graph",
  );
  const specifier = "../dev/browser/" + "upload-probe.js";
  return import(specifier) as Promise<Record<string, unknown>>;
};

const fakeSurface = (retirement?: Promise<void>) => {
  const calls: string[] = [];
  const queue = {
    writeBuffer(
      this: unknown,
      _buffer: object,
      _bufferOffset: number,
      _data: Uint8Array,
      _dataOffset = 0,
      size?: number,
    ) {
      assert.equal(this, queue);
      calls.push(`write:${size ?? _data.byteLength - _dataOffset}`);
    },
    async onSubmittedWorkDone(this: unknown) {
      assert.equal(this, queue);
      calls.push("retire:start");
      await retirement;
      calls.push("retire:end");
    },
    submit(this: unknown) {
      assert.equal(this, queue);
      calls.push("submit");
    },
  };
  const device = {
    features: new Set(["shader-f16"]),
    limits: { maxBufferSize: 256 * MIB, maxStorageBufferBindingSize: 256 * MIB },
    queue,
    destroy(this: unknown) {
      assert.equal(this, device);
      calls.push("destroy");
    },
  };
  const adapter = {
    features: new Set(["shader-f16"]),
    limits: device.limits,
    async requestDevice(this: unknown) {
      assert.equal(this, adapter);
      calls.push("requestDevice");
      return device;
    },
  };
  const gpu = {
    async requestAdapter(this: unknown) {
      assert.equal(this, gpu);
      calls.push("requestAdapter");
      return adapter;
    },
  };
  return { surface: { gpu }, calls };
};

const createProbe = async (input: Record<string, unknown>) => {
  const module = await loadProbeModule();
  const create = module.createLocalWeightUploadProbe;
  assert.equal(typeof create, "function");
  return (create as (value: Record<string, unknown>) => {
    readonly surface: { readonly gpu: { requestAdapter(): Promise<unknown> } };
    beginWeightsUpload(): void;
    endWeightsUpload(): void;
  })(input);
};

test("dev queue probe reports exact split-tensor shard boundaries and preserves method receivers", async () => {
  const fake = fakeSurface();
  const observed: ProbeEvent[] = [];
  const probe = await createProbe({
    surface: fake.surface,
    manifest,
    bufferShardBytes: 128 * MIB,
    uploadLaneBytes: 8 * MIB,
    retireAfterEachWrite: false,
    onEvent: (event: ProbeEvent) => observed.push(event),
  });
  const adapter = await probe.surface.gpu.requestAdapter() as {
    requestDevice(): Promise<{ queue: {
      writeBuffer(buffer: object, offset: number, data: Uint8Array, dataOffset: number, size: number): void;
      onSubmittedWorkDone(): Promise<void>;
      submit(commands: readonly unknown[]): void;
    }; destroy(): void }>;
  };
  const device = await adapter.requestDevice();

  device.queue.writeBuffer({}, 0, new Uint8Array(4), 0, 4);
  probe.beginWeightsUpload();
  device.queue.writeBuffer({}, 0, new Uint8Array(8), 0, 8);
  device.queue.writeBuffer({}, 0, new Uint8Array(8), 0, 8);
  await device.queue.onSubmittedWorkDone();
  device.queue.writeBuffer({}, 0, new Uint8Array(8), 0, 8);
  await device.queue.onSubmittedWorkDone();
  probe.endWeightsUpload();
  device.queue.submit([]);
  device.destroy();

  assert.deepEqual(observed.map((event) => event.ordinal), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.deepEqual(observed.map((event) => event.stage), [
    "before_write",
    "after_write",
    "before_write",
    "after_write",
    "after_retire",
    "after_retire",
    "before_write",
    "after_write",
    "after_retire",
  ]);
  assert.deepEqual(
    observed.filter((event) => event.stage === "before_write").map((event) => ({
      shardIndex: event.shardIndex,
      shardCount: event.shardCount,
      segmentIndex: event.segmentIndex,
      segmentCount: event.segmentCount,
      globalOffset: event.globalOffset,
      byteCount: event.byteCount,
    })),
    [
      { shardIndex: 0, shardCount: 2, segmentIndex: 0, segmentCount: 2, globalOffset: 0, byteCount: 8 },
      { shardIndex: 0, shardCount: 2, segmentIndex: 1, segmentCount: 2, globalOffset: 8, byteCount: 8 },
      { shardIndex: 1, shardCount: 2, segmentIndex: 0, segmentCount: 1, globalOffset: 16, byteCount: 8 },
    ],
  );
  assert.equal(
    observed.every((event) =>
      event.bufferShardBytes === 128 * MIB &&
      event.uploadLaneBytes === 8 * MIB &&
      event.retireAfterEachWrite === false
    ),
    true,
  );
  assert.deepEqual(Object.keys(observed[0] ?? {}).sort(), [
    "bufferShardBytes",
    "byteCount",
    "globalOffset",
    "ordinal",
    "retireAfterEachWrite",
    "segmentCount",
    "segmentIndex",
    "shardCount",
    "shardIndex",
    "stage",
    "uploadLaneBytes",
  ]);
  assert.doesNotMatch(
    JSON.stringify(observed),
    /tensor|token_embd|output_norm|url|path|prompt|response|secret|stack/i,
  );
  assert.deepEqual(fake.calls, [
    "requestAdapter",
    "requestDevice",
    "write:4",
    "write:8",
    "write:8",
    "retire:start",
    "retire:end",
    "write:8",
    "retire:start",
    "retire:end",
    "submit",
    "destroy",
  ]);
});

test("dev queue probe retires accepted writes after deactivation without controlling the queue", async () => {
  let releaseRetirement!: () => void;
  const retirement = new Promise<void>((resolve) => { releaseRetirement = resolve; });
  const fake = fakeSurface(retirement);
  const observed: ProbeEvent[] = [];
  const probe = await createProbe({
    surface: fake.surface,
    manifest,
    bufferShardBytes: 64 * MIB,
    uploadLaneBytes: 16 * MIB,
    retireAfterEachWrite: true,
    onEvent(event: ProbeEvent) {
      observed.push(event);
      if (event.stage === "after_write") throw new Error("private observer failure");
    },
  });
  const adapter = await probe.surface.gpu.requestAdapter() as {
    requestDevice(): Promise<{ queue: {
      writeBuffer(buffer: object, offset: number, data: Uint8Array, dataOffset: number, size: number): void;
      onSubmittedWorkDone(): Promise<void>;
    } }>;
  };
  const device = await adapter.requestDevice();

  probe.beginWeightsUpload();
  assert.doesNotThrow(() =>
    device.queue.writeBuffer({}, 0, new Uint8Array(8), 0, 8)
  );
  const retiring = device.queue.onSubmittedWorkDone();
  probe.endWeightsUpload();
  assert.deepEqual(observed.map((event) => event.stage), ["before_write", "after_write"]);
  releaseRetirement();
  await retiring;

  assert.deepEqual(observed.map((event) => event.stage), [
    "before_write",
    "after_write",
    "after_retire",
  ]);
  assert.equal(observed[2]?.bufferShardBytes, 64 * MIB);
  assert.equal(observed[2]?.uploadLaneBytes, 16 * MIB);
  assert.equal(observed[2]?.retireAfterEachWrite, true);
});

test("dev queue probe fails observation closed while preserving unexpected queue traffic", async () => {
  const fake = fakeSurface();
  const observed: ProbeEvent[] = [];
  const probe = await createProbe({
    surface: fake.surface,
    manifest,
    bufferShardBytes: 128 * MIB,
    uploadLaneBytes: 8 * MIB,
    retireAfterEachWrite: false,
    onEvent: (event: ProbeEvent) => observed.push(event),
  });
  const adapter = await probe.surface.gpu.requestAdapter() as {
    requestDevice(): Promise<{ queue: {
      writeBuffer(buffer: object, offset: number, data: Uint8Array, dataOffset: number, size: number): void;
      onSubmittedWorkDone(): Promise<void>;
    } }>;
  };
  const device = await adapter.requestDevice();

  probe.beginWeightsUpload();
  assert.doesNotThrow(() =>
    device.queue.writeBuffer({}, 0, new Uint8Array(24), 0, 24)
  );
  await assert.doesNotReject(device.queue.onSubmittedWorkDone());
  probe.endWeightsUpload();

  assert.deepEqual(observed, []);
  assert.deepEqual(fake.calls.slice(-4), [
    "requestDevice",
    "write:24",
    "retire:start",
    "retire:end",
  ]);
});
