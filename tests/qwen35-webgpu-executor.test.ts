import assert from "node:assert/strict";
import test from "node:test";

import {
  Qwen35WebGpuExecutor,
  type Qwen35WebGpuDevice,
  type Qwen35OwnedWebGpuBuffer,
} from "../src/qwen35-webgpu-executor.js";

interface FakeBuffer extends Qwen35OwnedWebGpuBuffer {
  readonly id: string;
  readonly bytes: ArrayBuffer;
  destroyed: boolean;
}

function fakeDevice(options: {
  readonly bufferError?: boolean;
  readonly compilationError?: boolean;
  readonly destroyError?: boolean;
  readonly validationError?: boolean;
  readonly pipelineGate?: Promise<void>;
  readonly validationGate?: Promise<void>;
  readonly writeError?: boolean;
} = {}): {
  readonly device: Qwen35WebGpuDevice;
  readonly events: string[];
  readonly writes: { readonly offset: number; readonly bytes: number[] }[];
  readonly buffers: FakeBuffer[];
} {
  const events: string[] = [];
  const writes: { offset: number; bytes: number[] }[] = [];
  const buffers: FakeBuffer[] = [];
  let nextBuffer = 0;
  const device: Qwen35WebGpuDevice = {
    limits: {
      minStorageBufferOffsetAlignment: 256,
      minUniformBufferOffsetAlignment: 256,
      maxStorageBufferBindingSize: 1_073_741_824,
      maxUniformBufferBindingSize: 65_536,
      maxComputeWorkgroupsPerDimension: 65_535,
    },
    queue: {
      writeBuffer(_buffer, offset, data, dataOffset, size) {
        if (options.writeError === true) {
          throw new Error("private write detail");
        }
        const buffer = _buffer as FakeBuffer;
        new Uint8Array(buffer.bytes, offset, size).set(
          new Uint8Array(data, dataOffset, size),
        );
        writes.push({
          offset,
          bytes: Array.from(new Uint8Array(data, dataOffset, size)),
        });
      },
      submit() {
        events.push("submit");
      },
      async onSubmittedWorkDone() {
        events.push("done");
      },
    },
    createShaderModule() {
      events.push("module");
      return {
        async getCompilationInfo() {
          return {
            messages: options.compilationError === true
              ? [{ type: "error", message: "private compiler detail" }]
              : [],
          };
        },
      };
    },
    pushErrorScope(filter) {
      events.push(`push:${filter}`);
    },
    async popErrorScope() {
      events.push("pop");
      await options.validationGate;
      return options.validationError === true ? { message: "private" } : null;
    },
    async createComputePipelineAsync() {
      events.push("pipeline");
      await options.pipelineGate;
      return {
        getBindGroupLayout() {
          return {};
        },
      };
    },
    createBindGroup(descriptor) {
      events.push(`bind:${descriptor.entries.length}`);
      return {};
    },
    createCommandEncoder() {
      events.push("encoder");
      return {
        beginComputePass() {
          events.push("pass");
          return {
            setPipeline() {
              events.push("set-pipeline");
            },
            setBindGroup() {
              events.push("set-bind-group");
            },
            dispatchWorkgroups(x, y, z) {
              events.push(`dispatch:${x},${y},${z}`);
            },
            end() {
              events.push("end-pass");
            },
          };
        },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          new Uint8Array(
            (destination as FakeBuffer).bytes,
            destinationOffset,
            size,
          ).set(new Uint8Array((source as FakeBuffer).bytes, sourceOffset, size));
          events.push(`copy:${size}`);
        },
        finish() {
          events.push("finish");
          return {};
        },
      };
    },
    createBuffer(descriptor) {
      if (options.bufferError === true) {
        throw new Error("private allocation detail");
      }
      const buffer: FakeBuffer = {
        id: `buffer-${nextBuffer++}`,
        bytes: new ArrayBuffer(descriptor.size),
        destroyed: false,
        destroy() {
          this.destroyed = true;
          if (options.destroyError === true) {
            throw new Error("private destroy detail");
          }
        },
        async mapAsync() {
          events.push("map");
        },
        getMappedRange() {
          return this.bytes;
        },
        unmap() {
          events.push("unmap");
        },
      };
      buffers.push(buffer);
      return buffer;
    },
  };
  return { device, events, writes, buffers };
}

test("compiles each exact model kernel once and submits bound physical views", async () => {
  const { device, events } = fakeDevice();
  const executor = new Qwen35WebGpuExecutor(device);
  const buffer = { destroy() {} };
  const dispatch = {
    kernel: {
      id: "qwen35-test-kernel",
      source: "@compute @workgroup_size(1) fn main() {}",
      entryPoint: "main",
    },
    bindings: [
      { binding: 0, kind: "storage", buffer, offset: 256, size: 512 },
      { binding: 1, kind: "uniform", buffer, offset: 0, size: 16 },
    ],
    workgroups: { x: 7, y: 2, z: 1 },
  } as const;

  await executor.dispatch(dispatch);
  await executor.dispatch(dispatch);

  assert.equal(events.filter((event) => event === "module").length, 1);
  assert.equal(events.filter((event) => event === "pipeline").length, 1);
  assert.equal(events.filter((event) => event === "bind:2").length, 1);
  assert.equal(events.filter((event) => event === "submit").length, 2);
  assert.equal(
    events.filter((event) => event === "dispatch:7,2,1").length,
    2,
  );
});

test("encodes a model step as one compute pass and one queue submission", async () => {
  const { device, events } = fakeDevice();
  const executor = new Qwen35WebGpuExecutor(device);
  const buffer = { destroy() {} };

  await executor.dispatchBatch([
    {
      kernel: {
        id: "qwen35-first-kernel",
        source: "@compute @workgroup_size(1) fn main() {}",
        entryPoint: "main",
      },
      bindings: [{ binding: 0, kind: "storage", buffer, offset: 0, size: 16 }],
      workgroups: { x: 4, y: 1, z: 1 },
    },
    {
      kernel: {
        id: "qwen35-second-kernel",
        source: "@compute @workgroup_size(1) fn main() {}",
        entryPoint: "main",
      },
      bindings: [{ binding: 0, kind: "storage", buffer, offset: 256, size: 16 }],
      workgroups: { x: 8, y: 1, z: 1 },
    },
  ]);

  assert.equal(events.filter((event) => event === "pass").length, 1);
  assert.equal(events.filter((event) => event === "submit").length, 1);
  assert.deepEqual(
    events.filter((event) => event.startsWith("dispatch:")),
    ["dispatch:4,1,1", "dispatch:8,1,1"],
  );
});

test("creates aligned uniform storage and fences before destruction", async () => {
  const { device, events, writes, buffers } = fakeDevice();
  const executor = new Qwen35WebGpuExecutor(device);
  const uniform = executor.createUniform(
    "qwen35-layer-params",
    Uint32Array.of(3, 7, 11),
  );

  assert.equal(uniform.byteLength, 12);
  assert.deepEqual(writes, [{
    offset: 0,
    bytes: [3, 0, 0, 0, 7, 0, 0, 0, 11, 0, 0, 0],
  }]);
  assert.equal(buffers[0]?.destroyed, false);
  await executor.dispose();
  assert.equal(events.at(-1), "done");
  assert.equal(buffers[0]?.destroyed, true);
});

test("uses explicit uniform slots and updates a selected slot in queue order", () => {
  const { device, writes, buffers } = fakeDevice();
  const executor = new Qwen35WebGpuExecutor(device);
  const first = executor.createUniform("qwen35-position", Uint32Array.of(1, 0, 0, 0));
  const second = executor.createUniform("qwen35-position", Uint32Array.of(2, 0, 0, 0));
  executor.updateUniform(first, Uint32Array.of(3, 0, 0, 0));

  assert.notEqual(first, second);
  assert.equal(buffers.length, 2);
  assert.deepEqual(writes.map(({ bytes }) => bytes[0]), [1, 2, 3]);
  assert.throws(
    () => executor.updateUniform(first, Uint32Array.of(1)),
    { code: "webgpu-uniform-size-mismatch" },
  );
});

test("sanitizes uniform upload failures and fails the executor closed", async () => {
  const { device, buffers } = fakeDevice({ writeError: true });
  const executor = new Qwen35WebGpuExecutor(device);

  assert.throws(
    () => executor.createUniform("qwen35-position", Uint32Array.of(1, 0, 0, 0)),
    {
      code: "webgpu-uniform-upload-failed",
      message: "Qwen3.5 WebGPU uniform upload failed",
    },
  );
  assert.equal(buffers[0]?.destroyed, true);
  assert.throws(
    () => executor.createUniform("qwen35-position", Uint32Array.of(2, 0, 0, 0)),
    { code: "webgpu-executor-poisoned" },
  );
  await executor.dispose();
});

test("sanitizes uniform allocation failures and fails the executor closed", async () => {
  const { device } = fakeDevice({ bufferError: true });
  const executor = new Qwen35WebGpuExecutor(device);

  assert.throws(
    () => executor.createUniform("qwen35-position", Uint32Array.of(1, 0, 0, 0)),
    {
      code: "webgpu-uniform-allocation-failed",
      message: "Qwen3.5 WebGPU uniform allocation failed",
    },
  );
  assert.throws(
    () => executor.createUniform("qwen35-position", Uint32Array.of(2, 0, 0, 0)),
    { code: "webgpu-executor-poisoned" },
  );
  await executor.dispose();
});

test("rejects invalid binding ranges before command encoding", async () => {
  const { device, events } = fakeDevice();
  const executor = new Qwen35WebGpuExecutor(device);

  await assert.rejects(
    executor.dispatch({
      kernel: {
        id: "qwen35-test-kernel",
        source: "@compute @workgroup_size(1) fn main() {}",
        entryPoint: "main",
      },
      bindings: [{
        binding: 0,
        kind: "storage",
        buffer: { destroy() {} },
        offset: 2,
        size: 4,
      }],
      workgroups: { x: 1, y: 1, z: 1 },
    }),
    { code: "webgpu-binding-invalid" },
  );
  assert.equal(events.includes("encoder"), false);
  assert.equal(events.includes("submit"), false);
});

test("enforces live storage and uniform binding alignment", async () => {
  const { device, events } = fakeDevice();
  const executor = new Qwen35WebGpuExecutor(device);

  await assert.rejects(executor.dispatch({
    kernel: {
      id: "qwen35-test-kernel",
      source: "@compute @workgroup_size(1) fn main() {}",
      entryPoint: "main",
    },
    bindings: [{
      binding: 0,
      kind: "storage",
      buffer: { destroy() {} },
      offset: 4,
      size: 4,
    }],
    workgroups: { x: 1, y: 1, z: 1 },
  }), { code: "webgpu-binding-invalid" });
  assert.equal(events.includes("module"), false);
});

test("fails closed on asynchronous WebGPU validation errors", async () => {
  const { device, events } = fakeDevice({ validationError: true });
  const executor = new Qwen35WebGpuExecutor(device);
  const request = {
    kernel: {
      id: "qwen35-test-kernel",
      source: "@compute @workgroup_size(1) fn main() {}",
      entryPoint: "main",
    },
    bindings: [],
    workgroups: { x: 1, y: 1, z: 1 },
  } as const;

  await assert.rejects(executor.dispatch(request), {
    code: "webgpu-dispatch-validation-failed",
    message: "Qwen3.5 WebGPU dispatch validation failed",
  });
  assert.deepEqual(
    events.filter((event) => event === "push:validation" || event === "pop"),
    ["push:validation", "pop"],
  );
  await assert.rejects(executor.dispatch(request), {
    code: "webgpu-executor-poisoned",
  });
  await assert.doesNotReject(executor.submittedWorkDone());
  await assert.doesNotReject(executor.dispose());
  assert.equal(events.filter((event) => event === "done").length, 2);
});

test("pipeline identity includes WGSL source", async () => {
  const { device, events } = fakeDevice();
  const executor = new Qwen35WebGpuExecutor(device);
  const base = {
    bindings: [] as const,
    workgroups: { x: 1, y: 1, z: 1 },
  };

  await executor.dispatch({
    ...base,
    kernel: {
      id: "qwen35-test-kernel",
      source: "@compute @workgroup_size(1) fn main() {}",
      entryPoint: "main",
    },
  });
  await executor.dispatch({
    ...base,
    kernel: {
      id: "qwen35-test-kernel",
      source: "@compute @workgroup_size(2) fn main() {}",
      entryPoint: "main",
    },
  });

  assert.equal(events.filter((event) => event === "pipeline").length, 2);
});

test("disposal waits for in-flight pipeline compilation", async () => {
  let releasePipeline!: () => void;
  const pipelineGate = new Promise<void>((resolve) => {
    releasePipeline = resolve;
  });
  const { device, events } = fakeDevice({ pipelineGate });
  const executor = new Qwen35WebGpuExecutor(device);
  const request = {
    kernel: {
      id: "qwen35-test-kernel",
      source: "@compute @workgroup_size(1) fn main() {}",
      entryPoint: "main",
    },
    bindings: [],
    workgroups: { x: 1, y: 1, z: 1 },
  } as const;
  const dispatch = executor.dispatch(request);
  while (!events.includes("pipeline")) await Promise.resolve();

  let disposed = false;
  const disposal = executor.dispose().then(() => {
    disposed = true;
  });
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(disposed, false);
  releasePipeline();

  await assert.rejects(dispatch, { code: "webgpu-executor-disposed" });
  await disposal;
  assert.equal(disposed, true);
});

test("disposal waits for an in-flight dispatch validation scope", async () => {
  let releaseValidation!: () => void;
  const validationGate = new Promise<void>((resolve) => {
    releaseValidation = resolve;
  });
  const { device, events } = fakeDevice({ validationGate });
  const executor = new Qwen35WebGpuExecutor(device);
  const dispatch = executor.dispatch({
    kernel: {
      id: "qwen35-test-kernel",
      source: "@compute @workgroup_size(1) fn main() {}",
      entryPoint: "main",
    },
    bindings: [],
    workgroups: { x: 1, y: 1, z: 1 },
  });
  while (!events.includes("pop")) await Promise.resolve();

  let disposed = false;
  const disposal = executor.dispose().then(() => {
    disposed = true;
  });
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(disposed, false);
  releaseValidation();

  await dispatch;
  await disposal;
  assert.equal(disposed, true);
});

test("reads back only the selected u32 and destroys its temporary buffer", async () => {
  const { device, events, buffers } = fakeDevice();
  const executor = new Qwen35WebGpuExecutor(device);
  const source = device.createBuffer({ label: "source", size: 16, usage: 0 });
  device.queue.writeBuffer(
    source,
    4,
    Uint32Array.of(248_069).buffer,
    0,
    4,
  );

  const value = await executor.readU32(source, 4);

  assert.equal(value, 248_069);
  assert.equal(events.includes("copy:4"), true);
  assert.equal(events.includes("map"), true);
  assert.equal(events.includes("unmap"), true);
  assert.equal(buffers.at(-1)?.destroyed, true);
});

test("does not report a selected token when readback cleanup fails", async () => {
  const { device } = fakeDevice({ destroyError: true });
  const executor = new Qwen35WebGpuExecutor(device);
  const source = device.createBuffer({ label: "source", size: 4, usage: 0 });

  await assert.rejects(executor.readU32(source, 0), {
    code: "webgpu-readback-cleanup-failed",
    message: "Qwen3.5 WebGPU readback cleanup failed",
  });
  await assert.rejects(executor.dispose(), {
    code: "webgpu-cleanup-failed",
  });
});

test("sanitizes shader compiler failures and permanently blocks submissions after dispose", async () => {
  const { device, events } = fakeDevice({ compilationError: true });
  const executor = new Qwen35WebGpuExecutor(device);
  const request = {
    kernel: {
      id: "qwen35-test-kernel",
      source: "@compute @workgroup_size(1) fn main() {}",
      entryPoint: "main",
    },
    bindings: [],
    workgroups: { x: 1, y: 1, z: 1 },
  } as const;

  await assert.rejects(executor.dispatch(request), {
    code: "webgpu-kernel-compile-failed",
    message: "Qwen3.5 WebGPU kernel compilation failed",
  });
  assert.equal(events.includes("pipeline"), false);
  assert.equal(events.includes("submit"), false);

  await executor.dispose();
  await assert.rejects(executor.dispatch(request), {
    code: "webgpu-executor-disposed",
  });
});
