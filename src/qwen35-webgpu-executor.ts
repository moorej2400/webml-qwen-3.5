import { diagnosticError, isSafeDiagnosticCode } from "./diagnostics.js";

const GPU_BUFFER_USAGE_COPY_DST = 0x0008;
const GPU_BUFFER_USAGE_UNIFORM = 0x0040;

/** Opaque buffer identity accepted for binding without transferring ownership. */
export type Qwen35WebGpuBuffer = object;

export interface Qwen35OwnedWebGpuBuffer {
  destroy(): void;
  mapAsync?(mode: number): Promise<void>;
  getMappedRange?(): ArrayBuffer;
  unmap?(): void;
}

interface Qwen35ShaderModule {
  getCompilationInfo(): Promise<{
    readonly messages: Iterable<{ readonly type: string; readonly message: string }>;
  }>;
}

interface Qwen35ComputePipeline {
  getBindGroupLayout(index: number): unknown;
}

interface Qwen35ComputePass {
  setPipeline(pipeline: Qwen35ComputePipeline): void;
  setBindGroup(index: number, bindGroup: unknown): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

interface Qwen35CommandEncoder {
  beginComputePass(): Qwen35ComputePass;
  copyBufferToBuffer?(
    source: Qwen35WebGpuBuffer,
    sourceOffset: number,
    destination: Qwen35WebGpuBuffer,
    destinationOffset: number,
    size: number,
  ): void;
  finish(): unknown;
}

export interface Qwen35WebGpuDevice {
  readonly limits: {
    readonly minStorageBufferOffsetAlignment: number;
    readonly minUniformBufferOffsetAlignment: number;
    readonly maxStorageBufferBindingSize: number;
    readonly maxUniformBufferBindingSize: number;
    readonly maxComputeWorkgroupsPerDimension: number;
  };
  readonly queue: {
    writeBuffer(
      buffer: Qwen35WebGpuBuffer,
      bufferOffset: number,
      data: ArrayBuffer,
      dataOffset: number,
      size: number,
    ): void;
    submit(commands: readonly unknown[]): void;
    onSubmittedWorkDone(): Promise<void>;
  };
  pushErrorScope(filter: "validation"): void;
  popErrorScope(): Promise<{ readonly message?: string } | null>;
  createShaderModule(descriptor: {
    readonly label: string;
    readonly code: string;
  }): Qwen35ShaderModule;
  createComputePipelineAsync(descriptor: {
    readonly label: string;
    readonly layout: "auto";
    readonly compute: {
      readonly module: Qwen35ShaderModule;
      readonly entryPoint: string;
    };
  }): Promise<Qwen35ComputePipeline>;
  createBindGroup(descriptor: {
    readonly label: string;
    readonly layout: unknown;
    readonly entries: readonly {
      readonly binding: number;
      readonly resource: {
        readonly buffer: Qwen35WebGpuBuffer;
        readonly offset: number;
        readonly size: number;
      };
    }[];
  }): unknown;
  createCommandEncoder(descriptor?: { readonly label: string }): Qwen35CommandEncoder;
  createBuffer(descriptor: {
    readonly label: string;
    readonly size: number;
    readonly usage: number;
  }): Qwen35OwnedWebGpuBuffer;
}

export interface Qwen35KernelSource {
  readonly id: string;
  readonly source: string;
  readonly entryPoint: string;
}

export interface Qwen35BufferBinding {
  readonly binding: number;
  readonly kind: "storage" | "uniform";
  readonly buffer: Qwen35WebGpuBuffer;
  readonly offset: number;
  readonly size: number;
}

export interface Qwen35DispatchRequest {
  readonly kernel: Qwen35KernelSource;
  readonly bindings: readonly Qwen35BufferBinding[];
  readonly workgroups: {
    readonly x: number;
    readonly y: number;
    readonly z: number;
  };
}

export interface Qwen35OwnedUniform {
  readonly buffer: Qwen35OwnedWebGpuBuffer;
  readonly byteLength: number;
}

interface Qwen35KernelSnapshot {
  readonly identity: number;
  readonly id: string;
  readonly source: string;
  readonly entryPoint: string;
}

function requirePositiveInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function validateDispatch(
  request: Qwen35DispatchRequest,
  limits: Qwen35WebGpuDevice["limits"],
): void {
  if (
    !isSafeDiagnosticCode(request.kernel.id) ||
    !/^[A-Za-z_][A-Za-z0-9_]*$/.test(request.kernel.entryPoint) ||
    request.kernel.source.length === 0 ||
    !requirePositiveInteger(request.workgroups.x) ||
    !requirePositiveInteger(request.workgroups.y) ||
    !requirePositiveInteger(request.workgroups.z) ||
    request.workgroups.x > limits.maxComputeWorkgroupsPerDimension ||
    request.workgroups.y > limits.maxComputeWorkgroupsPerDimension ||
    request.workgroups.z > limits.maxComputeWorkgroupsPerDimension
  ) {
    throw diagnosticError(
      "webgpu-dispatch-invalid",
      "Qwen3.5 WebGPU dispatch is invalid",
    );
  }
  const bindings = new Set<number>();
  for (const binding of request.bindings) {
    const knownKind = binding.kind === "storage" || binding.kind === "uniform";
    const requiredAlignment = binding.kind === "storage"
      ? limits.minStorageBufferOffsetAlignment
      : limits.minUniformBufferOffsetAlignment;
    const maximumSize = binding.kind === "storage"
      ? limits.maxStorageBufferBindingSize
      : limits.maxUniformBufferBindingSize;
    const valid =
      knownKind &&
      Number.isSafeInteger(binding.binding) &&
      binding.binding >= 0 &&
      Number.isSafeInteger(binding.offset) &&
      binding.offset >= 0 &&
      requirePositiveInteger(requiredAlignment) &&
      binding.offset % requiredAlignment === 0 &&
      requirePositiveInteger(binding.size) &&
      Number.isSafeInteger(binding.offset + binding.size) &&
      binding.size % 4 === 0 &&
      binding.size <= maximumSize &&
      !bindings.has(binding.binding);
    if (!valid) {
      throw diagnosticError(
        "webgpu-binding-invalid",
        "Qwen3.5 WebGPU binding range is invalid",
      );
    }
    bindings.add(binding.binding);
  }
}

/** Executes only the fixed Qwen3.5 command plans assembled by the model driver. */
export class Qwen35WebGpuExecutor {
  readonly #device: Qwen35WebGpuDevice;
  readonly #pipelines = new Map<number, Promise<Qwen35ComputePipeline>>();
  readonly #bindGroups = new Map<string, unknown>();
  readonly #bufferIds = new WeakMap<object, number>();
  readonly #kernelSnapshots = new WeakMap<object, Qwen35KernelSnapshot>();
  readonly #uniforms = new WeakSet<object>();
  readonly #ownedBuffers = new Set<Qwen35OwnedWebGpuBuffer>();
  #nextBufferId = 1;
  #nextKernelIdentity = 1;
  #disposed = false;
  #poisoned = false;
  #disposePromise: Promise<void> | null = null;

  constructor(device: Qwen35WebGpuDevice) {
    this.#device = device;
  }

  createUniform(label: string, values: ArrayBufferView<ArrayBuffer>): Qwen35OwnedUniform {
    this.#assertUsable();
    if (!isSafeDiagnosticCode(label)) {
      throw diagnosticError(
        "webgpu-uniform-invalid",
        "Qwen3.5 WebGPU uniform data is invalid",
      );
    }
    this.#validateUniformValues(values);
    if (this.#ownedBuffers.size >= 2_048) {
      throw diagnosticError(
        "webgpu-uniform-capacity-exceeded",
        "Qwen3.5 WebGPU uniform slots must be reused",
      );
    }
    const buffer = this.#device.createBuffer({
      label,
      size: values.byteLength,
      usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
    });
    this.#ownedBuffers.add(buffer);
    const uniform = Object.freeze({ buffer, byteLength: values.byteLength });
    this.#uniforms.add(uniform);
    this.updateUniform(uniform, values);
    return uniform;
  }

  /** Updates one explicit slot; callers allocate distinct slots used in one batch. */
  updateUniform(
    uniform: Qwen35OwnedUniform,
    values: ArrayBufferView<ArrayBuffer>,
  ): void {
    this.#assertUsable();
    this.#validateUniformValues(values);
    if (!this.#uniforms.has(uniform) || uniform.byteLength !== values.byteLength) {
      throw diagnosticError(
        "webgpu-uniform-size-mismatch",
        "Qwen3.5 WebGPU uniform size changed",
      );
    }
    this.#device.queue.writeBuffer(
      uniform.buffer,
      0,
      values.buffer,
      values.byteOffset,
      values.byteLength,
    );
  }

  async dispatch(request: Qwen35DispatchRequest): Promise<void> {
    await this.dispatchBatch([request]);
  }

  /** Encodes one model step without paying one queue submission per kernel. */
  async dispatchBatch(requests: readonly Qwen35DispatchRequest[]): Promise<void> {
    this.#assertUsable();
    if (requests.length === 0) {
      throw diagnosticError(
        "webgpu-dispatch-invalid",
        "Qwen3.5 WebGPU dispatch is invalid",
      );
    }
    for (const request of requests) validateDispatch(request, this.#device.limits);
    const pipelines = await Promise.all(
      requests.map((request) => this.#pipeline(request.kernel)),
    );
    this.#assertUsable();
    this.#device.pushErrorScope("validation");
    try {
      const encoder = this.#device.createCommandEncoder({
        label: "qwen35-model-step",
      });
      const pass = encoder.beginComputePass();
      requests.forEach((request, index) => {
        const pipeline = pipelines[index];
        if (pipeline === undefined) {
          throw diagnosticError(
            "webgpu-dispatch-invalid",
            "Qwen3.5 WebGPU dispatch is invalid",
          );
        }
        const bindGroup = this.#bindGroup(request, pipeline);
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
          request.workgroups.x,
          request.workgroups.y,
          request.workgroups.z,
        );
      });
      pass.end();
      this.#device.queue.submit([encoder.finish()]);
    } catch {
      await this.#retireFailedScope();
      throw diagnosticError(
        "webgpu-dispatch-validation-failed",
        "Qwen3.5 WebGPU dispatch validation failed",
      );
    }
    let validationError: { readonly message?: string } | null;
    try {
      validationError = await this.#device.popErrorScope();
    } catch {
      this.#poisoned = true;
      throw diagnosticError(
        "webgpu-dispatch-validation-failed",
        "Qwen3.5 WebGPU dispatch validation failed",
      );
    }
    if (validationError !== null) {
      this.#poisoned = true;
      throw diagnosticError(
        "webgpu-dispatch-validation-failed",
        "Qwen3.5 WebGPU dispatch validation failed",
      );
    }
  }

  async submittedWorkDone(): Promise<void> {
    await this.#device.queue.onSubmittedWorkDone();
  }

  dispose(): Promise<void> {
    if (this.#disposePromise !== null) return this.#disposePromise;
    this.#disposed = true;
    this.#disposePromise = this.#cleanup();
    return this.#disposePromise;
  }

  #assertUsable(): void {
    if (this.#disposed) {
      throw diagnosticError(
        "webgpu-executor-disposed",
        "Qwen3.5 WebGPU executor is disposed",
      );
    }
    if (this.#poisoned) {
      throw diagnosticError(
        "webgpu-executor-poisoned",
        "Qwen3.5 WebGPU executor must be disposed",
      );
    }
  }

  #pipeline(kernel: Qwen35KernelSource): Promise<Qwen35ComputePipeline> {
    const snapshot = this.#kernelSnapshot(kernel);
    const cached = this.#pipelines.get(snapshot.identity);
    if (cached !== undefined) return cached;
    const compiling = this.#compile(snapshot);
    this.#pipelines.set(snapshot.identity, compiling);
    return compiling;
  }

  #kernelSnapshot(kernel: Qwen35KernelSource): Qwen35KernelSnapshot {
    const object = kernel as object;
    const existing = this.#kernelSnapshots.get(object);
    if (existing !== undefined) {
      if (
        existing.id !== kernel.id ||
        existing.source !== kernel.source ||
        existing.entryPoint !== kernel.entryPoint
      ) {
        throw diagnosticError(
          "webgpu-kernel-mutated",
          "Qwen3.5 WebGPU kernel identity changed",
        );
      }
      return existing;
    }
    const snapshot = Object.freeze({
      identity: this.#nextKernelIdentity,
      id: kernel.id,
      source: kernel.source,
      entryPoint: kernel.entryPoint,
    });
    this.#nextKernelIdentity += 1;
    this.#kernelSnapshots.set(object, snapshot);
    return snapshot;
  }

  #bindGroup(
    request: Qwen35DispatchRequest,
    pipeline: Qwen35ComputePipeline,
  ): unknown {
    const bindings = request.bindings.map((binding) => [
      binding.binding,
      binding.kind,
      this.#bufferId(binding.buffer),
      binding.offset,
      binding.size,
    ].join(":"));
    const key = `${this.#kernelSnapshot(request.kernel).identity}\u0000${bindings.join("|")}`;
    const cached = this.#bindGroups.get(key);
    if (cached !== undefined) return cached;
    const bindGroup = this.#device.createBindGroup({
      label: `${request.kernel.id}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: request.bindings.map((binding) => ({
        binding: binding.binding,
        resource: {
          buffer: binding.buffer,
          offset: binding.offset,
          size: binding.size,
        },
      })),
    });
    this.#bindGroups.set(key, bindGroup);
    return bindGroup;
  }

  #bufferId(buffer: Qwen35WebGpuBuffer): number {
    const object = buffer as object;
    const existing = this.#bufferIds.get(object);
    if (existing !== undefined) return existing;
    const created = this.#nextBufferId;
    this.#nextBufferId += 1;
    this.#bufferIds.set(object, created);
    return created;
  }

  async #retireFailedScope(): Promise<void> {
    this.#poisoned = true;
    try {
      await this.#device.popErrorScope();
    } catch {
      // The sanitized dispatch failure remains the public error.
    }
  }

  #validateUniformValues(values: ArrayBufferView<ArrayBuffer>): void {
    if (
      values.byteLength === 0 ||
      values.byteLength % 4 !== 0 ||
      values.byteLength > this.#device.limits.maxUniformBufferBindingSize
    ) {
      throw diagnosticError(
        "webgpu-uniform-invalid",
        "Qwen3.5 WebGPU uniform data is invalid",
      );
    }
  }

  async #cleanup(): Promise<void> {
    let failed = false;
    // Compilation is device-owned asynchronous work even before a queue
    // submission exists. Retire every cached attempt before lock transfer.
    await Promise.allSettled([...this.#pipelines.values()]);
    try {
      await this.#device.queue.onSubmittedWorkDone();
    } catch {
      failed = true;
    }
    for (const buffer of this.#ownedBuffers) {
      try {
        buffer.destroy();
      } catch {
        failed = true;
      }
    }
    this.#ownedBuffers.clear();
    this.#bindGroups.clear();
    this.#pipelines.clear();
    if (failed) {
      throw diagnosticError(
        "webgpu-cleanup-failed",
        "Qwen3.5 WebGPU cleanup did not complete",
      );
    }
  }

  async #compile(kernel: Qwen35KernelSnapshot): Promise<Qwen35ComputePipeline> {
    const module = this.#device.createShaderModule({
      label: kernel.id,
      code: kernel.source,
    });
    let info: Awaited<ReturnType<Qwen35ShaderModule["getCompilationInfo"]>>;
    try {
      info = await module.getCompilationInfo();
    } catch {
      throw diagnosticError(
        "webgpu-kernel-compile-failed",
        "Qwen3.5 WebGPU kernel compilation failed",
      );
    }
    if (Array.from(info.messages).some((message) => message.type === "error")) {
      throw diagnosticError(
        "webgpu-kernel-compile-failed",
        "Qwen3.5 WebGPU kernel compilation failed",
      );
    }
    try {
      return await this.#device.createComputePipelineAsync({
        label: kernel.id,
        layout: "auto",
        compute: { module, entryPoint: kernel.entryPoint },
      });
    } catch {
      throw diagnosticError(
        "webgpu-kernel-compile-failed",
        "Qwen3.5 WebGPU kernel compilation failed",
      );
    }
  }
}
