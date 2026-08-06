import { diagnosticError, isSafeDiagnosticCode } from "./diagnostics.js";

const GPU_BUFFER_USAGE_COPY_DST = 0x0008;
const GPU_BUFFER_USAGE_UNIFORM = 0x0040;
const GPU_BUFFER_USAGE_MAP_READ = 0x0001;
const GPU_MAP_MODE_READ = 0x0001;

// The fixed ABI bound still covers both mutually exclusive language tails:
// resident execution uses two reduction kernels, while disk-backed execution
// replaces them with one staged reduction kernel. Vision adds its fixed set.
const QWEN35_FIXED_ABI_KERNEL_CAPACITY = 39;

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

export interface Qwen35CommandEncoder {
  beginComputePass(): Qwen35ComputePass;
  clearBuffer(
    buffer: Qwen35WebGpuBuffer,
    offset?: number,
    size?: number,
  ): void;
  copyBufferToBuffer(
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
  readonly contentIdentity: number;
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
  readonly #kernelContentIdentities = new Map<string, number>();
  readonly #uniforms = new WeakSet<object>();
  readonly #ownedBuffers = new Set<Qwen35OwnedWebGpuBuffer>();
  readonly #operations = new Set<Promise<unknown>>();
  #nextBufferId = 1;
  #nextKernelContentIdentity = 1;
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
    let buffer: Qwen35OwnedWebGpuBuffer;
    try {
      buffer = this.#device.createBuffer({
        label,
        size: values.byteLength,
        usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
      });
    } catch {
      this.#poisoned = true;
      throw diagnosticError(
        "webgpu-uniform-allocation-failed",
        "Qwen3.5 WebGPU uniform allocation failed",
      );
    }
    this.#ownedBuffers.add(buffer);
    const uniform = Object.freeze({ buffer, byteLength: values.byteLength });
    this.#uniforms.add(uniform);
    try {
      this.updateUniform(uniform, values);
    } catch (error) {
      this.#uniforms.delete(uniform);
      this.#releaseOwnedBuffer(buffer);
      throw error;
    }
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
    try {
      this.#device.queue.writeBuffer(
        uniform.buffer,
        0,
        values.buffer,
        values.byteOffset,
        values.byteLength,
      );
    } catch {
      this.#poisoned = true;
      throw diagnosticError(
        "webgpu-uniform-upload-failed",
        "Qwen3.5 WebGPU uniform upload failed",
      );
    }
  }

  async dispatch(request: Qwen35DispatchRequest): Promise<void> {
    await this.dispatchBatch([request]);
  }

  /** Encodes one model step without paying one queue submission per kernel. */
  dispatchBatch(requests: readonly Qwen35DispatchRequest[]): Promise<void> {
    return this.#track(this.#dispatchBatch(requests));
  }

  async #dispatchBatch(requests: readonly Qwen35DispatchRequest[]): Promise<void> {
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
    try {
      this.#device.pushErrorScope("validation");
    } catch {
      this.#poisoned = true;
      throw diagnosticError(
        "webgpu-dispatch-validation-failed",
        "Qwen3.5 WebGPU dispatch validation failed",
      );
    }
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

  /** Drops bindings that may retain caller-owned rolling GPU buffers. */
  releaseBindGroups(): void {
    this.#assertUsable();
    this.#bindGroups.clear();
  }

  /** Reads only the GPU-selected token or count, never a vocabulary score tile. */
  readU32(source: Qwen35WebGpuBuffer, byteOffset: number): Promise<number> {
    return this.#track(this.#readU32(source, byteOffset));
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
    const cached = this.#pipelines.get(snapshot.contentIdentity);
    if (cached !== undefined) return cached;
    const compiling = this.#compile(snapshot);
    this.#pipelines.set(snapshot.contentIdentity, compiling);
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
    // JSON array encoding is unambiguous even when WGSL contains separators.
    // Interning keeps later bind-group keys small while retaining exactly one
    // strong cache entry per semantic kernel in the fixed model program.
    const contentKey = JSON.stringify([
      kernel.id,
      kernel.entryPoint,
      kernel.source,
    ]);
    let contentIdentity = this.#kernelContentIdentities.get(contentKey);
    if (contentIdentity === undefined) {
      if (
        this.#kernelContentIdentities.size >=
          QWEN35_FIXED_ABI_KERNEL_CAPACITY
      ) {
        this.#poisoned = true;
        throw diagnosticError(
          "webgpu-kernel-capacity-exceeded",
          "Qwen3.5 WebGPU kernel capacity was exceeded",
        );
      }
      contentIdentity = this.#nextKernelContentIdentity;
      this.#nextKernelContentIdentity += 1;
      this.#kernelContentIdentities.set(contentKey, contentIdentity);
    }
    const snapshot = Object.freeze({
      contentIdentity,
      id: kernel.id,
      source: kernel.source,
      entryPoint: kernel.entryPoint,
    });
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
    const key = `${this.#kernelSnapshot(request.kernel).contentIdentity}\u0000${bindings.join("|")}`;
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

  #track<T>(operation: Promise<T>): Promise<T> {
    this.#operations.add(operation);
    void operation.then(
      () => this.#operations.delete(operation),
      () => this.#operations.delete(operation),
    );
    return operation;
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

  async #readU32(
    source: Qwen35WebGpuBuffer,
    byteOffset: number,
  ): Promise<number> {
    this.#assertUsable();
    if (
      !Number.isSafeInteger(byteOffset) ||
      byteOffset < 0 ||
      byteOffset % 4 !== 0
    ) {
      throw diagnosticError(
        "webgpu-readback-invalid",
        "Qwen3.5 WebGPU readback range is invalid",
      );
    }
    let readback: Qwen35OwnedWebGpuBuffer | null = null;
    try {
      this.#device.pushErrorScope("validation");
    } catch {
      this.#poisoned = true;
      throw diagnosticError(
        "webgpu-readback-failed",
        "Qwen3.5 WebGPU readback failed",
      );
    }
    try {
      readback = this.#device.createBuffer({
        label: "qwen35-selected-u32",
        size: 4,
        usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
      });
      this.#ownedBuffers.add(readback);
      const encoder = this.#device.createCommandEncoder({
        label: "qwen35-selected-u32-copy",
      });
      encoder.copyBufferToBuffer(source, byteOffset, readback, 0, 4);
      this.#device.queue.submit([encoder.finish()]);
    } catch {
      await this.#retireFailedScope();
      this.#releaseOwnedBuffer(readback);
      throw diagnosticError(
        "webgpu-readback-failed",
        "Qwen3.5 WebGPU readback failed",
      );
    }
    let validationError: { readonly message?: string } | null;
    try {
      validationError = await this.#device.popErrorScope();
    } catch {
      this.#poisoned = true;
      this.#releaseOwnedBuffer(readback);
      throw diagnosticError(
        "webgpu-readback-failed",
        "Qwen3.5 WebGPU readback failed",
      );
    }
    if (validationError !== null) {
      this.#poisoned = true;
      this.#releaseOwnedBuffer(readback);
      throw diagnosticError(
        "webgpu-readback-failed",
        "Qwen3.5 WebGPU readback failed",
      );
    }
    let value: number;
    try {
      if (
        readback.mapAsync === undefined ||
        readback.getMappedRange === undefined ||
        readback.unmap === undefined
      ) {
        throw new Error("readback methods unavailable");
      }
      await readback.mapAsync(GPU_MAP_MODE_READ);
      const mapped = readback.getMappedRange();
      if (mapped.byteLength < 4) throw new Error("readback range is short");
      value = new DataView(mapped).getUint32(0, true);
      readback.unmap();
    } catch {
      this.#poisoned = true;
      this.#releaseOwnedBuffer(readback);
      throw diagnosticError(
        "webgpu-readback-failed",
        "Qwen3.5 WebGPU readback failed",
      );
    }
    if (!this.#releaseOwnedBuffer(readback)) {
      throw diagnosticError(
        "webgpu-readback-cleanup-failed",
        "Qwen3.5 WebGPU readback cleanup failed",
      );
    }
    return value;
  }

  #releaseOwnedBuffer(buffer: Qwen35OwnedWebGpuBuffer | null): boolean {
    if (buffer === null || !this.#ownedBuffers.has(buffer)) return true;
    try {
      buffer.destroy();
      this.#ownedBuffers.delete(buffer);
      return true;
    } catch {
      this.#poisoned = true;
      return false;
    }
  }

  async #cleanup(): Promise<void> {
    let failed = false;
    // Error-scope retirement and other executor work must finish before the
    // device queue or the origin-wide model lock can transfer ownership.
    await Promise.allSettled([...this.#operations]);
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
    this.#kernelContentIdentities.clear();
    if (failed) {
      throw diagnosticError(
        "webgpu-cleanup-failed",
        "Qwen3.5 WebGPU cleanup did not complete",
      );
    }
  }

  async #compile(kernel: Qwen35KernelSnapshot): Promise<Qwen35ComputePipeline> {
    let module: Qwen35ShaderModule;
    try {
      module = this.#device.createShaderModule({
        label: kernel.id,
        code: kernel.source,
      });
    } catch {
      throw diagnosticError(
        "webgpu-kernel-compile-failed",
        "Qwen3.5 WebGPU kernel compilation failed",
      );
    }
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
