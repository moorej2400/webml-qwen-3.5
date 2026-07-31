import { diagnosticError, isSafeDiagnosticCode } from "./diagnostics.js";

const GPU_BUFFER_USAGE_COPY_DST = 0x0008;
const GPU_BUFFER_USAGE_UNIFORM = 0x0040;

export interface Qwen35WebGpuBuffer {
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
  }): Qwen35WebGpuBuffer;
}

export interface Qwen35KernelSource {
  readonly id: string;
  readonly source: string;
  readonly entryPoint: string;
}

export interface Qwen35BufferBinding {
  readonly binding: number;
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
  readonly buffer: Qwen35WebGpuBuffer;
  readonly byteLength: number;
}

function requirePositiveInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function validateDispatch(request: Qwen35DispatchRequest): void {
  if (
    !isSafeDiagnosticCode(request.kernel.id) ||
    !/^[A-Za-z_][A-Za-z0-9_]*$/.test(request.kernel.entryPoint) ||
    request.kernel.source.length === 0 ||
    !requirePositiveInteger(request.workgroups.x) ||
    !requirePositiveInteger(request.workgroups.y) ||
    !requirePositiveInteger(request.workgroups.z)
  ) {
    throw diagnosticError(
      "webgpu-dispatch-invalid",
      "Qwen3.5 WebGPU dispatch is invalid",
    );
  }
  const bindings = new Set<number>();
  for (const binding of request.bindings) {
    const valid =
      Number.isSafeInteger(binding.binding) &&
      binding.binding >= 0 &&
      Number.isSafeInteger(binding.offset) &&
      binding.offset >= 0 &&
      binding.offset % 4 === 0 &&
      requirePositiveInteger(binding.size) &&
      binding.size % 4 === 0 &&
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
  readonly #pipelines = new Map<string, Promise<Qwen35ComputePipeline>>();
  readonly #ownedBuffers = new Set<Qwen35WebGpuBuffer>();
  #disposed = false;

  constructor(device: Qwen35WebGpuDevice) {
    this.#device = device;
  }

  createUniform(label: string, values: ArrayBufferView<ArrayBuffer>): Qwen35OwnedUniform {
    this.#assertUsable();
    if (
      !isSafeDiagnosticCode(label) ||
      values.byteLength === 0 ||
      values.byteLength % 4 !== 0
    ) {
      throw diagnosticError(
        "webgpu-uniform-invalid",
        "Qwen3.5 WebGPU uniform data is invalid",
      );
    }
    const buffer = this.#device.createBuffer({
      label,
      size: values.byteLength,
      usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
    });
    this.#ownedBuffers.add(buffer);
    this.#device.queue.writeBuffer(
      buffer,
      0,
      values.buffer,
      values.byteOffset,
      values.byteLength,
    );
    return Object.freeze({ buffer, byteLength: values.byteLength });
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
    for (const request of requests) validateDispatch(request);
    const pipelines = await Promise.all(
      requests.map((request) => this.#pipeline(request.kernel)),
    );
    this.#assertUsable();
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
  }

  async submittedWorkDone(): Promise<void> {
    this.#assertUsable();
    await this.#device.queue.onSubmittedWorkDone();
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    for (const buffer of this.#ownedBuffers) buffer.destroy();
    this.#ownedBuffers.clear();
    this.#pipelines.clear();
  }

  #assertUsable(): void {
    if (this.#disposed) {
      throw diagnosticError(
        "webgpu-executor-disposed",
        "Qwen3.5 WebGPU executor is disposed",
      );
    }
  }

  #pipeline(kernel: Qwen35KernelSource): Promise<Qwen35ComputePipeline> {
    const key = `${kernel.id}\u0000${kernel.entryPoint}`;
    const cached = this.#pipelines.get(key);
    if (cached !== undefined) return cached;
    const compiling = this.#compile(kernel);
    this.#pipelines.set(key, compiling);
    return compiling;
  }

  async #compile(kernel: Qwen35KernelSource): Promise<Qwen35ComputePipeline> {
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
