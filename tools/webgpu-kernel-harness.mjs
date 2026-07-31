import {
  LANGUAGE_GEMV_KERNELS,
  gemvCpu,
  planGemvDispatch,
} from "../dist/src/mixed-gemv.js";
import {
  repackNativeQ4K,
  repackNativeQ5K,
  repackNativeQ6K,
  repackNativeQ8_0,
} from "../dist/src/mixed-quant.js";
import { repackNativeQ3K } from "../dist/src/q3k.js";
import { QWEN_PRIMITIVE_KERNELS } from "../dist/src/qwen-primitives.js";
import { PACKED_EMBEDDING_KERNELS } from "../dist/src/qwen-embedding.js";
import { validateParity } from "./webgpu-parity.mjs";

function deterministicBytes(length, seed) {
  let state = seed >>> 0;
  return Uint8Array.from({ length }, () => {
    state = Math.imul(state ^ (state >>> 15), 0x2c1b3c6d) >>> 0;
    state = Math.imul(state ^ (state >>> 12), 0x297a2d39) >>> 0;
    return (state ^ (state >>> 15)) & 255;
  });
}

function setHalf(bytes, offset, bits = 0x3c00) {
  new DataView(bytes.buffer, bytes.byteOffset).setUint16(offset, bits, true);
}

function packedFixture(layout, rowIndex, blockIndex) {
  if (layout === "f32") {
    const values = Float32Array.of(
      (rowIndex + 1) * 2.5 + (blockIndex + 1) * 0.75,
    );
    return new Uint8Array(values.buffer.slice(0));
  }
  const nativeBytes = {
    "q8-0-36": 34,
    "q3-k-112": 110,
    "q4-k-144": 144,
    "q5-k-176": 176,
    "q6-k-212": 210,
  }[layout];
  const seed =
    nativeBytes * 101 + (rowIndex + 1) * 1009 + (blockIndex + 1) * 917;
  const native = deterministicBytes(nativeBytes, seed);
  if (layout === "q8-0-36") {
    setHalf(native, 0, 0x3800);
    return repackNativeQ8_0(native);
  }
  if (layout === "q3-k-112") {
    setHalf(native, 108);
    return repackNativeQ3K(native);
  }
  if (layout === "q4-k-144") {
    setHalf(native, 0);
    setHalf(native, 2, 0x3800);
    return repackNativeQ4K(native);
  }
  if (layout === "q5-k-176") {
    setHalf(native, 0);
    setHalf(native, 2, 0x3800);
    return repackNativeQ5K(native);
  }
  setHalf(native, 208);
  return repackNativeQ6K(native);
}

function storageBuffer(device, bytes, usage) {
  const buffer = device.createBuffer({
    size: Math.max(4, Math.ceil(bytes.byteLength / 4) * 4),
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(bytes);
  buffer.unmap();
  return buffer;
}

async function runKernel(device, kernel) {
  const module = device.createShaderModule({
    label: kernel.id,
    code: kernel.source,
  });
  const compilation = await module.getCompilationInfo();
  const errors = Array.from(compilation.messages).filter(
    (message) => message.type === "error",
  );
  if (errors.length > 0) {
    throw new Error(
      `${kernel.id}: ${errors.map((message) => message.message).join("; ")}`,
    );
  }

  const rows = 3;
  const blocksPerRow = 2;
  const columns = kernel.abi.valuesPerBlock * blocksPerRow;
  const packedByteOffset = 32;
  const packed = new Uint8Array(
    packedByteOffset + kernel.abi.bytesPerBlock * blocksPerRow * rows,
  );
  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    for (let blockIndex = 0; blockIndex < blocksPerRow; blockIndex += 1) {
      const block = packedFixture(kernel.layout, rowIndex, blockIndex);
      const blockOffset =
        packedByteOffset +
        (rowIndex * blocksPerRow + blockIndex) * kernel.abi.bytesPerBlock;
      packed.set(block, blockOffset);
    }
  }
  const activation = Float32Array.from(
    { length: columns },
    (_, index) => ((index * 17 + 3) % 29 - 14) / 16,
  );
  const expected = gemvCpu(kernel.layout, packed, activation, {
    rows,
    columns,
    packedByteOffset,
  });
  if (new Set(expected).size !== rows) {
    throw new Error(`${kernel.id}: row fixtures did not produce distinct output`);
  }
  const outputRowOffset = 2;
  const plan = planGemvDispatch({
    layout: kernel.layout,
    localRows: rows,
    columns,
    packedByteOffset,
    outputRowOffset,
    maxWorkgroupsPerDimension: 2,
  });
  if (plan.workgroups.x !== 2 || plan.workgroups.y !== 2) {
    throw new Error(`${kernel.id}: harness did not force a 2D dispatch`);
  }
  const sentinel = -12_345.5;
  const outputSlots = outputRowOffset + rows + 1;
  const expectedOutput = new Float32Array(outputSlots).fill(sentinel);
  expectedOutput.set(expected, outputRowOffset);

  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "packed_gemv" },
  });
  const weights = storageBuffer(
    device,
    packed,
    GPUBufferUsage.STORAGE,
  );
  const inputs = storageBuffer(
    device,
    new Uint8Array(activation.buffer),
    GPUBufferUsage.STORAGE,
  );
  const initialOutput = new Float32Array(outputSlots).fill(sentinel);
  const output = storageBuffer(
    device,
    new Uint8Array(initialOutput.buffer),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const uniforms = storageBuffer(
    device,
    new Uint8Array(
      Uint32Array.of(
        rows,
        columns,
        plan.uniforms.blocksPerRow,
        plan.uniforms.weightWordOffset,
        plan.uniforms.outputRowOffset,
      ).buffer,
    ),
    GPUBufferUsage.UNIFORM,
  );
  const readback = device.createBuffer({
    size: outputSlots * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: weights } },
      { binding: 1, resource: { buffer: inputs } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: uniforms } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(
    plan.workgroups.x,
    plan.workgroups.y,
    plan.workgroups.z,
  );
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, outputSlots * 4);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const actual = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  for (const buffer of [weights, inputs, output, uniforms, readback]) {
    buffer.destroy();
  }
  validateParity(kernel.id, expectedOutput, actual);
  return {
    id: kernel.id,
    blocksPerRow,
    outputRowOffset,
    workgroups: plan.workgroups,
    expected: Array.from(expectedOutput),
    actual: Array.from(actual),
  };
}

async function compilePrimitive(device, kernel) {
  const module = device.createShaderModule({
    label: `primitive-compile-${kernel.id}`,
    code: kernel.source,
  });
  const compilation = await module.getCompilationInfo();
  const errors = Array.from(compilation.messages).filter(
    (message) => message.type === "error",
  );
  if (errors.length > 0) {
    throw new Error(
      `${kernel.id}: ${errors.map((message) => message.message).join("; ")}`,
    );
  }
  await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return { id: kernel.id, status: "compiled" };
}

export async function runWebGpuKernelHarness() {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU did not provide an adapter");
  const device = await adapter.requestDevice();
  const results = [];
  for (const kernel of LANGUAGE_GEMV_KERNELS) {
    results.push(await runKernel(device, kernel));
  }
  for (const kernel of QWEN_PRIMITIVE_KERNELS) {
    results.push(await compilePrimitive(device, kernel));
  }
  for (const kernel of PACKED_EMBEDDING_KERNELS) {
    results.push(await compilePrimitive(device, kernel));
  }
  device.destroy();
  return results;
}

const output = document.querySelector("#results");
try {
  const results = await runWebGpuKernelHarness();
  output.textContent = JSON.stringify({ status: "pass", results }, null, 2);
  document.documentElement.dataset.status = "pass";
} catch (error) {
  output.textContent = JSON.stringify(
    { status: "error", message: String(error) },
    null,
    2,
  );
  document.documentElement.dataset.status = "error";
}
