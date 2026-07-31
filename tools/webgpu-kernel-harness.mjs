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
import {
  QWEN_PRIMITIVE_KERNELS,
  attentionOutputGateCpu,
  fusedSwiGluCpu,
  partialMropeCpu,
  qkRmsNormPerHeadCpu,
  residualAddCpu,
  rmsNormCpu,
  siluCpu,
  stableTiledTopK,
} from "../dist/src/qwen-primitives.js";
import {
  PACKED_EMBEDDING_KERNELS,
  embeddingCpu,
  planPackedEmbeddingRow,
} from "../dist/src/qwen-embedding.js";
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

async function createPipeline(device, kernel) {
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
  return device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

function floatBytes(values) {
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

function uintBytes(values) {
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

function paramsBytes(byteLength, writes) {
  const bytes = new Uint8Array(byteLength);
  const view = new DataView(bytes.buffer);
  for (const [kind, offset, value] of writes) {
    view[kind](offset, value, true);
  }
  return bytes;
}

async function dispatchAndRead(
  device,
  kernel,
  entries,
  outputs,
  workgroups,
) {
  const pipeline = await createPipeline(device, kernel);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });
  const readbacks = outputs.map(({ byteLength }) =>
    device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
  );
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroups.x, workgroups.y, workgroups.z);
  pass.end();
  outputs.forEach(({ buffer, byteLength }, index) => {
    encoder.copyBufferToBuffer(buffer, 0, readbacks[index], 0, byteLength);
  });
  device.queue.submit([encoder.finish()]);
  const result = [];
  for (const readback of readbacks) {
    await readback.mapAsync(GPUMapMode.READ);
    result.push(readback.getMappedRange().slice(0));
    readback.unmap();
  }
  for (const buffer of new Set([
    ...entries.map((entry) => entry.resource.buffer),
    ...readbacks,
  ])) {
    buffer.destroy();
  }
  return result;
}

function outputFixture(cpuValues, sentinel = -12_345.5) {
  const outputRowOffset = 64;
  const expected = new Float32Array(
    outputRowOffset + cpuValues.length + 1,
  ).fill(sentinel);
  expected.set(cpuValues, outputRowOffset);
  return {
    expected,
    initial: new Float32Array(expected.length).fill(sentinel),
    outputRowOffset,
  };
}

function vectorPrimitiveFixture(operation) {
  const input = Float32Array.of(3, -4, 0.5, -2);
  switch (operation) {
    case "rms-norm": {
      const weight = Float32Array.of(1, 0.75, 1.25, 0.5);
      return {
        inputs: [input, weight],
        expected: rmsNormCpu(input, weight, Math.fround(1e-6)),
        params: paramsBytes(16, [
          ["setUint32", 0, input.length],
          ["setUint32", 4, input.length],
          ["setFloat32", 8, Math.fround(1e-6)],
        ]),
      };
    }
    case "residual-add": {
      const residual = Float32Array.of(-1, 2, 3, -4);
      return {
        inputs: [input, residual],
        expected: residualAddCpu(input, residual),
        params: uintBytes(Uint32Array.of(input.length, 0, 0, 0)),
      };
    }
    case "silu":
      return {
        inputs: [input],
        expected: Float32Array.from(input, siluCpu),
        params: uintBytes(Uint32Array.of(input.length, 0, 0, 0)),
      };
    case "swiglu": {
      const up = Float32Array.of(2, 0.5, -3, 4);
      return {
        inputs: [input, up],
        expected: fusedSwiGluCpu(input, up),
        params: uintBytes(Uint32Array.of(input.length, 0, 0, 0)),
      };
    }
    case "attention-output-gate": {
      const gate = Float32Array.of(0, 1, -1, 2);
      return {
        inputs: [input, gate],
        expected: attentionOutputGateCpu(input, gate),
        params: uintBytes(Uint32Array.of(input.length, 0, 0, 0)),
      };
    }
    case "qk-rms-norm": {
      const weight = Float32Array.of(1, 0.75);
      return {
        inputs: [input, weight],
        expected: qkRmsNormPerHeadCpu(input, weight, {
          headCount: 2,
          headDimension: 2,
          epsilon: Math.fround(1e-6),
        }),
        params: paramsBytes(16, [
          ["setUint32", 0, input.length],
          ["setUint32", 4, 2],
          ["setFloat32", 8, Math.fround(1e-6)],
        ]),
      };
    }
    default:
      throw new Error(`No vector fixture for ${operation}`);
  }
}

async function runVectorPrimitive(device, kernel) {
  const fixture = vectorPrimitiveFixture(kernel.operation);
  const output = outputFixture(fixture.expected);
  const outputBuffer = storageBuffer(
    device,
    floatBytes(output.initial),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const inputBuffers = fixture.inputs.map((input) =>
    storageBuffer(device, floatBytes(input), GPUBufferUsage.STORAGE),
  );
  const uniformBuffer = storageBuffer(
    device,
    fixture.params,
    GPUBufferUsage.UNIFORM,
  );
  const entries = inputBuffers.map((buffer, binding) => ({
    binding,
    resource: { buffer },
  }));
  const outputBinding = kernel.abi.bindings.output;
  entries.push({
    binding: outputBinding,
    resource: {
      buffer: outputBuffer,
      offset: output.outputRowOffset * 4,
      size: (fixture.expected.length + 1) * 4,
    },
  });
  entries.push({
    binding: kernel.abi.bindings.uniforms,
    resource: { buffer: uniformBuffer },
  });
  const [actualBytes] = await dispatchAndRead(
    device,
    kernel,
    entries,
    [{ buffer: outputBuffer, byteLength: output.expected.byteLength }],
    { x: 1, y: 1, z: 1 },
  );
  const actual = new Float32Array(actualBytes);
  validateParity(kernel.id, output.expected, actual);
  return {
    id: kernel.id,
    status: "executed",
    outputRowOffset: output.outputRowOffset,
  };
}

async function runMropePrimitive(device, kernel) {
  const input = Float32Array.from(
    { length: 256 },
    (_, index) => Math.fround(((index * 19) % 31 - 15) / 7),
  );
  const positions = [16_384, 16_383, 16_382];
  const expectedValues = partialMropeCpu(input, {
    headCount: 1,
    headDimension: 256,
    rotaryDimension: 64,
    sections: [11, 11, 10],
    positions: [16_384, 16_383, 16_382],
    theta: 10_000_000,
  });
  const output = outputFixture(expectedValues);
  const inputBuffer = storageBuffer(
    device,
    floatBytes(input),
    GPUBufferUsage.STORAGE,
  );
  const outputBuffer = storageBuffer(
    device,
    floatBytes(output.initial),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const sectionsBuffer = storageBuffer(
    device,
    uintBytes(Uint32Array.of(11, 11, 10)),
    GPUBufferUsage.STORAGE,
  );
  const positionsBuffer = storageBuffer(
    device,
    uintBytes(Uint32Array.from(positions)),
    GPUBufferUsage.STORAGE,
  );
  const uniformBuffer = storageBuffer(
    device,
    paramsBytes(32, [
      ["setUint32", 0, input.length],
      ["setUint32", 4, 256],
      ["setUint32", 8, 64],
      ["setUint32", 12, 32],
      ["setFloat32", 16, 10_000_000],
    ]),
    GPUBufferUsage.UNIFORM,
  );
  const [actualBytes] = await dispatchAndRead(
    device,
    kernel,
    [
      { binding: 0, resource: { buffer: inputBuffer } },
      {
        binding: 1,
        resource: {
          buffer: outputBuffer,
          offset: output.outputRowOffset * 4,
          size: (expectedValues.length + 1) * 4,
        },
      },
      { binding: 2, resource: { buffer: sectionsBuffer } },
      { binding: 3, resource: { buffer: positionsBuffer } },
      { binding: 4, resource: { buffer: uniformBuffer } },
    ],
    [{ buffer: outputBuffer, byteLength: output.expected.byteLength }],
    { x: 1, y: 1, z: 1 },
  );
  const actual = new Float32Array(actualBytes);
  // Portable WebGPU trigonometric built-ins are approximate even after both
  // paths use the same explicit f32 range reduction.
  validateParity(kernel.id, output.expected, actual, 1e-3);
  return {
    id: kernel.id,
    status: "executed",
    positions: [16_384, 16_383, 16_382],
    outputRowOffset: output.outputRowOffset,
  };
}

async function runTopKPrimitive(device, kernel) {
  const scores = Float32Array.of(Number.NaN, 5, 5, Infinity, -2);
  const k = 4;
  const cpu = stableTiledTopK([{ startIndex: 0, scores }], k);
  const sentinel = -12_345.5;
  const outputRowOffset = 64;
  const expectedScores = new Float32Array(outputRowOffset + k + 1).fill(
    sentinel,
  );
  const expectedIndices = new Uint32Array(outputRowOffset + k + 1).fill(
    0xffff_ffff,
  );
  cpu.forEach((entry, index) => {
    expectedScores[outputRowOffset + index] = entry.score;
    expectedIndices[outputRowOffset + index] = entry.index;
  });
  const scoreBuffer = storageBuffer(
    device,
    floatBytes(scores),
    GPUBufferUsage.STORAGE,
  );
  const outputScores = storageBuffer(
    device,
    floatBytes(new Float32Array(expectedScores.length).fill(sentinel)),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const outputIndices = storageBuffer(
    device,
    uintBytes(new Uint32Array(expectedIndices.length).fill(0xffff_ffff)),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const validCount = storageBuffer(
    device,
    uintBytes(Uint32Array.of(0xffff_ffff)),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const uniforms = storageBuffer(
    device,
    uintBytes(Uint32Array.of(scores.length, k, 0, 0)),
    GPUBufferUsage.UNIFORM,
  );
  const bindingSize = (k + 1) * 4;
  const [scoreBytes, indexBytes, countBytes] = await dispatchAndRead(
    device,
    kernel,
    [
      { binding: 0, resource: { buffer: scoreBuffer } },
      {
        binding: 1,
        resource: {
          buffer: outputScores,
          offset: outputRowOffset * 4,
          size: bindingSize,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: outputIndices,
          offset: outputRowOffset * 4,
          size: bindingSize,
        },
      },
      { binding: 3, resource: { buffer: validCount } },
      { binding: 4, resource: { buffer: uniforms } },
    ],
    [
      { buffer: outputScores, byteLength: expectedScores.byteLength },
      { buffer: outputIndices, byteLength: expectedIndices.byteLength },
      { buffer: validCount, byteLength: 4 },
    ],
    { x: 1, y: 1, z: 1 },
  );
  const actualScores = new Float32Array(scoreBytes);
  const actualIndices = new Uint32Array(indexBytes);
  const actualValidCount = new Uint32Array(countBytes)[0];
  validateParity(kernel.id, expectedScores, actualScores);
  if (
    actualValidCount !== cpu.length ||
    actualIndices.some((value, index) => value !== expectedIndices[index])
  ) {
    throw new Error(`${kernel.id}: GPU top-k indices or validCount differ`);
  }
  return {
    id: kernel.id,
    status: "executed",
    validCount: actualValidCount,
    outputRowOffset,
  };
}

async function runPrimitive(device, kernel) {
  if (kernel.operation === "partial-mrope") {
    return runMropePrimitive(device, kernel);
  }
  if (kernel.operation === "top-k-merge") {
    return runTopKPrimitive(device, kernel);
  }
  return runVectorPrimitive(device, kernel);
}

async function runEmbedding(device, kernel) {
  const rows = 3;
  const tokenId = 1;
  const blocksPerRow = 2;
  const embeddingLength = kernel.abi.valuesPerBlock * blocksPerRow;
  const packed = new Uint8Array(
    rows * blocksPerRow * kernel.abi.bytesPerBlock,
  );
  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    for (let blockIndex = 0; blockIndex < blocksPerRow; blockIndex += 1) {
      packed.set(
        packedFixture(kernel.storageType, rowIndex, blockIndex),
        (rowIndex * blocksPerRow + blockIndex) * kernel.abi.bytesPerBlock,
      );
    }
  }
  const plan = planPackedEmbeddingRow({
    ggmlType: kernel.ggmlType,
    storageType: kernel.storageType,
    tokenId,
    vocabSize: rows,
    embeddingLength,
    maxComputeWorkgroupsPerDimension:
      device.limits.maxComputeWorkgroupsPerDimension,
  });
  const expectedValues = embeddingCpu(
    kernel.storageType,
    packed,
    tokenId,
    { vocabSize: rows, embeddingLength },
  );
  const output = outputFixture(expectedValues);
  const packedBuffer = storageBuffer(
    device,
    packed,
    GPUBufferUsage.STORAGE,
  );
  const outputBuffer = storageBuffer(
    device,
    floatBytes(output.initial),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const uniforms = storageBuffer(
    device,
    uintBytes(
      Uint32Array.of(
        plan.packedByteOffset / 4,
        plan.outputElements,
        blocksPerRow,
        0,
      ),
    ),
    GPUBufferUsage.UNIFORM,
  );
  const [actualBytes] = await dispatchAndRead(
    device,
    kernel,
    [
      { binding: 0, resource: { buffer: packedBuffer } },
      {
        binding: 1,
        resource: {
          buffer: outputBuffer,
          offset: output.outputRowOffset * 4,
          size: (expectedValues.length + 1) * 4,
        },
      },
      { binding: 2, resource: { buffer: uniforms } },
    ],
    [{ buffer: outputBuffer, byteLength: output.expected.byteLength }],
    plan.workgroups,
  );
  const actual = new Float32Array(actualBytes);
  validateParity(kernel.id, output.expected, actual);
  return {
    id: kernel.id,
    status: "executed",
    packedByteOffset: plan.packedByteOffset,
    outputRowOffset: output.outputRowOffset,
  };
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
    results.push(await runPrimitive(device, kernel));
  }
  for (const kernel of PACKED_EMBEDDING_KERNELS) {
    results.push(await runEmbedding(device, kernel));
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
