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
import { QWEN35_HYBRID_KERNELS } from "../dist/src/hybrid-kernels.js";
import {
  packFloat16PairCpu,
  qwen35OnlineAttentionHeadCpu,
  splitQwen35QueryGateProjection,
} from "../dist/src/full-attention.js";
import {
  deltaNetRecurrentHeadStepCpu,
  qwen35DeltaNetParametersCpu,
} from "../dist/src/gated-deltanet.js";
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

async function runFullAttentionPrepare(device, kernel) {
  const position = 16_383;
  const capacity = 16_384;
  const positions = [7, 5, 3];
  const queryGate = new Float32Array(16 * 512);
  for (let head = 0; head < 16; head += 1) {
    const base = head * 512;
    for (let lane = 0; lane < 256; lane += 1) {
      queryGate[base + lane] = ((head * 13 + lane * 7) % 31 - 15) / 8;
      queryGate[base + 256 + lane] = (head * 256 + lane) / 4096;
    }
  }
  const key = Float32Array.from(
    { length: 4 * 256 },
    (_, index) => ((index * 11) % 37 - 18) / 9,
  );
  const value = Float32Array.from(
    { length: 4 * 256 },
    (_, index) => ((index * 5) % 23 - 11) / 7,
  );
  const queryNormWeight = Float32Array.from(
    { length: 256 },
    (_, lane) => 0.75 + (lane % 7) / 10,
  );
  const keyNormWeight = Float32Array.from(
    { length: 256 },
    (_, lane) => 0.8 + (lane % 5) / 8,
  );
  const split = splitQwen35QueryGateProjection(queryGate);
  const normalizedQuery = qkRmsNormPerHeadCpu(
    split.query,
    queryNormWeight,
    {
      headCount: 16,
      headDimension: 256,
      epsilon: Math.fround(1e-6),
    },
  );
  const expectedQuery = partialMropeCpu(normalizedQuery, {
    headCount: 16,
    headDimension: 256,
    rotaryDimension: 64,
    sections: [11, 11, 10],
    positions,
    theta: 10_000_000,
  });
  const normalizedKey = qkRmsNormPerHeadCpu(key, keyNormWeight, {
    headCount: 4,
    headDimension: 256,
    epsilon: Math.fround(1e-6),
  });
  const expectedKey = partialMropeCpu(normalizedKey, {
    headCount: 4,
    headDimension: 256,
    rotaryDimension: 64,
    sections: [11, 11, 10],
    positions,
    theta: 10_000_000,
  });

  const wordsPerToken = 512;
  const suffixSentinel = 0xdead_beef;
  const cacheWords = capacity * wordsPerToken;
  const packedKeyCache = new Uint32Array(cacheWords + 2).fill(suffixSentinel);
  const packedValueCache = new Uint32Array(cacheWords + 2).fill(suffixSentinel);
  const expectedPackedKey = new Uint32Array(wordsPerToken + 2).fill(
    suffixSentinel,
  );
  const expectedPackedValue = new Uint32Array(wordsPerToken + 2).fill(
    suffixSentinel,
  );
  for (let scalar = 0; scalar < 4 * 256; scalar += 2) {
    expectedPackedKey[scalar / 2] = packFloat16PairCpu(
      expectedKey[scalar],
      expectedKey[scalar + 1],
    );
    expectedPackedValue[scalar / 2] = packFloat16PairCpu(
      value[scalar],
      value[scalar + 1],
    );
  }

  const pipeline = await createPipeline(device, kernel);
  const queryGateBuffer = storageBuffer(
    device,
    floatBytes(queryGate),
    GPUBufferUsage.STORAGE,
  );
  const keyBuffer = storageBuffer(device, floatBytes(key), GPUBufferUsage.STORAGE);
  const valueBuffer = storageBuffer(
    device,
    floatBytes(value),
    GPUBufferUsage.STORAGE,
  );
  const queryNormBuffer = storageBuffer(
    device,
    floatBytes(queryNormWeight),
    GPUBufferUsage.STORAGE,
  );
  const keyNormBuffer = storageBuffer(
    device,
    floatBytes(keyNormWeight),
    GPUBufferUsage.STORAGE,
  );
  const preparedOutput = storageBuffer(
    device,
    floatBytes(new Float32Array(16 * 512)),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const packedKeyBuffer = storageBuffer(
    device,
    uintBytes(packedKeyCache),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const packedValueBuffer = storageBuffer(
    device,
    uintBytes(packedValueCache),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const uniforms = storageBuffer(
    device,
    uintBytes(
      Uint32Array.of(position, capacity, ...positions, 0, 0, 0),
    ),
    GPUBufferUsage.UNIFORM,
  );
  const entries = [
    queryGateBuffer,
    keyBuffer,
    valueBuffer,
    queryNormBuffer,
    keyNormBuffer,
    preparedOutput,
    packedKeyBuffer,
    packedValueBuffer,
    uniforms,
  ].map((buffer, binding) => ({ binding, resource: { buffer } }));
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });
  const preparedReadback = device.createBuffer({
    size: queryGate.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const rowByteLength = expectedPackedKey.byteLength;
  const keyReadback = device.createBuffer({
    size: rowByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const valueReadback = device.createBuffer({
    size: rowByteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(16);
  pass.end();
  encoder.copyBufferToBuffer(
    preparedOutput,
    0,
    preparedReadback,
    0,
    queryGate.byteLength,
  );
  const cacheByteOffset = position * wordsPerToken * 4;
  encoder.copyBufferToBuffer(
    packedKeyBuffer,
    cacheByteOffset,
    keyReadback,
    0,
    rowByteLength,
  );
  encoder.copyBufferToBuffer(
    packedValueBuffer,
    cacheByteOffset,
    valueReadback,
    0,
    rowByteLength,
  );
  device.queue.submit([encoder.finish()]);
  await Promise.all([
    preparedReadback.mapAsync(GPUMapMode.READ),
    keyReadback.mapAsync(GPUMapMode.READ),
    valueReadback.mapAsync(GPUMapMode.READ),
  ]);
  const actualPrepared = new Float32Array(
    preparedReadback.getMappedRange().slice(0),
  );
  const actualSplit = splitQwen35QueryGateProjection(actualPrepared);
  const actualPackedKey = new Uint32Array(
    keyReadback.getMappedRange().slice(0),
  );
  const actualPackedValue = new Uint32Array(
    valueReadback.getMappedRange().slice(0),
  );
  for (const readback of [
    preparedReadback,
    keyReadback,
    valueReadback,
  ]) {
    readback.unmap();
  }
  validateParity(kernel.id, expectedQuery, actualSplit.query, 1e-3);
  validateParity(`${kernel.id}-gate`, split.gate, actualSplit.gate);
  if (
    actualPackedKey.some(
      (word, index) => word !== expectedPackedKey[index],
    ) ||
    actualPackedValue.some(
      (word, index) => word !== expectedPackedValue[index],
    )
  ) {
    throw new Error(`${kernel.id}: packed K/V row or suffix differs`);
  }
  for (const buffer of [
    ...entries.map((entry) => entry.resource.buffer),
    preparedReadback,
    keyReadback,
    valueReadback,
  ]) {
    buffer.destroy();
  }
  return {
    id: kernel.id,
    status: "executed",
    position,
    suffixSentinel,
  };
}

async function runFullAttentionOnline(device, kernel) {
  const tokenCount = 2;
  const prepared = new Float32Array(16 * 512);
  const keys = new Float32Array(tokenCount * 4 * 256);
  const values = new Float32Array(keys.length);
  for (let head = 0; head < 16; head += 1) {
    const base = head * 512;
    for (let lane = 0; lane < 256; lane += 1) {
      prepared[base + lane] = ((head * 5 + lane * 3) % 17 - 8) / 8;
      prepared[base + 256 + lane] = (head - 8) / 4;
    }
  }
  for (let token = 0; token < tokenCount; token += 1) {
    for (let kvHead = 0; kvHead < 4; kvHead += 1) {
      const base = (token * 4 + kvHead) * 256;
      for (let lane = 0; lane < 256; lane += 1) {
        keys[base + lane] = ((token * 7 + kvHead * 5 + lane) % 9 - 4) / 4;
        values[base + lane] =
          ((token * 11 + kvHead * 13 + lane * 3) % 15 - 7) / 2;
      }
    }
  }
  const packedKeys = new Uint32Array(keys.length / 2);
  const packedValues = new Uint32Array(values.length / 2);
  for (let scalar = 0; scalar < keys.length; scalar += 2) {
    packedKeys[scalar / 2] = packFloat16PairCpu(
      keys[scalar],
      keys[scalar + 1],
    );
    packedValues[scalar / 2] = packFloat16PairCpu(
      values[scalar],
      values[scalar + 1],
    );
  }
  const expectedValues = new Float32Array(16 * 256);
  for (let queryHead = 0; queryHead < 16; queryHead += 1) {
    const queryBase = queryHead * 512;
    const kvHead = Math.floor(queryHead / 4);
    const headKeys = new Float32Array(tokenCount * 256);
    const headValues = new Float32Array(headKeys.length);
    for (let token = 0; token < tokenCount; token += 1) {
      const source = (token * 4 + kvHead) * 256;
      headKeys.set(keys.subarray(source, source + 256), token * 256);
      headValues.set(values.subarray(source, source + 256), token * 256);
    }
    const attention = qwen35OnlineAttentionHeadCpu(
      prepared.subarray(queryBase, queryBase + 256),
      headKeys,
      headValues,
    );
    const gate = prepared.subarray(queryBase + 256, queryBase + 512);
    expectedValues.set(
      attentionOutputGateCpu(attention, gate),
      queryHead * 256,
    );
  }
  const output = outputFixture(expectedValues);
  const preparedBuffer = storageBuffer(
    device,
    floatBytes(prepared),
    GPUBufferUsage.STORAGE,
  );
  const keyBuffer = storageBuffer(
    device,
    uintBytes(packedKeys),
    GPUBufferUsage.STORAGE,
  );
  const valueBuffer = storageBuffer(
    device,
    uintBytes(packedValues),
    GPUBufferUsage.STORAGE,
  );
  const outputBuffer = storageBuffer(
    device,
    floatBytes(output.initial),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const uniforms = storageBuffer(
    device,
    uintBytes(Uint32Array.of(tokenCount, 1, 2, 0)),
    GPUBufferUsage.UNIFORM,
  );
  const [actualBytes] = await dispatchAndRead(
    device,
    kernel,
    [
      { binding: 0, resource: { buffer: preparedBuffer } },
      { binding: 1, resource: { buffer: keyBuffer } },
      { binding: 2, resource: { buffer: valueBuffer } },
      {
        binding: 3,
        resource: {
          buffer: outputBuffer,
          offset: output.outputRowOffset * 4,
          size: (expectedValues.length + 1) * 4,
        },
      },
      { binding: 4, resource: { buffer: uniforms } },
    ],
    [{ buffer: outputBuffer, byteLength: output.expected.byteLength }],
    { x: 16, y: 1, z: 1 },
  );
  validateParity(kernel.id, output.expected, new Float32Array(actualBytes), 1e-3);
  return { id: kernel.id, status: "executed", tokenCount };
}

async function runDeltaNetConv(device, kernel) {
  const raw = Float32Array.from(
    { length: 8192 },
    (_, index) => ((index * 5) % 19 - 9) / 8,
  );
  const weights = Float32Array.from(
    { length: 8192 * 4 },
    (_, index) => ((index * 7) % 13 - 6) / 16,
  );
  const initialState = Float32Array.from(
    { length: 8192 * 4 },
    (_, index) => ((index * 11) % 17 - 8) / 32,
  );
  const expectedState = new Float32Array(initialState);
  const expectedValues = new Float32Array(8192);
  for (let channel = 0; channel < 8192; channel += 1) {
    const base = channel * 4;
    expectedState[base] = initialState[base + 1];
    expectedState[base + 1] = initialState[base + 2];
    expectedState[base + 2] = initialState[base + 3];
    expectedState[base + 3] = raw[channel];
    let sum = Math.fround(0);
    for (let tap = 0; tap < 4; tap += 1) {
      sum = Math.fround(
        sum + Math.fround(expectedState[base + tap] * weights[base + tap]),
      );
    }
    expectedValues[channel] = siluCpu(sum);
  }
  const output = outputFixture(expectedValues);
  const rawBuffer = storageBuffer(device, floatBytes(raw), GPUBufferUsage.STORAGE);
  const weightBuffer = storageBuffer(
    device,
    floatBytes(weights),
    GPUBufferUsage.STORAGE,
  );
  const stateBuffer = storageBuffer(
    device,
    floatBytes(initialState),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const outputBuffer = storageBuffer(
    device,
    floatBytes(output.initial),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const [actualOutputBytes, actualStateBytes] = await dispatchAndRead(
    device,
    kernel,
    [
      { binding: 0, resource: { buffer: rawBuffer } },
      { binding: 1, resource: { buffer: weightBuffer } },
      { binding: 2, resource: { buffer: stateBuffer } },
      {
        binding: 3,
        resource: {
          buffer: outputBuffer,
          offset: output.outputRowOffset * 4,
          size: (expectedValues.length + 1) * 4,
        },
      },
    ],
    [
      { buffer: outputBuffer, byteLength: output.expected.byteLength },
      { buffer: stateBuffer, byteLength: expectedState.byteLength },
    ],
    { x: 128, y: 1, z: 1 },
  );
  const actualState = new Float32Array(actualStateBytes);
  validateParity(kernel.id, output.expected, new Float32Array(actualOutputBytes));
  validateParity(`${kernel.id}-state`, expectedState, actualState);
  const stateMutation = actualState[0] !== initialState[0];
  if (!stateMutation) throw new Error(`${kernel.id}: state did not mutate`);
  return { id: kernel.id, status: "executed", stateMutation };
}

async function runDeltaNetParameters(device, kernel) {
  const betaInput = Float32Array.from(
    { length: 32 },
    (_, head) => (head - 16) / 5,
  );
  const a = Float32Array.from(
    { length: 32 },
    (_, head) => (head % 7) / 4,
  );
  const dt = Float32Array.from(
    { length: 32 },
    (_, head) => -(head % 5) / 6,
  );
  const ssmA = Float32Array.from(
    { length: 32 },
    (_, head) => -0.25 - head / 64,
  );
  const expectedBeta = new Float32Array(32);
  const expectedDecay = new Float32Array(32);
  for (let head = 0; head < 32; head += 1) {
    const result = qwen35DeltaNetParametersCpu(
      betaInput[head],
      a[head],
      dt[head],
      ssmA[head],
    );
    expectedBeta[head] = result.beta;
    expectedDecay[head] = result.decay;
  }
  const betaOutput = outputFixture(expectedBeta);
  const decayOutput = outputFixture(expectedDecay);
  const inputBuffers = [betaInput, a, dt, ssmA].map((values) =>
    storageBuffer(device, floatBytes(values), GPUBufferUsage.STORAGE),
  );
  const betaBuffer = storageBuffer(
    device,
    floatBytes(betaOutput.initial),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const decayBuffer = storageBuffer(
    device,
    floatBytes(decayOutput.initial),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const [actualBetaBytes, actualDecayBytes] = await dispatchAndRead(
    device,
    kernel,
    [
      ...inputBuffers.map((buffer, binding) => ({
        binding,
        resource: { buffer },
      })),
      {
        binding: 4,
        resource: {
          buffer: betaBuffer,
          offset: betaOutput.outputRowOffset * 4,
          size: (expectedBeta.length + 1) * 4,
        },
      },
      {
        binding: 5,
        resource: {
          buffer: decayBuffer,
          offset: decayOutput.outputRowOffset * 4,
          size: (expectedDecay.length + 1) * 4,
        },
      },
    ],
    [
      { buffer: betaBuffer, byteLength: betaOutput.expected.byteLength },
      { buffer: decayBuffer, byteLength: decayOutput.expected.byteLength },
    ],
    { x: 1, y: 1, z: 1 },
  );
  validateParity(
    `${kernel.id}-beta`,
    betaOutput.expected,
    new Float32Array(actualBetaBytes),
    1e-3,
  );
  validateParity(
    `${kernel.id}-decay`,
    decayOutput.expected,
    new Float32Array(actualDecayBytes),
    1e-3,
  );
  return { id: kernel.id, status: "executed" };
}

function normalizeDeltaHead(values, offset, query) {
  let sum = Math.fround(0);
  for (let lane = 0; lane < 128; lane += 1) {
    const value = values[offset + lane];
    sum = Math.fround(sum + Math.fround(value * value));
  }
  let scale = Math.fround(
    1 / Math.sqrt(Math.fround(sum + Math.fround(1e-6))),
  );
  if (query) {
    scale = Math.fround(scale * Math.fround(1 / Math.sqrt(128)));
  }
  return Float32Array.from(
    { length: 128 },
    (_, lane) => Math.fround(values[offset + lane] * scale),
  );
}

async function runDeltaNetRecurrent(device, kernel) {
  const qkv = new Float32Array(8192);
  for (let qkHead = 0; qkHead < 16; qkHead += 1) {
    for (let lane = 0; lane < 128; lane += 1) {
      qkv[qkHead * 128 + lane] =
        ((qkHead * 7 + lane * 3) % 11 - 5) / 8;
      qkv[2048 + qkHead * 128 + lane] =
        ((qkHead * 5 + lane * 2) % 13 - 6) / 8;
    }
  }
  for (let valueHead = 0; valueHead < 32; valueHead += 1) {
    for (let lane = 0; lane < 128; lane += 1) {
      qkv[4096 + valueHead * 128 + lane] =
        ((valueHead * 11 + lane * 5) % 17 - 8) / 4;
    }
  }
  const beta = Float32Array.from(
    { length: 32 },
    (_, head) => 0.2 + (head % 5) / 10,
  );
  const decay = Float32Array.from(
    { length: 32 },
    (_, head) => 0.7 + (head % 3) / 20,
  );
  const initialState = new Float32Array(32 * 128 * 128);
  for (let head = 0; head < 32; head += 1) {
    const base = head * 128 * 128;
    initialState[base + (head % 128) * 128 + ((head * 3) % 128)] =
      (head + 1) / 64;
    initialState[base + ((head + 17) % 128) * 128 + ((head * 7) % 128)] =
      -(head + 1) / 96;
  }
  const expectedState = new Float32Array(initialState);
  const expectedValues = new Float32Array(4096);
  for (let valueHead = 0; valueHead < 32; valueHead += 1) {
    const qkHead = valueHead % 16;
    const query = normalizeDeltaHead(qkv, qkHead * 128, true);
    const key = normalizeDeltaHead(qkv, 2048 + qkHead * 128, false);
    const value = qkv.subarray(
      4096 + valueHead * 128,
      4096 + (valueHead + 1) * 128,
    );
    const state = expectedState.subarray(
      valueHead * 128 * 128,
      (valueHead + 1) * 128 * 128,
    );
    expectedValues.set(
      deltaNetRecurrentHeadStepCpu(
        state,
        query,
        key,
        value,
        beta[valueHead],
        decay[valueHead],
      ),
      valueHead * 128,
    );
  }
  const output = outputFixture(expectedValues);
  const qkvBuffer = storageBuffer(device, floatBytes(qkv), GPUBufferUsage.STORAGE);
  const betaBuffer = storageBuffer(
    device,
    floatBytes(beta),
    GPUBufferUsage.STORAGE,
  );
  const decayBuffer = storageBuffer(
    device,
    floatBytes(decay),
    GPUBufferUsage.STORAGE,
  );
  const stateBuffer = storageBuffer(
    device,
    floatBytes(initialState),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const outputBuffer = storageBuffer(
    device,
    floatBytes(output.initial),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const [actualOutputBytes, actualStateBytes] = await dispatchAndRead(
    device,
    kernel,
    [
      { binding: 0, resource: { buffer: qkvBuffer } },
      { binding: 1, resource: { buffer: betaBuffer } },
      { binding: 2, resource: { buffer: decayBuffer } },
      { binding: 3, resource: { buffer: stateBuffer } },
      {
        binding: 4,
        resource: {
          buffer: outputBuffer,
          offset: output.outputRowOffset * 4,
          size: (expectedValues.length + 1) * 4,
        },
      },
    ],
    [
      { buffer: outputBuffer, byteLength: output.expected.byteLength },
      { buffer: stateBuffer, byteLength: expectedState.byteLength },
    ],
    { x: 32, y: 1, z: 1 },
  );
  const actualState = new Float32Array(actualStateBytes);
  validateParity(
    kernel.id,
    output.expected,
    new Float32Array(actualOutputBytes),
    1e-3,
  );
  validateParity(`${kernel.id}-state`, expectedState, actualState, 1e-3);
  const stateMutation = actualState[0] !== initialState[0];
  if (!stateMutation) throw new Error(`${kernel.id}: state did not mutate`);
  return { id: kernel.id, status: "executed", stateMutation };
}

async function runDeltaNetGatedNorm(device, kernel) {
  const recurrent = Float32Array.from(
    { length: 4096 },
    (_, index) => ((index * 7) % 19 - 9) / 6,
  );
  const z = Float32Array.from(
    { length: 4096 },
    (_, index) => ((index * 5) % 13 - 6) / 5,
  );
  const normWeight = Float32Array.from(
    { length: 128 },
    (_, lane) => 0.75 + (lane % 7) / 10,
  );
  const expectedValues = new Float32Array(4096);
  for (let head = 0; head < 32; head += 1) {
    const base = head * 128;
    let sum = Math.fround(0);
    for (let lane = 0; lane < 128; lane += 1) {
      const value = recurrent[base + lane];
      sum = Math.fround(sum + Math.fround(value * value));
    }
    const inverseRms = Math.fround(
      1 /
        Math.sqrt(
          Math.fround(
            Math.fround(sum / 128) + Math.fround(1e-6),
          ),
        ),
    );
    for (let lane = 0; lane < 128; lane += 1) {
      const index = base + lane;
      expectedValues[index] = Math.fround(
        Math.fround(
          Math.fround(recurrent[index] * inverseRms) * normWeight[lane],
        ) * siluCpu(z[index]),
      );
    }
  }
  const output = outputFixture(expectedValues);
  const recurrentBuffer = storageBuffer(
    device,
    floatBytes(recurrent),
    GPUBufferUsage.STORAGE,
  );
  const zBuffer = storageBuffer(device, floatBytes(z), GPUBufferUsage.STORAGE);
  const weightBuffer = storageBuffer(
    device,
    floatBytes(normWeight),
    GPUBufferUsage.STORAGE,
  );
  const outputBuffer = storageBuffer(
    device,
    floatBytes(output.initial),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const [actualBytes] = await dispatchAndRead(
    device,
    kernel,
    [
      { binding: 0, resource: { buffer: recurrentBuffer } },
      { binding: 1, resource: { buffer: zBuffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
      {
        binding: 3,
        resource: {
          buffer: outputBuffer,
          offset: output.outputRowOffset * 4,
          size: (expectedValues.length + 1) * 4,
        },
      },
    ],
    [{ buffer: outputBuffer, byteLength: output.expected.byteLength }],
    { x: 32, y: 1, z: 1 },
  );
  validateParity(kernel.id, output.expected, new Float32Array(actualBytes), 1e-3);
  return { id: kernel.id, status: "executed" };
}

async function runHybridKernel(device, kernel) {
  switch (kernel.key.operation) {
    case "full-attention-prepare":
      return runFullAttentionPrepare(device, kernel);
    case "full-attention-online":
      return runFullAttentionOnline(device, kernel);
    case "deltanet-conv":
      return runDeltaNetConv(device, kernel);
    case "deltanet-parameters":
      return runDeltaNetParameters(device, kernel);
    case "deltanet-recurrent":
      return runDeltaNetRecurrent(device, kernel);
    case "deltanet-gated-norm":
      return runDeltaNetGatedNorm(device, kernel);
    default:
      throw new Error("No hybrid kernel fixture");
  }
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
  for (const kernel of QWEN35_HYBRID_KERNELS) {
    results.push(await runHybridKernel(device, kernel));
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
