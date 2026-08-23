import {
  LANGUAGE_GEMV_KERNELS,
  gemvCpu,
  planGemvDispatch,
} from "../dist/src/mixed-gemv.js?v=41";
import {
  repackNativeQ4K,
  repackNativeQ5K,
  repackNativeQ6K,
  repackNativeQ8_0,
} from "../dist/src/mixed-quant.js?v=41";
import { repackNativeQ3K } from "../dist/src/q3k.js?v=41";
import {
  repackNativeQ3KBrowser,
  repackNativeQ3KFusedBrowser,
  repackNativeQ4KBrowser,
  repackNativeQ5KBrowser,
  repackNativeQ6KBrowser,
} from "../dist/src/browser-quant.js?v=41";
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
} from "../dist/src/qwen-primitives.js?v=41";
import {
  PACKED_EMBEDDING_KERNELS,
  embeddingCpu,
  planPackedEmbeddingRow,
} from "../dist/src/qwen-embedding.js?v=41";
import { QWEN35_HYBRID_KERNELS } from "../dist/src/hybrid-kernels.js?v=41";
import {
  assembleQwen35StagedFinalTokenCommand,
  assembleQwen35StagedLogitsTileGpuCommands,
} from "../dist/src/qwen35-logits-dispatch.js?v=41";
import {
  QWEN35_VISION_FOUNDATION_KERNELS,
  visionAddLearnedPositionCpu,
  visionApply2dRopeCpu,
  visionPatchConv3dCpu,
  visionPrepare2dRopeCpu,
} from "../dist/src/qwen35-vision-foundation-kernels.js?v=41";
import {
  QWEN35_VISION_LAYER_KERNELS,
  visionLayerNormCpu,
  visionLinearBf16Cpu,
  visionOnlineAttentionCpu,
  visionTanhGeluCpu,
} from "../dist/src/qwen35-vision-layer-kernels.js?v=41";
import {
  QWEN35_VISION_MERGER_KERNELS,
  visionExactGeluCpu,
} from "../dist/src/qwen35-vision-merger-kernels.js?v=41";
import {
  packFloat16PairCpu,
  qwen35OnlineAttentionHeadCpu,
  splitQwen35QueryGateProjection,
  unpackFloat16PairCpu,
} from "../dist/src/full-attention.js?v=41";
import {
  deltaNetRecurrentHeadStepCpu,
  qwen35DeltaNetParametersCpu,
} from "../dist/src/gated-deltanet.js?v=41";
import { validateParity } from "./webgpu-parity.mjs?v=39";

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

function activationBytesForKernel(kernel, activation) {
  if (
    kernel.layout === "q5-k-fused-f32-224" &&
    kernel.profile === "mobile-f16-subgroup"
  ) {
    const packed = new Uint32Array(activation.length / 2);
    for (let block = 0; block < activation.length / 256; block += 1) {
      const blockBase = block * 256;
      for (let half = 0; half < 2; half += 1) {
        for (let lane = 0; lane < 32; lane += 1) {
          const destination = block * 128 + half * 64 + lane;
          const source = blockBase + half * 128 + lane;
          packed[destination] = packFloat16PairCpu(
            activation[source], activation[source + 32],
          );
          packed[destination + 32] = packFloat16PairCpu(
            activation[source + 64], activation[source + 96],
          );
        }
      }
    }
    return new Uint8Array(packed.buffer);
  }
  if (kernel.profile === "mobile-f16-subgroup" && [
    "q3-k-112", "q3-k-nibble-148", "q4-k-144", "q5-k-176", "q6-k-212",
  ].includes(kernel.layout)) {
    const packed = new Uint32Array(activation.length / 2);
    for (let word = 0; word < packed.length; word += 1) {
      packed[word] = packFloat16PairCpu(
        activation[word * 2],
        activation[word * 2 + 1],
      );
    }
    return new Uint8Array(packed.buffer);
  }

  return new Uint8Array(activation.buffer);
}

function cpuActivationForKernel(kernel, activation) {
  const usesPackedHalf = kernel.profile === "mobile-f16-subgroup" && [
    "q8-0-36",
    "q3-k-112",
    "q3-k-nibble-148",
    "q4-k-144",
    "q5-k-176",
    "q6-k-212",
    "q3-k-fused-f32-192",
    "q5-k-fused-f32-224",
  ].includes(kernel.layout);
  if (!usesPackedHalf) return activation;
  return Float32Array.from(activation, (value) => {
    const packed = packFloat16PairCpu(value, 0);
    return unpackFloat16PairCpu(packed)[0];
  });
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
    "q3-k-nibble-148": 110,
    "q3-k-fused-f32-192": 110,
    "q4-k-144": 144,
    "q4-k-fused-f32-192": 144,
    "q5-k-176": 176,
    "q5-k-fused-f32-224": 176,
    "q6-k-212": 210,
    "q6-k-fused-f32-256": 210,
  }[layout];
  if (nativeBytes === undefined) {
    throw new Error(`Unsupported packed fixture layout: ${String(layout)}`);
  }
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
  if (layout === "q3-k-nibble-148") {
    setHalf(native, 108);
    return repackNativeQ3KBrowser(native);
  }
  if (layout === "q3-k-fused-f32-192") {
    setHalf(native, 108);
    return repackNativeQ3KFusedBrowser(native);
  }
  if (layout === "q4-k-144" || layout === "q4-k-fused-f32-192") {
    setHalf(native, 0);
    setHalf(native, 2, 0x3800);
    return layout === "q4-k-144"
      ? repackNativeQ4K(native)
      : repackNativeQ4KBrowser(native);
  }
  if (layout === "q5-k-176" || layout === "q5-k-fused-f32-224") {
    setHalf(native, 0);
    setHalf(native, 2, 0x3800);
    return layout === "q5-k-176"
      ? repackNativeQ5K(native)
      : repackNativeQ5KBrowser(native);
  }
  setHalf(native, 208);
  return layout === "q6-k-212"
    ? repackNativeQ6K(native)
    : repackNativeQ6KBrowser(native);
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

  // Force a 2D dispatch for every current kernel family. This also covers
  // fused kernels whose workgroups own multiple output rows.
  const rows = kernel.abi.rowsPerWorkgroup * 2 + 1;
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
  const expected = gemvCpu(
    kernel.layout,
    packed,
    cpuActivationForKernel(kernel, activation),
    {
    rows,
    columns,
    packedByteOffset,
    },
  );
  if (new Set(expected).size !== rows) {
    throw new Error(`${kernel.id}: row fixtures did not produce distinct output`);
  }
  const outputRowOffset = 2;
  const genericPlan = planGemvDispatch({
    layout: kernel.layout,
    profile: kernel.profile,
    localRows: rows,
    columns,
    packedByteOffset,
    outputRowOffset,
    maxWorkgroupsPerDimension: 2,
  });
  // A layout can have both a portable reference kernel and a specialized
  // mobile kernel. The runtime planner selects its production specialization
  // by layout, but this harness must dispatch the exact kernel under test.
  const kernelDispatchRows = Math.ceil(rows / kernel.abi.rowsPerWorkgroup);
  const kernelWorkgroupX = Math.min(kernelDispatchRows, 2);
  const plan = {
    ...genericPlan,
    workgroups: {
      x: kernelWorkgroupX,
      y: Math.ceil(kernelDispatchRows / kernelWorkgroupX),
      z: 1,
    },
  };
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
    activationBytesForKernel(kernel, activation),
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
  validateParity(
    kernel.id,
    expectedOutput,
    actual,
    // These optimized kernels intentionally multiply in FP16. Keep their
    // synthetic stress tolerance separate from the exact portable oracle.
    kernel.profile === "mobile-f16-subgroup" ? 2e-2 : 2e-4,
  );
  return {
    id: kernel.id,
    blocksPerRow,
    outputRowOffset,
    workgroups: plan.workgroups,
    expected: Array.from(expectedOutput),
    actual: Array.from(actual),
  };
}

async function benchmarkProductionGemv(
  device,
  { layout, rows, columns, iterations = 30 },
) {
  const kernel = LANGUAGE_GEMV_KERNELS.find((candidate) =>
    candidate.layout === layout &&
    candidate.profile === "mobile-f16-subgroup"
  );
  if (!kernel) throw new Error(`production ${layout} benchmark kernel is unavailable`);
  const blocksPerRow = columns / kernel.abi.valuesPerBlock;
  const weightBytes = rows * blocksPerRow * kernel.abi.bytesPerBlock;
  const plan = planGemvDispatch({
    layout: kernel.layout,
    profile: kernel.profile,
    localRows: rows,
    columns,
  });
  const module = device.createShaderModule({ label: `${kernel.id}-benchmark`, code: kernel.source });
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "packed_gemv" },
  });
  const weights = device.createBuffer({ size: weightBytes, usage: GPUBufferUsage.STORAGE });
  const activationValues = Float32Array.from(
    { length: columns }, (_, index) => ((index % 31) - 15) / 32,
  );
  const activation = storageBuffer(
    device,
    activationBytesForKernel(kernel, activationValues),
    GPUBufferUsage.STORAGE,
  );
  const output = device.createBuffer({
    size: rows * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const uniforms = storageBuffer(
    device,
    new Uint8Array(Uint32Array.of(rows, columns, blocksPerRow, 0, 0).buffer),
    GPUBufferUsage.UNIFORM,
  );
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: weights } },
      { binding: 1, resource: { buffer: activation } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: uniforms } },
    ],
  });
  const submit = async (iterations) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      pass.dispatchWorkgroups(plan.workgroups.x, plan.workgroups.y, 1);
    }
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, 4);
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    new Uint32Array(readback.getMappedRange())[0];
    readback.unmap();
    return performance.now() - started;
  };
  await submit(Math.min(3, iterations));
  const durationMs = await submit(iterations);
  for (const buffer of [weights, activation, output, uniforms, readback]) {
    buffer.destroy();
  }
  return {
    id: `qwen35-production-${layout}${
      rows > 100_000 ? "-vocabulary" : ""
    }-gemv-benchmark`,
    status: "executed",
    rows,
    columns,
    weightBytes,
    iterations,
    durationMs,
    millisecondsPerDispatch: durationMs / iterations,
    effectiveWeightGigabytesPerSecond:
      (weightBytes * iterations) / durationMs / 1_000_000,
  };
}

async function benchmarkProductionGemvs(device) {
  const specifications = [
    { layout: "q3-k-fused-f32-192", rows: 9_216, columns: 2_560 },
    { layout: "q4-k-fused-f32-192", rows: 2_560, columns: 4_096 },
    { layout: "q5-k-fused-f32-224", rows: 2_560, columns: 9_216 },
    { layout: "q6-k-fused-f32-256", rows: 8_192, columns: 2_560 },
    {
      layout: "q6-k-fused-f32-256",
      rows: 248_320,
      columns: 2_560,
      iterations: 3,
    },
    { layout: "q8-0-36", rows: 2_560, columns: 4_096 },
  ];
  const results = [];
  for (const specification of specifications) {
    results.push(await benchmarkProductionGemv(device, specification));
  }
  return results;
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
    compute: { module, entryPoint: kernel.entryPoint },
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
      size: output.expected.byteLength - output.outputRowOffset * 4,
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

async function runResidualRmsPrimitive(device, kernel) {
  const input = Float32Array.of(3, -4, 0.5, -2);
  const residual = Float32Array.of(-1, 2, 3, -4);
  const weight = Float32Array.of(1, 0.75, 1.25, 0.5);
  const combined = residualAddCpu(input, residual);
  const normalized = rmsNormCpu(combined, weight, Math.fround(1e-6));
  const residualOutput = outputFixture(combined);
  const normalizedOutput = outputFixture(normalized);
  const residualBuffer = storageBuffer(device, floatBytes(residual), GPUBufferUsage.STORAGE);
  const weightBuffer = storageBuffer(device, floatBytes(weight), GPUBufferUsage.STORAGE);
  const residualOutputBuffer = storageBuffer(
    device,
    floatBytes(residualOutput.initial),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const normalizedOutputBuffer = storageBuffer(
    device,
    floatBytes(new Float32Array(normalizedOutput.initial).map((value, index) =>
      index >= normalizedOutput.outputRowOffset &&
        index < normalizedOutput.outputRowOffset + input.length
        ? input[index - normalizedOutput.outputRowOffset]
        : value
    )),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const uniformBuffer = storageBuffer(
    device,
    paramsBytes(16, [
      ["setUint32", 0, input.length],
      ["setUint32", 4, input.length],
      ["setFloat32", 8, Math.fround(1e-6)],
    ]),
    GPUBufferUsage.UNIFORM,
  );
  const outputOffset = residualOutput.outputRowOffset * 4;
  const outputSize = residualOutput.expected.byteLength - outputOffset;
  const [residualBytes, normalizedBytes] = await dispatchAndRead(
    device,
    kernel,
    [
      {
        binding: 0,
        resource: {
          buffer: normalizedOutputBuffer,
          offset: outputOffset,
          size: outputSize,
        },
      },
      { binding: 1, resource: { buffer: residualBuffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
      {
        binding: 3,
        resource: { buffer: residualOutputBuffer, offset: outputOffset, size: outputSize },
      },
      { binding: 4, resource: { buffer: uniformBuffer } },
    ],
    [
      { buffer: residualOutputBuffer, byteLength: residualOutput.expected.byteLength },
      { buffer: normalizedOutputBuffer, byteLength: normalizedOutput.expected.byteLength },
    ],
    { x: 1, y: 1, z: 1 },
  );
  validateParity(
    `${kernel.id}-residual`,
    residualOutput.expected,
    new Float32Array(residualBytes),
  );
  validateParity(
    `${kernel.id}-normalized`,
    normalizedOutput.expected,
    new Float32Array(normalizedBytes),
  );
  return {
    id: kernel.id,
    status: "executed",
    outputRowOffset: residualOutput.outputRowOffset,
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
          size: output.expected.byteLength - output.outputRowOffset * 4,
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
  if (kernel.operation === "residual-rms-norm") {
    return runResidualRmsPrimitive(device, kernel);
  }
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
          size: output.expected.byteLength - output.outputRowOffset * 4,
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
  value[0] = 58_832;
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
        keys[base + lane] = ((token * 7 + kvHead * 5 + lane) % 9 - 4) / 7;
        values[base + lane] =
          ((token * 11 + kvHead * 13 + lane * 3) % 15 - 7) / 11;
      }
    }
  }
  const packedKeys = new Uint32Array(keys.length / 2);
  const packedValues = new Uint32Array(values.length / 2);
  const quantizedKeys = new Float32Array(keys.length);
  const quantizedValues = new Float32Array(values.length);
  for (let scalar = 0; scalar < keys.length; scalar += 2) {
    packedKeys[scalar / 2] = packFloat16PairCpu(
      keys[scalar],
      keys[scalar + 1],
    );
    packedValues[scalar / 2] = packFloat16PairCpu(
      values[scalar],
      values[scalar + 1],
    );
    quantizedKeys.set(
      unpackFloat16PairCpu(packedKeys[scalar / 2]),
      scalar,
    );
    quantizedValues.set(
      unpackFloat16PairCpu(packedValues[scalar / 2]),
      scalar,
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
      headKeys.set(
        quantizedKeys.subarray(source, source + 256),
        token * 256,
      );
      headValues.set(
        quantizedValues.subarray(source, source + 256),
        token * 256,
      );
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
  const onlineStateBuffer = storageBuffer(
    device,
    floatBytes(new Float32Array(16 * 2)),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const uniforms = storageBuffer(
    device,
    uintBytes(Uint32Array.of(tokenCount, 0, 1, 0)),
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
          size: output.expected.byteLength - output.outputRowOffset * 4,
        },
      },
      { binding: 4, resource: { buffer: onlineStateBuffer } },
      { binding: 5, resource: { buffer: uniforms } },
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
          size: output.expected.byteLength - output.outputRowOffset * 4,
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
          size: betaOutput.expected.byteLength - betaOutput.outputRowOffset * 4,
        },
      },
      {
        binding: 5,
        resource: {
          buffer: decayBuffer,
          offset: decayOutput.outputRowOffset * 4,
          size: decayOutput.expected.byteLength - decayOutput.outputRowOffset * 4,
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

async function runDeltaNetRecurrent(device, kernel, fusedGatedNorm = false) {
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
  const z = Float32Array.from(
    { length: 4096 },
    (_, index) => ((index * 5) % 13 - 6) / 5,
  );
  const normWeight = Float32Array.from(
    { length: 128 },
    (_, lane) => 0.75 + (lane % 7) / 10,
  );
  const outputValues = fusedGatedNorm
    ? new Float32Array(expectedValues.length)
    : expectedValues;
  if (fusedGatedNorm) {
    for (let head = 0; head < 32; head += 1) {
      const base = head * 128;
      let sum = Math.fround(0);
      for (let lane = 0; lane < 128; lane += 1) {
        const value = expectedValues[base + lane];
        sum = Math.fround(sum + Math.fround(value * value));
      }
      const inverseRms = Math.fround(
        1 / Math.sqrt(Math.fround(Math.fround(sum / 128) + Math.fround(1e-6))),
      );
      for (let lane = 0; lane < 128; lane += 1) {
        const index = base + lane;
        outputValues[index] = Math.fround(
          Math.fround(
            Math.fround(expectedValues[index] * inverseRms) * normWeight[lane],
          ) * siluCpu(z[index]),
        );
      }
    }
  }
  const output = outputFixture(outputValues);
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
  const zBuffer = storageBuffer(device, floatBytes(z), GPUBufferUsage.STORAGE);
  const normWeightBuffer = storageBuffer(
    device,
    floatBytes(normWeight),
    GPUBufferUsage.STORAGE,
  );
  const bindings = fusedGatedNorm
    ? [
        { binding: 0, resource: { buffer: qkvBuffer } },
        { binding: 1, resource: { buffer: betaBuffer } },
        { binding: 2, resource: { buffer: decayBuffer } },
        { binding: 3, resource: { buffer: stateBuffer } },
        { binding: 4, resource: { buffer: zBuffer } },
        { binding: 5, resource: { buffer: normWeightBuffer } },
        {
          binding: 6,
          resource: {
            buffer: outputBuffer,
            offset: output.outputRowOffset * 4,
            size: output.expected.byteLength - output.outputRowOffset * 4,
          },
        },
      ]
    : [
        { binding: 0, resource: { buffer: qkvBuffer } },
        { binding: 1, resource: { buffer: betaBuffer } },
        { binding: 2, resource: { buffer: decayBuffer } },
        { binding: 3, resource: { buffer: stateBuffer } },
        {
          binding: 4,
          resource: {
            buffer: outputBuffer,
            offset: output.outputRowOffset * 4,
            size: output.expected.byteLength - output.outputRowOffset * 4,
          },
        },
      ];
  const [actualOutputBytes, actualStateBytes] = await dispatchAndRead(
    device,
    kernel,
    bindings,
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
          size: output.expected.byteLength - output.outputRowOffset * 4,
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
    case "deltanet-recurrent-gated-norm":
      return runDeltaNetRecurrent(device, kernel, true);
    default:
      throw new Error("No hybrid kernel fixture");
  }
}

async function runVisionFoundationKernel(device, kernel) {
  const uniform = (words) => storageBuffer(device, uintBytes(Uint32Array.from(words)), GPUBufferUsage.UNIFORM);
  if (kernel.key.operation === "vision-patch-conv3d") {
    // This is deliberately sparse. Every channel and temporal slice feeds
    // both checked output rows, so parity detects a C/T/H/W packing error.
    const patches = new Float32Array(1536);
    const weights0 = new Float32Array(16 * 16 * 3 * 1024);
    const weights1 = new Float32Array(weights0.length);
    const bias = Float32Array.from({ length: 1024 }, (_, i) => Math.fround(i / 1024));
    const convWeightIndex = (channel, row, column, hidden) =>
      column + 16 * (row + 16 * (channel + 3 * hidden));
    for (let channel = 0; channel < 3; channel += 1) {
      const row0 = channel * 3;
      const column0 = channel * 5;
      const row1 = 15 - channel * 2;
      const column1 = 14 - channel * 3;
      patches[channel * 512 + row0 * 16 + column0] = Math.fround(channel + 1.25);
      patches[channel * 512 + 256 + row1 * 16 + column1] = Math.fround(channel + 2.5);
      for (const hidden of [0, 1]) {
        weights0[convWeightIndex(channel, row0, column0, hidden)] = Math.fround((hidden + 1) * (channel + 0.5));
        weights1[convWeightIndex(channel, row1, column1, hidden)] = Math.fround((hidden + 1.5) * (channel + 0.75));
      }
    }
    const expected = visionPatchConv3dCpu({ patches, patchCount: 1, hiddenSize: 1024, patchSize: 16, temporalPatchSize: 2, weightsTemporalZero: weights0, weightsTemporalOne: weights1, bias });
    const output = storageBuffer(device, floatBytes(new Float32Array(1024)), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const [actual] = await dispatchAndRead(device, kernel, [
      { binding: 0, resource: { buffer: storageBuffer(device, floatBytes(patches), GPUBufferUsage.STORAGE) } },
      { binding: 1, resource: { buffer: storageBuffer(device, floatBytes(weights0), GPUBufferUsage.STORAGE) } }, { binding: 2, resource: { buffer: storageBuffer(device, floatBytes(weights1), GPUBufferUsage.STORAGE) } },
      { binding: 3, resource: { buffer: storageBuffer(device, floatBytes(bias), GPUBufferUsage.STORAGE) } }, { binding: 4, resource: { buffer: output } }, { binding: 5, resource: { buffer: uniform([1, 0, 0, 0]) } },
    ], [{ buffer: output, byteLength: expected.byteLength }], { x: 16, y: 1, z: 1 });
    validateParity(kernel.id, expected, new Float32Array(actual));
  } else if (kernel.key.operation === "vision-add-learned-position") {
    // A 4x4 grid reaches fractional align-corners coordinates. This split
    // falls within the taps of the row-31, column-15.66 sample.
    const embeddings = Float32Array.from({ length: 16 * 1024 }, (_, i) => Math.fround((i % 17) / 17));
    const table = Float32Array.from(
      { length: 2304 * 1024 },
      (_, i) => Math.fround((((i * 17) % 101) - 50) / 13),
    );
    const split = (31 * 48 + 16) * 1024;
    const expected = visionAddLearnedPositionCpu({ embeddings, gridHeight: 4, gridWidth: 4, hiddenSize: 1024, tableHeight: 48, tableWidth: 48, table });
    const output = storageBuffer(device, floatBytes(embeddings), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const [actual] = await dispatchAndRead(device, kernel, [
      { binding: 0, resource: { buffer: output } }, { binding: 1, resource: { buffer: storageBuffer(device, floatBytes(table.subarray(0, split)), GPUBufferUsage.STORAGE) } }, { binding: 2, resource: { buffer: storageBuffer(device, floatBytes(table.subarray(split)), GPUBufferUsage.STORAGE) } }, { binding: 3, resource: { buffer: uniform([16, 4, 4, split]) } },
    ], [{ buffer: output, byteLength: expected.byteLength }], { x: 16, y: 16, z: 1 }); validateParity(kernel.id, expected, new Float32Array(actual));
  } else if (kernel.key.operation === "vision-prepare-2d-rope") {
    // The 16,000-patch grid exercises long-axis merge coordinates and the
    // shader's pow/cos path while keeping the readback at about four MiB.
    const expected = visionPrepare2dRopeCpu({ gridHeight: 10, gridWidth: 1_600, headDimension: 64 }); const output = storageBuffer(device, floatBytes(new Float32Array(expected.values.length)), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const [actual] = await dispatchAndRead(device, kernel, [{ binding: 0, resource: { buffer: output } }, { binding: 1, resource: { buffer: uniform([16_000, 10, 1_600, 0]) } }], [{ buffer: output, byteLength: expected.values.byteLength }], { x: 1, y: 16_000, z: 1 }); validateParity(kernel.id, expected.values, new Float32Array(actual));
  } else {
    const prepared = visionPrepare2dRopeCpu({ gridHeight: 10, gridWidth: 1_600, headDimension: 64 }); const query = Float32Array.from({ length: 1024 }, (_, i) => Math.fround(i / 97)); const key = Float32Array.from(query, (v) => Math.fround(v * 0.5));
    // The final patch is coordinate (9, 1599), so this checks long-axis 2D rotation.
    const nonIdentityRope = prepared.values.subarray((16_000 - 1) * 64, 16_000 * 64);
    const expected = visionApply2dRopeCpu({ query, key, rope: nonIdentityRope, patchCount: 1, headCount: 16, headDimension: 64 });
    const q = storageBuffer(device, floatBytes(query), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC); const k = storageBuffer(device, floatBytes(key), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const results = await dispatchAndRead(device, kernel, [{ binding: 0, resource: { buffer: q } }, { binding: 1, resource: { buffer: k } }, { binding: 2, resource: { buffer: storageBuffer(device, floatBytes(nonIdentityRope), GPUBufferUsage.STORAGE) } }, { binding: 3, resource: { buffer: uniform([1, 0, 0, 0]) } }], [{ buffer: q, byteLength: 4096 }, { buffer: k, byteLength: 4096 }], { x: 1, y: 1, z: 16 }); validateParity(`${kernel.id}-q`, expected.query, new Float32Array(results[0])); validateParity(`${kernel.id}-k`, expected.key, new Float32Array(results[1]));
  }
  return { id: kernel.id, status: "executed" };
}

function bf16Fixture(length) {
  return Uint16Array.from({ length }, (_, index) => 0x3f00 + (index % 5) * 0x0080);
}

function f32Bits(value) {
  return new Uint32Array(new Float32Array([value]).buffer)[0];
}

async function runVisionLayerKernel(device, kernel) {
  const uniform = (words) => storageBuffer(device, uintBytes(Uint32Array.from(words)), GPUBufferUsage.UNIFORM);
  if (kernel.key.operation === "vision-layernorm") {
    const input = Float32Array.from({ length: 1024 }, (_, i) => Math.fround(1_000 + Math.sin(i * 0.17) * 1.5));
    const weight = Float32Array.from({ length: 1024 }, (_, i) => Math.fround(0.5 + (i % 7) / 10)); const bias = Float32Array.from({ length: 1024 }, (_, i) => Math.fround((i % 5) / 20));
    const expected = visionLayerNormCpu({ input, tokenCount: 1, hiddenSize: 1024, weight, bias }); const output = storageBuffer(device, floatBytes(new Float32Array(1024)), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const [actual] = await dispatchAndRead(device, kernel, [{ binding: 0, resource: { buffer: storageBuffer(device, floatBytes(input), GPUBufferUsage.STORAGE) } }, { binding: 1, resource: { buffer: storageBuffer(device, floatBytes(weight), GPUBufferUsage.STORAGE) } }, { binding: 2, resource: { buffer: storageBuffer(device, floatBytes(bias), GPUBufferUsage.STORAGE) } }, { binding: 3, resource: { buffer: output } }, { binding: 4, resource: { buffer: uniform([1, 1024, f32Bits(1e-6), 0]) } }], [{ buffer: output, byteLength: expected.byteLength }], { x: 1, y: 1, z: 1 }); validateParity(kernel.id, expected, new Float32Array(actual), 1e-4);
  } else if (kernel.key.operation === "vision-bf16-linear" || kernel.key.operation === "vision-qkv-bf16-linear") {
    const qkv = kernel.key.operation === "vision-qkv-bf16-linear"; const outputWidth = qkv ? 3072 : 1024; const tokenCount = qkv ? 2 : 1; const input = Float32Array.from({ length: tokenCount * 1024 }, (_, i) => Math.fround(((i % 17) - 8) / 17)); const weight = bf16Fixture(1024 * outputWidth); const bias = Float32Array.from({ length: outputWidth }, (_, i) => Math.fround((i % 11) / 11));
    const tokenMajor = visionLinearBf16Cpu({ input, tokenCount, inputWidth: 1024, outputWidth, weight, bias }); const expected = new Float32Array(tokenMajor.length);
    if (qkv) { for (let branch = 0; branch < 3; branch += 1) for (let token = 0; token < tokenCount; token += 1) expected.set(tokenMajor.subarray(token * 3072 + branch * 1024, token * 3072 + (branch + 1) * 1024), (branch * tokenCount + token) * 1024); if (expected.every((value, index) => value === tokenMajor[index])) throw new Error(`${kernel.id}: planar QKV fixture did not differ from token-major QKV`); } else expected.set(tokenMajor);
    const output = storageBuffer(device, floatBytes(new Float32Array(expected.length)), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC); const params = qkv ? [tokenCount, 0, 0, 0] : [tokenCount, 1024, 1024, 0];
    const [actual] = await dispatchAndRead(device, kernel, [{ binding: 0, resource: { buffer: storageBuffer(device, floatBytes(input), GPUBufferUsage.STORAGE) } }, { binding: 1, resource: { buffer: storageBuffer(device, uintBytes(weight), GPUBufferUsage.STORAGE) } }, { binding: 2, resource: { buffer: storageBuffer(device, floatBytes(bias), GPUBufferUsage.STORAGE) } }, { binding: 3, resource: { buffer: output } }, { binding: 4, resource: { buffer: uniform(params) } }], [{ buffer: output, byteLength: expected.byteLength }], { x: outputWidth / 64, y: tokenCount, z: 1 }); validateParity(kernel.id, expected, new Float32Array(actual), 1e-3);
  } else if (kernel.key.operation === "vision-online-attention") {
    const query = Float32Array.from({ length: 2 * 1024 }, (_, i) => Math.fround(((i % 29) - 14) / 29)); const key = Float32Array.from({ length: query.length }, (_, i) => Math.fround(((i % 23) - 11) / 23)); const value = Float32Array.from({ length: query.length }, (_, i) => Math.fround(((i % 19) - 9) / 19)); const offsets = Uint32Array.of(0, 2);
    const expected = visionOnlineAttentionCpu({ query, key, value, tokenCount: 2, headCount: 16, headDimension: 64, segmentOffsets: offsets }); const output = storageBuffer(device, floatBytes(new Float32Array(expected.length)), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const [actual] = await dispatchAndRead(device, kernel, [{ binding: 0, resource: { buffer: storageBuffer(device, floatBytes(query), GPUBufferUsage.STORAGE) } }, { binding: 1, resource: { buffer: storageBuffer(device, floatBytes(key), GPUBufferUsage.STORAGE) } }, { binding: 2, resource: { buffer: storageBuffer(device, floatBytes(value), GPUBufferUsage.STORAGE) } }, { binding: 3, resource: { buffer: storageBuffer(device, uintBytes(offsets), GPUBufferUsage.STORAGE) } }, { binding: 4, resource: { buffer: output } }, { binding: 5, resource: { buffer: uniform([2, 1, 0, 0]) } }], [{ buffer: output, byteLength: expected.byteLength }], { x: 1, y: 2, z: 16 });
    validateParity(kernel.id, expected, new Float32Array(actual), 1e-4);
  } else if (kernel.key.operation === "vision-tanh-gelu") {
    const values = Float32Array.from({ length: 4096 }, (_, i) => Math.fround(((i % 31) - 15) / 5)); const expected = visionTanhGeluCpu(values); const output = storageBuffer(device, floatBytes(values), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const [actual] = await dispatchAndRead(device, kernel, [{ binding: 0, resource: { buffer: output } }, { binding: 1, resource: { buffer: uniform([4096, 0, 0, 0]) } }], [{ buffer: output, byteLength: expected.byteLength }], { x: 64, y: 1, z: 1 }); validateParity(kernel.id, expected, new Float32Array(actual), 1e-5);
  } else {
    const input = Float32Array.from({ length: 1024 }, (_, i) => Math.fround(i / 97)); const update = Float32Array.from({ length: 1024 }, (_, i) => Math.fround(-i / 193)); const expected = Float32Array.from(input, (value, i) => Math.fround(value + update[i])); const output = storageBuffer(device, floatBytes(input), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const [actual] = await dispatchAndRead(device, kernel, [{ binding: 0, resource: { buffer: output } }, { binding: 1, resource: { buffer: storageBuffer(device, floatBytes(update), GPUBufferUsage.STORAGE) } }, { binding: 2, resource: { buffer: uniform([1024, 0, 0, 0]) } }], [{ buffer: output, byteLength: expected.byteLength }], { x: 16, y: 1, z: 1 }); validateParity(kernel.id, expected, new Float32Array(actual));
  }
  return { id: kernel.id, status: "executed" };
}

async function runVisionMergerKernel(device, kernel) {
  const values = Float32Array.from({ length: 4096 }, (_, index) => Math.fround(((index % 31) - 15) / 5));
  const expected = visionExactGeluCpu(values);
  const output = storageBuffer(device, floatBytes(values), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const uniforms = storageBuffer(device, uintBytes(Uint32Array.of(1, 0, 0, 0)), GPUBufferUsage.UNIFORM);
  const [actual] = await dispatchAndRead(device, kernel, [{ binding: 0, resource: { buffer: output } }, { binding: 1, resource: { buffer: uniforms } }], [{ buffer: output, byteLength: expected.byteLength }], { x: 64, y: 1, z: 1 });
  validateParity(kernel.id, expected, new Float32Array(actual), 2e-6);
  return { id: kernel.id, status: "executed" };
}

function stagedWorkspace(logitsBuffer) {
  const byteLength = 1_024 * 4;
  const resource = Object.freeze({
    kind: "logits-tile",
    scalarType: "f32",
    elementCount: 1_024,
    bytes: BigInt(byteLength),
    usage: GPUBufferUsage.STORAGE,
    byteLength,
    binding: Object.freeze({ buffer: logitsBuffer, offset: 0, size: byteLength }),
  });
  return Object.freeze({
    get(kind) {
      if (kind !== "logits-tile") throw new Error(`Unexpected staged resource: ${kind}`);
      return resource;
    },
  });
}

function stagedBindingEntries(command) {
  return command.bindings.map(({ binding, buffer, offset, size }) => ({
    binding,
    resource: { buffer, offset, size },
  }));
}

async function runStagedLogitsGpuSelection(device) {
  const limits = {
    minStorageBufferOffsetAlignment: 256,
    minUniformBufferOffsetAlignment: 256,
    maxStorageBufferBindingSize: 1 << 30,
    maxUniformBufferBindingSize: 65_536,
    maxComputeWorkgroupsPerDimension: 65_535,
  };
  // Eight complete tiles cross the production four-tile staging cadence
  // without turning a deterministic parity check into a full 248K-vocabulary
  // benchmark. The production planner intentionally accepts a partial tile
  // only at the model's real vocabulary tail, so a synthetic early tail would
  // test an invalid state rather than the runtime contract.
  const tileCount = 8;
  const tileRows = 1_024;
  const rowBytes = 2_560;
  const blocksPerRow = 10;
  const packed = new Uint8Array(rowBytes * tileRows);
  for (let rowIndex = 0; rowIndex < tileRows; rowIndex += 1) {
    for (let blockIndex = 0; blockIndex < blocksPerRow; blockIndex += 1) {
      packed.set(
        packedFixture("q6-k-fused-f32-256", rowIndex, blockIndex),
        rowIndex * rowBytes + blockIndex * 256,
      );
    }
  }
  const normalizedHidden = Float32Array.from(
    { length: 2_560 },
    (_, index) => Math.fround(((index * 17 + 3) % 29 - 14) / 16),
  );
  const expectedScores = gemvCpu("q6-k-fused-f32-256", packed, normalizedHidden, {
    rows: tileRows,
    columns: 2_560,
    packedByteOffset: 0,
  });
  let expectedWinnerRow = 0;
  for (let row = 1; row < expectedScores.length; row += 1) {
    if (
      expectedScores[row] > expectedScores[expectedWinnerRow] ||
      (expectedScores[row] === expectedScores[expectedWinnerRow] && row < expectedWinnerRow)
    ) {
      expectedWinnerRow = row;
    }
  }
  const logitsBuffer = storageBuffer(
    device,
    new Uint8Array(tileRows * 4),
    GPUBufferUsage.STORAGE,
  );
  const normalizedHiddenBuffer = storageBuffer(
    device,
    floatBytes(normalizedHidden),
    GPUBufferUsage.STORAGE,
  );
  const packedTileBuffer = storageBuffer(
    device,
    packed,
    GPUBufferUsage.STORAGE,
  );
  // The production final kernel reduces the fixed 243 candidate slots. Pad
  // inactive fixture slots with negative infinity so the same reducer remains
  // valid while the harness runs only its bounded representative tile set.
  const candidateWords = new Uint32Array(256 * 2);
  for (let slot = 0; slot < 256; slot += 1) {
    candidateWords[slot * 2] = f32Bits(Number.NEGATIVE_INFINITY);
    candidateWords[slot * 2 + 1] = 0xffff_ffff;
  }
  const candidateBuffer = storageBuffer(
    device,
    uintBytes(candidateWords),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const finalUniformOffset = tileCount * 2 * 256;
  const uniformBuffer = storageBuffer(
    device,
    new Uint8Array(finalUniformOffset + 256),
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  const selectedTokenBuffer = storageBuffer(
    device,
    uintBytes(Uint32Array.of(0xffff_ffff)),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const tileCommands = [];
  for (let tileIndex = 0; tileIndex < tileCount; tileIndex += 1) {
    const firstRow = tileIndex * tileRows;
    const rowCount = tileRows;
    const tile = {
      tensorName: "token_embd.weight",
      storageType: "q6-k-fused-f32-256",
      firstRow,
      rowCount,
      rowBytes,
      buffer: packedTileBuffer,
      bufferOffset: 0,
      byteLength: rowBytes * rowCount,
    };
    const assembled = assembleQwen35StagedLogitsTileGpuCommands({
      tile,
      normalizedHidden: {
        buffer: normalizedHiddenBuffer,
        offset: 0,
        byteLength: 10_240,
      },
      workspace: stagedWorkspace(logitsBuffer),
      candidateOutput: {
        buffer: candidateBuffer,
        offset: 0,
        byteLength: 256 * 8,
      },
      candidateSlot: tileIndex,
      limits,
      uniforms: [
        { buffer: uniformBuffer, offset: tileIndex * 512, byteLength: 20 },
        { buffer: uniformBuffer, offset: tileIndex * 512 + 256, byteLength: 20 },
      ],
    });
    const gemvCommand = assembled.commands[0];
    const candidateCommand = assembled.commands[1];
    if (gemvCommand === undefined || candidateCommand === undefined) {
      throw new Error("Staged logits commands are incomplete");
    }
    device.queue.writeBuffer(
      uniformBuffer,
      tileIndex * 512,
      uintBytes(Uint32Array.from(gemvCommand.uniformWords)),
    );
    device.queue.writeBuffer(
      uniformBuffer,
      tileIndex * 512 + 256,
      uintBytes(Uint32Array.from(candidateCommand.uniformWords)),
    );
    tileCommands.push({ gemvCommand, candidateCommand });
  }
  const finalCommand = assembleQwen35StagedFinalTokenCommand({
    candidateOutput: {
      buffer: candidateBuffer,
      offset: 0,
      byteLength: 256 * 8,
    },
    selectedToken: {
      buffer: selectedTokenBuffer,
      offset: 0,
      byteLength: 4,
    },
    limits,
    uniform: { buffer: uniformBuffer, offset: finalUniformOffset, byteLength: 20 },
  });
  device.queue.writeBuffer(
    uniformBuffer,
    finalUniformOffset,
    uintBytes(Uint32Array.from(finalCommand.uniformWords)),
  );
  const gemvPipeline = await createPipeline(device, tileCommands[0].gemvCommand.kernel);
  const candidatePipeline = await createPipeline(device, tileCommands[0].candidateCommand.kernel);
  const finalPipeline = await createPipeline(device, finalCommand.kernel);
  const finalBindGroup = device.createBindGroup({
    layout: finalPipeline.getBindGroupLayout(0),
    entries: stagedBindingEntries(finalCommand),
  });
  const candidateReadback = device.createBuffer({
    size: candidateWords.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const selectedReadback = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (const { gemvCommand, candidateCommand } of tileCommands) {
    pass.setPipeline(gemvPipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: gemvPipeline.getBindGroupLayout(0),
      entries: stagedBindingEntries(gemvCommand),
    }));
    pass.dispatchWorkgroups(
      gemvCommand.workgroups.x,
      gemvCommand.workgroups.y,
      gemvCommand.workgroups.z,
    );
    pass.setPipeline(candidatePipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: candidatePipeline.getBindGroupLayout(0),
      entries: stagedBindingEntries(candidateCommand),
    }));
    pass.dispatchWorkgroups(
      candidateCommand.workgroups.x,
      candidateCommand.workgroups.y,
      candidateCommand.workgroups.z,
    );
  }
  pass.setPipeline(finalPipeline);
  pass.setBindGroup(0, finalBindGroup);
  pass.dispatchWorkgroups(1, 1, 1);
  pass.end();
  encoder.copyBufferToBuffer(candidateBuffer, 0, candidateReadback, 0, candidateWords.byteLength);
  encoder.copyBufferToBuffer(selectedTokenBuffer, 0, selectedReadback, 0, 4);
  device.queue.submit([encoder.finish()]);
  await candidateReadback.mapAsync(GPUMapMode.READ);
  const actualCandidates = new Uint32Array(candidateReadback.getMappedRange().slice(0));
  candidateReadback.unmap();
  await selectedReadback.mapAsync(GPUMapMode.READ);
  const selectedToken = new Uint32Array(selectedReadback.getMappedRange().slice(0))[0];
  selectedReadback.unmap();
  const actualScore = new Float32Array(actualCandidates.buffer)[0];
  if (
    !Number.isFinite(actualScore) ||
    Math.abs(actualScore - expectedScores[expectedWinnerRow]) > 1e-3 ||
    actualCandidates[1] !== expectedWinnerRow
  ) {
    throw new Error(
      `staged candidate mismatch: ${actualScore}, ${actualCandidates[1]}, expected ${expectedScores[expectedWinnerRow]}, ${expectedWinnerRow}`,
    );
  }
  if (selectedToken !== expectedWinnerRow) {
    throw new Error(`staged final token mismatch: ${selectedToken}`);
  }
  for (const buffer of [
    logitsBuffer,
    normalizedHiddenBuffer,
    packedTileBuffer,
    candidateBuffer,
    uniformBuffer,
    selectedTokenBuffer,
    candidateReadback,
    selectedReadback,
  ]) buffer.destroy();
  return {
    id: "qwen35-staged-logits-gpu-selection",
    status: "executed",
    candidateToken: actualCandidates[1],
    selectedToken,
  };
}

export async function runWebGpuKernelHarness({ onProgress = () => {} } = {}) {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU did not provide an adapter");
  // Keep the dev harness on the production contract. A default device leaves
  // optional WGSL extensions disabled, which made the optimized kernels look
  // invalid even on an adapter that supports the features.
  const requiredFeatures = ["shader-f16", "subgroups"];
  const missingFeatures = requiredFeatures.filter(
    (feature) => !adapter.features.has(feature),
  );
  if (missingFeatures.length > 0) {
    throw new Error(
      `This WebGPU adapter does not meet the Qwen runtime requirement: ${missingFeatures.join(", ")}`,
    );
  }
  const device = await adapter.requestDevice({ requiredFeatures });
  const results = [{
    id: "qwen35-wgsl-language-features",
    packed4x8IntegerDotProduct:
      navigator.gpu.wgslLanguageFeatures?.has(
        "packed_4x8_integer_dot_product",
      ) === true,
    subgroups: navigator.gpu.wgslLanguageFeatures?.has("subgroups") === true,
    adapterSubgroups: adapter.features.has("subgroups"),
    subgroupLimits: [adapter.limits.minSubgroupSize, adapter.limits.maxSubgroupSize],
  }];
  for (const kernel of LANGUAGE_GEMV_KERNELS) {
    onProgress(kernel.id);
    results.push(await runKernel(device, kernel));
  }
  for (const kernel of QWEN_PRIMITIVE_KERNELS) {
    onProgress(kernel.id);
    results.push(await runPrimitive(device, kernel));
  }
  for (const kernel of PACKED_EMBEDDING_KERNELS) {
    onProgress(kernel.id);
    results.push(await runEmbedding(device, kernel));
  }
  for (const kernel of QWEN35_HYBRID_KERNELS) {
    onProgress(kernel.id);
    results.push(await runHybridKernel(device, kernel));
  }
  for (const kernel of QWEN35_VISION_FOUNDATION_KERNELS) {
    onProgress(kernel.id);
    results.push(await runVisionFoundationKernel(device, kernel));
  }
  for (const kernel of QWEN35_VISION_LAYER_KERNELS) {
    onProgress(kernel.id);
    results.push(await runVisionLayerKernel(device, kernel));
  }
  for (const kernel of QWEN35_VISION_MERGER_KERNELS) {
    onProgress(kernel.id);
    results.push(await runVisionMergerKernel(device, kernel));
  }
  onProgress("qwen35-staged-logits-gpu-selection");
  results.push(await runStagedLogitsGpuSelection(device));
  onProgress("qwen35-production-gemv-benchmarks");
  results.push(...await benchmarkProductionGemvs(device));
  device.destroy();
  return results;
}

const output = document.querySelector("#results");
try {
  const results = await runWebGpuKernelHarness({
    onProgress: (kernel) => {
      output.textContent = JSON.stringify({ status: "running", kernel }, null, 2);
    },
  });
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
