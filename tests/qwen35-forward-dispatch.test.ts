import assert from "node:assert/strict";
import test from "node:test";

import { GgmlType } from "../src/gguf.js";
import {
  planQwen35PackedEmbeddingDispatch,
  planQwen35VisualEmbeddingDispatch,
  planQwen35PackedGemvDispatches,
  planQwen35TwinQ3GemvDispatches,
  planQwen35TiedLogitsGeometry,
  planQwen35TiedLogitsDispatches,
  type Qwen35ForwardDeviceLimits,
} from "../src/qwen35-forward-dispatch.js";
import type {
  Qwen35TensorWeightView,
  Qwen35WeightDirectoryView,
} from "../src/qwen35-weight-directory.js";
import type { Qwen35DispatchRequest } from "../src/qwen35-webgpu-executor.js";
import type { Qwen35StagedPackedRows } from "../src/qwen35-disk-backed-tied-embedding.js";

interface StagedForwardSubject {
  planQwen35StagedPackedEmbeddingDispatch(input: {
    readonly rows: Qwen35StagedPackedRows;
    readonly output: { readonly buffer: object; readonly offset: number; readonly byteLength: number };
    readonly uniform: { readonly buffer: object; readonly offset: number; readonly byteLength: number };
    readonly limits: Qwen35ForwardDeviceLimits;
  }): ReturnType<typeof planQwen35PackedEmbeddingDispatch>;
}

async function stagedForwardSubject(): Promise<StagedForwardSubject> {
  return await import("../src/qwen35-forward-dispatch.js") as unknown as StagedForwardSubject;
}

const limits: Qwen35ForwardDeviceLimits = {
  minStorageBufferOffsetAlignment: 256,
  minUniformBufferOffsetAlignment: 256,
  maxStorageBufferBindingSize: 1 << 30,
  maxUniformBufferBindingSize: 65_536,
  maxComputeWorkgroupsPerDimension: 65_535,
  supportsSubgroups: true,
};

test("plans a projected visual-token row into the language hidden workspace", () => {
  const source = { buffer: {}, offset: 256, byteLength: 2_560 * 4 };
  const output = { buffer: {}, offset: 0, byteLength: 2_560 * 4 };
  const uniform = { buffer: {}, offset: 0, byteLength: 16 };
  const plan = planQwen35VisualEmbeddingDispatch({ source, output, uniform, limits });
  assert.equal(plan.kernel.id, "qwen35-visual-embedding-f32");
  assert.deepEqual(plan.workgroups, { x: 10, y: 1, z: 1 });
  assert.deepEqual(plan.uniformWords, [2_560, 0, 0, 0]);
  assert.equal(plan.bindings[0]!.offset, source.offset);
  assert.equal(plan.bindings[1]!.offset, output.offset);
});

function tensor(input: {
  readonly name: string;
  readonly shape: readonly number[];
  readonly ggmlType: number;
  readonly storageType: string;
  readonly rowBytes: number;
  readonly splits: readonly number[];
  readonly buffers?: readonly object[];
  readonly bufferByteOffset?: number;
}): Qwen35TensorWeightView {
  let firstRow = 0;
  let tensorByteOffset = 0;
  const buffers = input.buffers ?? input.splits.map(() => ({}));
  return Object.freeze({
    name: input.name,
    shape: Object.freeze([...input.shape]),
    ggmlType: input.ggmlType,
    storageType: input.storageType,
    rowBytes: input.rowBytes,
    rowCount: input.splits.reduce((sum, rows) => sum + rows, 0),
    logicalBytes: BigInt(input.rowBytes) * BigInt(
      input.splits.reduce((sum, rows) => sum + rows, 0),
    ),
    physicalRows: Object.freeze(input.splits.map((rowCount, index) => {
      const view = Object.freeze({
        buffer: buffers[index]!,
        firstRow,
        rowCount,
        tensorByteOffset,
        bufferByteOffset: input.bufferByteOffset ?? 0,
        byteLength: input.rowBytes * rowCount,
      });
      firstRow += rowCount;
      tensorByteOffset += input.rowBytes * rowCount;
      return view;
    })),
  });
}

function directory(
  tensors: readonly Qwen35TensorWeightView[],
): Qwen35WeightDirectoryView {
  const byName = new Map(tensors.map((item) => [item.name, item] as const));
  return Object.freeze({
    size: byName.size,
    tensors: Object.freeze([...tensors]),
    logicalBytes: tensors.reduce((sum, item) => sum + item.logicalBytes, 0n),
    allocatedBytes: tensors.reduce((sum, item) => sum + item.logicalBytes, 0n),
    get: (name: string) => byName.get(name),
    entries: () => byName.entries(),
    [Symbol.iterator]: () => byName[Symbol.iterator](),
  });
}

function slice(buffer: object, byteLength: number, offset = 0) {
  return { buffer, offset, byteLength };
}

function consumeRequest(request: Qwen35DispatchRequest): Qwen35DispatchRequest {
  return request;
}

test("fuses equal Q3 projections that share one activation", () => {
  const firstWeights = {};
  const secondWeights = {};
  const activation = {};
  const packedActivation = {};
  const output = {};
  const uniform = {};
  const first = tensor({
    name: "blk.0.ffn_gate.weight",
    shape: [2_560, 9_216],
    ggmlType: GgmlType.Q3_K,
    storageType: "q3-k-fused-f32-192",
    rowBytes: 1_920,
    splits: [9_216],
    buffers: [firstWeights],
  });
  const second = tensor({
    name: "blk.0.ffn_up.weight",
    shape: [2_560, 9_216],
    ggmlType: GgmlType.Q3_K,
    storageType: "q3-k-fused-f32-192",
    rowBytes: 1_920,
    splits: [9_216],
    buffers: [secondWeights],
  });

  const plans = planQwen35TwinQ3GemvDispatches({
    weights: directory([first, second]),
    firstTensorName: first.name,
    secondTensorName: second.name,
    activation: slice(activation, 2_560 * 4),
    packedActivation: slice(packedActivation, 2_560 * 2),
    output: slice(output, 9_216 * 4),
    uniform: slice(uniform, 20),
    limits,
  });

  assert.ok(plans !== null);
  assert.equal(plans.length, 1);
  assert.equal(plans[0]!.kernel.id, "q3-k-fused-f32-192-swiglu-gemv-mobile-f16-subgroup");
  assert.match(plans[0]!.kernel.source, /row_sums/);
  assert.deepEqual(plans[0]!.workgroups, { x: 2_304, y: 1, z: 1 });
  assert.deepEqual(plans[0]!.uniformWords, [9_216, 2_560, 10, 0, 0]);
  assert.equal(plans[0]!.bindings[0]!.buffer, firstWeights);
  assert.equal(plans[0]!.bindings[1]!.buffer, secondWeights);
  assert.equal(plans[0]!.bindings[2]!.buffer, activation);
  assert.equal(plans[0]!.bindings[3]!.buffer, output);
  assert.equal(plans[0]!.bindings[4]!.buffer, uniform);
  assert.match(plans[0]!.kernel.source, /output\[logical_row\].*exp\(-gate\).*up/);
});

test("uses a portable fused Q3 SwiGLU path without subgroup feature evidence", () => {
  const first = tensor({
    name: "blk.0.ffn_gate.weight",
    shape: [2_560, 9_216],
    ggmlType: GgmlType.Q3_K,
    storageType: "q3-k-fused-f32-192",
    rowBytes: 1_920,
    splits: [9_216],
  });
  const second = tensor({
    name: "blk.0.ffn_up.weight",
    shape: [2_560, 9_216],
    ggmlType: GgmlType.Q3_K,
    storageType: "q3-k-fused-f32-192",
    rowBytes: 1_920,
    splits: [9_216],
  });
  const activation = {};
  const plans = planQwen35TwinQ3GemvDispatches({
    weights: directory([first, second]),
    firstTensorName: first.name,
    secondTensorName: second.name,
    activation: slice(activation, 2_560 * 4),
    packedActivation: slice({}, 2_560 * 2),
    output: slice({}, 9_216 * 4),
    uniform: slice({}, 20),
    limits: { ...limits, supportsSubgroups: false },
  });

  assert.ok(plans !== null);
  assert.equal(plans.length, 1);
  assert.equal(plans[0]!.kernel.id, "q3-k-fused-f32-192-swiglu-gemv-portable-f32");
  assert.equal(plans[0]!.bindings[2]!.buffer, activation);
  assert.deepEqual(plans[0]!.workgroups, { x: 9_216, y: 1, z: 1 });
  assert.doesNotMatch(plans[0]!.kernel.source, /enable subgroups|subgroupAdd/);
});

test("locates one packed embedding row inside its physical buffer", () => {
  const first = {};
  const second = {};
  const weights = directory([tensor({
    name: "token_embd.weight",
    shape: [2_560, 248_320],
    ggmlType: GgmlType.Q6_K,
    storageType: "q6-k-212",
    rowBytes: 2_120,
    splits: [2, 248_318],
    buffers: [first, second],
  })]);
  const output = {};
  const uniform = {};

  const plan = planQwen35PackedEmbeddingDispatch({
    weights,
    tokenId: 3,
    output: slice(output, 10_240),
    uniform: slice(uniform, 16),
    limits,
  });

  assert.equal(consumeRequest(plan), plan);
  assert.equal(plan.kernel.id, "q6-k-212-embedding-row-shared-portable-f32");
  assert.equal(plan.bindings[0]?.buffer, second);
  assert.deepEqual(plan.bindings, [
    { binding: 0, kind: "storage", buffer: second, offset: 2_048, size: 2_192 },
    { binding: 1, kind: "storage", buffer: output, offset: 0, size: 10_240 },
    { binding: 2, kind: "uniform", buffer: uniform, offset: 0, size: 16 },
  ]);
  assert.deepEqual(plan.uniformWords, [18, 2_560, 10, 0]);
  assert.deepEqual(plan.workgroups, { x: 10, y: 1, z: 1 });
});

test("binds only an aligned embedding-row window inside a large arena", () => {
  const weightBuffer = {};
  const plan = planQwen35PackedEmbeddingDispatch({
    weights: directory([tensor({
      name: "token_embd.weight",
      shape: [2_560, 248_320],
      ggmlType: GgmlType.Q6_K,
      storageType: "q6-k-212",
      rowBytes: 2_120,
      splits: [248_320],
      buffers: [weightBuffer],
    })]),
    tokenId: 1_025,
    output: slice({}, 10_240),
    uniform: slice({}, 16),
    limits: { ...limits, maxStorageBufferBindingSize: 16_384 },
  });

  assert.deepEqual(plan.bindings[0], {
    binding: 0,
    kind: "storage",
    buffer: weightBuffer,
    offset: 2_172_928,
    size: 2_192,
  });
  assert.deepEqual(plan.uniformWords, [18, 2_560, 10, 0]);
});

test("aligns a staged Q6_K cache row before binding its packed prefix", async () => {
  const subject = await stagedForwardSubject();
  assert.equal(
    typeof subject.planQwen35StagedPackedEmbeddingDispatch,
    "function",
    "the staged packed-row planner is required by the disk-backed cache",
  );
  const packedCache = {};
  const output = {};
  const uniform = {};
  const plan = subject.planQwen35StagedPackedEmbeddingDispatch({
    rows: {
      tensorName: "token_embd.weight",
      storageType: "q6-k-fused-f32-256",
      firstRow: 17,
      rowCount: 1,
      rowBytes: 2_560,
      buffer: packedCache,
      bufferOffset: 2_240,
      byteLength: 2_560,
    },
    output: slice(output, 10_240),
    uniform: slice(uniform, 16),
    limits,
  });

  assert.equal(plan.kernel.id, "q6-k-fused-f32-256-embedding-row-shared-portable-f32");
  assert.deepEqual(plan.bindings, [
    { binding: 0, kind: "storage", buffer: packedCache, offset: 2_048, size: 2_752 },
    { binding: 1, kind: "storage", buffer: output, offset: 0, size: 10_240 },
    { binding: 2, kind: "uniform", buffer: uniform, offset: 0, size: 16 },
  ]);
  assert.deepEqual(plan.uniformWords, [48, 2_560, 10, 0]);
  assert.deepEqual(plan.workgroups, { x: 10, y: 1, z: 1 });
});

test("plans every physical matrix row range for all six packed layouts", () => {
  const layouts = [
    [GgmlType.F32, "f32", 1, 4],
    [GgmlType.Q8_0, "q8-0-36", 32, 36],
    [GgmlType.Q3_K, "q3-k-112", 256, 112],
    [GgmlType.Q4_K, "q4-k-144", 256, 144],
    [GgmlType.Q5_K, "q5-k-176", 256, 176],
    [GgmlType.Q6_K, "q6-k-212", 256, 212],
  ] as const;

  for (const [ggmlType, storageType, valuesPerBlock, bytesPerBlock] of layouts) {
    const columns = storageType === "q8-0-36" ? 512 : valuesPerBlock * 2;
    const blocksPerRow = columns / valuesPerBlock;
    const rowBytes = bytesPerBlock * blocksPerRow;
    const weightBuffers = [{}, {}];
    const activation = {};
    const output = {};
    const uniforms = [{}, {}];
    const plans = planQwen35PackedGemvDispatches({
      weights: directory([tensor({
        name: "blk.0.ffn_down.weight",
        shape: [columns, 3],
        ggmlType,
        storageType,
        rowBytes,
        splits: [1, 2],
        buffers: weightBuffers,
      })]),
      tensorName: "blk.0.ffn_down.weight",
      activation: slice(activation, columns * 4),
      ...(storageType !== "f32"
        ? { packedActivation: slice({}, columns * 2) }
        : {}),
      output: slice(output, 12),
      uniforms: uniforms.map((buffer) => slice(buffer, 20)),
      limits,
    });

    const usesFastKernel = storageType !== "f32";
    const usesSeparatePack = usesFastKernel && storageType !== "q8-0-36";
    assert.equal(plans.length, usesSeparatePack ? 3 : 2, storageType);
    const matrixPlans = usesSeparatePack ? plans.slice(1) : plans;
    const profile = usesFastKernel
      ? "mobile-f16-subgroup"
      : "portable-f32";
    assert.equal(matrixPlans[0]?.kernel.id, `${storageType}-gemv-shared-${profile}`);
    assert.equal(matrixPlans[0]?.kernel.entryPoint, "packed_gemv");
    assert.deepEqual(matrixPlans.map((plan) => plan.uniformWords), [
      [1, columns, blocksPerRow, 0, 0],
      [2, columns, blocksPerRow, 0, 1],
    ]);
    assert.deepEqual(matrixPlans.map((plan) => plan.workgroups), [
      { x: 1, y: 1, z: 1 },
      { x: usesFastKernel ? 1 : 2, y: 1, z: 1 },
    ]);
    assert.deepEqual(matrixPlans.map((plan) => plan.bindings[0]?.buffer), weightBuffers);
    assert.equal(consumeRequest(matrixPlans[0]!), matrixPlans[0]);
  }
});

test("selects the portable Q8 GEMV when the device has no subgroups", () => {
  const columns = 512;
  const weights = directory([tensor({
    name: "blk.0.ffn_down.weight",
    shape: [columns, 1],
    ggmlType: GgmlType.Q8_0,
    storageType: "q8-0-36",
    rowBytes: 576,
    splits: [1],
    buffers: [{}],
  })]);
  const plans = planQwen35PackedGemvDispatches({
    weights,
    tensorName: "blk.0.ffn_down.weight",
    activation: slice({}, columns * 4),
    output: slice({}, 4),
    uniforms: [slice({}, 20)],
    limits: { ...limits, supportsSubgroups: false },
  });

  assert.equal(plans.length, 1);
  assert.equal(plans[0]!.kernel.id, "q8-0-36-gemv-shared-portable-f32");
  assert.doesNotMatch(plans[0]!.kernel.source, /enable subgroups/);
});

test("converts F32 activation inside a browser fused Q3 GEMV", () => {
  const weightBuffer = {};
  const activation = {};
  const packedActivation = {};
  const output = {};
  const uniform = {};
  const plans = planQwen35PackedGemvDispatches({
    weights: directory([tensor({
      name: "blk.0.ffn_gate.weight",
      shape: [2_560, 2],
      ggmlType: GgmlType.Q3_K,
      storageType: "q3-k-fused-f32-192",
      rowBytes: 1_920,
      splits: [2],
      buffers: [weightBuffer],
    })]),
    tensorName: "blk.0.ffn_gate.weight",
    activation: slice(activation, 10_240),
    packedActivation: slice(packedActivation, 5_120),
    output: slice(output, 8),
    uniforms: [slice(uniform, 20)],
    limits,
  });

  assert.equal(plans.length, 1);
  assert.equal(
    plans[0]?.kernel.id,
    "q3-k-fused-f32-192-gemv-shared-mobile-f16-subgroup",
  );
  assert.equal(plans[0]?.bindings[1]?.buffer, activation);
  assert.equal(plans[0]?.bindings[1]?.size, 10_240);
  assert.match(plans[0]?.kernel.source ?? "", /activation: array<f32>/);
});

test("converts F32 activation inside a browser Q8 GEMV", () => {
  const packedActivation = {};
  const plans = planQwen35PackedGemvDispatches({
    weights: directory([tensor({
      name: "blk.0.ssm_out.weight",
      shape: [4_096, 2_560],
      ggmlType: GgmlType.Q8_0,
      storageType: "q8-0-36",
      rowBytes: 4_608,
      splits: [2_560],
      buffers: [{}],
    })]),
    tensorName: "blk.0.ssm_out.weight",
    activation: slice({}, 4_096 * 4),
    packedActivation: slice(packedActivation, 4_096 * 2),
    output: slice({}, 2_560 * 4),
    uniforms: [slice({}, 20)],
    limits,
  });
  assert.equal(plans.length, 1);
  assert.equal(plans[0]?.bindings[1]?.size, 4_096 * 4);
  assert.match(plans[0]?.kernel.source ?? "", /activation: array<f32>/);
});

test("packs one lane-major FP16 activation before a browser fused Q5 GEMV", () => {
  for (const [ggmlType, storageType, rowBytes] of [
    [GgmlType.Q5_K, "q5-k-fused-f32-224", 8_064],
  ] as const) {
    const packedActivation = {};
    const plans = planQwen35PackedGemvDispatches({
      weights: directory([tensor({
        name: "blk.0.ffn_down.weight",
        shape: [9_216, 2_560],
        ggmlType,
        storageType,
        rowBytes,
        splits: [2_560],
        buffers: [{}],
      })]),
      tensorName: "blk.0.ffn_down.weight",
      activation: slice({}, 9_216 * 4),
      packedActivation: slice(packedActivation, 9_216 * 2),
      output: slice({}, 2_560 * 4),
      uniforms: [slice({}, 20)],
      limits,
    });

    assert.equal(plans.length, 2);
    assert.equal(plans[0]?.kernel.id, "qwen35-pack-lane-f16-activation");
    assert.deepEqual(plans[0]?.workgroups, { x: 18, y: 1, z: 1 });
    assert.equal(plans[1]?.kernel.id, `${storageType}-gemv-shared-mobile-f16-subgroup`);
    assert.equal(plans[1]?.bindings[1]?.buffer, packedActivation);
    assert.match(plans[1]?.kernel.source ?? "", /activation: array<u32>/);
  }
});

test("tiles tied logits without a vocabulary-sized output buffer", () => {
  const firstEmbeddingBuffer = {};
  const secondEmbeddingBuffer = {};
  const forgedOutput = {};
  const logitsTile = {};
  const weights = directory([
    tensor({
      name: "token_embd.weight",
      shape: [2_560, 248_320],
      ggmlType: GgmlType.Q4_K,
      storageType: "q4-k-144",
      rowBytes: 1_440,
      splits: [1_500, 246_820],
      buffers: [firstEmbeddingBuffer, secondEmbeddingBuffer],
    }),
    tensor({
      name: "output.weight",
      shape: [2_560, 248_320],
      ggmlType: GgmlType.Q4_K,
      storageType: "q4-k-144",
      rowBytes: 1_440,
      splits: [248_320],
      buffers: [forgedOutput],
    }),
  ]);

  const embedding = planQwen35PackedEmbeddingDispatch({
    weights,
    tokenId: 0,
    output: slice({}, 10_240),
    uniform: slice({}, 16),
    limits,
  });
  const logits = planQwen35TiedLogitsDispatches({
    weights,
    activation: slice({}, 10_240),
    output: slice(logitsTile, 4_096),
    uniforms: Array.from({ length: 244 }, () => slice({}, 20)),
    limits,
  });

  assert.equal(embedding.bindings[0]?.buffer, firstEmbeddingBuffer);
  assert.deepEqual(logits.slice(0, 4).map((plan) => ({
    tileIndex: plan.tileIndex,
    vocabularyStart: plan.vocabularyStart,
    tileRows: plan.tileRows,
    pieceOutputOffset: plan.pieceOutputOffset,
    pieceRows: plan.pieceRows,
    completesTile: plan.completesTile,
    weightBuffer: plan.bindings[0]?.buffer,
    weightOffset: plan.bindings[0]?.offset,
    weightBytes: plan.bindings[0]?.size,
    outputBuffer: plan.bindings[2]?.buffer,
    outputBytes: plan.bindings[2]?.size,
    uniforms: plan.uniformWords,
  })), [
    {
      tileIndex: 0,
      vocabularyStart: 0,
      tileRows: 1_024,
      pieceOutputOffset: 0,
      pieceRows: 1_024,
      completesTile: true,
      weightBuffer: firstEmbeddingBuffer,
      weightOffset: 0,
      weightBytes: 1_474_560,
      outputBuffer: logitsTile,
      outputBytes: 4_096,
      uniforms: [1_024, 2_560, 10, 0, 0],
    },
    {
      tileIndex: 1,
      vocabularyStart: 1_024,
      tileRows: 1_024,
      pieceOutputOffset: 0,
      pieceRows: 476,
      completesTile: false,
      weightBuffer: firstEmbeddingBuffer,
      weightOffset: 1_474_560,
      weightBytes: 685_440,
      outputBuffer: logitsTile,
      outputBytes: 4_096,
      uniforms: [476, 2_560, 10, 0, 0],
    },
    {
      tileIndex: 1,
      vocabularyStart: 1_024,
      tileRows: 1_024,
      pieceOutputOffset: 476,
      pieceRows: 548,
      completesTile: true,
      weightBuffer: secondEmbeddingBuffer,
      weightOffset: 0,
      weightBytes: 789_120,
      outputBuffer: logitsTile,
      outputBytes: 4_096,
      uniforms: [548, 2_560, 10, 0, 476],
    },
    {
      tileIndex: 2,
      vocabularyStart: 2_048,
      tileRows: 1_024,
      pieceOutputOffset: 0,
      pieceRows: 1_024,
      completesTile: true,
      weightBuffer: secondEmbeddingBuffer,
      weightOffset: 788_992,
      weightBytes: 1_474_688,
      outputBuffer: logitsTile,
      outputBytes: 4_096,
      uniforms: [1_024, 2_560, 10, 32, 0],
    },
  ]);
  assert.equal(logits.every(
    (plan) => plan.bindings[0]?.buffer !== forgedOutput,
  ), true);
  assert.equal(logits.length, 244);
  assert.equal(logits.filter((plan) => plan.completesTile).length, 243);
  assert.equal(new Set(logits.map((plan) => plan.tileIndex)).size, 243);
  assert.deepEqual(
    (({ tileIndex, vocabularyStart, tileRows, pieceOutputOffset, pieceRows, completesTile }) => ({
      tileIndex,
      vocabularyStart,
      tileRows,
      pieceOutputOffset,
      pieceRows,
      completesTile,
    }))(logits.at(-1)!),
    {
      tileIndex: 242,
      vocabularyStart: 247_808,
      tileRows: 262,
      pieceOutputOffset: 0,
      pieceRows: 262,
      completesTile: true,
    },
  );
  assert.equal(
    Math.max(...logits.map(
      (plan) => plan.vocabularyStart + plan.pieceOutputOffset + plan.pieceRows,
    )),
    248_070,
  );
  assert.equal(
    logits.at(-1)!.vocabularyStart +
      logits.at(-1)!.pieceOutputOffset +
      logits.at(-1)!.pieceRows - 1,
    248_069,
  );
});

test("derives tied-logits uniform capacity before buffers are allocated", () => {
  const geometry = planQwen35TiedLogitsGeometry({
    weights: directory([tensor({
      name: "token_embd.weight",
      shape: [2_560, 248_320],
      ggmlType: GgmlType.Q4_K,
      storageType: "q4-k-144",
      rowBytes: 1_440,
      splits: [1_500, 246_820],
    })]),
    limits,
  });

  assert.deepEqual(geometry, {
    modelRows: 248_320,
    decodableRows: 248_070,
    logicalTileRows: 1_024,
    mathematicalTileCount: 243,
    finalTileRows: 262,
    physicalPieceCount: 244,
    reductionDispatchCount: 244,
    uniformCount: 488,
  });
  assert.equal(Object.isFrozen(geometry), true);
});

test("reduces tied-logits tiles to the aligned device binding limit", () => {
  const weightBuffer = {};
  const logits = planQwen35TiedLogitsDispatches({
    weights: directory([tensor({
      name: "token_embd.weight",
      shape: [2_560, 248_320],
      ggmlType: GgmlType.Q4_K,
      storageType: "q4-k-144",
      rowBytes: 1_440,
      splits: [248_320],
      buffers: [weightBuffer],
    })]),
    activation: slice({}, 10_240),
    output: slice({}, 4_096),
    uniforms: Array.from({ length: 485 }, () => slice({}, 20)),
    limits: { ...limits, maxStorageBufferBindingSize: 737_280 },
  });

  assert.deepEqual(logits.slice(0, 2).map((plan) => ({
    tileIndex: plan.tileIndex,
    vocabularyStart: plan.vocabularyStart,
    tileRows: plan.tileRows,
    pieceOutputOffset: plan.pieceOutputOffset,
    pieceRows: plan.pieceRows,
    completesTile: plan.completesTile,
    weightOffset: plan.bindings[0]?.offset,
    weightBytes: plan.bindings[0]?.size,
    outputBytes: plan.bindings[2]?.size,
  })), [
    {
      tileIndex: 0,
      vocabularyStart: 0,
      tileRows: 1_024,
      pieceOutputOffset: 0,
      pieceRows: 512,
      completesTile: false,
      weightOffset: 0,
      weightBytes: 737_280,
      outputBytes: 4_096,
    },
    {
      tileIndex: 0,
      vocabularyStart: 0,
      tileRows: 1_024,
      pieceOutputOffset: 512,
      pieceRows: 512,
      completesTile: true,
      weightOffset: 737_280,
      weightBytes: 737_280,
      outputBytes: 4_096,
    },
  ]);
  assert.equal(logits.every((plan) => plan.bindings[0]?.buffer === weightBuffer), true);
  assert.equal(logits.length, 485);
  assert.equal(logits.filter((plan) => plan.completesTile).length, 243);
});

test("requires the tiled path for the tied embedding projection", () => {
  const weights = directory([tensor({
    name: "token_embd.weight",
    shape: [256, 2],
    ggmlType: GgmlType.Q4_K,
    storageType: "q4-k-144",
    rowBytes: 144,
    splits: [2],
  })]);

  assert.throws(
    () => planQwen35PackedGemvDispatches({
      weights,
      tensorName: "token_embd.weight",
      activation: slice({}, 1_024),
      output: slice({}, 8),
      uniforms: [slice({}, 20)],
      limits,
    }),
    { code: "forward-tied-logits-required" },
  );
  assert.throws(
    () => planQwen35PackedEmbeddingDispatch({
      weights,
      tokenId: 0,
      output: slice({}, 1_024),
      uniform: slice({}, 16),
      limits,
    }),
    { code: "forward-embedding-shape-invalid" },
  );
  assert.throws(
    () => planQwen35TiedLogitsDispatches({
      weights,
      activation: slice({}, 1_024),
      output: slice({}, 8),
      uniforms: [slice({}, 20)],
      limits,
    }),
    { code: "forward-embedding-shape-invalid" },
  );
});

test("requires distinct uniform ranges for every batched dispatch", () => {
  const projection = directory([tensor({
    name: "blk.0.ffn_down.weight",
    shape: [64, 2],
    ggmlType: GgmlType.F32,
    storageType: "f32",
    rowBytes: 256,
    splits: [1, 1],
  })]);
  const sharedUniformBuffer = {};
  const projectionInput = {
    weights: projection,
    tensorName: "blk.0.ffn_down.weight",
    activation: slice({}, 256),
    output: slice({}, 8),
    limits,
  } as const;

  assert.throws(
    () => planQwen35PackedGemvDispatches({
      ...projectionInput,
      uniforms: [
        slice(sharedUniformBuffer, 20),
        slice(sharedUniformBuffer, 20),
      ],
    }),
    { code: "forward-uniform-alias-invalid" },
  );

  const disjoint = planQwen35PackedGemvDispatches({
    ...projectionInput,
    uniforms: [
      slice(sharedUniformBuffer, 20),
      slice(sharedUniformBuffer, 20, 256),
    ],
  });
  assert.deepEqual(
    disjoint.map((plan) => plan.bindings[3]?.offset),
    [0, 256],
  );

  const tiedWeights = directory([tensor({
    name: "token_embd.weight",
    shape: [2_560, 248_320],
    ggmlType: GgmlType.Q4_K,
    storageType: "q4-k-144",
    rowBytes: 1_440,
    splits: [248_320],
  })]);
  const overlappingUniforms = Array.from(
    { length: 243 },
    (_, index) => slice(
      sharedUniformBuffer,
      20,
      index === 0 ? 0 : index === 1 ? 4 : index * 32,
    ),
  );
  assert.throws(
    () => planQwen35TiedLogitsDispatches({
      weights: tiedWeights,
      activation: slice({}, 10_240),
      output: slice({}, 4_096),
      uniforms: overlappingUniforms,
      limits: { ...limits, minUniformBufferOffsetAlignment: 4 },
    }),
    { code: "forward-uniform-alias-invalid" },
  );
});

test("returns deeply immutable requests and uniform words", () => {
  const plan = planQwen35PackedEmbeddingDispatch({
    weights: directory([tensor({
      name: "token_embd.weight",
      shape: [2_560, 248_320],
      ggmlType: GgmlType.F32,
      storageType: "f32",
      rowBytes: 10_240,
      splits: [248_320],
    })]),
    tokenId: 0,
    output: slice({}, 10_240),
    uniform: slice({}, 16),
    limits,
  });

  assert.equal(Object.isFrozen(plan), true);
  assert.equal(Object.isFrozen(plan.kernel), true);
  assert.equal(Object.isFrozen(plan.bindings), true);
  assert.equal(Object.isFrozen(plan.bindings[0]), true);
  assert.equal(Object.isFrozen(plan.uniformWords), true);
  assert.equal(Object.isFrozen(plan.workgroups), true);
});

test("rejects malformed physical rows, binding limits, and missing uniform slots", () => {
  const validEmbedding = tensor({
    name: "token_embd.weight",
    shape: [2_560, 248_320],
    ggmlType: GgmlType.F32,
    storageType: "f32",
    rowBytes: 10_240,
    splits: [1, 248_319],
  });
  const validProjection = tensor({
    name: "blk.0.ffn_down.weight",
    shape: [64, 2],
    ggmlType: GgmlType.F32,
    storageType: "f32",
    rowBytes: 256,
    splits: [1, 1],
  });
  const malformed = {
    ...validEmbedding,
    physicalRows: Object.freeze([
      validEmbedding.physicalRows[0]!,
      Object.freeze({ ...validEmbedding.physicalRows[1]!, firstRow: 2 }),
    ]),
  };
  assert.throws(
    () => planQwen35PackedEmbeddingDispatch({
      weights: directory([malformed]),
      tokenId: 1,
      output: slice({}, 10_240),
      uniform: slice({}, 16),
      limits,
    }),
    { code: "forward-weight-views-invalid" },
  );

  assert.throws(
    () => planQwen35PackedEmbeddingDispatch({
      weights: directory([validEmbedding]),
      tokenId: 0,
      output: slice({}, 10_240, 4),
      uniform: slice({}, 16),
      limits,
    }),
    { code: "forward-binding-invalid" },
  );

  const misalignedWeight = tensor({
    name: "token_embd.weight",
    shape: [2_560, 248_320],
    ggmlType: GgmlType.F32,
    storageType: "f32",
    rowBytes: 10_240,
    splits: [248_320],
    bufferByteOffset: 4,
  });
  assert.throws(
    () => planQwen35PackedEmbeddingDispatch({
      weights: directory([misalignedWeight]),
      tokenId: 0,
      output: slice({}, 10_240),
      uniform: slice({}, 16),
      limits,
    }),
    { code: "forward-binding-invalid" },
  );

  assert.throws(
    () => planQwen35PackedGemvDispatches({
      weights: directory([validProjection]),
      tensorName: "blk.0.ffn_down.weight",
      activation: slice({}, 256),
      output: slice({}, 8),
      uniforms: [slice({}, 20), slice({}, 20)],
      limits: { ...limits, maxStorageBufferBindingSize: 128 },
    }),
    { code: "forward-binding-invalid" },
  );

  assert.throws(
    () => planQwen35PackedGemvDispatches({
      weights: directory([validProjection]),
      tensorName: "blk.0.ffn_down.weight",
      activation: slice({}, 256),
      output: slice({}, 8),
      uniforms: [slice({}, 20)],
      limits,
    }),
    { code: "forward-uniform-count-invalid" },
  );

  const sharedInputOutput = {};
  assert.throws(
    () => planQwen35PackedGemvDispatches({
      weights: directory([validProjection]),
      tensorName: "blk.0.ffn_down.weight",
      activation: slice(sharedInputOutput, 256),
      output: slice(sharedInputOutput, 8),
      uniforms: [slice({}, 20), slice({}, 20)],
      limits,
    }),
    { code: "forward-buffer-alias-invalid" },
  );

  const sharedPhysicalBuffer = {};
  const overlappingPhysical = tensor({
    name: "blk.0.ffn_down.weight",
    shape: [64, 2],
    ggmlType: GgmlType.F32,
    storageType: "f32",
    rowBytes: 256,
    splits: [1, 1],
    buffers: [sharedPhysicalBuffer, sharedPhysicalBuffer],
  });
  assert.throws(
    () => planQwen35PackedGemvDispatches({
      weights: directory([overlappingPhysical]),
      tensorName: "blk.0.ffn_down.weight",
      activation: slice({}, 256),
      output: slice({}, 8),
      uniforms: [slice({}, 20), slice({}, 20)],
      limits,
    }),
    { code: "forward-weight-views-invalid" },
  );

  const weightBuffer = {};
  const singleViewProjection = tensor({
    name: "blk.0.ffn_down.weight",
    shape: [64, 2],
    ggmlType: GgmlType.F32,
    storageType: "f32",
    rowBytes: 256,
    splits: [2],
    buffers: [weightBuffer],
  });
  assert.throws(
    () => planQwen35PackedGemvDispatches({
      weights: directory([singleViewProjection]),
      tensorName: "blk.0.ffn_down.weight",
      activation: slice({}, 256),
      output: slice(weightBuffer, 8),
      uniforms: [slice({}, 20)],
      limits,
    }),
    { code: "forward-buffer-alias-invalid" },
  );

  const sharedUniformOutput = {};
  assert.throws(
    () => planQwen35PackedGemvDispatches({
      weights: directory([singleViewProjection]),
      tensorName: "blk.0.ffn_down.weight",
      activation: slice({}, 256),
      output: slice(sharedUniformOutput, 8),
      uniforms: [slice(sharedUniformOutput, 20)],
      limits,
    }),
    { code: "forward-buffer-alias-invalid" },
  );
});
