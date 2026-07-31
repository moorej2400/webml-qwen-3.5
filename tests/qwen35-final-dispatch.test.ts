import assert from "node:assert/strict";
import test from "node:test";

import { GgmlType } from "../src/gguf.js";
import type {
  Qwen35ActivationResourceKind,
  Qwen35ActivationResourceView,
} from "../src/qwen35-activation-workspace.js";
import type { Qwen35Invocation, Qwen35Program } from "../src/qwen35-program.js";
import type {
  Qwen35TensorWeightView,
  Qwen35WeightDirectoryView,
} from "../src/qwen35-weight-directory.js";

const limits = {
  minStorageBufferOffsetAlignment: 256,
  minUniformBufferOffsetAlignment: 256,
  maxStorageBufferBindingSize: 1 << 30,
  maxUniformBufferBindingSize: 65_536,
  maxComputeWorkgroupsPerDimension: 65_535,
} as const;

function fixture() {
  const invocation = Object.freeze({
    kind: "rms-norm" as const,
    layer: "final" as const,
    site: "final" as const,
    weight: "output_norm.weight",
    epsilon: 1e-6,
    fp32Accumulation: true as const,
  });
  const logits = Object.freeze({
    kind: "tiled-tied-logits" as const,
    weight: "token_embd.weight" as const,
    tiedWeightOwner: "embedding" as const,
    modelRows: 248_320 as const,
    decodableRows: 248_070 as const,
    columns: 2_560 as const,
    logicalTileRows: 1_024 as const,
    mathematicalTileCount: 243 as const,
    finalTileRows: 262 as const,
  });
  const reduction = Object.freeze({
    kind: "greedy-logits-reduction" as const,
    kernels: Object.freeze(["logits-tile-top-1", "indexed-top-1"] as const),
    mathematicalTileCount: 243 as const,
    candidatesPerTile: 1 as const,
    candidateCount: 243 as const,
    candidateCapacity: 256 as const,
    selectedTokenReadback: Object.freeze({
      resource: "selected-token" as const,
      scalarType: "u32" as const,
      elementCount: 1 as const,
      byteOffset: 0 as const,
      noSelectionSentinel: 0xffff_ffff,
    }),
    runnable: true as const,
  });
  const program = Object.freeze({
    model: "qwen35-4b",
    invocations: Object.freeze([invocation, logits, reduction]) as readonly Qwen35Invocation[],
    tensorBindings: Object.freeze({}),
    runnable: true,
  }) as unknown as Qwen35Program;
  const weightBuffer = {};
  const weight = Object.freeze({
    name: "output_norm.weight",
    shape: Object.freeze([2_560]),
    ggmlType: GgmlType.F32,
    storageType: "f32" as const,
    rowBytes: 10_240,
    rowCount: 1,
    logicalBytes: 10_240n,
    physicalRows: Object.freeze([Object.freeze({
      buffer: weightBuffer,
      firstRow: 0,
      rowCount: 1,
      tensorByteOffset: 0,
      bufferByteOffset: 0,
      byteLength: 10_240,
    })]),
  }) satisfies Qwen35TensorWeightView;
  const byName = new Map([[weight.name, weight]] as const);
  const weights = Object.freeze({
    size: 1,
    tensors: Object.freeze([weight]),
    logicalBytes: 10_240n,
    allocatedBytes: 10_240n,
    get: (name: string) => byName.get(name),
    entries: () => byName.entries(),
    [Symbol.iterator]: () => byName[Symbol.iterator](),
  }) satisfies Qwen35WeightDirectoryView;
  const hiddenBuffer = {};
  const normalizedBuffer = {};
  const views = new Map<Qwen35ActivationResourceKind, Qwen35ActivationResourceView>([
    ["packed-embedding-output", Object.freeze({
      kind: "packed-embedding-output",
      scalarType: "f32",
      elementCount: 2_560,
      bytes: 10_240n,
      usage: 0x0088,
      byteLength: 10_240,
      binding: Object.freeze({ buffer: hiddenBuffer, offset: 0 as const, size: 10_240 }),
    })],
    ["normalized-hidden", Object.freeze({
      kind: "normalized-hidden",
      scalarType: "f32",
      elementCount: 2_560,
      bytes: 10_240n,
      usage: 0x0088,
      byteLength: 10_240,
      binding: Object.freeze({ buffer: normalizedBuffer, offset: 0 as const, size: 10_240 }),
    })],
  ]);
  return {
    program,
    invocation,
    weights,
    workspace: Object.freeze({ get(kind: Qwen35ActivationResourceKind) {
      const view = views.get(kind);
      if (view === undefined) throw new Error("missing fixture resource");
      return view;
    } }),
    limits,
    uniform: Object.freeze({ buffer: {}, offset: 0, byteLength: 16 }),
  };
}

async function planner() {
  const module = await import("../src/qwen35-final-dispatch.js").catch(() => ({}));
  assert.equal(typeof module.planQwen35FinalNormDispatch, "function");
  return module.planQwen35FinalNormDispatch as (input: ReturnType<typeof fixture>) => {
    readonly uniformCount: 1;
    readonly command: {
      readonly stage: "final-rms";
      readonly bindings: readonly { readonly binding: number; readonly buffer: object }[];
      readonly uniformWords: readonly number[];
      readonly workgroups: { readonly x: number };
    };
  };
}

function assertSanitizedDiagnostic(
  callback: () => unknown,
  code: string,
  privateDetail: string,
): void {
  assert.throws(callback, (error: unknown) => {
    assert.equal((error as { code?: unknown }).code, code);
    assert.doesNotMatch((error as Error).message, new RegExp(privateDetail, "i"));
    return true;
  });
}

test("assembles the exact final output RMS command", async () => {
  const input = fixture();
  const plan = (await planner())(input);
  const epsilon = new ArrayBuffer(4);
  new DataView(epsilon).setFloat32(0, 1e-6, true);

  assert.equal(plan.uniformCount, 1);
  assert.equal(plan.command.stage, "final-rms");
  assert.deepEqual(plan.command.bindings.map(({ binding }) => binding), [0, 1, 2, 3]);
  assert.equal(plan.command.bindings[0]!.buffer,
    input.workspace.get("packed-embedding-output").binding.buffer);
  assert.equal(plan.command.bindings[2]!.buffer,
    input.workspace.get("normalized-hidden").binding.buffer);
  assert.deepEqual(plan.command.uniformWords, [
    2_560,
    2_560,
    new DataView(epsilon).getUint32(0, true),
    0,
  ]);
  assert.equal(plan.command.workgroups.x, 10);
  assert.equal(Object.isFrozen(plan), true);
  assert.equal(Object.isFrozen(plan.command), true);
  assert.equal(Object.isFrozen(plan.command.bindings), true);
  assert.equal(Object.isFrozen(plan.command.uniformWords), true);
});

test("rejects program, direct weight, and alias drift", async () => {
  const plan = await planner();
  const valid = fixture();
  assert.throws(
    () => plan({
      ...valid,
      program: Object.freeze({ ...valid.program, runnable: false }) as unknown as Qwen35Program,
    }),
    { code: "final-norm-program-invalid" },
  );
  assert.throws(
    () => plan({
      ...valid,
      program: Object.freeze({
        ...valid.program,
        blockedBy: undefined,
      }) as unknown as Qwen35Program,
    }),
    { code: "final-norm-program-invalid" },
  );
  assert.throws(
    () => plan({
      ...valid,
      program: Object.freeze({
        ...valid.program,
        blockedBy: "weight-orchestration",
      }) as unknown as Qwen35Program,
    }),
    { code: "final-norm-program-invalid" },
  );
  assert.throws(
    () => plan({ ...valid, invocation: Object.freeze({ ...valid.invocation, site: "input" }) }),
    /program|final/i,
  );
  const invalidWeight = Object.freeze({
    ...valid.weights.get("output_norm.weight")!,
    ggmlType: GgmlType.Q3_K,
    storageType: "q3-k-112" as const,
  });
  const invalidByName = new Map([[invalidWeight.name, invalidWeight]] as const);
  assert.throws(
    () => plan({
      ...valid,
      weights: Object.freeze({
        ...valid.weights,
        tensors: Object.freeze([invalidWeight]),
        get: (name: string) => invalidByName.get(name),
        entries: () => invalidByName.entries(),
        [Symbol.iterator]: () => invalidByName[Symbol.iterator](),
      }),
    }),
    /F32|weight/i,
  );

  const shared = {};
  const aliasViews = new Map<Qwen35ActivationResourceKind, Qwen35ActivationResourceView>([
    ["packed-embedding-output", Object.freeze({
      ...valid.workspace.get("packed-embedding-output"),
      binding: Object.freeze({ buffer: shared, offset: 0 as const, size: 10_240 }),
    })],
    ["normalized-hidden", Object.freeze({
      ...valid.workspace.get("normalized-hidden"),
      binding: Object.freeze({ buffer: shared, offset: 0 as const, size: 10_240 }),
    })],
  ]);
  assert.throws(
    () => plan({
      ...valid,
      workspace: Object.freeze({ get(kind: Qwen35ActivationResourceKind) {
        const view = aliasViews.get(kind);
        if (view === undefined) throw new Error("missing fixture resource");
        return view;
      } }),
    }),
    /alias|overlap/i,
  );
});

test("rejects a forged final RMS discriminator and every fixed logits ABI field", async () => {
  const plan = await planner();
  const valid = fixture();
  const logits = valid.program.invocations[1]!;
  const reduction = valid.program.invocations[2]!;
  const forgedInvocation = Object.freeze({ ...valid.invocation, kind: "gemv" });
  const forgedInvocationProgram = Object.freeze({
    ...valid.program,
    invocations: Object.freeze([forgedInvocation, logits, reduction]),
  }) as unknown as Qwen35Program;
  assert.throws(
    () => plan({
      ...valid,
      program: forgedInvocationProgram,
      invocation: forgedInvocation as unknown as typeof valid.invocation,
    }),
    { code: "final-norm-program-invalid" },
  );

  const forgedFields: readonly (readonly [string, unknown])[] = [
    ["kind", "gemv"],
    ["weight", "forged.weight"],
    ["tiedWeightOwner", "logits"],
    ["modelRows", 248_319],
    ["decodableRows", 248_069],
    ["columns", 2_559],
    ["logicalTileRows", 1_023],
    ["mathematicalTileCount", 242],
    ["finalTileRows", 261],
  ];
  for (const [field, value] of forgedFields) {
    const forgedLogits = Object.freeze({ ...logits, [field]: value });
    const program = Object.freeze({
      ...valid.program,
      invocations: Object.freeze([valid.invocation, forgedLogits, reduction]),
    }) as unknown as Qwen35Program;
    assert.throws(
      () => plan({ ...valid, program }),
      { code: "final-norm-program-invalid" },
      `forged logits field ${field}`,
    );
  }
});

test("rejects every forged reduction and selected-token readback ABI field", async () => {
  const plan = await planner();
  const valid = fixture();
  const logits = valid.program.invocations[1]!;
  const reduction = valid.program.invocations[2]!;
  const forgedFields: readonly (readonly [string, unknown])[] = [
    ["kind", "residual-add"],
    ["kernels", Object.freeze(["indexed-top-1", "logits-tile-top-1"])],
    ["kernels", undefined],
    ["mathematicalTileCount", 242],
    ["candidatesPerTile", 2],
    ["candidateCount", 242],
    ["candidateCapacity", 255],
    ["selectedTokenReadback", undefined],
    ["runnable", false],
  ];
  for (const [field, value] of forgedFields) {
    const forgedReduction = Object.freeze({ ...reduction, [field]: value });
    const program = Object.freeze({
      ...valid.program,
      invocations: Object.freeze([valid.invocation, logits, forgedReduction]),
    }) as unknown as Qwen35Program;
    assert.throws(
      () => plan({ ...valid, program }),
      { code: "final-norm-program-invalid" },
      `forged reduction field ${field}`,
    );
  }

  if (reduction.kind !== "greedy-logits-reduction") throw new Error("bad fixture");
  const readbackFields: readonly (readonly [string, unknown])[] = [
    ["resource", "normalized-hidden"],
    ["scalarType", "f32"],
    ["elementCount", 2],
    ["byteOffset", 4],
    ["noSelectionSentinel", 0],
  ];
  for (const [field, value] of readbackFields) {
    const forgedReduction = Object.freeze({
      ...reduction,
      selectedTokenReadback: Object.freeze({
        ...reduction.selectedTokenReadback,
        [field]: value,
      }),
    });
    const program = Object.freeze({
      ...valid.program,
      invocations: Object.freeze([valid.invocation, logits, forgedReduction]),
    }) as unknown as Qwen35Program;
    assert.throws(
      () => plan({ ...valid, program }),
      { code: "final-norm-program-invalid" },
      `forged selected-token readback field ${field}`,
    );
  }
});

test("sanitizes caller weight and workspace lookup failures", async () => {
  const plan = await planner();
  const valid = fixture();
  const privateDetail = "private-final-model-path";
  assertSanitizedDiagnostic(
    () => plan({
      ...valid,
      weights: Object.freeze({
        ...valid.weights,
        get() {
          throw new Error(privateDetail);
        },
      }),
    }),
    "final-norm-weight-invalid",
    privateDetail,
  );
  assertSanitizedDiagnostic(
    () => plan({
      ...valid,
      workspace: Object.freeze({
        get() {
          throw new Error(privateDetail);
        },
      }),
    }),
    "final-norm-workspace-invalid",
    privateDetail,
  );
});
