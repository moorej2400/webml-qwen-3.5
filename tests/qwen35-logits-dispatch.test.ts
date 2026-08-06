import assert from "node:assert/strict";
import test from "node:test";

import { GgmlType } from "../src/gguf.js";
import type {
  Qwen35ActivationResourceKind,
  Qwen35ActivationResourceView,
} from "../src/qwen35-activation-workspace.js";
import type { Qwen35TiedLogitsDispatchPlan } from "../src/qwen35-forward-dispatch.js";
import {
  assembleQwen35TiledLogitsCommands,
  planQwen35TiledLogitsUniformCount,
  validateQwen35TiedLogitsDispatchGroups,
} from "../src/qwen35-logits-dispatch.js";
import type {
  Qwen35TensorWeightView,
  Qwen35WeightDirectoryView,
} from "../src/qwen35-weight-directory.js";
import type { Qwen35DispatchRequest } from "../src/qwen35-webgpu-executor.js";
import type { Qwen35StagedPackedRows } from "../src/qwen35-disk-backed-tied-embedding.js";

interface StagedLogitsSubject {
  assembleQwen35StagedLogitsTileGpuCommands(input: {
    readonly tile: Qwen35StagedPackedRows;
    readonly normalizedHidden: { readonly buffer: object; readonly offset: number; readonly byteLength: number };
    readonly workspace: ReturnType<typeof workspace>;
    readonly candidateOutput: { readonly buffer: object; readonly offset: number; readonly byteLength: number };
    readonly candidateSlot: number;
    readonly limits: typeof limits;
    readonly uniforms: readonly { readonly buffer: object; readonly offset: number; readonly byteLength: number }[];
  }): {
    readonly commands: readonly ({
      readonly kind: "logits-gemv-piece" | "logits-tile-top-1";
      readonly tileIndex: number;
      readonly uniformWords: readonly number[];
    } & Qwen35DispatchRequest)[];
    readonly uniformCount: 2;
  };
  assembleQwen35StagedFinalTokenCommand(input: {
    readonly candidateOutput: { readonly buffer: object; readonly offset: number; readonly byteLength: number };
    readonly selectedToken: { readonly buffer: object; readonly offset: number; readonly byteLength: number };
    readonly limits: typeof limits;
    readonly uniform: { readonly buffer: object; readonly offset: number; readonly byteLength: number };
  }): {
    readonly kind: "indexed-top-1";
    readonly tileIndex: null;
    readonly uniformWords: readonly number[];
  } & Qwen35DispatchRequest;
  assembleQwen35StagedLogitsTileCommands(input: {
    readonly tile: Qwen35StagedPackedRows;
    readonly normalizedHidden: { readonly buffer: object; readonly offset: number; readonly byteLength: number };
    readonly workspace: ReturnType<typeof workspace>;
    readonly candidateOutput: { readonly buffer: object; readonly offset: number; readonly byteLength: number };
    readonly limits: typeof limits;
    readonly uniforms: readonly { readonly buffer: object; readonly offset: number; readonly byteLength: number }[];
  }): {
    readonly commands: readonly ({
      readonly kind: "logits-gemv-piece" | "logits-tile-top-1";
      readonly tileIndex: number;
      readonly uniformWords: readonly number[];
    } & Qwen35DispatchRequest)[];
    readonly uniformCount: 2;
    readonly candidateScoreReadback: {
      readonly buffer: object;
      readonly offset: number;
      readonly byteLength: 4;
      readonly scalarType: "f32";
    };
    readonly candidateTokenReadback: {
      readonly buffer: object;
      readonly offset: number;
      readonly byteLength: 4;
      readonly scalarType: "u32";
    };
  };
}

async function stagedLogitsSubject(): Promise<StagedLogitsSubject> {
  return await import("../src/qwen35-logits-dispatch.js") as unknown as StagedLogitsSubject;
}

const limits = {
  minStorageBufferOffsetAlignment: 256,
  minUniformBufferOffsetAlignment: 256,
  maxStorageBufferBindingSize: 1 << 30,
  maxUniformBufferBindingSize: 65_536,
  maxComputeWorkgroupsPerDimension: 65_535,
} as const;

function tensor(input: {
  readonly splits: readonly number[];
  readonly buffers?: readonly object[];
}): Qwen35TensorWeightView {
  const rowBytes = 1_440;
  const buffers = input.buffers ?? input.splits.map(() => ({}));
  let firstRow = 0;
  let tensorByteOffset = 0;
  return Object.freeze({
    name: "token_embd.weight",
    shape: Object.freeze([2_560, 248_320]),
    ggmlType: GgmlType.Q4_K,
    storageType: "q4-k-144",
    rowBytes,
    rowCount: 248_320,
    logicalBytes: BigInt(rowBytes) * 248_320n,
    physicalRows: Object.freeze(input.splits.map((rowCount, index) => {
      const view = Object.freeze({
        buffer: buffers[index]!,
        firstRow,
        rowCount,
        tensorByteOffset,
        bufferByteOffset: 0,
        byteLength: rowCount * rowBytes,
      });
      firstRow += rowCount;
      tensorByteOffset += rowCount * rowBytes;
      return view;
    })),
  });
}

function directory(weight: Qwen35TensorWeightView): Qwen35WeightDirectoryView {
  const byName = new Map([[weight.name, weight]] as const);
  return Object.freeze({
    size: 1,
    tensors: Object.freeze([weight]),
    logicalBytes: weight.logicalBytes,
    allocatedBytes: weight.logicalBytes,
    get: (name: string) => byName.get(name),
    entries: () => byName.entries(),
    [Symbol.iterator]: () => byName[Symbol.iterator](),
  });
}

const RESOURCE_SPECS = {
  "logits-tile": ["f32", 1_024, 0x0088],
  "top-k-scores": ["f32", 256, 0x0088],
  "top-k-indices": ["u32", 256, 0x0088],
  "selected-token": ["u32", 1, 0x008c],
} as const;

function workspace(overrides: Partial<Record<
  keyof typeof RESOURCE_SPECS,
  Partial<Qwen35ActivationResourceView>
>> = {}) {
  const resources = new Map<Qwen35ActivationResourceKind, Qwen35ActivationResourceView>();
  for (const [kind, [scalarType, elementCount, usage]] of Object.entries(
    RESOURCE_SPECS,
  ) as [keyof typeof RESOURCE_SPECS, (typeof RESOURCE_SPECS)[keyof typeof RESOURCE_SPECS]][]) {
    const byteLength = elementCount * 4;
    resources.set(kind, Object.freeze({
      kind,
      scalarType,
      elementCount,
      bytes: BigInt(byteLength),
      usage,
      byteLength,
      binding: Object.freeze({ buffer: {}, offset: 0, size: byteLength }),
      ...overrides[kind],
    }) as Qwen35ActivationResourceView);
  }
  return Object.freeze({
    get(kind: Qwen35ActivationResourceKind): Qwen35ActivationResourceView {
      const resource = resources.get(kind);
      if (resource === undefined) throw new Error("missing test resource");
      return resource;
    },
  });
}

function uniformSlices(count: number, buffer: object = {}) {
  return Array.from({ length: count }, (_, index) => ({
    buffer,
    offset: index * 256,
    byteLength: 20,
  }));
}

function asRequest(request: Qwen35DispatchRequest): Qwen35DispatchRequest {
  return request;
}

test("assembles physical pieces, one winner per tile, and one final token", () => {
  const logitsWorkspace = workspace();
  const normalizedHidden = { buffer: {}, offset: 0, byteLength: 10_240 };
  // The row-1500 split cuts mathematical tile one into two physical pieces.
  const result = assembleQwen35TiledLogitsCommands({
    weights: directory(tensor({ splits: [1_500, 246_820] })),
    normalizedHidden,
    workspace: logitsWorkspace,
    limits,
    uniforms: uniformSlices(488),
  });

  assert.equal(result.uniformCount, 488);
  assert.equal(result.commands.length, 488);
  assert.deepEqual(result.commands.slice(0, 5).map((command) => ({
    kind: command.kind,
    tileIndex: command.tileIndex,
    uniformWords: command.uniformWords,
  })), [
    { kind: "logits-gemv-piece", tileIndex: 0, uniformWords: [1_024, 2_560, 10, 0, 0] },
    { kind: "logits-tile-top-1", tileIndex: 0, uniformWords: [1_024, 0, 0, 0] },
    { kind: "logits-gemv-piece", tileIndex: 1, uniformWords: [476, 2_560, 10, 0, 0] },
    { kind: "logits-gemv-piece", tileIndex: 1, uniformWords: [548, 2_560, 10, 0, 476] },
    { kind: "logits-tile-top-1", tileIndex: 1, uniformWords: [1_024, 1_024, 1, 0] },
  ]);
  const tileWinners = result.commands.filter(
    (command) => command.kind === "logits-tile-top-1",
  );
  assert.equal(tileWinners.length, 243);
  assert.deepEqual(tileWinners.at(-1)?.uniformWords, [262, 247_808, 242, 0]);
  assert.deepEqual(result.commands.at(-1), {
    kind: "indexed-top-1",
    tileIndex: null,
    kernel: result.commands.at(-1)!.kernel,
    bindings: result.commands.at(-1)!.bindings,
    uniformWords: [243, 0, 0, 0],
    workgroups: { x: 1, y: 1, z: 1 },
  });
  assert.equal(asRequest(result.commands[0]!), result.commands[0]);
  assert.equal(asRequest(result.commands.at(-1)!), result.commands.at(-1));

  const logitsTile = logitsWorkspace.get("logits-tile").binding.buffer;
  assert.equal(result.commands[0]!.bindings[2]?.buffer, logitsTile);
  assert.equal(result.commands[2]!.bindings[2]?.buffer, logitsTile);
  assert.equal(result.commands[3]!.bindings[2]?.buffer, logitsTile);
  assert.equal(result.commands[4]!.bindings[0]?.buffer, logitsTile);
  assert.deepEqual(result.selectedTokenReadback, {
    buffer: logitsWorkspace.get("selected-token").binding.buffer,
    offset: 0,
    byteLength: 4,
    scalarType: "u32",
  });
  assert.equal("scoresReadback" in result, false);
  assert.equal("logitsReadback" in result, false);

  assert.equal(Object.isFrozen(result), true);
  assert.equal(Object.isFrozen(result.commands), true);
  assert.equal(result.commands.every(Object.isFrozen), true);
  assert.equal(result.commands.every((command) => Object.isFrozen(command.bindings)), true);
  assert.equal(result.commands.every((command) => Object.isFrozen(command.uniformWords)), true);
  assert.equal(Object.isFrozen(result.selectedTokenReadback), true);
});

test("exposes the exact uniform count without caller-owned uniform buffers", () => {
  assert.equal(planQwen35TiledLogitsUniformCount({
    weights: directory(tensor({ splits: [1_500, 246_820] })),
    limits,
  }), 488);
  assert.equal(planQwen35TiledLogitsUniformCount({
    weights: directory(tensor({ splits: [248_320] })),
    limits,
  }), 487);
});

test("assembles one staged Q6_K logits tile with score and token readback", async () => {
  const subject = await stagedLogitsSubject();
  assert.equal(
    typeof subject.assembleQwen35StagedLogitsTileCommands,
    "function",
    "the disk-backed output path requires a real staged-tile scorer",
  );
  const packedTile = {};
  const candidateOutput = {};
  const logitsWorkspace = workspace();
  const result = subject.assembleQwen35StagedLogitsTileCommands({
    tile: {
      tensorName: "token_embd.weight",
      storageType: "q6-k-212",
      firstRow: 247_808,
      rowCount: 262,
      rowBytes: 2_120,
      buffer: packedTile,
      bufferOffset: 0,
      byteLength: 2_120 * 262,
    },
    normalizedHidden: { buffer: {}, offset: 0, byteLength: 10_240 },
    workspace: logitsWorkspace,
    candidateOutput: { buffer: candidateOutput, offset: 0, byteLength: 8 },
    limits,
    uniforms: uniformSlices(2),
  });

  assert.equal(result.uniformCount, 2);
  assert.deepEqual(result.commands.map(({ kind, tileIndex, uniformWords }) => ({
    kind,
    tileIndex,
    uniformWords,
  })), [
    {
      kind: "logits-gemv-piece",
      tileIndex: 242,
      uniformWords: [262, 2_560, 10, 0, 0],
    },
    {
      kind: "logits-tile-top-1",
      tileIndex: 242,
      uniformWords: [262, 247_808, 0, 0],
    },
  ]);
  assert.deepEqual(result.commands[0]!.bindings[0], {
    binding: 0,
    kind: "storage",
    buffer: packedTile,
    offset: 0,
    size: 2_120 * 262,
  });
  assert.deepEqual(result.candidateScoreReadback, {
    buffer: candidateOutput,
    offset: 0,
    byteLength: 4,
    scalarType: "f32",
  });
  assert.deepEqual(result.candidateTokenReadback, {
    buffer: candidateOutput,
    offset: 4,
    byteLength: 4,
    scalarType: "u32",
  });
});

test("assembles staged logits into GPU candidate slots without per-tile readbacks", async () => {
  const subject = await stagedLogitsSubject();
  const packedTile = {};
  const candidateOutput = {};
  const result = subject.assembleQwen35StagedLogitsTileGpuCommands({
    tile: {
      tensorName: "token_embd.weight",
      storageType: "q6-k-212",
      firstRow: 247_808,
      rowCount: 262,
      rowBytes: 2_120,
      buffer: packedTile,
      bufferOffset: 0,
      byteLength: 2_120 * 262,
    },
    normalizedHidden: { buffer: {}, offset: 0, byteLength: 10_240 },
    workspace: workspace(),
    candidateOutput: { buffer: candidateOutput, offset: 0, byteLength: 256 * 8 },
    candidateSlot: 242,
    limits,
    uniforms: uniformSlices(2),
  });

  assert.equal(result.uniformCount, 2);
  const reduction = result.commands.at(-1)!;
  assert.equal(reduction.kind, "logits-tile-top-1");
  assert.equal(reduction.tileIndex, 242);
  assert.deepEqual(reduction.uniformWords, [262, 247_808, 242, 0]);
  assert.match(reduction.kernel.source, /@workgroup_size\(64\)/);
  assert.match(reduction.kernel.source, /workgroupBarrier\(\)/);
  assert.equal("candidateScoreReadback" in result, false);
  assert.equal("candidateTokenReadback" in result, false);
});

test("assembles one final staged token reduction command", async () => {
  const subject = await stagedLogitsSubject();
  const candidateOutput = {};
  const selectedToken = {};
  const command = subject.assembleQwen35StagedFinalTokenCommand({
    candidateOutput: { buffer: candidateOutput, offset: 0, byteLength: 256 * 8 },
    selectedToken: { buffer: selectedToken, offset: 0, byteLength: 4 },
    limits,
    uniform: uniformSlices(1)[0]!,
  });

  assert.equal(command.kind, "indexed-top-1");
  assert.equal(command.tileIndex, null);
  assert.deepEqual(command.uniformWords, [243, 0, 0, 0]);
  assert.match(command.kernel.source, /@workgroup_size\(64\)/);
  assert.match(command.kernel.source, /workgroupBarrier\(\)/);
});

test("rejects missing, extra, and aliased uniform slots", () => {
  const input = {
    weights: directory(tensor({ splits: [1_500, 246_820] })),
    normalizedHidden: { buffer: {}, offset: 0, byteLength: 10_240 },
    workspace: workspace(),
    limits,
  } as const;
  for (const count of [487, 489]) {
    assert.throws(
      () => assembleQwen35TiledLogitsCommands({
        ...input,
        uniforms: uniformSlices(count),
      }),
      { code: "logits-dispatch-uniform-count-invalid" },
    );
  }

  const aliased = uniformSlices(488);
  aliased[487] = aliased[0]!;
  assert.throws(
    () => assembleQwen35TiledLogitsCommands({ ...input, uniforms: aliased }),
    { code: "logits-dispatch-uniform-alias-invalid" },
  );

  const partiallyOverlapping = Array.from({ length: 488 }, (_, index) => ({
    buffer: aliased[0]!.buffer,
    offset: index === 487 ? 4 : index * 32,
    byteLength: 20,
  }));
  assert.throws(
    () => assembleQwen35TiledLogitsCommands({
      ...input,
      uniforms: partiallyOverlapping,
      limits: { ...limits, minUniformBufferOffsetAlignment: 4 },
    }),
    { code: "logits-dispatch-uniform-alias-invalid" },
  );
});

test("rejects workspace drift and cross-resource write aliases", () => {
  const base = {
    weights: directory(tensor({ splits: [248_320] })),
    normalizedHidden: { buffer: {}, offset: 0, byteLength: 10_240 },
    limits,
    uniforms: uniformSlices(487),
  } as const;
  assert.throws(
    () => assembleQwen35TiledLogitsCommands({
      ...base,
      workspace: workspace({
        "top-k-indices": { scalarType: "f32" },
      }),
    }),
    { code: "logits-dispatch-workspace-invalid" },
  );

  const shared = {};
  assert.throws(
    () => assembleQwen35TiledLogitsCommands({
      ...base,
      workspace: workspace({
        "top-k-scores": {
          binding: Object.freeze({ buffer: shared, offset: 0, size: 1_024 }),
        },
        "top-k-indices": {
          binding: Object.freeze({ buffer: shared, offset: 0, size: 1_024 }),
        },
      }),
    }),
    { code: "logits-dispatch-buffer-alias-invalid" },
  );

  const hiddenBuffer = {};
  assert.throws(
    () => assembleQwen35TiledLogitsCommands({
      ...base,
      normalizedHidden: { buffer: hiddenBuffer, offset: 0, byteLength: 10_240 },
      workspace: workspace({
        "top-k-scores": {
          binding: Object.freeze({ buffer: hiddenBuffer, offset: 0, size: 1_024 }),
        },
      }),
    }),
    { code: "logits-dispatch-buffer-alias-invalid" },
  );

  const sharedWeightWorkspace = {};
  assert.throws(
    () => assembleQwen35TiledLogitsCommands({
      ...base,
      weights: directory(tensor({
        splits: [248_320],
        buffers: [sharedWeightWorkspace],
      })),
      workspace: workspace({
        "top-k-scores": {
          binding: Object.freeze({
            buffer: sharedWeightWorkspace,
            offset: 0,
            size: 1_024,
          }),
        },
      }),
    }),
    { code: "logits-dispatch-buffer-alias-invalid" },
  );

  const sharedWeightUniform = {};
  assert.throws(
    () => assembleQwen35TiledLogitsCommands({
      ...base,
      weights: directory(tensor({
        splits: [248_320],
        buffers: [sharedWeightUniform],
      })),
      workspace: workspace(),
      uniforms: uniformSlices(487, sharedWeightUniform),
    }),
    { code: "logits-dispatch-buffer-alias-invalid" },
  );
});

test("rejects noncontiguous groups and invalid completion markers", () => {
  const assembled = assembleQwen35TiledLogitsCommands({
    weights: directory(tensor({ splits: [1_500, 246_820] })),
    normalizedHidden: { buffer: {}, offset: 0, byteLength: 10_240 },
    workspace: workspace(),
    limits,
    uniforms: uniformSlices(488),
  });
  const pieces = assembled.commands.filter(
    (command) => command.kind === "logits-gemv-piece",
  ) as Qwen35TiedLogitsDispatchPlan[];

  assert.throws(
    () => validateQwen35TiedLogitsDispatchGroups([
      pieces[0]!,
      { ...pieces[2]!, pieceOutputOffset: 1 },
      ...pieces.slice(3),
    ]),
    { code: "logits-dispatch-group-invalid" },
  );
  assert.throws(
    () => validateQwen35TiedLogitsDispatchGroups([
      { ...pieces[0]!, completesTile: false },
      ...pieces.slice(1),
    ]),
    { code: "logits-dispatch-group-invalid" },
  );
  assert.throws(
    () => validateQwen35TiedLogitsDispatchGroups([
      pieces[0]!,
      { ...pieces[1]!, completesTile: true },
      ...pieces.slice(2),
    ]),
    { code: "logits-dispatch-group-invalid" },
  );
  assert.throws(
    () => validateQwen35TiedLogitsDispatchGroups([
      pieces[0]!,
      { ...pieces[0]! },
      ...pieces.slice(1),
    ]),
    { code: "logits-dispatch-group-invalid" },
  );
});
